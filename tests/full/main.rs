//! Test `boltzmann-solver` to calculate the baryon asymmetry generated through
//! leptogenesis in the standard type-I seesaw.

#![allow(clippy::single_element_loop)]

mod common;
mod model;

use crate::model::{interaction, LeptogenesisModel};
#[allow(unused_imports)]
use boltzmann_solver::{
    model::interaction::{FourParticle, ThreeParticle},
    prelude::*,
};
use common::{into_interaction_box, n1f1, n3f1, n3f3};
use ndarray::prelude::*;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::{error, fmt, fs, io, sync::RwLock};

/// Solve the Boltzmann equations and return the final values.
///
/// The model function is specified by `model`, and optionally a CSV writer can
/// be given to record the progress of Boltzmann equations.
pub fn solve<W, S>(
    mut builder: SolverBuilder<LeptogenesisModel>,
    names: &[S],
    csv: Option<csv::Writer<W>>,
) -> Result<(Array1<f64>, Array1<f64>), Box<dyn error::Error>>
where
    W: io::Write + 'static,
    S: AsRef<str> + fmt::Display,
{
    // Set equilibrium conditions for the vector bosons.
    builder = builder
        .in_equilibrium([
            LeptogenesisModel::static_particle_idx("A", 0).unwrap(),
            LeptogenesisModel::static_particle_idx("W", 0).unwrap(),
            LeptogenesisModel::static_particle_idx("G", 0).unwrap(),
        ])
        .no_asymmetry([
            LeptogenesisModel::static_particle_idx("A", 0).unwrap(),
            LeptogenesisModel::static_particle_idx("W", 0).unwrap(),
            LeptogenesisModel::static_particle_idx("G", 0).unwrap(),
        ]);

    // Add the logger if needed
    if let Some(mut csv) = csv {
        // If we have a CSV file to write to, track the number densities as they
        // evolve.

        // Write the headers
        csv.write_field("step")?;
        csv.write_field("beta")?;
        for name in names {
            csv.write_field(format!("n-{}", name))?;
            csv.write_field(format!("dn-{}", name))?;
            csv.write_field(format!("na-{}", name))?;
            csv.write_field(format!("dna-{}", name))?;
            csv.write_field(format!("eq-{}", name))?;
        }
        csv.write_record(None::<&[u8]>)?;

        // Wrap the CSV into a RefCell as the logger is shared across threads
        // and offers mutability to the CSV.
        let csv = RwLock::new(csv);

        builder = builder.logger(move |c, dn, dna| {
            // Write out the current step and inverse temperature
            let mut csv = csv.write().unwrap();
            csv.write_field(format!("{}", c.step)).unwrap();
            csv.write_field(format!("{:e}", c.beta)).unwrap();
            // Write all the number densities
            for i in 0..c.n.len() {
                csv.write_field(format!("{:e}", c.n[i])).unwrap();
                csv.write_field(format!("{:e}", dn[i])).unwrap();
                csv.write_field(format!("{:e}", c.na[i])).unwrap();
                csv.write_field(format!("{:e}", dna[i])).unwrap();
                csv.write_field(format!("{:e}", c.eq[i])).unwrap();
            }
            csv.write_record(None::<&[u8]>).unwrap();

            if c.n.iter().chain(c.na.iter()).any(|n| !n.is_finite()) {
                csv.flush().unwrap();
                log::error!("Non-finite number density at step {}.", c.step);
                panic!("Non-finite number density at step {}.", c.step)
            }
        });
    } else {
        // If we're not tracking the number density, simply make sure that all
        // the number densities are always finite, or quite.
        builder = builder.logger(|c, _dn, _dna| {
            if c.n.iter().chain(c.na.iter()).any(|n| !n.is_finite()) {
                log::error!("Non-finite number density at step {}.", c.step);
                panic!("Non-finite number density at step {}.", c.step)
            }
        });
    }

    // builder = builder.step_precision(1e-3, 1e0);

    // Build and run the solver
    let mut solver = builder.build()?;
    Ok(solver.solve()?)
}

#[test]
fn particle_indices() {
    common::setup_logging(1);

    let model = LeptogenesisModel::zero();

    for (i, p) in model.particles().iter().enumerate() {
        let name = &p.name;
        if name.len() == 1 {
            assert_eq!(Ok(i), LeptogenesisModel::static_particle_idx(name, 0));
            assert_eq!(Ok(i), model.particle_idx(name, 0));
        } else if name.len() == 2 {
            let mut chars = name.chars();
            let head = chars.next().unwrap();
            let idx = chars.next().unwrap() as usize - 49;
            assert_eq!(
                Ok(i),
                LeptogenesisModel::static_particle_idx(&head.to_string(), idx)
            );
            assert_eq!(Ok(i), model.particle_idx(&head.to_string(), idx));
        }
    }
}

/// Run a particular interaction with the specified `$name` and outputting data
/// into `$csv`.  The three- and four-particle interactions are specified in
/// `$interactions3` and `$interactions4` respectively.
///
/// The `$filter` is a function used to filter the interactions (e.g. one
/// generation or not).
///
/// A pre-function that takes in and returns the [`ModelBuilder`] instance can
/// be given in `$pre` to modify initial conditions.  A check on the final
/// number densities can done in the function `$post`.
macro_rules! run {
    {$name:ident, $csv:expr, $filter:expr, $interactions3:expr, $interactions4:expr} => {
        run!{$name, $csv,$filter, $interactions3, $interactions4, std::convert::identity, |_n, _na| {}}
    };
    {$name:ident, $csv:expr, $filter:expr, $interactions3:expr, $interactions4:expr, $post:expr} => {
        run!{$name, $csv,$filter, $interactions3, $interactions4, std::convert::identity, $post}
    };
    {$name:ident, $csv:expr, $filter:expr, $interactions3:expr, $interactions4:expr, $pre:expr, $post:expr} => {
        #[test]
        #[allow(unused_variables)]
        pub fn $name() -> Result<(), Box<dyn error::Error>> {
            common::setup_logging(2);

            // Create the CSV file
            let output_dir = common::output_dir("full");
            let csv = csv::Writer::from_path(output_dir.join(concat!($csv, ".", "csv")))?;

            let mut model = LeptogenesisModel::zero();
            for i in $interactions3 {
                model
                    .interactions
                    .extend(i().drain(..).filter($filter).map(into_interaction_box));
            }
            for i in $interactions4 {
                model
                    .interactions
                    .extend(i().drain(..).filter($filter).map(into_interaction_box));
            }

            // Collect the names now as SolverBuilder takes ownership of the
            // model later.
            let names: Vec<_> = model.particles().iter().map(|p| p.name.clone()).collect();

            let mut builder = SolverBuilder::new()
                .model(model)
                .beta_range(1e-12, 1e0)
                .fast_interaction(true);

            builder = $pre(builder);

            let (n, na) = solve(builder, &names, Some(csv))?;

            log::info!("Final number density:\n{:.3e}", n);
            log::info!("Final number density asymmetry:\n{:.3e}", na);

            $post(n, na);

            Ok(())
        }
    };
}

/// Runs the same as the [`run!`] macro with the three filters: [`n1f1`],
/// [`n3f1`], [`n3f3`].
macro_rules! run_all {
    {$name:ident, $interactions3:expr, $interactions4:expr} => {
        run_all!{$name, $interactions3, $interactions4, std::convert::identity, |_n, _na| {}}
    };
    {$name:ident, $interactions3:expr, $interactions4:expr, $post:expr} => {
        run_all!{$name, $interactions3, $interactions4, std::convert::identity, $post}
    };
    {$name:ident, $interactions3:expr, $interactions4:expr, $pre:expr, $post:expr} => {
        mod $name {
            use super::*;

            run! {
                n1f1,
                concat!(stringify!($name), "_n1f1"),
                super::n1f1,
                $interactions3,
                $interactions4, $pre, $post
            }
            run! {
                n3f1,
                concat!(stringify!($name), "_n3f1"),
                super::n3f1,
                $interactions3,
                $interactions4, $pre, $post
            }
            run! {
                n3f3,
                concat!(stringify!($name), "_n3f3"),
                super::n3f3,
                $interactions3,
                $interactions4, $pre, $post
            }
        }
    };
}

run_all! {
    decay,
    [
        interaction::hln,
        // interaction::ffa,
        // interaction::ffw,
        // interaction::ffg,
    ],
    // [
    //     interaction::hhww,
    //     interaction::hhaa,
    //     interaction::hhaw,
    // ]
    std::iter::empty::<&dyn Fn() -> Vec<FourParticle<LeptogenesisModel>>>()
}

// #[cfg(not(debug_assertions))]
run_all! {
    washout,
    [
        interaction::ffa,
        // interaction::ffw,
        // interaction::ffg,
    ],
    [
        // interaction::hhww,
        // interaction::hhaa,
        // interaction::hhaw,
        interaction::hhll1,
        // interaction::hhll2,
        // interaction::hhen,
        // interaction::nhla,
        interaction::nhlw,
        // interaction::quln,
        // interaction::qdln,
        // interaction::leln,
        // interaction::lnln,
    ],
    |builder: SolverBuilder<LeptogenesisModel>| {
        builder.initial_asymmetries([
            (LeptogenesisModel::static_particle_idx("N", 0).unwrap(), 1e-2),
            (LeptogenesisModel::static_particle_idx("N", 1).unwrap(), 1e-3),
            (LeptogenesisModel::static_particle_idx("N", 2).unwrap(), 1e-4),
        ])
    },
    |_n, _na| {}
}

#[cfg(not(debug_assertions))]
run_all! {
    full,
    [
        interaction::hqu,
        interaction::hqd,
        interaction::hle,
        interaction::hln,
        // interaction::hhw,
        // interaction::hha,
        interaction::ffa,
        // interaction::ffw,
        // interaction::ffg,
    ],
    [
        // interaction::hhww,
        // interaction::hhaa,
        interaction::hhaw,
        interaction::hhll1,
        // interaction::hhll2,
        // interaction::hhen,
        // interaction::nhla,
        interaction::nhlw,
        // interaction::quln,
        // interaction::qdln,
        // interaction::leln,
        // interaction::lnln,
    ]
}

#[cfg(feature = "serde")]
#[test]
fn evolution() -> Result<(), Box<dyn error::Error>> {
    common::setup_logging(1);

    // Create the CSV files
    let output_dir = common::output_dir("full");

    let mut data = [Vec::new(), Vec::new(), Vec::new()];
    let filters = [n1f1, n3f1, n3f3];

    for &beta in Array1::geomspace(1e-17, 1e-2, 1024).unwrap().iter() {
        for i in 0..data.len() {
            let mut model = LeptogenesisModel::zero();
            for interaction in [
                interaction::hle,
                interaction::hln,
                interaction::hqu,
                interaction::hqd,
                interaction::hha,
                interaction::hhw,
                interaction::ffa,
                interaction::ffw,
                interaction::ffg,
            ] {
                model.interactions.extend(
                    interaction()
                        .drain(..)
                        .filter(filters[i])
                        .map(common::into_interaction_box),
                );
            }
            model.set_beta(beta);
            model.update_widths();
            data[i].push(model);
        }
    }

    serde_json::to_writer(
        io::BufWriter::new(fs::File::create(output_dir.join("evolution_n1f1.json"))?),
        &data[0],
    )?;
    serde_json::to_writer(
        io::BufWriter::new(fs::File::create(output_dir.join("evolution_n3f1.json"))?),
        &data[1],
    )?;
    serde_json::to_writer(
        io::BufWriter::new(fs::File::create(output_dir.join("evolution_n3f3.json"))?),
        &data[2],
    )?;

    Ok(())
}

/// Test that gauge couplings keep the Higgs in equilibrium
#[test]
pub fn higgs_equilibrium() -> Result<(), Box<dyn error::Error>> {
    common::setup_logging(1);

    // Create the CSV file
    let output_dir = common::output_dir("full/higgs_equilibrium");

    let v = Array1::geomspace(1e-20, 1e-0, 100).unwrap().to_vec();
    #[cfg(feature = "parallel")]
    let iter = v.into_par_iter().enumerate();
    #[cfg(not(feature = "parallel"))]
    let iter = v.into_iter().enumerate();

    iter.for_each(|(i, beta)| {
        let csv = csv::Writer::from_path(output_dir.join(format!("{:03}.csv", i))).unwrap();

        let mut model = LeptogenesisModel::zero();
        // for i in &[interaction::hha, interaction::hhw] {
        //     model
        //         .interactions
        //         .extend(i().drain(..).map(into_interaction_box));
        // }
        for i in &[interaction::hhaa, interaction::hhww, interaction::hhaw] {
            model
                .interactions
                .extend(i().drain(..).map(into_interaction_box));
        }

        let p_h = model.particle_idx("H", 0).unwrap();
        let n_eq = model.particles()[p_h].normalized_number_density(0.0, beta);

        // Collect the names now as SolverBuilder takes ownership of the model
        // later.
        let names: Vec<_> = model.particles().iter().map(|p| p.name.clone()).collect();

        let builder = SolverBuilder::new()
            .model(model)
            .beta_range(beta, beta * 1e3)
            .initial_densities([(p_h, 1.1 * n_eq)])
            .fast_interaction(true);

        solve(builder, &names, Some(csv)).unwrap();
    });

    Ok(())
}

/// Test that gauge couplings keep the lepton doublets in equilibrium
#[test]
pub fn lepton_equilibrium() -> Result<(), Box<dyn error::Error>> {
    common::setup_logging(1);

    // Create the CSV file
    let output_dir = common::output_dir("full/lepton_equilibrium");

    let v = Array1::geomspace(1e-20, 1e-0, 100).unwrap().to_vec();
    #[cfg(feature = "parallel")]
    let iter = v.into_par_iter().enumerate();
    #[cfg(not(feature = "parallel"))]
    let iter = v.into_iter().enumerate();

    iter.for_each(|(i, beta)| {
        let csv = csv::Writer::from_path(output_dir.join(format!("{:03}.csv", i))).unwrap();

        let mut model = LeptogenesisModel::zero();
        for i in &[
            interaction::ffa,
            interaction::ffw,
            // interaction::ffg,
        ] {
            model
                .interactions
                .extend(i().drain(..).filter(n1f1).map(into_interaction_box));
        }

        let p_l = model.particle_idx("L", 0).unwrap();
        let n_eq = model.particles()[p_l].normalized_number_density(0.0, beta);

        // Collect the names now as SolverBuilder takes ownership of the model
        // later.
        let names: Vec<_> = model.particles().iter().map(|p| p.name.clone()).collect();

        let builder = SolverBuilder::new()
            .model(model)
            .beta_range(beta, beta * 1e3)
            .initial_densities([(p_l, 1.1 * n_eq)])
            .fast_interaction(true);

        solve(builder, &names, Some(csv)).unwrap();
    });

    Ok(())
}

#[test]
fn gammas() -> Result<(), Box<dyn error::Error>> {
    common::setup_logging(2);

    let output_dir = common::output_dir("full");

    // Create two copies of the model for both solvers
    let mut model = LeptogenesisModel::zero();
    for i in &[
        interaction::hle,
        interaction::hqu,
        interaction::hqd,
        interaction::hln,
        interaction::hha,
        interaction::hhw,
        interaction::ffa,
        interaction::ffw,
        interaction::ffg,
    ] {
        model
            .interactions
            .extend(i().drain(..).map(into_interaction_box));
    }

    #[cfg(not(debug_assertions))]
    for i in &[
        interaction::hhww,
        interaction::hhaa,
        interaction::hhaw,
        interaction::hhll1,
        interaction::hhll2,
        interaction::hhen,
        interaction::nhla,
        interaction::nhlw,
        interaction::quln,
        interaction::qdln,
        interaction::leln,
        interaction::lnln,
    ] {
        model
            .interactions
            .extend(i().drain(..).filter(n1f1).map(into_interaction_box));
    }

    let mut solver = SolverBuilder::new()
        .model(model)
        .beta_range(1e-14, 1e0)
        .build()
        .unwrap();

    let mut csv = csv::Writer::from_path({
        let mut path = output_dir.clone();
        path.push("gamma.csv");
        path
    })?;
    let (header, data) = solver.gammas(1024, true);
    csv.serialize(header)?;
    for row in data.axis_iter(Axis(0)) {
        csv.serialize(row.as_slice().unwrap())?;
    }

    let mut csv = csv::Writer::from_path({
        let mut path = output_dir;
        path.push("asymmetry.csv");
        path
    })?;
    let (header, data) = solver.asymmetries(1024, true);
    csv.serialize(header)?;
    for row in data.axis_iter(Axis(0)) {
        csv.serialize(row.as_slice().unwrap())?;
    }

    Ok(())
}
