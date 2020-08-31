//! Test `boltzmann-solver` to calculate the baryon asymmetry generated through
//! leptogenesis in the standard type-I seesaw.

mod common;
mod model;

use crate::model::{interaction, LeptogenesisModel};
use boltzmann_solver::prelude::*;
use common::{into_interaction_box, one_generation};
#[cfg(not(debug_assertions))]
use itertools::iproduct;
use ndarray::prelude::*;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::{error, fmt, fs, io, sync::RwLock};

/// Common initialization function, used to setup common logging for all
/// functions.
fn init() {
    common::setup_logging(0);
}

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
    builder = builder.in_equilibrium(
        [
            LeptogenesisModel::static_particle_idx("A", 0).unwrap(),
            LeptogenesisModel::static_particle_idx("W", 0).unwrap(),
            LeptogenesisModel::static_particle_idx("G", 0).unwrap(),
        ]
        .iter()
        .cloned(),
    );
    builder = builder.no_asymmetry(
        [
            LeptogenesisModel::static_particle_idx("A", 0).unwrap(),
            LeptogenesisModel::static_particle_idx("W", 0).unwrap(),
            LeptogenesisModel::static_particle_idx("G", 0).unwrap(),
        ]
        .iter()
        .cloned(),
    );

    // Add the logger if needed
    if let Some(mut csv) = csv {
        // If we have a CSV file to write to, track the number densities as they
        // evolve.

        // Write the headers
        csv.write_field("step")?;
        csv.write_field("beta")?;
        for name in names {
            csv.write_field(format!("{}", name))?;
            csv.write_field(format!("Δ{}", name))?;
            csv.write_field(format!("({})", name))?;
        }
        csv.write_record(None::<&[u8]>)?;

        // Wrap the CSV into a RefCell as the logger is shared across threads
        // and offers mutability to the CSV.
        let csv = RwLock::new(csv);

        builder = builder.logger(move |c| {
            // Write out the current step and inverse temperature
            let mut csv = csv.write().unwrap();
            csv.write_field(format!("{}", c.step)).unwrap();
            csv.write_field(format!("{:e}", c.beta)).unwrap();
            // Write all the number densities
            for i in 0..c.n.len() {
                csv.write_field(format!("{:e}", c.n[i])).unwrap();
                csv.write_field(format!("{:e}", c.na[i])).unwrap();
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
        builder = builder.logger(|c| {
            if c.n.iter().chain(c.na.iter()).any(|n| !n.is_finite()) {
                log::error!("Non-finite number density at step {}.", c.step);
                panic!("Non-finite number density at step {}.", c.step)
            }
        });
    }

    // Build and run the solver
    let mut solver = builder.build()?;
    Ok(solver.solve())
}

#[test]
fn particle_indices() {
    init();

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

/// Test the effects of the right-handed neutrino decay on its own in the
/// 1-generation case.
#[test]
pub fn decay_1() -> Result<(), Box<dyn error::Error>> {
    init();

    // Create the CSV file
    let output_dir = common::output_dir("full");
    let csv = csv::Writer::from_path(output_dir.join("decay_1.csv"))?;

    // Get the solution
    let mut model = LeptogenesisModel::zero();
    for i in &[
        interaction::hln,
        interaction::hhw,
        interaction::hha,
        interaction::ffa,
        interaction::ffw,
        interaction::ffg,
    ] {
        model.interactions.extend(
            i().drain(..)
                .filter(one_generation)
                .map(into_interaction_box),
        );
    }
    for i in &[interaction::hhww, interaction::hhaa, interaction::hhaw] {
        model.interactions.extend(
            i().drain(..)
                .filter(one_generation)
                .map(into_interaction_box),
        );
    }

    // Collect the names now as SolverBuilder takes ownership of the model
    // later.
    let names: Vec<_> = model.particles().iter().map(|p| p.name.clone()).collect();

    let builder = SolverBuilder::new().model(model).beta_range(1e-17, 1e-3);

    let (n, na) = solve(builder, &names, Some(csv))?;

    // Check that the solution is fine
    println!("Final number density: {:.3e}", n);
    println!("Final number density asymmetry: {:.3e}", na);

    let nai = na[LeptogenesisModel::static_particle_idx("L", 0).unwrap()].abs();
    assert!(1e-14 < nai && nai < 1e-10);
    assert!(n[LeptogenesisModel::static_particle_idx("N", 0).unwrap()] < 1e-8);

    Ok(())
}

/// Test the effects of the right-handed neutrino decay on its own in the
/// 3-generation case.
#[test]
pub fn decay_3() -> Result<(), Box<dyn error::Error>> {
    init();

    // Create the CSV file
    let output_dir = common::output_dir("full");
    let csv = csv::Writer::from_path(output_dir.join("decay_3.csv"))?;

    // Get the solution
    let mut model = LeptogenesisModel::zero();
    for i in &[
        interaction::hln,
        interaction::hhw,
        interaction::hha,
        interaction::ffa,
        interaction::ffw,
        interaction::ffg,
    ] {
        model
            .interactions
            .extend(i().drain(..).map(common::into_interaction_box));
    }
    for i in &[interaction::hhww, interaction::hhaa, interaction::hhaw] {
        model.interactions.extend(
            i().drain(..)
                .filter(one_generation)
                .map(into_interaction_box),
        );
    }

    // Collect the names now as SolverBuilder takes ownership of the model
    // later.
    let names: Vec<_> = model.particles().iter().map(|p| p.name.clone()).collect();

    let builder = SolverBuilder::new().model(model).beta_range(1e-17, 1e-3);

    let (n, na) = solve(builder, &names, Some(csv))?;

    // Check that the solution is fine
    println!("Final number density: {:.3e}", n);
    println!("Final number density asymmetry: {:.3e}", na);
    let mut lepton_asymmetry = 0.0;
    for i in 0..3 {
        lepton_asymmetry += na[LeptogenesisModel::static_particle_idx("L", i).unwrap()].abs();
        assert!(n[LeptogenesisModel::static_particle_idx("N", i).unwrap()] < 1e-8);
    }
    assert!(1e-10 < lepton_asymmetry.abs() && lepton_asymmetry.abs() < 1e-6);

    Ok(())
}

/// Test the effects of a washout term on its own in the 1-generation case.
#[test]
#[cfg(not(debug_assertions))]
pub fn washout_1() -> Result<(), Box<dyn error::Error>> {
    init();

    // Create the CSV file
    let output_dir = common::output_dir("full");
    let csv = csv::Writer::from_path(output_dir.join("washout_1.csv"))?;

    // Get the solution
    let mut model = LeptogenesisModel::zero();
    for i in &[
        interaction::hha,
        interaction::hhw,
        interaction::ffa,
        interaction::ffw,
        interaction::ffg,
    ] {
        model.interactions.extend(
            i().drain(..)
                .filter(one_generation)
                .map(common::into_interaction_box),
        );
    }
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
        model.interactions.extend(
            i().drain(..)
                .filter(one_generation)
                .map(into_interaction_box),
        );
    }

    // Collect the names now as SolverBuilder takes ownership of the model
    // later.
    let names: Vec<_> = model.particles().iter().map(|p| p.name.clone()).collect();

    let builder = SolverBuilder::new()
        .initial_asymmetries(vec![(
            LeptogenesisModel::static_particle_idx("L", 0).unwrap(),
            1e-2,
        )])
        .model(model)
        .beta_range(1e-17, 1e-3);

    let (n, na) = solve(builder, &names, Some(csv))?;

    // Check that the solution is fine
    println!("Final number density: {:.3e}", n);
    println!("Final number density asymmetry: {:.3e}", na);

    // FIXME
    // let nai = na[LeptogenesisModel::static_particle_idx("L", 0).unwrap()].abs();
    // assert!(nai < 1e-5);

    Ok(())
}

/// Test the effects of a washout terms on their own in the 3-generation case.
#[test]
#[cfg(not(debug_assertions))]
pub fn washout_3() -> Result<(), Box<dyn error::Error>> {
    init();

    // Create the CSV file
    let output_dir = common::output_dir("full");
    let csv = csv::Writer::from_path(output_dir.join("washout_3.csv"))?;

    // Get the solution
    let mut model = LeptogenesisModel::zero();
    for i in &[
        interaction::hha,
        interaction::hhw,
        interaction::ffa,
        interaction::ffw,
        interaction::ffg,
    ] {
        model
            .interactions
            .extend(i().drain(..).map(common::into_interaction_box));
    }
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
            .extend(i().drain(..).map(common::into_interaction_box));
    }

    // Collect the names now as SolverBuilder takes ownership of the model
    // later.
    let names: Vec<_> = model.particles().iter().map(|p| p.name.clone()).collect();

    let builder = SolverBuilder::new()
        .initial_asymmetries(vec![
            (
                LeptogenesisModel::static_particle_idx("L", 0).unwrap(),
                1e-2,
            ),
            (
                LeptogenesisModel::static_particle_idx("L", 1).unwrap(),
                2e-2,
            ),
            (
                LeptogenesisModel::static_particle_idx("L", 2).unwrap(),
                3e-2,
            ),
        ])
        .model(model)
        .beta_range(1e-17, 1e-3);

    let (n, na) = solve(builder, &names, Some(csv))?;

    // Check that the solution is fine
    println!("Final number density: {:.3e}", n);
    println!("Final number density asymmetry: {:.3e}", na);

    // FIXME
    // let nai = na[LeptogenesisModel::static_particle_idx("L", 0).unwrap()].abs();
    // assert!(nai < 1e-5);

    Ok(())
}

#[test]
#[cfg(not(debug_assertions))]
pub fn full_1() -> Result<(), Box<dyn error::Error>> {
    init();

    // Create the CSV files
    let output_dir = common::output_dir("full");
    let csv = csv::Writer::from_path(output_dir.join("full_1.csv"))?;

    // Get the solution
    let mut model = LeptogenesisModel::zero();
    for i in &[
        interaction::hqu,
        interaction::hqd,
        interaction::hle,
        interaction::hln,
        interaction::hhw,
        interaction::hha,
        interaction::ffa,
        interaction::ffw,
        interaction::ffg,
    ] {
        model.interactions.extend(
            i().drain(..)
                .filter(one_generation)
                .map(into_interaction_box),
        );
    }
    for i in &[
        interaction::hhww,
        interaction::hhaa,
        interaction::hhaw,
        interaction::hhll1,
        // interaction::hhll2, // problematic
        // interaction::hhen, // problematic
        // interaction::nhla, // problematic
        // interaction::nhlw, // problematic
        interaction::quln,
        interaction::qdln,
        // interaction::leln, // problematic
        interaction::lnln,
    ] {
        model.interactions.extend(
            i().drain(..)
                .filter(one_generation)
                .map(into_interaction_box),
        );
    }

    // Collect the names now as SolverBuilder takes ownership of the model
    // later.
    let names: Vec<_> = model.particles().iter().map(|p| p.name.clone()).collect();

    let builder = SolverBuilder::new().model(model).beta_range(1e-17, 1e-3);

    let (n, na) = solve(builder, &names, Some(csv))?;

    // Check that the solution is fine
    println!("Final number density: {:.3e}", n);
    println!("Final number density asymmetry: {:.3e}", na);

    let nai = na[LeptogenesisModel::static_particle_idx("L", 0).unwrap()].abs();
    assert!(1e-10 < nai && nai < 1e-5);
    assert!(n[LeptogenesisModel::static_particle_idx("N", 0).unwrap()] < 1e-8);

    Ok(())
}

#[test]
#[cfg(not(debug_assertions))]
pub fn full_3() -> Result<(), Box<dyn error::Error>> {
    init();

    // Create the CSV file
    let output_dir = common::output_dir("full");
    let csv = csv::Writer::from_path(output_dir.join("full_3.csv"))?;

    // Get the solution
    let mut model = LeptogenesisModel::zero();
    for i in &[
        interaction::hln,
        interaction::hhw,
        interaction::hha,
        interaction::ffa,
        interaction::ffw,
        interaction::ffg,
    ] {
        model
            .interactions
            .extend(i().drain(..).map(common::into_interaction_box));
    }
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
        model.interactions.extend(
            i().drain(..)
                .filter(one_generation)
                .map(into_interaction_box),
        );
    }

    // Collect the names now as SolverBuilder takes ownership of the model
    // later.
    let names: Vec<_> = model.particles().iter().map(|p| p.name.clone()).collect();

    let builder = SolverBuilder::new().model(model).beta_range(1e-17, 1e-3);

    let (n, na) = solve(builder, &names, Some(csv))?;

    // Check that the solution is fine
    println!("Final number density: {:.3e}", n);
    println!("Final number density asymmetry: {:.3e}", na);

    let nai = na[LeptogenesisModel::static_particle_idx("L", 0).unwrap()].abs();
    assert!(1e-10 < nai && nai < 1e-5);
    for i in 0..3 {
        let nai = na[LeptogenesisModel::static_particle_idx("L", i).unwrap()].abs();
        assert!(1e-10 < nai && nai < 1e-5);
        assert!(n[LeptogenesisModel::static_particle_idx("N", i).unwrap()] < 1e-8);
    }

    Ok(())
}

#[cfg(feature = "serde")]
#[test]
fn evolution() -> Result<(), Box<dyn error::Error>> {
    init();

    // Create the CSV files
    let output_dir = common::output_dir("full");

    let mut data = Vec::new();

    for &beta in Array1::geomspace(1e-17, 1e-2, 1024).unwrap().into_iter() {
        let mut model = LeptogenesisModel::zero();
        for i in &[
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
            model
                .interactions
                .extend(i().drain(..).map(common::into_interaction_box));
        }
        model.set_beta(beta);
        model.update_widths();
        data.push(model);
    }

    serde_json::to_writer(
        io::BufWriter::new(fs::File::create(output_dir.join("evolution.json"))?),
        &data,
    )?;

    Ok(())
}

/// Test that gauge couplings keep the Higgs in equilibrium
#[test]
pub fn higgs_equilibrium() -> Result<(), Box<dyn error::Error>> {
    init();

    // Create the CSV file
    let output_dir = common::output_dir("full/higgs_equilibrium");

    let v: Vec<(usize, f64)> = Array1::geomspace(1e-17, 1e-4, 50)
        .unwrap()
        .iter()
        .cloned()
        .enumerate()
        .collect();

    #[cfg(feature = "parallel")]
    let iter = v.into_par_iter();
    #[cfg(not(feature = "parallel"))]
    let iter = v.into_iter();

    iter.for_each(|(i, beta)| {
        let csv = csv::Writer::from_path(output_dir.join(format!("{:03}.csv", i))).unwrap();

        let mut model = LeptogenesisModel::zero();
        for i in &[interaction::hha, interaction::hhw] {
            model
                .interactions
                .extend(i().drain(..).map(into_interaction_box));
        }
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
            .beta_range(beta, beta * 10.0)
            .initial_densities(vec![(p_h, 1.1 * n_eq)]);

        solve(builder, &names, Some(csv)).unwrap();
    });

    Ok(())
}

/// Test that gauge couplings keep the lepton doublets in equilibrium
#[test]
pub fn lepton_equilibrium() -> Result<(), Box<dyn error::Error>> {
    init();

    // Create the CSV file
    let output_dir = common::output_dir("full/lepton_equilibrium");

    let v: Vec<(usize, f64)> = Array1::geomspace(1e-17, 1e-4, 50)
        .unwrap()
        .iter()
        .cloned()
        .enumerate()
        .collect();

    #[cfg(feature = "parallel")]
    let iter = v.into_par_iter();
    #[cfg(not(feature = "parallel"))]
    let iter = v.into_iter();

    iter.for_each(|(i, beta)| {
        let csv = csv::Writer::from_path(output_dir.join(format!("{:03}.csv", i))).unwrap();

        let mut model = LeptogenesisModel::zero();
        for i in &[interaction::ffa, interaction::ffw, interaction::ffg] {
            model.interactions.extend(
                i().drain(..)
                    .filter(one_generation)
                    .map(into_interaction_box),
            );
        }

        let p_l = model.particle_idx("L", 0).unwrap();
        let n_eq = model.particles()[p_l].normalized_number_density(0.0, beta);

        // Collect the names now as SolverBuilder takes ownership of the model
        // later.
        let names: Vec<_> = model.particles().iter().map(|p| p.name.clone()).collect();

        let builder = SolverBuilder::new()
            .model(model)
            .beta_range(beta, beta * 10.0)
            .initial_densities(vec![(p_l, 1.1 * n_eq)]);

        solve(builder, &names, Some(csv)).unwrap();
    });

    Ok(())
}

#[test]
fn gammas() -> Result<(), Box<dyn error::Error>> {
    init();

    let output_dir = common::output_dir("full");

    // Create two copies of the model for both solvers
    let mut models = vec![LeptogenesisModel::zero(), LeptogenesisModel::zero()];
    for model in &mut models {
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
            model.interactions.extend(
                i().drain(..)
                    // .filter(one_generation)
                    .map(into_interaction_box),
            );
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
            model.interactions.extend(
                i().drain(..)
                    // .filter(one_generation)
                    .map(into_interaction_box),
            );
        }
    }

    let mut solvers: Vec<_> = models
        .into_iter()
        .enumerate()
        .map(|(i, model)| {
            SolverBuilder::new()
                .model(model)
                .beta_range(1e-18, 1e-2)
                .precompute(i == 0)
                .build()
                .unwrap()
        })
        .collect();

    let mut csv = csv::Writer::from_path({
        let mut path = output_dir.clone();
        path.push("gamma.csv");
        path
    })?;
    let (header, data) = solvers[0].gammas(1024, true);
    csv.serialize(header)?;
    for row in data.axis_iter(Axis(0)) {
        csv.serialize(row.as_slice().unwrap())?;
    }

    let mut csv = csv::Writer::from_path({
        let mut path = output_dir;
        path.push("asymmetry.csv");
        path
    })?;
    let (header, data) = solvers[0].asymmetries(1024, true);
    csv.serialize(header)?;
    for row in data.axis_iter(Axis(0)) {
        csv.serialize(row.as_slice().unwrap())?;
    }

    // let mut csv = csv::Writer::from_path({
    //     let mut path = output_dir.clone();
    //     path.push("gamma_raw.csv");
    //     path
    // })?;
    // let (header, data) = solvers[1].gammas(1024, true);
    // csv.serialize(header)?;
    // for row in data.axis_iter(Axis(0)) {
    //     csv.serialize(row.as_slice().unwrap())?;
    // }

    // let mut csv = csv::Writer::from_path({
    //     let mut path = output_dir;
    //     path.push("asymmetry.csv");
    //     path
    // })?;
    // let (header, data) = solvers[1].asymmetries(1024, true);
    // csv.serialize(header)?;
    // for row in data.axis_iter(Axis(0)) {
    //     csv.serialize(row.as_slice().unwrap())?;
    // }

    Ok(())
}
