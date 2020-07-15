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
            LeptogenesisModel::particle_idx("A", 0).unwrap(),
            LeptogenesisModel::particle_idx("W", 0).unwrap(),
            LeptogenesisModel::particle_idx("G", 0).unwrap(),
            // LeptogenesisModel::particle_idx("H", 0).unwrap(),
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
            assert_eq!(Ok(i), LeptogenesisModel::particle_idx(name, 0));
        } else if name.len() == 2 {
            let mut chars = name.chars();
            let head = chars.next().unwrap();
            let idx = chars.next().unwrap() as usize - 49;
            assert_eq!(
                Ok(i),
                LeptogenesisModel::particle_idx(&head.to_string(), idx)
            );
        }
    }
}

/// Test the effects of the right-handed neutrino decay on its own in the
/// 1-generation case.
#[test]
pub fn decay_only_1gen() -> Result<(), Box<dyn error::Error>> {
    init();

    // Create the CSV file
    let output_dir = common::output_dir("leptogenesis/decay_only/1gen");
    let csv = csv::Writer::from_path(output_dir.join("n.csv"))?;

    // Get the solution
    let mut model = LeptogenesisModel::zero();
    for i in &[
        interaction::hln,
        interaction::hhw,
        interaction::hha,
        interaction::ffa,
        interaction::ffw,
    ] {
        model.interactions.extend(
            i().drain(..)
                .filter(one_generation)
                .map(into_interaction_box),
        );
    }
    for i in &[interaction::hhww, interaction::hhaa] {
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

    let nai = na[LeptogenesisModel::particle_idx("L", 0).unwrap()].abs();
    assert!(1e-14 < nai && nai < 1e-10);
    assert!(n[LeptogenesisModel::particle_idx("N", 0).unwrap()] < 1e-8);

    Ok(())
}

/// Test the effects of the right-handed neutrino decay on its own in the
/// 3-generation case.
#[test]
pub fn decay_only_3gen() -> Result<(), Box<dyn error::Error>> {
    init();

    // Create the CSV file
    let output_dir = common::output_dir("leptogenesis/decay_only/3gen");
    let csv = csv::Writer::from_path(output_dir.join("n.csv"))?;

    // Get the solution
    let mut model = LeptogenesisModel::zero();
    for i in &[
        interaction::hln,
        interaction::hhw,
        interaction::hha,
        interaction::ffa,
        interaction::ffw,
    ] {
        model
            .interactions
            .extend(i().drain(..).map(common::into_interaction_box));
    }
    for i in &[interaction::hhww, interaction::hhaa] {
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
    for i in 0..3 {
        let nai = na[LeptogenesisModel::particle_idx("L", i).unwrap()].abs();
        assert!(1e-10 < nai && nai < 1e-5);
        assert!(n[LeptogenesisModel::particle_idx("N", i).unwrap()] < 1e-8);
    }

    Ok(())
}

/// Test the effects of a washout term on its own in the 1-generation case.
#[test]
#[cfg(not(debug_assertions))]
pub fn washout_only_1gen() -> Result<(), Box<dyn error::Error>> {
    init();

    // Create the CSV file
    let output_dir = common::output_dir("leptogenesis/washout_only/1gen");
    let csv = csv::Writer::from_path(output_dir.join("n.csv"))?;

    // Get the solution
    let mut model = LeptogenesisModel::zero();
    for i in &[
        interaction::hha,
        interaction::hhw,
        interaction::ffa,
        interaction::ffw,
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
        interaction::hhll1,
        interaction::hhll2,
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
            LeptogenesisModel::particle_idx("L", 0).unwrap(),
            1e-2,
        )])
        .model(model)
        .beta_range(1e-17, 1e-3);

    let (n, na) = solve(builder, &names, Some(csv))?;

    // Check that the solution is fine
    println!("Final number density: {:.3e}", n);
    println!("Final number density asymmetry: {:.3e}", na);

    // FIXME
    // let nai = na[LeptogenesisModel::particle_idx("L", 0).unwrap()].abs();
    // assert!(nai < 1e-5);

    Ok(())
}

/// Test the effects of a washout terms on their own in the 3-generation case.
#[test]
#[cfg(not(debug_assertions))]
pub fn washout_only_3gen() -> Result<(), Box<dyn error::Error>> {
    init();

    // Create the CSV file
    let output_dir = common::output_dir("leptogenesis/washout_only/3gen");
    let csv = csv::Writer::from_path(output_dir.join("n.csv"))?;

    // Get the solution
    let mut model = LeptogenesisModel::zero();
    for i in &[
        interaction::hha,
        interaction::hhw,
        interaction::ffa,
        interaction::ffw,
    ] {
        model
            .interactions
            .extend(i().drain(..).map(common::into_interaction_box));
    }
    for i in &[
        interaction::hhww,
        interaction::hhaa,
        interaction::hhll1,
        interaction::hhll2,
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
            (LeptogenesisModel::particle_idx("L", 0).unwrap(), 1e-2),
            (LeptogenesisModel::particle_idx("L", 1).unwrap(), 2e-2),
            (LeptogenesisModel::particle_idx("L", 2).unwrap(), 3e-2),
        ])
        .model(model)
        .beta_range(1e-17, 1e-3);

    let (n, na) = solve(builder, &names, Some(csv))?;

    // Check that the solution is fine
    println!("Final number density: {:.3e}", n);
    println!("Final number density asymmetry: {:.3e}", na);

    // FIXME
    // let nai = na[LeptogenesisModel::particle_idx("L", 0).unwrap()].abs();
    // assert!(nai < 1e-5);

    Ok(())
}

#[test]
#[cfg(not(debug_assertions))]
pub fn decay_washout_1gen() -> Result<(), Box<dyn error::Error>> {
    init();

    // Create the CSV files
    let output_dir = common::output_dir("leptogenesis/decay_washout/1gen");
    let csv = csv::Writer::from_path(output_dir.join("n.csv"))?;

    // Get the solution
    let mut model = LeptogenesisModel::zero();
    for i in &[
        interaction::hln,
        interaction::hhw,
        interaction::hha,
        interaction::ffa,
        interaction::ffw,
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
        interaction::hhll1,
        interaction::hhll2,
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

    let nai = na[LeptogenesisModel::particle_idx("L", 0).unwrap()].abs();
    assert!(1e-10 < nai && nai < 1e-5);
    assert!(n[LeptogenesisModel::particle_idx("N", 0).unwrap()] < 1e-8);

    Ok(())
}

#[test]
#[cfg(not(debug_assertions))]
pub fn decay_washout_3gen() -> Result<(), Box<dyn error::Error>> {
    init();

    // Create the CSV file
    let output_dir = common::output_dir("leptogenesis/decay_washout/3gen");
    let csv = csv::Writer::from_path(output_dir.join("n.csv"))?;

    // Get the solution
    let mut model = LeptogenesisModel::zero();
    for i in &[
        interaction::hln,
        interaction::hhw,
        interaction::hha,
        interaction::ffa,
        interaction::ffw,
    ] {
        model
            .interactions
            .extend(i().drain(..).map(common::into_interaction_box));
    }
    for i in &[
        interaction::hhww,
        interaction::hhaa,
        interaction::hhll1,
        interaction::hhll2,
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

    let nai = na[LeptogenesisModel::particle_idx("L", 0).unwrap()].abs();
    assert!(1e-10 < nai && nai < 1e-5);
    for i in 0..3 {
        let nai = na[LeptogenesisModel::particle_idx("L", i).unwrap()].abs();
        assert!(1e-10 < nai && nai < 1e-5);
        assert!(n[LeptogenesisModel::particle_idx("N", i).unwrap()] < 1e-8);
    }

    Ok(())
}

/// Scan some parameter space and store the B-L from the scan in a CSV file.
#[test]
#[ignore]
#[cfg(not(debug_assertions))]
pub fn scan() -> Result<(), Box<dyn error::Error>> {
    init();

    // Create the CSV file
    let output_dir = common::output_dir("leptogenesis");
    let mut csv = csv::Writer::from_path(output_dir.join("scan.csv"))?;

    // Write the header for the CSV
    csv.serialize(("y", "m", "B-L", "N1"))?;

    for (&y, &m) in iproduct!(
        Array1::linspace(-8.0, -2.0, 8).into_iter(),
        Array1::linspace(6.0, 14.0, 8).into_iter()
    ) {
        // Convert y and m from the exponent to their actual values
        let y = 10f64.powf(y);
        let m = 10f64.powf(m);

        // Adjust the model with the necessary Yukawa and mass
        let mut model = LeptogenesisModel::zero();
        model.yv.mapv_inplace(|yi| yi / 1e-4 * y);
        model.mn[0] = m;

        let names: Vec<_> = model.particles().iter().map(|p| p.name.clone()).collect();
        let builder = SolverBuilder::new().model(model).beta_range(1e-17, 1e-3);

        let (n, na) = solve(builder, &names, None::<csv::Writer<fs::File>>)?;

        // Write to the CSV file
        csv.serialize((
            format!("{:e}", y),
            format!("{:e}", m),
            format!(
                "{:e}",
                iproduct!(
                    [
                        ("Q", 1.0 / 3.0),
                        ("u", 1.0 / 3.0),
                        ("d", 1.0 / 3.0),
                        ("L", -1.0),
                        ("e", -1.0)
                    ]
                    .iter(),
                    0..3
                )
                .map(|((p, f), i)| f * na[LeptogenesisModel::particle_idx(p, i).unwrap()])
                .sum::<f64>()
            ),
            format!("{:e}", n[LeptogenesisModel::particle_idx("N", 0).unwrap()]),
            format!("{:e}", n[LeptogenesisModel::particle_idx("N", 1).unwrap()]),
            format!("{:e}", n[LeptogenesisModel::particle_idx("N", 2).unwrap()]),
        ))?;
    }

    Ok(())
}

#[cfg(feature = "serde")]
#[test]
fn evolution() -> Result<(), Box<dyn error::Error>> {
    init();

    // Create the CSV files
    let output_dir = common::output_dir("leptogenesis");

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
    let output_dir = common::output_dir("leptogenesis/higgs_equilibrium");

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
        for i in &[interaction::hhaa, interaction::hhww] {
            model
                .interactions
                .extend(i().drain(..).map(into_interaction_box));
        }

        let p_h = LeptogenesisModel::particle_idx("H", 0).unwrap();
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
    let output_dir = common::output_dir("leptogenesis/lepton_equilibrium");

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
        for i in &[interaction::ffa, interaction::ffw] {
            model.interactions.extend(
                i().drain(..)
                    .filter(one_generation)
                    .map(into_interaction_box),
            );
        }

        let p_l = LeptogenesisModel::particle_idx("L", 0).unwrap();
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

    let output_dir = common::output_dir("leptogenesis/gamma");

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
        ] {
            model.interactions.extend(
                i().drain(..)
                    // .filter(one_generation)
                    .map(into_interaction_box),
            );
        }

        #[cfg(not(debug_assertions))]
        for i in &[
            interaction::hhll1,
            interaction::hhll2,
            interaction::nlqd,
            interaction::nlqu,
            interaction::hhaa,
            interaction::hhww,
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

    let mut precomputed = csv::Writer::from_path({
        let mut path = output_dir.clone();
        path.push("gamma_pre.csv");
        path
    })?;
    let (header, data) = solvers[0].gammas(1024, true);
    precomputed.serialize(header)?;
    for row in data.axis_iter(Axis(0)) {
        precomputed.serialize(row.as_slice().unwrap())?;
    }

    let mut precomputed_asymmetry = csv::Writer::from_path({
        let mut path = output_dir.clone();
        path.push("asymmetry_pre.csv");
        path
    })?;
    let (header, data) = solvers[0].asymmetries(1024, true);
    precomputed_asymmetry.serialize(header)?;
    for row in data.axis_iter(Axis(0)) {
        precomputed_asymmetry.serialize(row.as_slice().unwrap())?;
    }

    let mut non_precomputed = csv::Writer::from_path({
        let mut path = output_dir.clone();
        path.push("gamma_nopre.csv");
        path
    })?;
    let (header, data) = solvers[1].gammas(1024, true);
    non_precomputed.serialize(header)?;
    for row in data.axis_iter(Axis(0)) {
        non_precomputed.serialize(row.as_slice().unwrap())?;
    }

    let mut non_precomputed_asymmetry = csv::Writer::from_path({
        let mut path = output_dir;
        path.push("asymmetry_nopre.csv");
        path
    })?;
    let (header, data) = solvers[1].asymmetries(1024, true);
    non_precomputed_asymmetry.serialize(header)?;
    for row in data.axis_iter(Axis(0)) {
        non_precomputed_asymmetry.serialize(row.as_slice().unwrap())?;
    }

    Ok(())
}
