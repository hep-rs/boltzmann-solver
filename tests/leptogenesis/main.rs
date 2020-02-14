//! Test `boltzmann-solver` to calculate the baryon asymmetry generated through
//! leptogenesis in the standard type-I seesaw.

mod common;
mod model;

use crate::model::{interaction, LeptogenesisModel};
use boltzmann_solver::prelude::*;
#[cfg(not(debug_assertions))]
use boltzmann_solver::{
    statistic::{Statistic, Statistics},
    utilities::spline::rec_geomspace,
};
#[cfg(not(debug_assertions))]
use itertools::iproduct;
use ndarray::prelude::*;
#[cfg(not(debug_assertions))]
use rayon::prelude::*;
#[cfg(not(debug_assertions))]
use std::fs::File;
use std::{error, fmt, io, sync::RwLock};

/// Box an interaction
#[cfg(feature = "parallel")]
fn into_interaction_box<I, M>(interaction: I) -> Box<dyn Interaction<M> + Sync>
where
    I: Interaction<M> + Sync + 'static,
    M: Model,
{
    Box::new(interaction)
}

/// Box an interaction
#[cfg(not(feature = "parallel"))]
fn into_interaction_box<I, M>(interaction: I) -> Box<dyn Interaction<M>>
where
    I: Interaction<M> + 'static,
    M: Model,
{
    Box::new(interaction)
}

/// Filter iteraction based whether they involve first-generation particles only
/// or not.
fn one_generation<I, M>(interaction: &I) -> bool
where
    I: Interaction<M>,
    M: Model,
{
    let ptcl = interaction.particles_idx();
    ptcl.incoming.iter().chain(&ptcl.outgoing).all(|i| match i {
        1 | 2 | 3 | 4 | 5 | 8 | 11 | 14 | 17 | 20 => true,
        _ => false,
    })
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
    common::setup_logging(1);

    // Create the CSV file
    let output_dir = common::output_dir("leptogenesis/decay_only/1gen");
    let csv = csv::Writer::from_path(output_dir.join("n.csv"))?;

    // Get the solution
    let mut model = LeptogenesisModel::zero();
    model.interactions.extend(
        interaction::hln()
            .drain(..)
            .filter(one_generation)
            .map(into_interaction_box),
    );

    // Collect the names now as SolverBuilder takes ownership of the model
    // later.
    let names: Vec<_> = model.particles().iter().map(|p| p.name.clone()).collect();

    // Prevent any asymmetry from being generated in the N_i and H.
    let no_asymmetry: Vec<usize> = (0..3)
        .map(|i| LeptogenesisModel::particle_idx("N", i).unwrap())
        .collect();

    let builder = SolverBuilder::new()
        .no_asymmetry(no_asymmetry)
        .model(model)
        .beta_range(1e-17, 1e-3)
        .step_precision(1e-6, 1e-1);

    let (n, na) = solve(builder, &names, Some(csv))?;

    // Check that the solution is fine
    println!("Final number density: {:.3e}", n);
    println!("Final number density asymmetry: {:.3e}", na);

    let nai = na[LeptogenesisModel::particle_idx("L", 0).unwrap()].abs();
    assert!(1e-10 < nai && nai < 1e-5);
    assert!(n[LeptogenesisModel::particle_idx("N", 0).unwrap()] < 1e-8);

    Ok(())
}

/// Test the effects of the right-handed neutrino decay on its own in the
/// 3-generation case.
#[test]
pub fn decay_only_3gen() -> Result<(), Box<dyn error::Error>> {
    // common::setup_logging(2);

    // Create the CSV file
    let output_dir = common::output_dir("leptogenesis/decay_only/3gen");
    let csv = csv::Writer::from_path(output_dir.join("n.csv"))?;

    // Get the solution
    let mut model = LeptogenesisModel::zero();
    model
        .interactions
        .extend(interaction::hln().drain(..).map(into_interaction_box));

    // Collect the names now as SolverBuilder takes ownership of the model
    // later.
    let names: Vec<_> = model.particles().iter().map(|p| p.name.clone()).collect();

    // Prevent any asymmetry from being generated in the N_i and H.
    let no_asymmetry: Vec<usize> = (0..3)
        .map(|i| LeptogenesisModel::particle_idx("N", i).unwrap())
        .collect();

    let builder = SolverBuilder::new()
        .no_asymmetry(no_asymmetry)
        .model(model)
        .beta_range(1e-17, 1e-3)
        .step_precision(1e-6, 1e-1);

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
    // common::setup_logging(2);

    // Create the CSV file
    let output_dir = common::output_dir("leptogenesis/washout_only/1gen");
    let csv = csv::Writer::from_path(output_dir.join("n.csv"))?;

    // Get the solution
    let mut model = LeptogenesisModel::zero();
    model.interactions.extend(
        interaction::hhll1()
            .drain(..)
            .filter(one_generation)
            .map(into_interaction_box),
    );

    // Collect the names now as SolverBuilder takes ownership of the model
    // later.
    let names: Vec<_> = model.particles().iter().map(|p| p.name.clone()).collect();

    // Prevent any asymmetry from being generated in the N_i and H.
    let no_asymmetry: Vec<usize> = (0..3)
        .map(|i| LeptogenesisModel::particle_idx("N", i).unwrap())
        .collect();

    // Add a primordial asymmetry in L1
    let mut initial_asymmetries = Array1::zeros(model.particles().len());
    initial_asymmetries[LeptogenesisModel::particle_idx("L", 0).unwrap()] = 1.0;

    let builder = SolverBuilder::new()
        .no_asymmetry(no_asymmetry)
        .initial_asymmetries(initial_asymmetries)
        .model(model)
        .beta_range(1e-17, 1e-3)
        .step_precision(1e-6, 1e-1);

    let (n, na) = solve(builder, &names, Some(csv))?;

    // Check that the solution is fine
    println!("Final number density: {:.3e}", n);
    println!("Final number density asymmetry: {:.3e}", na);

    let nai = na[LeptogenesisModel::particle_idx("L", 0).unwrap()].abs();
    assert!(nai < 1e-5);

    Ok(())
}

#[test]
#[cfg(not(debug_assertions))]
pub fn decay_washout_1gen() -> Result<(), Box<dyn error::Error>> {
    // common::setup_logging(2);

    // Create the CSV files
    let output_dir = common::output_dir("leptogenesis/decay_washout/1gen");
    let csv = csv::Writer::from_path(output_dir.join("n.csv"))?;

    // Get the solution
    let mut model = LeptogenesisModel::zero();
    model.interactions.extend(
        interaction::hln()
            .drain(..)
            .filter(one_generation)
            .map(into_interaction_box),
    );
    model.interactions.extend(
        interaction::hhll1()
            .drain(..)
            .filter(one_generation)
            .map(into_interaction_box),
    );

    // Collect the names now as SolverBuilder takes ownership of the model
    // later.
    let names: Vec<_> = model.particles().iter().map(|p| p.name.clone()).collect();

    let no_asymmetry: Vec<usize> = (0..3)
        .map(|i| LeptogenesisModel::particle_idx("N", i).unwrap())
        .collect();

    let builder = SolverBuilder::new()
        .no_asymmetry(no_asymmetry)
        .model(model)
        .beta_range(1e-17, 1e-3);

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
    // common::setup_logging(2);

    // Create the CSV file
    let output_dir = common::output_dir("leptogenesis/decay_washout/3gen");
    let csv = csv::Writer::from_path(output_dir.join("n.csv"))?;

    // Get the solution
    let mut model = LeptogenesisModel::zero();
    model
        .interactions
        .extend(interaction::hln().drain(..).map(into_interaction_box));
    model
        .interactions
        .extend(interaction::hhll1().drain(..).map(into_interaction_box));
    model
        .interactions
        .extend(interaction::hhll2().drain(..).map(into_interaction_box));

    // Collect the names now as SolverBuilder takes ownership of the model
    // later.
    let names: Vec<_> = model.particles().iter().map(|p| p.name.clone()).collect();

    let no_asymmetry: Vec<usize> = (0..3)
        .map(|i| LeptogenesisModel::particle_idx("N", i).unwrap())
        .collect();

    let builder = SolverBuilder::new()
        .no_asymmetry(no_asymmetry)
        .model(model)
        .beta_range(1e-17, 1e-3);

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
    // common::setup_logging(2);

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

        let no_asymmetry: Vec<usize> = (0..3)
            .map(|i| LeptogenesisModel::particle_idx("N", i).unwrap())
            .collect();

        let names: Vec<_> = model.particles().iter().map(|p| p.name.clone()).collect();
        let builder = SolverBuilder::new()
            .no_asymmetry(no_asymmetry)
            .model(model)
            .beta_range(1e-17, 1e-3);

        let (n, na) = solve(builder, &names, None::<csv::Writer<File>>)?;

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

#[test]
fn masses_widths() -> Result<(), Box<dyn error::Error>> {
    // Setup logging
    // common::setup_logging(2);

    let mut model = LeptogenesisModel::zero();
    model
        .interactions
        .extend(interaction::hle().drain(..).map(into_interaction_box));
    model
        .interactions
        .extend(interaction::hln().drain(..).map(into_interaction_box));
    model
        .interactions
        .extend(interaction::hqu().drain(..).map(into_interaction_box));
    model
        .interactions
        .extend(interaction::hqd().drain(..).map(into_interaction_box));

    // Create the CSV files
    let output_dir = common::output_dir("leptogenesis");

    let mut widths = csv::Writer::from_path(output_dir.join("width.csv"))?;
    widths.write_field("beta")?;
    for p in model.particles() {
        widths.write_field(&p.name)?;
    }
    widths.write_record(None::<&[u8]>)?;

    let mut masses = csv::Writer::from_path(output_dir.join("mass.csv"))?;
    masses.write_field("beta")?;
    for p in model.particles() {
        masses.write_field(&p.name)?;
    }
    masses.write_record(None::<&[u8]>)?;

    for &beta in Array1::geomspace(1e-17, 1e-2, 1024).unwrap().into_iter() {
        model.set_beta(beta);
        model.update_widths();

        widths.write_field(format!("{:e}", beta))?;
        for p in model.particles() {
            widths.write_field(format!("{:e}", p.width))?;
        }
        widths.write_record(None::<&[u8]>)?;

        masses.write_field(format!("{:e}", beta))?;
        for p in model.particles() {
            masses.write_field(format!("{:e}", p.mass))?;
        }
        masses.write_record(None::<&[u8]>)?;
    }

    Ok(())
}

#[test]
#[cfg(not(debug_assertions))]
fn gammas() -> Result<(), Box<dyn error::Error>> {
    // Setup logging
    // common::setup_logging(2);

    let beta = (1e-17, 1e-2);
    let mut model_precomp = LeptogenesisModel::zero();
    model_precomp.interactions.extend(
        interaction::hln()
            .drain(..)
            .filter(one_generation)
            .map(into_interaction_box),
    );
    model_precomp.interactions.extend(
        interaction::hhll1()
            .drain(..)
            .filter(one_generation)
            .map(into_interaction_box),
    );
    // model_precomp.interactions.extend(
    //     interaction::hhll2()
    //         .drain(..)
    //         .filter(one_generation)
    //         .map(into_interaction_box),
    // );

    let mut model_no_precomp = LeptogenesisModel::zero();
    model_no_precomp.interactions.extend(
        interaction::hln()
            .drain(..)
            .filter(one_generation)
            .map(into_interaction_box),
    );
    model_no_precomp.interactions.extend(
        interaction::hhll1()
            .drain(..)
            .filter(one_generation)
            .map(into_interaction_box),
    );
    // model_no_precomp.interactions.extend(
    //     interaction::hhll2()
    //         .drain(..)
    //         .filter(one_generation)
    //         .map(into_interaction_box),
    // );

    // Precompute gamma for model1 only
    const N: u32 = 10;
    for (i, &beta) in vec![0.98 * beta.0, 0.99 * beta.0, 1.01 * beta.1, 1.02 * beta.1]
        .iter()
        .chain(&rec_geomspace(1e-17, 1e-2, N))
        .enumerate()
    {
        if i % 64 == 0 {
            log::debug!("Precomputing step {} / {}", i, 2usize.pow(N) + 4);
        } else {
            log::trace!("Precomputing step {} / {}", i, 2usize.pow(N) + 4);
        }

        model_precomp.set_beta(beta);
        let c = model_precomp.as_context();

        model_precomp
            .interactions()
            .par_iter()
            .for_each(|interaction| {
                interaction.gamma(&c);
            });
    }

    // Create the CSV files
    let output_dir_precomp = common::output_dir("leptogenesis/gamma/precomp");
    let output_dir_no_precomp = common::output_dir("leptogenesis/gamma/noprecomp");
    let mut normalization_csv = csv::Writer::from_path({
        let output_dir = common::output_dir("leptogenesis/gamma");
        let mut path = output_dir.clone();
        path.push("normalization.csv");
        path
    })?;
    normalization_csv.serialize(("beta", "H", "n1", "normalization"))?;

    let mut csvs_precomp = Vec::new();
    for interaction in model_precomp.interactions() {
        let particles = interaction.particles();
        let mut csv = csv::Writer::from_path({
            let mut path = output_dir_precomp.clone();
            path.push(format!(
                "{:?}:{:?}.csv",
                particles.incoming, particles.outgoing
            ));
            path
        })?;
        csv.serialize(("beta", "gamma"))?;
        csvs_precomp.push(RwLock::new(csv));
    }

    let mut csvs_no_precomp = Vec::new();
    for interaction in model_no_precomp.interactions() {
        let particles = interaction.particles();
        let mut csv = csv::Writer::from_path({
            let mut path = output_dir_no_precomp.clone();
            path.push(format!(
                "{:?}:{:?}.csv",
                particles.incoming, particles.outgoing
            ));
            path
        })?;
        csv.serialize(("beta", "gamma"))?;
        csvs_no_precomp.push(RwLock::new(csv));
    }

    // Write out all the outputs
    for &beta in Array1::geomspace(beta.0, beta.1, 100).unwrap().iter() {
        model_precomp.set_beta(beta);
        model_no_precomp.set_beta(beta);

        let c = model_precomp.as_context();
        normalization_csv.serialize((
            c.beta,
            c.hubble_rate,
            Statistic::BoseEinstein.massless_number_density(0.0, beta),
            c.normalization,
        ))?;

        model_precomp
            .interactions()
            .par_iter()
            .zip(&mut csvs_precomp)
            .for_each(|(interaction, csv)| {
                csv.write()
                    .unwrap()
                    .serialize((beta, interaction.gamma(&c)))
                    .unwrap();
            });

        model_no_precomp
            .interactions()
            .par_iter()
            .zip(&mut csvs_no_precomp)
            .for_each(|(interaction, csv)| {
                csv.write()
                    .unwrap()
                    .serialize((beta, interaction.gamma(&c)))
                    .unwrap();
            })
    }

    Ok(())
}
