//! Test `boltzmann-solver` to calculate the baryon asymmetry generated through
//! leptogenesis in the standard type-I seesaw.

mod common;
mod model;

pub use crate::model::LeptogenesisModel;
use boltzmann_solver::{prelude::*, utilities::spline::rec_geomspace};
use itertools::iproduct;
use ndarray::prelude::*;
use std::{fs::File, io, sync::RwLock};

/// Solve the Boltzmann equations and return the final values.
///
/// The model function is specified by `model`, and optionally a CSV writer can
/// be given to record the progress of Boltzmann equations.
pub fn solve<W>(
    mut builder: SolverBuilder<LeptogenesisModel>,
    names: &Vec<&str>,
    csv: Option<csv::Writer<W>>,
) -> Result<(Array1<f64>, Array1<f64>), Box<dyn std::error::Error>>
where
    W: io::Write + 'static,
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

        builder.logger(move |c| {
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
        builder.logger(|c| {
            if c.n.iter().chain(c.na.iter()).any(|n| !n.is_finite()) {
                log::error!("Non-finite number density at step {}.", c.step);
                panic!("Non-finite number density at step {}.", c.step)
            }
        })
    }

    // Build and run the solver
    let mut solver = builder.build().expect("Error while building the solver.");
    Ok(solver.solve())
}

/// Test a single run, and log the evolution in a CSV file.
#[test]
#[ignore]
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    common::setup_logging(2);

    // Create the CSV file
    let output_dir = common::output_dir("leptogenesis");
    let csv = csv::Writer::from_path(output_dir.join("n.csv"))?;

    // Get the solution
    let model = LeptogenesisModel::zero();
    let names = model.particles().iter().map(|p| p.name).collect();

    let mut no_asymmetry: Vec<usize> = (0..3)
        .map(|i| LeptogenesisModel::particle_idx("N", i).unwrap())
        .collect();
    no_asymmetry.push(LeptogenesisModel::particle_idx("H", 0).unwrap());

    let builder = SolverBuilder::new()
        .no_asymmetry(no_asymmetry)
        .model(model)
        .beta_range(1e-17, 1e-3);

    let (n, na) = solve(builder, &names, Some(csv))?;

    // Check that the solution is fine
    log::info!("Final number density: {:.3e}", n);
    log::info!("Final number density asymmetry: {:.3e}", na);
    for i in 0..3 {
        let nai = na[LeptogenesisModel::particle_idx("L", i).unwrap()].abs();
        assert!(1e-10 < nai && nai < 1e-5);
        assert!(n[LeptogenesisModel::particle_idx("N", i).unwrap()] < 1e-9);
    }

    Ok(())
}

/// Scan some parameter space and store the B-L from the scan in a CSV file.
#[test]
#[ignore]
pub fn scan() -> Result<(), Box<dyn std::error::Error>> {
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

        let mut no_asymmetry: Vec<usize> = (0..3)
            .map(|i| LeptogenesisModel::particle_idx("N", i).unwrap())
            .collect();
        no_asymmetry.push(LeptogenesisModel::particle_idx("H", 0).unwrap());

        let names = model.particles().iter().map(|p| p.name).collect();
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
                    .into_iter(),
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
#[ignore]
fn widths() -> Result<(), Box<dyn std::error::Error>> {
    // Setup logging
    // common::setup_logging(2);

    let mut model = LeptogenesisModel::zero();

    // Create the CSV files
    let output_dir = common::output_dir("leptogenesis");

    let mut gauge = csv::Writer::from_path(output_dir.join("gauge.csv"))?;
    gauge.serialize(["beta", "g1", "g2", "g3"])?;

    let mut widths = csv::Writer::from_path(output_dir.join("width.csv"))?;
    widths.write_field("beta")?;
    for p in model.particles() {
        widths.write_field(p.name)?;
    }
    widths.write_record(None::<&[u8]>)?;

    let mut masses = csv::Writer::from_path(output_dir.join("mass.csv"))?;
    masses.write_field("beta")?;
    for p in model.particles() {
        masses.write_field(p.name)?;
    }
    masses.write_record(None::<&[u8]>)?;

    for beta in rec_geomspace(1e-20, 1e-3, 10).into_iter() {
        model.set_beta(beta);
        model.update_widths();

        gauge.serialize([beta, model.sm.g1, model.sm.g2, model.sm.g3])?;

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
#[ignore]
fn gammas() -> Result<(), Box<dyn std::error::Error>> {
    // Setup logging
    common::setup_logging(2);

    let mut model = LeptogenesisModel::zero();

    // Create the CSV files
    let output_dir = common::output_dir("leptogenesis");
    let mut gammas = csv::Writer::from_path(output_dir.join("gammas.csv"))?;
    gammas.write_field("beta")?;
    for interaction in model.interactions() {
        for ps in interaction.particles() {
            gammas.write_field(format!("{:?}", ps))?;
        }
    }
    gammas.write_record(None::<&[u8]>)?;

    // for beta in rec_geomspace(1e-20, 1e-3, 10).into_iter() {
    for &beta in Array1::geomspace(1e-20, 1e-3, 40).unwrap().into_iter() {
        log::warn!("β = {:e}", beta);
        model.set_beta(beta);
        let c = model.as_context();

        gammas.write_field(format!("{:e}", beta))?;
        for interaction in model.interactions().iter() {
            let g = interaction.gamma(&c);
            if interaction.is_two_particle() {
                gammas.write_field(format!("{:e}", g[0]))?;
            }
            if interaction.is_three_particle() {
                gammas.write_field(format!("{:e}", g[0]))?;
                gammas.write_field(format!("{:e}", g[1]))?;
                gammas.write_field(format!("{:e}", g[2]))?;
            }
            if interaction.is_four_particle() {
                gammas.write_field(format!("{:e}", g[0]))?;
                gammas.write_field(format!("{:e}", g[1]))?;
                gammas.write_field(format!("{:e}", g[2]))?;
            }
        }
        gammas.write_record(None::<&[u8]>)?;
    }

    Ok(())
}

#[test]
#[ignore]
fn particle_indices() {
    let model = LeptogenesisModel::zero();

    for (i, p) in model.particles().iter().enumerate() {
        let name = p.name;
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
