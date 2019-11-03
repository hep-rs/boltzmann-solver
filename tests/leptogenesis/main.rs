//! Test `boltzmann-solver` to calculate the baryon asymmetry generated through
//! leptogenesis in the standard type-I seesaw.

mod common;
mod model;

pub use crate::model::LeptogenesisModel;
use boltzmann_solver::prelude::*;
use itertools::iproduct;
use ndarray::prelude::*;
use std::{fs::File, io, sync::RwLock};

/// Solve the Boltzmann equations and return the final values.
///
/// The model function is specified by `model`, and optionally a CSV writer can
/// be given to record the progress of Boltzmann equations.
pub fn solve<W>(model: LeptogenesisModel, csv: Option<csv::Writer<W>>) -> (Array1<f64>, Array1<f64>)
where
    W: io::Write + 'static,
{
    // Set up the universe in which we'll run the Boltzmann equations
    let beta_start = 1e-17;
    let beta_end = 1e-3;
    let names: Vec<_> = model.particles().iter().map(|p| p.name).collect();

    let mut initial_asymmetries = Array1::zeros(model.particles().len());
    // initial_asymmetries[LeptogenesisModel::particle_idx("N", 0).unwrap() as usize] = 1.0;
    initial_asymmetries[LeptogenesisModel::particle_idx("L", 0).unwrap() as usize] = 1e-3;

    // Create the SolverBuilder and set base parameters
    let mut solver_builder = SolverBuilder::new()
        .initial_asymmetries(initial_asymmetries)
        .no_asymmetry((0..3).map(|i| LeptogenesisModel::particle_idx("N", i).unwrap()))
        .model(model)
        .beta_range(beta_start, beta_end);

    if let Some(mut csv) = csv {
        // If we have a CSV file to write to, track the number densities as they
        // evolve.

        // Write the headers
        csv.write_field("step").unwrap();
        csv.write_field("beta").unwrap();
        for name in names {
            csv.write_field(format!("{}", name)).unwrap();
            csv.write_field(format!("Δ{}", name)).unwrap();
            csv.write_field(format!("({})", name)).unwrap();
        }
        csv.write_record(None::<&[u8]>).unwrap();

        // Wrap the CSV into a RefCell as the logger is shared across threads
        // and offers mutability to the CSV.
        let csv = RwLock::new(csv);

        solver_builder.logger(move |c| {
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
        solver_builder.logger(|c| {
            if c.n.iter().chain(c.na.iter()).any(|n| !n.is_finite()) {
                log::error!("Non-finite number density at step {}.", c.step);
                panic!("Non-finite number density at step {}.", c.step)
            }
        })
    }

    // Build and run the solver
    let mut solver = solver_builder
        .build()
        .expect("Error while building the solver.");
    solver.solve()
}

/// Test a single run, and log the evolution in a CSV file.
#[test]
pub fn run() {
    common::setup_logging(2);

    // Create the CSV file
    let output_dir = common::output_dir("leptogenesis");
    let csv = csv::Writer::from_path(output_dir.join("n.csv")).unwrap();

    // Get the solution
    let model = LeptogenesisModel::new();
    let (n, na) = solve(model, Some(csv));

    // Check that the solution is fine
    log::info!("Final number density: {:.3e}", n);
    log::info!("Final number density asymmetry: {:.3e}", na);
    for i in 0..3 {
        let nai = na[LeptogenesisModel::particle_idx("L", i).unwrap()].abs();
        assert!(1e-10 < nai && nai < 1e-5);
        assert!(n[LeptogenesisModel::particle_idx("N", i).unwrap()] < 1e-9);
    }
}

/// Scan some parameter space and store the B-L from the scan in a CSV file.
#[test]
pub fn scan() {
    // common::setup_logging(2);

    // Create the CSV file
    let output_dir = common::output_dir("leptogenesis");
    let mut csv = csv::Writer::from_path(output_dir.join("scan.csv")).unwrap();

    // Write the header for the CSV
    csv.serialize(("y", "m", "B-L", "N1")).unwrap();

    for (&y, &m) in iproduct!(
        Array1::linspace(-8.0, -2.0, 8).into_iter(),
        Array1::linspace(6.0, 14.0, 8).into_iter()
    ) {
        // Convert y and m from the exponent to their actual values
        let y = 10f64.powf(y);
        let m = 10f64.powf(m);

        // Adjust the model with the necessary Yukawa and mass
        let mut model = LeptogenesisModel::new();
        model.yv.mapv_inplace(|yi| yi / 1e-4 * y);
        model.mn[0] = m;
        let (n, na) = solve(model, None::<csv::Writer<File>>);

        // Write to the CSV file
        csv.serialize((
            format!("{:e}", y),
            format!("{:e}", m),
            format!(
                "{:e}",
                (0..3)
                    .map(|i| na[LeptogenesisModel::particle_idx("L", i).unwrap()])
                    .sum::<f64>()
            ),
            format!("{:e}", n[LeptogenesisModel::particle_idx("N", 0).unwrap()]),
            format!("{:e}", n[LeptogenesisModel::particle_idx("N", 1).unwrap()]),
            format!("{:e}", n[LeptogenesisModel::particle_idx("N", 2).unwrap()]),
        ))
        .unwrap();
    }
}
