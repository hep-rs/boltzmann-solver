//! Test `boltzmann-solver` to calculate the baryon asymmetry generated through
//! leptogenesis in the standard type-I seesaw.

mod common;
mod interaction;
mod model;

use boltzmann_solver::{
    solver::{number_density::SolverBuilder, Model},
    universe::StandardModel,
};
use itertools::iproduct;
use model::{p_i, LeptogenesisModel};
use ndarray::prelude::*;
use std::{fs::File, io, sync::RwLock};

/// Solve the Boltzmann equations and return the final values.
///
/// The model function is specified by `model`, and optionally a CSV writer can
/// be given to record the progress of Boltzmann equations.
pub fn solve<F, W>(model: F, csv: Option<csv::Writer<W>>) -> Array1<f64>
where
    F: Fn(f64) -> LeptogenesisModel + 'static,
    W: io::Write + 'static,
{
    // Set up the universe in which we'll run the Boltzmann equations
    let universe = StandardModel::new();
    let beta_start = 1e-17;
    let beta_end = 1e-3;
    let model_start = model(beta_start);

    // Create the SolverBuilder and set base parameters
    let mut solver_builder = SolverBuilder::new()
        .model(model)
        .equilibrium(vec![1])
        .beta_range(beta_start, beta_end)
        .initial_conditions(
            model_start
                .particles()
                .iter()
                .map(|p| p.normalized_number_density(0.0, beta_start)),
        );

    if let Some(mut csv) = csv {
        // If we have a CSV file to write to, track the number densities as they
        // evolve.

        // Write the headers
        csv.write_field("step").unwrap();
        csv.write_field("beta").unwrap();
        for name in &model::NAMES {
            csv.write_field(name).unwrap();
            csv.write_field(format!("({})", name)).unwrap();
            csv.write_field(format!("Δ{}", name)).unwrap();
        }
        csv.write_record(None::<&[u8]>).unwrap();

        // Wrap the CSV into a RefCell as the logger is shared across threads
        // and offers mutability to the CSV.
        let csv = RwLock::new(csv);

        solver_builder.logger(move |n, dn, c| {
            // Write out the current step and inverse temperature
            let mut csv = csv.write().unwrap();
            csv.write_field(format!("{}", c.step)).unwrap();
            csv.write_field(format!("{:e}", c.beta)).unwrap();
            // Write all the number densities
            for i in 0..n.len() {
                csv.write_field(format!("{:e}", n[i])).unwrap();
                csv.write_field(format!("{:e}", c.eq[i])).unwrap();
                csv.write_field(format!("{:e}", dn[i])).unwrap();
            }
            csv.write_record(None::<&[u8]>).unwrap();

            if n.iter().any(|n| !n.is_finite()) {
                panic!("Obtained a non-finite number.")
            }
        });
    } else {
        // If we're not tracking the number density, simply make sure that all
        // the number densities are always finite, or quite.
        solver_builder.logger(|n, _, c| {
            if n.iter().any(|n| !n.is_finite()) {
                log::error!("Non-finite number density at step {}.", c.step);
                panic!("Obtained a non-finite number.")
            }
        })
    }

    // Add each interaction to the solver.
    interaction::n_el_h(&mut solver_builder);

    // Build and run the solver
    let solver = solver_builder
        .build()
        .expect("Error while building the solver.");
    solver.solve(&universe)
}

/// Test a single run, and log the evolution in a CSV file.
#[test]
pub fn run() {
    // common::setup_logging(2);

    // Create the CSV file
    let output_dir = common::output_dir("leptogenesis");
    let csv = csv::Writer::from_path(output_dir.join("n.csv")).unwrap();

    // Get the solution
    let sol = solve(LeptogenesisModel::new, Some(csv));

    // Check that the solution is fine
    log::info!("Final number density: {:.3e}", sol);
    assert!(1e-10 < sol[p_i("B-L", 0)].abs() && sol[p_i("B-L", 0)].abs() < 1e-5);
    for i in 0..3 {
        assert!(sol[p_i("N", i)] < 1e-9);
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
        let model = move |beta: f64| {
            let mut model = LeptogenesisModel::new(beta);
            model.coupling.y_v.mapv_inplace(|yi| yi / 1e-4 * y);
            model.particles[p_i("N", 0)].set_mass(m);
            model.mass.n[0] = m;
            model.mass2.n[0] = m.powi(2);
            model
        };
        let sol = solve(model, None::<csv::Writer<File>>);

        // Verify that the solution is fine
        assert!(1e-20 < sol[p_i("B-L", 0)].abs() && sol[p_i("B-L", 0)].abs() < 1e-1);
        for i in 1..3 {
            assert!(sol[p_i("N", i)] < 1e-15);
        }

        // Write to the CSV file
        csv.serialize((
            format!("{:e}", y),
            format!("{:e}", m),
            format!("{:e}", sol[p_i("B-L", 0)]),
            format!("{:e}", sol[p_i("N", 0)]),
        ))
        .unwrap();
    }
}
