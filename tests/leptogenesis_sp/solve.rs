//! Setup the solver, adding the particles and interactions to it, setting up
//! the logger(s), and running it before returning the result.

use super::{interaction, model, model::LeptogenesisModel};
use boltzmann_solver::{
    solver::{number_density::NumberDensitySolver, Model, Solver},
    universe::StandardModel,
};
use ndarray::prelude::*;
use std::cell::RefCell;

/// Solve the Boltzmann equations for the given model.
///
/// This routine sets up the solve, runs it and returns the final array of
/// number densities.
pub fn solve<F: 'static>(f: F) -> Array1<f64>
where
    F: Fn(f64) -> LeptogenesisModel,
{
    // Set up the universe in which we'll run the Boltzmann equations
    let universe = StandardModel::new();
    let beta_start = 1e-17;
    let beta_end = 1e-3;
    let model = f(beta_start);

    let initial_conditions = model
        .particles()
        .iter()
        .map(|p| p.normalized_number_density(0.0, beta_start))
        .collect();

    // Create the Solver and set integration parameters
    let mut solver: NumberDensitySolver<LeptogenesisModel> = NumberDensitySolver::new()
        .beta_range(beta_start, beta_end)
        .error_tolerance(1e-1)
        .step_precision(1e-2, 5e-1)
        .initial_conditions(initial_conditions)
        .initialize();

    solver.model_fn(f);

    // Logging of number densities
    ////////////////////////////////////////////////////////////////////////////////
    let output_dir = crate::output_dir().join("sp");
    let csv = RefCell::new(csv::Writer::from_path(output_dir.join("n.csv")).unwrap());
    {
        let mut csv = csv.borrow_mut();
        csv.write_field("step").unwrap();
        csv.write_field("beta").unwrap();

        for name in &model::NAMES {
            csv.write_field(name).unwrap();
            csv.write_field(format!("({})", name)).unwrap();
            csv.write_field(format!("Δ{}", name)).unwrap();
        }

        csv.write_record(None::<&[u8]>).unwrap();
    }

    solver.set_logger(move |n, dn, c| {
        let mut csv = csv.borrow_mut();
        csv.write_field(format!("{}", c.step)).unwrap();
        csv.write_field(format!("{:.15e}", c.beta)).unwrap();

        for i in 0..n.len() {
            csv.write_field(format!("{:.3e}", n[i])).unwrap();
            csv.write_field(format!("{:.3e}", c.eq_n[i])).unwrap();
            csv.write_field(format!("{:.3e}", dn[i])).unwrap();
        }

        csv.write_record(None::<&[u8]>).unwrap();
    });

    // Interactions
    ////////////////////////////////////////////////////////////////////////////////

    interaction::n_el_h(&mut solver);
    interaction::n_el_ql_qr(&mut solver);
    interaction::equilibrium(&mut solver);

    // Run solver
    ////////////////////////////////////////////////////////////////////////////////

    solver.solve(&universe)
}
