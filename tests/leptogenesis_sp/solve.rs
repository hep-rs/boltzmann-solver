//! Setup the solver, adding the particles and interactions to it, setting up
//! the logger(s), and running it before returning the result.

use super::{interaction, model, model::LeptogenesisModel};
use boltzmann_solver::{
    solver::{number_density::NumberDensitySolver, Solver},
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

    // Create the Solver and set integration parameters
    let mut solver: NumberDensitySolver<LeptogenesisModel> = NumberDensitySolver::new()
        .beta_range(1e-17, 1e0)
        .error_tolerance(1e-1)
        .step_precision(1e-2, 5e-1)
        .initialize();

    solver.model_fn(f);

    // Logging of number densities
    ////////////////////////////////////////////////////////////////////////////////
    let csv = RefCell::new(csv::Writer::from_path("/tmp/leptogenesis_sp/n.csv").unwrap());

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

    // Run solver
    ////////////////////////////////////////////////////////////////////////////////

    solver.solve(&universe)
}
