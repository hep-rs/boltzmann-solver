//! Setup the solver, adding the particles and interactions to it, setting up
//! the logger(s), and running it before returning the result.

use super::{interaction, model, model::LeptogenesisModel};
use boltzmann_solver::{
    particle::Particle,
    solver::{number_density::NumberDensitySolver, InitialCondition, Solver},
    universe::StandardModel,
};
use ndarray::prelude::*;
use std::cell::RefCell;

/// Solve the Boltzmann equations for the given model.
///
/// This routine sets up the solve, runs it and returns the final array of
/// number densities.
pub fn solve<F: 'static>(model: LeptogenesisModel, f: F) -> Array1<f64>
where
    F: Fn(LeptogenesisModel) -> LeptogenesisModel,
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

    // Add the particles to the solver, using for initial condition either 0 or
    // equilibrium number density.
    solver.add_particle(
        Particle::new(model::NAMES[0].to_string(), 0, 0.0).set_dof(0.0),
        InitialCondition::Zero,
    );

    solver.add_particle(
        Particle::new(model::NAMES[1].to_string(), 1, model.mass.n[0]),
        InitialCondition::Equilibrium(0.0),
    );
    solver.add_particle(
        Particle::new(model::NAMES[2].to_string(), 1, model.mass.n[1]),
        InitialCondition::Equilibrium(0.0),
    );
    solver.add_particle(
        Particle::new(model::NAMES[3].to_string(), 1, model.mass.n[2]),
        InitialCondition::Equilibrium(0.0),
    );

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
        // if BETA_RANGE.0 < c.beta && c.beta < BETA_RANGE.1 {
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
