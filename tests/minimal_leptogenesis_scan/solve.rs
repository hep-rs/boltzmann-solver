mod interaction;

use crate::model::{VanillaLeptogenesisModel, PARTICLE_NAMES};
use boltzmann_solver::{
    particle::Particle,
    solver::{number_density::NumberDensitySolver, InitialCondition, Solver},
    universe::StandardModel,
};
use ndarray::prelude::*;

/// Solve the Boltzmann equations for the given model.
///
/// This routine sets up the solve, runs it and returns the final array of
/// number densities.
pub fn solve(model: VanillaLeptogenesisModel, y: f64, m: f64) -> Array1<f64> {
    // Set up the universe in which we'll run the Boltzmann equations
    let universe = StandardModel::new();

    // Create the Solver and set integration parameters
    let mut solver: NumberDensitySolver<VanillaLeptogenesisModel> = NumberDensitySolver::new()
        .beta_range(1e-17, 1e0)
        .initialize();

    solver.model_fn(move |mut model: VanillaLeptogenesisModel| {
        model.coupling.y_v.mapv_inplace(|yi| yi * y);
        model.mass.n[0] = m;
        model
    });

    // Add the particles to the solver, using for initial condition either 0 or
    // equilibrium number density.
    solver.add_particle(
        Particle::new(PARTICLE_NAMES[0].to_string(), 0, 0.0).set_dof(0.0),
        InitialCondition::Zero,
    );

    solver.add_particle(
        Particle::new(PARTICLE_NAMES[1].to_string(), 1, model.mass.n[0]),
        InitialCondition::Equilibrium(0.0),
    );
    solver.add_particle(
        Particle::new(PARTICLE_NAMES[2].to_string(), 1, model.mass.n[1]),
        InitialCondition::Equilibrium(0.0),
    );
    solver.add_particle(
        Particle::new(PARTICLE_NAMES[3].to_string(), 1, model.mass.n[2]),
        InitialCondition::Equilibrium(0.0),
    );

    // Interactions
    ////////////////////////////////////////////////////////////////////////////////

    interaction::n_el_h(&mut solver);
    interaction::n_el_ql_qr(&mut solver);

    // Run solver
    ////////////////////////////////////////////////////////////////////////////////

    solver.solve(&universe)
}
