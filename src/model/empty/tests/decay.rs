//! Tests with various parts of the calculation set to 1.

use super::{solve, BETA_START};
use crate::{
    model::{interaction, Empty, Model, Particle},
    solver::SolverBuilder,
    utilities::test::approx_eq,
};
use std::error;

#[test]
fn symmetric() -> Result<(), Box<dyn error::Error>> {
    // crate::utilities::test::setup_logging(4);

    let mut model = Empty::default();
    model.extend_particles(
        [1e5, 1e2, 1e1]
            .iter()
            .enumerate()
            .map(|(i, &m)| Particle::new(0, m, m / 100.0).name(format!("{}", i + 1))),
    );
    model.push_interaction(interaction::ThreeParticle::new(|_m| 1.0, 1, 2, 3));

    // Get the initial conditions
    model.set_beta(BETA_START);
    let c = model.as_context();
    let (n0, na0) = (c.eq, c.na);

    // Run the solver
    let (n, na) = solve("symmetric.csv", SolverBuilder::new().model(model))?;

    // Check initial number densities
    approx_eq(n0[1], 1.0, 4.0, 1e-50)?;
    approx_eq(n0[2], 1.0, 4.0, 1e-50)?;
    approx_eq(n0[3], 1.0, 4.0, 1e-50)?;

    // Check final number densities
    approx_eq(n[1], 0.0, 4.0, 1e-8)?;
    approx_eq(n[2], 2.0, 4.0, 1e-50)?;
    approx_eq(n[3], 2.0, 4.0, 1e-50)?;

    assert_eq!(na0, na);

    Ok(())
}

#[test]
fn asymmetric() -> Result<(), Box<dyn error::Error>> {
    // crate::utilities::test::setup_logging(4);

    let mut model = Empty::default();
    model.extend_particles(
        [1e5, 1e2, 1e1]
            .iter()
            .enumerate()
            .map(|(i, &m)| Particle::new(0, m, m / 100.0).name(format!("{}", i + 1))),
    );
    model
        .push_interaction(interaction::ThreeParticle::new(|_m| 1e-0, 1, 2, 3).asymmetry(|_m| 1e-3));

    // Get the initial conditions
    model.set_beta(BETA_START);
    let c = model.as_context();
    let (n0, _na0) = (c.eq, c.na);

    // Run the solver
    let (n, na) = solve("asymmetric.csv", SolverBuilder::new().model(model))?;

    // Check initial number densities
    approx_eq(n0[1], 1.0, 4.0, 1e-50)?;
    approx_eq(n0[2], 1.0, 4.0, 1e-50)?;
    approx_eq(n0[3], 1.0, 4.0, 1e-50)?;

    // Check final number densities
    approx_eq(n[1], 0.0, 4.0, 1e-8)?;
    approx_eq(n[2], 2.0, 4.0, 1e-50)?;
    approx_eq(n[3], 2.0, 4.0, 1e-50)?;

    approx_eq(na[1], -9.31e-29, 2.0, 0.0)?;
    approx_eq(na[2], 9.31e-29, 2.0, 0.0)?;
    approx_eq(na[3], 9.31e-29, 2.0, 0.0)?;

    // assert_eq!(na0, na);

    Ok(())
}

#[test]
fn chained_decay() -> Result<(), Box<dyn error::Error>> {
    // crate::utilities::test::setup_logging(4);

    let mut model = Empty::default();
    model.extend_particles(
        [1e7, 1e5, 1e2, 1e1]
            .iter()
            .enumerate()
            .map(|(i, &m)| Particle::new(0, m, m / 100.0).name(format!("{}", i + 1))),
    );
    model.push_interaction(interaction::ThreeParticle::new(|_m| 1e0, 1, 2, 3));
    model.push_interaction(interaction::ThreeParticle::new(|_m| 1e0, 2, 3, 4));

    // Get the initial conditions
    model.set_beta(BETA_START);
    let c = model.as_context();
    let (n0, na0) = (c.eq, c.na);

    // Run the solver
    let (n, na) = solve("chained_decay.csv", SolverBuilder::new().model(model))?;

    // Check initial number densities
    approx_eq(n0[1], 1.0, 4.0, 1e-50)?;
    approx_eq(n0[2], 1.0, 4.0, 1e-50)?;
    approx_eq(n0[3], 1.0, 4.0, 1e-50)?;
    approx_eq(n0[4], 1.0, 4.0, 1e-50)?;

    // Check final number densities
    approx_eq(n[1], 0.0, 4.0, 1e-3)?;
    approx_eq(n[2], 0.0, 4.0, 1e-3)?;
    approx_eq(n[3], 4.0, 2.0, 1e-50)?;
    approx_eq(n[4], 3.0, 2.0, 1e-50)?;

    assert_eq!(na0, na);

    Ok(())
}
