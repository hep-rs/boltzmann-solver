//! 2 to 2 interactions

use super::{solve, BETA_START};
use crate::{
    model::{interaction, Empty, Particle},
    prelude::*,
    solver::SolverBuilder,
    utilities::test::approx_eq,
};
use std::error;

#[test]
fn massless() -> Result<(), Box<dyn error::Error>> {
    // crate::utilities::test::setup_logging(4);

    let mut model = Empty::default();
    model.extend_particles(
        [0.0, 0.0, 0.0, 0.0]
            .iter()
            .enumerate()
            .map(|(i, &m)| Particle::new(0, m, m / 100.0).name(format!("{}", i + 1))),
    );
    model.push_interaction(interaction::FourParticle::new(
        |_m, _s, _t, _u| 1e-10,
        1,
        2,
        3,
        4,
    ));

    // Get the initial conditions
    model.set_beta(BETA_START);
    let c = model.as_context();
    let (mut n0, na0) = (c.eq, c.na);
    n0[1] = 0.4;
    n0[2] = 0.8;
    n0[3] = 1.3;
    n0[4] = 1.8;

    // Run the solver
    let (n, na) = solve(
        "scattering_massless.csv",
        SolverBuilder::new().model(model).initial_densities([
            (1, n0[1]),
            (2, n0[2]),
            (3, n0[3]),
            (4, n0[4]),
        ]),
    )?;

    approx_eq(n[1] * n[2], n[3] * n[4], 2.0, 0.0)?;

    assert_eq!(na0, na);

    Ok(())
}

#[test]
fn massless_eq_1122() -> Result<(), Box<dyn error::Error>> {
    // crate::utilities::test::setup_logging(4);

    let mut model = Empty::default();
    model.extend_particles(
        [0.0, 0.0, 0.0, 0.0]
            .iter()
            .enumerate()
            .map(|(i, &m)| Particle::new(0, m, m / 100.0).name(format!("{}", i + 1))),
    );
    model.push_interaction(interaction::FourParticle::new(
        |_m, _s, _t, _u| 1e-10,
        1,
        1,
        2,
        2,
    ));

    // Get the initial conditions
    model.set_beta(BETA_START);
    let c = model.as_context();
    let (mut n0, na0) = (c.eq, c.na);
    n0[1] = 0.4;

    // Run the solver
    let (n, na) = solve(
        "scattering_massless_eq_1122.csv",
        SolverBuilder::new()
            .model(model)
            .initial_densities([(1, n0[1]), (2, n0[2]), (3, n0[3]), (4, n0[4])])
            .in_equilibrium([2])
            .fast_interaction(true),
    )?;

    approx_eq(n[1] * n[2], n[3] * n[4], 2.0, 0.0)?;

    assert_eq!(na0, na);

    Ok(())
}

#[test]
fn massless_eq_1234() -> Result<(), Box<dyn error::Error>> {
    // crate::utilities::test::setup_logging(4);

    let mut model = Empty::default();
    model.extend_particles(
        [0.0, 0.0, 0.0, 0.0]
            .iter()
            .enumerate()
            .map(|(i, &m)| Particle::new(0, m, m / 100.0).name(format!("{}", i + 1))),
    );
    model.push_interaction(interaction::FourParticle::new(
        |_m, _s, _t, _u| 1e-10,
        1,
        2,
        3,
        4,
    ));

    // Get the initial conditions
    model.set_beta(BETA_START);
    let c = model.as_context();
    let (mut n0, na0) = (c.eq, c.na);
    n0[1] = 0.4;
    n0[2] = 0.8;

    // Run the solver
    let (n, na) = solve(
        "scattering_massless_eq_1234.csv",
        SolverBuilder::new()
            .model(model)
            .initial_densities([(1, n0[1]), (2, n0[2]), (3, n0[3]), (4, n0[4])])
            .in_equilibrium([3, 4]),
    )?;

    approx_eq(n[1] * n[2], n[3] * n[4], 2.0, 0.0)?;

    assert_eq!(na0, na);

    Ok(())
}

#[test]
fn m000() -> Result<(), Box<dyn error::Error>> {
    // crate::utilities::test::setup_logging(4);

    let mut model = Empty::default();
    model.extend_particles(
        [1e5, 0.0, 0.0, 0.0]
            .iter()
            .enumerate()
            .map(|(i, &m)| Particle::new(0, m, m / 100.0).name(format!("{}", i + 1))),
    );
    model.push_interaction(interaction::FourParticle::new(
        |_m, _s, _t, _u| 1e-10,
        1,
        2,
        3,
        4,
    ));

    // Get the initial conditions
    model.set_beta(BETA_START);
    let c = model.as_context();
    let (n0, na0) = (c.eq, c.na);

    // Run the solver
    let (_n, na) = solve("scattering_m000.csv", SolverBuilder::new().model(model))?;

    // Check initial number densities
    approx_eq(n0[1], 1.0, 4.0, 1e-50)?;
    approx_eq(n0[2], 1.0, 4.0, 1e-50)?;
    approx_eq(n0[3], 1.0, 4.0, 1e-50)?;
    approx_eq(n0[4], 1.0, 4.0, 1e-50)?;

    // Check final number densities
    // approx_eq(n[1] * n[2], n[3] * n[4], 4.0, 1e-10)?;

    assert_eq!(na0, na);

    Ok(())
}

#[test]
fn m0m0() -> Result<(), Box<dyn error::Error>> {
    // crate::utilities::test::setup_logging(4);

    let mut model = Empty::default();
    model.extend_particles(
        [1e5, 0.0, 1e4, 0.0]
            .iter()
            .enumerate()
            .map(|(i, &m)| Particle::new(0, m, m / 100.0).name(format!("{}", i + 1))),
    );
    model.push_interaction(interaction::FourParticle::new(
        |_m, _s, _t, _u| 1e-10,
        1,
        2,
        3,
        4,
    ));

    // Get the initial conditions
    model.set_beta(BETA_START);
    let c = model.as_context();
    let (n0, na0) = (c.eq, c.na);

    // Run the solver
    let (n, na) = solve("scattering_m0m0.csv", SolverBuilder::new().model(model))?;

    // The overall change in both n[1] and n[3] should be the same (but oppose sign)
    approx_eq(n[1] - n0[1], n0[3] - n[3], 8.0, 1e-50)?;

    // Check final number densities
    approx_eq(n[1], 5.866e-2, 4.0, 1e-4)?;
    approx_eq(n[1], n[2], 4.0, 1e-10)?;
    approx_eq(n[3], 1.9413, 4.0, 1e-4)?;
    approx_eq(n[3], n[4], 4.0, 1e-50)?;

    assert_eq!(na0, na);

    Ok(())
}

#[test]
fn mm00() -> Result<(), Box<dyn error::Error>> {
    // crate::utilities::test::setup_logging(4);

    let mut model = Empty::default();
    model.extend_particles(
        [1e5, 1e5, 0.0, 0.0]
            .iter()
            .enumerate()
            .map(|(i, &m)| Particle::new(0, m, m / 100.0).name(format!("{}", i + 1))),
    );
    model.push_interaction(interaction::FourParticle::new(
        |_m, _s, _t, _u| 1e-10,
        1,
        2,
        3,
        4,
    ));

    // Get the initial conditions
    model.set_beta(BETA_START);
    let c = model.as_context();
    let (n0, na0) = (c.eq, c.na);

    // Run the solver
    let (n, na) = solve("scattering_mm00.csv", SolverBuilder::new().model(model))?;

    // The overall change in both n[1] and n[3] should be the same (but oppose sign)
    approx_eq(n[1] - n0[1], n0[3] - n[3], 8.0, 1e-50)?;

    // Check final number densities
    approx_eq(n[1], 0.27587, 4.0, 1e-4)?;
    approx_eq(n[1], n[2], 4.0, 1e-10)?;
    approx_eq(n[3], 1.7241, 4.0, 1e-4)?;
    approx_eq(n[3], n[4], 4.0, 1e-50)?;

    assert_eq!(na0, na);

    Ok(())
}

#[test]
fn mm00_eq() -> Result<(), Box<dyn error::Error>> {
    // crate::utilities::test::setup_logging(4);

    let mut model = Empty::default();
    model.extend_particles(
        [1e5, 1e5, 0.0, 0.0]
            .iter()
            .enumerate()
            .map(|(i, &m)| Particle::new(0, m, m / 100.0).name(format!("{}", i + 1))),
    );
    model.push_interaction(interaction::FourParticle::new(
        |_m, _s, _t, _u| 1e-10,
        1,
        2,
        3,
        4,
    ));

    // Get the initial conditions
    model.set_beta(BETA_START);
    let c = model.as_context();
    let (_n0, na0) = (c.eq, c.na);

    // Run the solver
    let (n, na) = solve(
        "scattering_mm00_eq.csv",
        SolverBuilder::new()
            .model(model)
            .initial_densities([(1, 1.3), (2, 1.1)])
            .in_equilibrium([3, 4]),
    )?;

    // The overall change in both n[1] and n[2] should be the same
    approx_eq(n[1] - 1.3, n[2] - 1.1, 8.0, 1e-50)?;

    // Check final number densities
    approx_eq(n[1], 0.3727, 4.0, 1e-10)?;
    approx_eq(n[2], 0.1727, 4.0, 1e-10)?;
    approx_eq(n[3], 1.0, 4.0, 1e-10)?;
    approx_eq(n[4], 1.0, 4.0, 1e-10)?;

    assert_eq!(na0, na);

    Ok(())
}

#[test]
fn mmmm() -> Result<(), Box<dyn error::Error>> {
    // crate::utilities::test::setup_logging(4);

    let mut model = Empty::default();
    model.extend_particles(
        [1e5, 1e5, 1e5, 1e5]
            .iter()
            .enumerate()
            .map(|(i, &m)| Particle::new(0, m, m / 100.0).name(format!("{}", i + 1))),
    );
    model.push_interaction(interaction::FourParticle::new(
        |_m, _s, _t, _u| 1e-10,
        1,
        2,
        3,
        4,
    ));

    // Get the initial conditions
    model.set_beta(BETA_START);
    let c = model.as_context();
    let (_n0, na0) = (c.eq, c.na);

    // Run the solver
    let (n, na) = solve(
        "scattering_mmmm.csv",
        SolverBuilder::new().model(model).initial_densities([
            (1, 1.3),
            (2, 1.1),
            (3, 0.9),
            (4, 0.6),
        ]),
    )?;

    // Check final number densities
    approx_eq(n[1] * n[2], n[3] * n[4], 2.0, 1e-5)?;

    assert_eq!(na0, na);

    Ok(())
}
