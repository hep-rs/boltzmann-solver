//! Tests with various parts of the calculation set to 1.

use super::{solve, BETA_END, BETA_START};
use crate::{
    model::{interaction, interaction::RateDensity, Empty, Interaction, Model, Particle},
    solver::{Context, SolverBuilder},
    utilities::test::approx_eq,
};
use std::error;

#[test]
fn amplitude() -> Result<(), Box<dyn error::Error>> {
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
    let (n, na) = solve("unit_amplitude.csv", SolverBuilder::new().model(model))?;

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

/// Test interaction between particles 1, 2 and 3 with `$\gamma = 1$` such that
/// the Botlzmann equations are of the form:
///
/// ```math
/// \ddfrac{n_1}{\beta} = \frac{1}{n_{BE} H \beta} \left(
///   \frac{n_2 n_3}{n_2^{(0)} n_3^{(0)}} - \frac{n_1}{n_1^{(0)}}
/// \right)
/// ```
#[derive(Debug)]
pub struct Gamma {
    particles: interaction::Particles,
}

impl Gamma {
    pub fn new() -> Self {
        let particles = interaction::Particles::new([1], [2, 3]);
        Self { particles }
    }
}

impl Interaction<Empty> for Gamma {
    fn particles(&self) -> &interaction::Particles {
        &self.particles
    }

    fn width_enabled(&self) -> bool {
        false
    }

    fn gamma_enabled(&self) -> bool {
        true
    }

    fn gamma(&self, _context: &Context<Empty>, _real: bool) -> Option<f64> {
        Some(1.0)
    }
}

/// Test a decay with a unit squared amplitude, with 1 -> 2, 3.
#[test]
fn gamma() -> Result<(), Box<dyn error::Error>> {
    // crate::utilities::test::setup_logging(4);

    let mut model = Empty::default();
    model.extend_particles(
        [1e5, 1e2, 1e1]
            .iter()
            .enumerate()
            .map(|(i, &m)| Particle::new(0, m, m / 100.0).name(format!("{}", i + 1))),
    );
    model.push_interaction(Gamma::new());

    // Get the initial conditions
    model.set_beta(BETA_START);
    let c = model.as_context();
    let (n0, na0) = (c.eq, c.na);

    // Run the solver
    let (n, na) = solve("unit_gamma.csv", SolverBuilder::new().model(model))?;

    // Check initial number densities
    approx_eq(n0[1], 1.0, 4.0, 1e-50)?;
    approx_eq(n0[2], 1.0, 4.0, 1e-50)?;
    approx_eq(n0[3], 1.0, 4.0, 1e-50)?;

    // Check final number densities
    approx_eq(n[1], 0.0, 4.0, 1e-9)?;
    approx_eq(n[2], 2.0, 4.0, 1e-50)?;
    approx_eq(n[3], 2.0, 4.0, 1e-50)?;

    assert_eq!(na0, na);

    Ok(())
}

/// Test interaction between particles 1, 2 and 3 with `$\gamma = 1$` such that
/// the Botlzmann equations are of the form:
///
/// ```math
/// \ddfrac{n_1}{\beta} = 1
/// \right)
/// ```
///
/// This is done before the check for overshoots.
#[derive(Debug)]
pub struct Rate {
    particles: interaction::Particles,
}

impl Rate {
    pub fn new() -> Self {
        let particles = interaction::Particles::new([1], [2, 3]);
        Self { particles }
    }
}

impl Interaction<Empty> for Rate {
    fn particles(&self) -> &interaction::Particles {
        &self.particles
    }

    fn width_enabled(&self) -> bool {
        false
    }

    fn gamma_enabled(&self) -> bool {
        true
    }

    fn gamma(&self, _context: &Context<Empty>, _real: bool) -> Option<f64> {
        None
    }

    fn rate(&self, _context: &Context<Empty>) -> Option<RateDensity> {
        let mut rate = RateDensity::zero();
        rate.gamma = 1.0;
        rate.symmetric = 1.0;
        Some(rate)
    }
}

/// Test a decay with a unit squared amplitude, with 1 -> 2, 3.
#[test]
fn rate() -> Result<(), Box<dyn error::Error>> {
    // crate::utilities::test::setup_logging(4);

    let mut model = Empty::default();
    model.extend_particles(
        [1e5, 1e2, 1e1]
            .iter()
            .enumerate()
            .map(|(i, &m)| Particle::new(0, m, m / 100.0).name(format!("{}", i + 1))),
    );
    model.push_interaction(Rate::new());

    // Run the solver
    let (n, _na) = solve("unit_rate.csv", SolverBuilder::new().model(model))?;

    // Check final number densities
    approx_eq(n[1], 9.178e-4, 4.0, 1e-4)?;
    approx_eq(n[2], 1.9991, 4.0, 1e-50)?;
    approx_eq(n[3], 1.9991, 4.0, 1e-50)?;

    Ok(())
}

/// Test interaction between particles 1, 2 and 3 with `$\gamma = 1$` such that
/// the Botlzmann equations are of the form:
///
/// ```math
/// \ddfrac{n_1}{\beta} = 1
/// \right)
/// ```
///
/// This is done after the checks for overshoots.
#[derive(Debug)]
pub struct AdjustedRate {
    particles: interaction::Particles,
}

impl AdjustedRate {
    pub fn new() -> Self {
        let particles = interaction::Particles::new([1], [2, 3]);
        Self { particles }
    }
}

impl Interaction<Empty> for AdjustedRate {
    fn particles(&self) -> &interaction::Particles {
        &self.particles
    }

    fn width_enabled(&self) -> bool {
        false
    }

    fn gamma_enabled(&self) -> bool {
        true
    }

    fn gamma(&self, _context: &Context<Empty>, _real: bool) -> Option<f64> {
        None
    }

    fn rate(&self, _context: &Context<Empty>) -> Option<RateDensity> {
        unimplemented!()
    }

    fn adjusted_rate(&self, context: &Context<Empty>) -> Option<RateDensity> {
        let mut rate = RateDensity::zero();
        rate.symmetric = 1.0;
        Some(rate * context.step_size)
    }
}

/// Test a decay with a unit squared amplitude, with 1 -> 2, 3.
#[test]
fn adjusted_rate() -> Result<(), Box<dyn error::Error>> {
    // crate::utilities::test::setup_logging(4);

    let mut model = Empty::default();
    model.extend_particles(
        [1e5, 1e2, 1e1]
            .iter()
            .enumerate()
            .map(|(i, &m)| Particle::new(0, m, m / 100.0).name(format!("{}", i + 1))),
    );
    model.push_interaction(AdjustedRate::new());

    // Run the solver
    let (n, _na) = solve("unit_adjusted_rate.csv", SolverBuilder::new().model(model))?;

    // Check final number densities
    approx_eq(n[1], 1.0 - BETA_END, 4.0, 1e-9)?;
    approx_eq(n[2], 1.0 + BETA_END, 4.0, 1e-50)?;
    approx_eq(n[3], 1.0 + BETA_END, 4.0, 1e-50)?;

    Ok(())
}
