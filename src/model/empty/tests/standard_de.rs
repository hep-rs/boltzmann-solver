//! Differential equation unrelated to Boltzmann equation to ensure that the ODE
//! solver works as expected.

use super::{solve, BETA_END};
use crate::{
    model::{
        interaction::{InteractionParticles, RateDensity},
        Empty, Interaction, Particle,
    },
    solver::{Context, SolverBuilder},
    utilities::test::approx_eq,
};
use std::error;

/// First of the two coupled interactions:
///
/// ```math
/// \ddfrac{n_1}{\beta} = - n_2 \\
/// \ddfrac{n_2}{\beta} = n_1
/// ```
#[derive(Debug)]
pub struct SinCos1 {
    particles: InteractionParticles,
}

impl SinCos1 {
    pub fn new() -> Self {
        let particles = InteractionParticles::new(&[], &[1]);
        Self { particles }
    }
}

impl Interaction<Empty> for SinCos1 {
    fn particles(&self) -> &InteractionParticles {
        &self.particles
    }

    fn width_enabled(&self) -> bool {
        false
    }

    fn gamma_enabled(&self) -> bool {
        true
    }

    fn gamma(&self, _c: &Context<Empty>, _real: bool) -> Option<f64> {
        None
    }

    fn rate(&self, _c: &Context<Empty>) -> Option<RateDensity> {
        unimplemented!()
    }

    fn adjusted_rate(&self, c: &Context<Empty>) -> Option<RateDensity> {
        let mut rate = RateDensity::zero();
        rate.symmetric = c.n[2];
        Some(rate * c.step_size)
    }
}

/// Second of the two coupled interactions:
///
/// ```math
/// \ddfrac{n_1}{\beta} = - n_2 \\
/// \ddfrac{n_2}{\beta} = n_1
/// ```
#[derive(Debug)]
pub struct SinCos2 {
    particles: InteractionParticles,
}

impl SinCos2 {
    pub fn new() -> Self {
        let particles = InteractionParticles::new(&[], &[2]);
        Self { particles }
    }
}

impl Interaction<Empty> for SinCos2 {
    fn particles(&self) -> &InteractionParticles {
        &self.particles
    }

    fn width_enabled(&self) -> bool {
        false
    }

    fn gamma_enabled(&self) -> bool {
        true
    }

    fn gamma(&self, _c: &Context<Empty>, _real: bool) -> Option<f64> {
        None
    }

    fn rate(&self, _c: &Context<Empty>) -> Option<RateDensity> {
        unimplemented!()
    }

    fn adjusted_rate(&self, c: &Context<Empty>) -> Option<RateDensity> {
        let mut rate = RateDensity::zero();
        rate.symmetric -= c.n[1];
        Some(rate * c.step_size)
    }
}

/// Coupled rate to produce sin and cosine
#[test]
fn sin_cos() -> Result<(), Box<dyn error::Error>> {
    // crate::utilities::test::setup_logging(2);

    let mut model = Empty::default();
    model.push_particle(Particle::new(0, 1.0, 1.0).name("1"));
    model.push_particle(Particle::new(0, 1.0, 1.0).name("2"));
    model.push_interaction(SinCos1::new());
    model.push_interaction(SinCos2::new());

    // Run the solver
    let (n, _na) = solve(
        "sin_cos.csv",
        SolverBuilder::new()
            .model(model)
            .initial_densities([(1, 0.0), (2, 1.0)]),
    )?;

    // Check final number densities
    approx_eq(n[1], BETA_END.sin(), 3.0, 1e-10)?;
    approx_eq(n[2], BETA_END.cos(), 3.0, 1e-10)?;

    Ok(())
}

/// Stable brusselator
#[derive(Debug)]
pub struct BrusselatorStable1 {
    particles: InteractionParticles,
}

impl BrusselatorStable1 {
    pub fn new() -> Self {
        let particles = InteractionParticles::new(&[], &[1]);
        Self { particles }
    }
}

impl Interaction<Empty> for BrusselatorStable1 {
    fn particles(&self) -> &InteractionParticles {
        &self.particles
    }

    fn width_enabled(&self) -> bool {
        false
    }

    fn gamma_enabled(&self) -> bool {
        true
    }

    fn gamma(&self, _c: &Context<Empty>, _real: bool) -> Option<f64> {
        None
    }

    fn rate(&self, _c: &Context<Empty>) -> Option<RateDensity> {
        unimplemented!()
    }

    fn adjusted_rate(&self, c: &Context<Empty>) -> Option<RateDensity> {
        let mut rate = RateDensity::zero();
        rate.symmetric = 1.0 + c.n[1].powi(2) * c.n[2] - 1.7 * c.n[1] - c.n[1];
        Some(rate * c.step_size)
    }
}

/// Stable brusselatorStable
#[derive(Debug)]
pub struct BrusselatorStable2 {
    particles: InteractionParticles,
}

impl BrusselatorStable2 {
    pub fn new() -> Self {
        let particles = InteractionParticles::new(&[], &[2]);
        Self { particles }
    }
}

impl Interaction<Empty> for BrusselatorStable2 {
    fn particles(&self) -> &InteractionParticles {
        &self.particles
    }

    fn width_enabled(&self) -> bool {
        false
    }

    fn gamma_enabled(&self) -> bool {
        true
    }

    fn gamma(&self, _c: &Context<Empty>, _real: bool) -> Option<f64> {
        None
    }

    fn rate(&self, _c: &Context<Empty>) -> Option<RateDensity> {
        unimplemented!()
    }

    fn adjusted_rate(&self, c: &Context<Empty>) -> Option<RateDensity> {
        let mut rate = RateDensity::zero();
        rate.symmetric = 1.7 * c.n[1] - c.n[1].powi(2) * c.n[2];
        Some(rate * c.step_size)
    }
}

/// Coupled rate to produce sin and cosine
#[test]
fn brusselator_stable() -> Result<(), Box<dyn error::Error>> {
    // crate::utilities::test::setup_logging(2);

    let mut model = Empty::default();
    model.push_particle(Particle::new(0, 1.0, 1.0).name("1"));
    model.push_particle(Particle::new(0, 1.0, 1.0).name("2"));
    model.push_interaction(BrusselatorStable1::new());
    model.push_interaction(BrusselatorStable2::new());

    // Run the solver
    let (n, _na) = solve(
        "brusselator_stable.csv",
        SolverBuilder::new()
            .model(model)
            .initial_densities([(1, 1.0), (2, 1.0)]),
    )?;

    // Check final number densities
    approx_eq(n[1], 0.972_098, 3.0, 1e-10)?;
    approx_eq(n[2], 1.83244, 3.0, 1e-10)?;

    Ok(())
}
/// Stable brusselator
#[derive(Debug)]
pub struct BrusselatorUnstable1 {
    particles: InteractionParticles,
}

impl BrusselatorUnstable1 {
    pub fn new() -> Self {
        let particles = InteractionParticles::new(&[], &[1]);
        Self { particles }
    }
}

impl Interaction<Empty> for BrusselatorUnstable1 {
    fn particles(&self) -> &InteractionParticles {
        &self.particles
    }

    fn width_enabled(&self) -> bool {
        false
    }

    fn gamma_enabled(&self) -> bool {
        true
    }

    fn gamma(&self, _c: &Context<Empty>, _real: bool) -> Option<f64> {
        None
    }

    fn rate(&self, _c: &Context<Empty>) -> Option<RateDensity> {
        unimplemented!()
    }

    fn adjusted_rate(&self, c: &Context<Empty>) -> Option<RateDensity> {
        let mut rate = RateDensity::zero();
        rate.symmetric = 1.0 + c.n[1].powi(2) * c.n[2] - 3.0 * c.n[1] - c.n[1];
        Some(rate * c.step_size)
    }
}

/// Unstable brusselator
#[derive(Debug)]
pub struct BrusselatorUnstable2 {
    particles: InteractionParticles,
}

impl BrusselatorUnstable2 {
    pub fn new() -> Self {
        let particles = InteractionParticles::new(&[], &[2]);
        Self { particles }
    }
}

impl Interaction<Empty> for BrusselatorUnstable2 {
    fn particles(&self) -> &InteractionParticles {
        &self.particles
    }

    fn width_enabled(&self) -> bool {
        false
    }

    fn gamma_enabled(&self) -> bool {
        true
    }

    fn gamma(&self, _c: &Context<Empty>, _real: bool) -> Option<f64> {
        None
    }

    fn rate(&self, _c: &Context<Empty>) -> Option<RateDensity> {
        unimplemented!()
    }

    fn adjusted_rate(&self, c: &Context<Empty>) -> Option<RateDensity> {
        let mut rate = RateDensity::zero();
        rate.symmetric = 3.0 * c.n[1] - c.n[1].powi(2) * c.n[2];
        Some(rate * c.step_size)
    }
}

/// Coupled rate to produce sin and cosine
#[test]
fn brusselator_unstable() -> Result<(), Box<dyn error::Error>> {
    // crate::utilities::test::setup_logging(2);

    let mut model = Empty::default();
    model.push_particle(Particle::new(0, 1.0, 1.0).name("1"));
    model.push_particle(Particle::new(0, 1.0, 1.0).name("2"));
    model.push_interaction(BrusselatorUnstable1::new());
    model.push_interaction(BrusselatorUnstable2::new());

    // Run the solver
    let (n, _na) = solve(
        "brusselator_unstable.csv",
        SolverBuilder::new()
            .model(model)
            .initial_densities([(1, 1.0), (2, 1.0)]),
    )?;

    // Check final number densities
    approx_eq(n[1], 0.373_265, 3.0, 1e-10)?;
    approx_eq(n[2], 3.36134, 3.0, 1e-10)?;

    Ok(())
}
