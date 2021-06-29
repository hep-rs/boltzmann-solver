use crate::{
    model::{
        interaction,
        interaction::{InteractionParticles, RateDensity},
        EmptyModel, Interaction, Model, Particle,
    },
    solver::{Context, SolverBuilder},
    utilities::test::approx_eq,
};
use std::{env::temp_dir, error, fs, path::Path, sync::RwLock};

const BETA_START: f64 = 1e-10;
const BETA_END: f64 = 1e1;

/// Shorthand to create the CSV file in the appropriate directory and with
/// headers.
fn create_csv<P: AsRef<Path>>(
    p: P,
    n: usize,
) -> Result<RwLock<csv::Writer<fs::File>>, Box<dyn error::Error>> {
    let dir = temp_dir().join("boltzmann-solver").join("empty-model");
    fs::create_dir_all(&dir)?;

    let mut csv = csv::Writer::from_path(dir.join(p))?;
    csv.write_field("step")?;
    csv.write_field("beta")?;
    csv.write_field("dn")?;
    for i in 1..=n {
        csv.write_field(format!("n-{}", i))?;
        csv.write_field(format!("eq-{}", i))?;
    }
    csv.write_record(None::<&[u8]>)?;
    Ok(RwLock::new(csv))
}

#[test]
fn no_interaction() -> Result<(), Box<dyn error::Error>> {
    let mut model = EmptyModel::default();
    model.push_particle(Particle::new(0, 1e5, 1e1).name("1"));
    model.push_particle(Particle::new(0, 1e3, 1e1).name("2"));

    // Get the initial conditions
    model.set_beta(BETA_START);
    let c = model.as_context();
    let (n0, na0) = (c.eq, c.na);

    let (n, na) = SolverBuilder::new()
        .model(model)
        .beta_range(BETA_START, 1e1)
        .fast_interaction(true)
        .build()?
        .solve()?;

    // Check that number densities have not changed.
    assert_eq!(n0, n);
    assert_eq!(na0, na);

    Ok(())
}

#[test]
fn unit_amplitude() -> Result<(), Box<dyn error::Error>> {
    let mut model = EmptyModel::default();
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
    let csv = create_csv("unit_amplitude.csv", 3)?;
    let (n, na) = SolverBuilder::new()
        .model(model)
        .beta_range(BETA_START, BETA_END)
        .logger(move |c, dn, _dna| {
            let mut csv = csv.write().unwrap();
            if c.n.iter().any(|v| v.is_nan()) {
                csv.flush().unwrap();
                panic!("NaN at step {}.{}", c.step, c.substep);
            };
            csv.serialize((
                c.step, c.beta, dn[1], c.n[1], c.eq[1], c.n[2], c.eq[2], c.n[3], c.eq[3],
            ))
            .unwrap();
            csv.flush().unwrap();
        })
        .fast_interaction(true)
        .build()?
        .solve()?;

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
pub struct UnitGamma {
    particles: InteractionParticles,
}

impl UnitGamma {
    pub fn new() -> Self {
        let particles = InteractionParticles::new(&[1], &[2, 3]);
        Self { particles }
    }
}

impl Interaction<EmptyModel> for UnitGamma {
    fn particles(&self) -> &InteractionParticles {
        &self.particles
    }

    fn width_enabled(&self) -> bool {
        false
    }

    fn gamma_enabled(&self) -> bool {
        true
    }

    fn gamma(&self, _c: &Context<EmptyModel>, _real: bool) -> Option<f64> {
        Some(1.0)
    }
}

/// Test a decay with a unit squared amplitude, with 1 -> 2, 3.
#[test]
fn unit_gamma() -> Result<(), Box<dyn error::Error>> {
    let mut model = EmptyModel::default();
    model.extend_particles(
        [1e5, 1e2, 1e1]
            .iter()
            .enumerate()
            .map(|(i, &m)| Particle::new(0, m, m / 100.0).name(format!("{}", i + 1))),
    );
    model.push_interaction(UnitGamma::new());

    // Get the initial conditions
    model.set_beta(BETA_START);
    let c = model.as_context();
    let (n0, na0) = (c.eq, c.na);

    // Run the solver
    let csv = create_csv("unit_gamma.csv", 3)?;
    let (n, na) = SolverBuilder::new()
        .model(model)
        .beta_range(BETA_START, BETA_END)
        .logger(move |c, dn, _dna| {
            let mut csv = csv.write().unwrap();
            if c.n.iter().any(|v| v.is_nan()) {
                csv.flush().unwrap();
                panic!("NaN at step {}.{}", c.step, c.substep);
            };
            csv.serialize((
                c.step, c.beta, dn[1], c.n[1], c.eq[1], c.n[2], c.eq[2], c.n[3], c.eq[3],
            ))
            .unwrap();
        })
        .fast_interaction(true)
        .build()?
        .solve()?;

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
pub struct UnitRate {
    particles: InteractionParticles,
}

impl UnitRate {
    pub fn new() -> Self {
        let particles = InteractionParticles::new(&[1], &[2, 3]);
        Self { particles }
    }
}

impl Interaction<EmptyModel> for UnitRate {
    fn particles(&self) -> &InteractionParticles {
        &self.particles
    }

    fn width_enabled(&self) -> bool {
        false
    }

    fn gamma_enabled(&self) -> bool {
        true
    }

    fn gamma(&self, _c: &Context<EmptyModel>, _real: bool) -> Option<f64> {
        None
    }

    fn rate(&self, _c: &Context<EmptyModel>) -> Option<RateDensity> {
        let mut rate = RateDensity::zero();
        rate.gamma = 1.0;
        rate.symmetric = 1.0;
        Some(rate)
    }
}

/// Test a decay with a unit squared amplitude, with 1 -> 2, 3.
#[test]
fn unit_rate() -> Result<(), Box<dyn error::Error>> {
    let mut model = EmptyModel::default();
    model.extend_particles(
        [1e5, 1e2, 1e1]
            .iter()
            .enumerate()
            .map(|(i, &m)| Particle::new(0, m, m / 100.0).name(format!("{}", i + 1))),
    );
    model.push_interaction(UnitRate::new());

    // Run the solver
    let csv = create_csv("unit_rate.csv", 3)?;
    let (n, _na) = SolverBuilder::new()
        .model(model)
        .beta_range(BETA_START, BETA_END)
        .logger(move |c, dn, _dna| {
            let mut csv = csv.write().unwrap();
            if c.n.iter().any(|v| v.is_nan()) {
                csv.flush().unwrap();
                panic!("NaN at step {}.{}", c.step, c.substep);
            };
            csv.serialize((
                c.step, c.beta, dn[1], c.n[1], c.eq[1], c.n[2], c.eq[2], c.n[3], c.eq[3],
            ))
            .unwrap();
        })
        .fast_interaction(true)
        .build()?
        .solve()?;

    // Check final number densities
    approx_eq(n[1], 0.0, 4.0, 1e-4)?;
    approx_eq(n[2], 2.0, 4.0, 1e-50)?;
    approx_eq(n[3], 2.0, 4.0, 1e-50)?;

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
pub struct UnitAdjRate {
    particles: InteractionParticles,
}

impl UnitAdjRate {
    pub fn new() -> Self {
        let particles = InteractionParticles::new(&[1], &[2, 3]);
        Self { particles }
    }
}

impl Interaction<EmptyModel> for UnitAdjRate {
    fn particles(&self) -> &InteractionParticles {
        &self.particles
    }

    fn width_enabled(&self) -> bool {
        false
    }

    fn gamma_enabled(&self) -> bool {
        true
    }

    fn gamma(&self, _c: &Context<EmptyModel>, _real: bool) -> Option<f64> {
        None
    }

    fn rate(&self, _c: &Context<EmptyModel>) -> Option<RateDensity> {
        unimplemented!()
    }

    fn adjusted_rate(&self, c: &Context<EmptyModel>) -> Option<RateDensity> {
        let mut rate = RateDensity::zero();
        rate.symmetric = 1.0;
        Some(rate * c.step_size)
    }
}

/// Test a decay with a unit squared amplitude, with 1 -> 2, 3.
#[test]
fn unit_adj_rate() -> Result<(), Box<dyn error::Error>> {
    let mut model = EmptyModel::default();
    model.extend_particles(
        [1e5, 1e2, 1e1]
            .iter()
            .enumerate()
            .map(|(i, &m)| Particle::new(0, m, m / 100.0).name(format!("{}", i + 1))),
    );
    model.push_interaction(UnitAdjRate::new());

    // Run the solver
    let csv = create_csv("unit_adj_rate.csv", 3)?;
    let (n, _na) = SolverBuilder::new()
        .model(model)
        .beta_range(BETA_START, BETA_END)
        .logger(move |c, dn, _dna| {
            let mut csv = csv.write().unwrap();
            if c.n.iter().any(|v| v.is_nan()) {
                csv.flush().unwrap();
                panic!("NaN at step {}.{}", c.step, c.substep);
            };
            csv.serialize((
                c.step, c.beta, dn[1], c.n[1], c.eq[1], c.n[2], c.eq[2], c.n[3], c.eq[3],
            ))
            .unwrap();
        })
        .fast_interaction(true)
        .build()?
        .solve()?;

    // Check final number densities
    approx_eq(n[1], 1.0 - BETA_END, 4.0, 1e-9)?;
    approx_eq(n[2], 1.0 + BETA_END, 4.0, 1e-50)?;
    approx_eq(n[3], 1.0 + BETA_END, 4.0, 1e-50)?;

    Ok(())
}

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

impl Interaction<EmptyModel> for SinCos1 {
    fn particles(&self) -> &InteractionParticles {
        &self.particles
    }

    fn width_enabled(&self) -> bool {
        false
    }

    fn gamma_enabled(&self) -> bool {
        true
    }

    fn gamma(&self, _c: &Context<EmptyModel>, _real: bool) -> Option<f64> {
        None
    }

    fn rate(&self, _c: &Context<EmptyModel>) -> Option<RateDensity> {
        unimplemented!()
    }

    fn adjusted_rate(&self, c: &Context<EmptyModel>) -> Option<RateDensity> {
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

impl Interaction<EmptyModel> for SinCos2 {
    fn particles(&self) -> &InteractionParticles {
        &self.particles
    }

    fn width_enabled(&self) -> bool {
        false
    }

    fn gamma_enabled(&self) -> bool {
        true
    }

    fn gamma(&self, _c: &Context<EmptyModel>, _real: bool) -> Option<f64> {
        None
    }

    fn rate(&self, _c: &Context<EmptyModel>) -> Option<RateDensity> {
        unimplemented!()
    }

    fn adjusted_rate(&self, c: &Context<EmptyModel>) -> Option<RateDensity> {
        let mut rate = RateDensity::zero();
        rate.symmetric -= c.n[1];
        Some(rate * c.step_size)
    }
}

/// Coupled rate to produce sin and cosine
#[test]
fn sin_cos() -> Result<(), Box<dyn error::Error>> {
    let mut model = EmptyModel::default();
    model.push_particle(Particle::new(0, 1.0, 1.0).name("1"));
    model.push_particle(Particle::new(0, 1.0, 1.0).name("2"));
    model.push_interaction(SinCos1::new());
    model.push_interaction(SinCos2::new());

    // Run the solver
    let csv = create_csv("sin_cos.csv", 3)?;
    let (n, _na) = SolverBuilder::new()
        .model(model)
        .beta_range(BETA_START, BETA_END)
        .logger(move |c, dn, _dna| {
            let mut csv = csv.write().unwrap();
            if c.n.iter().any(|v| v.is_nan()) {
                csv.flush().unwrap();
                panic!("NaN at step {}.{}", c.step, c.substep);
            };
            csv.serialize((
                c.step, c.beta, dn[1], c.n[1], c.eq[1], c.n[2], c.eq[2], 0.0, 0.0,
            ))
            .unwrap();
        })
        .initial_densities([(1, 0.0), (2, 1.0)])
        .fast_interaction(true)
        .build()?
        .solve()?;

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

impl Interaction<EmptyModel> for BrusselatorStable1 {
    fn particles(&self) -> &InteractionParticles {
        &self.particles
    }

    fn width_enabled(&self) -> bool {
        false
    }

    fn gamma_enabled(&self) -> bool {
        true
    }

    fn gamma(&self, _c: &Context<EmptyModel>, _real: bool) -> Option<f64> {
        None
    }

    fn rate(&self, _c: &Context<EmptyModel>) -> Option<RateDensity> {
        unimplemented!()
    }

    fn adjusted_rate(&self, c: &Context<EmptyModel>) -> Option<RateDensity> {
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

impl Interaction<EmptyModel> for BrusselatorStable2 {
    fn particles(&self) -> &InteractionParticles {
        &self.particles
    }

    fn width_enabled(&self) -> bool {
        false
    }

    fn gamma_enabled(&self) -> bool {
        true
    }

    fn gamma(&self, _c: &Context<EmptyModel>, _real: bool) -> Option<f64> {
        None
    }

    fn rate(&self, _c: &Context<EmptyModel>) -> Option<RateDensity> {
        unimplemented!()
    }

    fn adjusted_rate(&self, c: &Context<EmptyModel>) -> Option<RateDensity> {
        let mut rate = RateDensity::zero();
        rate.symmetric = 1.7 * c.n[1] - c.n[1].powi(2) * c.n[2];
        Some(rate * c.step_size)
    }
}

/// Coupled rate to produce sin and cosine
#[test]
fn brusselator_stable() -> Result<(), Box<dyn error::Error>> {
    let mut model = EmptyModel::default();
    model.push_particle(Particle::new(0, 1.0, 1.0).name("1"));
    model.push_particle(Particle::new(0, 1.0, 1.0).name("2"));
    model.push_interaction(BrusselatorStable1::new());
    model.push_interaction(BrusselatorStable2::new());

    // Run the solver
    let csv = create_csv("brusselator_stable.csv", 3)?;
    let (n, _na) = SolverBuilder::new()
        .model(model)
        .beta_range(BETA_START, BETA_END)
        .logger(move |c, dn, _dna| {
            let mut csv = csv.write().unwrap();
            if c.n.iter().any(|v| v.is_nan()) {
                csv.flush().unwrap();
                panic!("NaN at step {}.{}", c.step, c.substep);
            };
            csv.serialize((
                c.step, c.beta, dn[1], c.n[1], c.eq[1], c.n[2], c.eq[2], 0.0, 0.0,
            ))
            .unwrap();
        })
        .initial_densities([(1, 1.0), (2, 1.0)])
        .fast_interaction(true)
        .build()?
        .solve()?;

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

impl Interaction<EmptyModel> for BrusselatorUnstable1 {
    fn particles(&self) -> &InteractionParticles {
        &self.particles
    }

    fn width_enabled(&self) -> bool {
        false
    }

    fn gamma_enabled(&self) -> bool {
        true
    }

    fn gamma(&self, _c: &Context<EmptyModel>, _real: bool) -> Option<f64> {
        None
    }

    fn rate(&self, _c: &Context<EmptyModel>) -> Option<RateDensity> {
        unimplemented!()
    }

    fn adjusted_rate(&self, c: &Context<EmptyModel>) -> Option<RateDensity> {
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

impl Interaction<EmptyModel> for BrusselatorUnstable2 {
    fn particles(&self) -> &InteractionParticles {
        &self.particles
    }

    fn width_enabled(&self) -> bool {
        false
    }

    fn gamma_enabled(&self) -> bool {
        true
    }

    fn gamma(&self, _c: &Context<EmptyModel>, _real: bool) -> Option<f64> {
        None
    }

    fn rate(&self, _c: &Context<EmptyModel>) -> Option<RateDensity> {
        unimplemented!()
    }

    fn adjusted_rate(&self, c: &Context<EmptyModel>) -> Option<RateDensity> {
        let mut rate = RateDensity::zero();
        rate.symmetric = 3.0 * c.n[1] - c.n[1].powi(2) * c.n[2];
        Some(rate * c.step_size)
    }
}

/// Coupled rate to produce sin and cosine
#[test]
fn brusselator_unstable() -> Result<(), Box<dyn error::Error>> {
    let mut model = EmptyModel::default();
    model.push_particle(Particle::new(0, 1.0, 1.0).name("1"));
    model.push_particle(Particle::new(0, 1.0, 1.0).name("2"));
    model.push_interaction(BrusselatorUnstable1::new());
    model.push_interaction(BrusselatorUnstable2::new());

    // Run the solver
    let csv = create_csv("brusselator_unstable.csv", 3)?;
    let (n, _na) = SolverBuilder::new()
        .model(model)
        .beta_range(BETA_START, BETA_END)
        .logger(move |c, dn, _dna| {
            let mut csv = csv.write().unwrap();
            if c.n.iter().any(|v| v.is_nan()) {
                csv.flush().unwrap();
                panic!("NaN at step {}.{}", c.step, c.substep);
            };
            csv.serialize((
                c.step, c.beta, dn[1], c.n[1], c.eq[1], c.n[2], c.eq[2], 0.0, 0.0,
            ))
            .unwrap();
        })
        .initial_densities([(1, 1.0), (2, 1.0)])
        .fast_interaction(true)
        .build()?
        .solve()?;

    // Check final number densities
    approx_eq(n[1], 0.373_265, 3.0, 1e-10)?;
    approx_eq(n[2], 3.36134, 3.0, 1e-10)?;

    Ok(())
}

#[test]
fn chained_decay() -> Result<(), Box<dyn error::Error>> {
    let mut model = EmptyModel::default();
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
    let csv = create_csv("chained_decay.csv", 4)?;
    let (n, na) = SolverBuilder::new()
        .model(model)
        .beta_range(BETA_START, BETA_END)
        .logger(move |c, dn, _dna| {
            let mut csv = csv.write().unwrap();
            if c.n.iter().any(|v| v.is_nan()) {
                csv.flush().unwrap();
                panic!("NaN at step {}.{}", c.step, c.substep);
            };
            csv.serialize((
                c.step, c.beta, dn[1], c.n[1], c.eq[1], c.n[2], c.eq[2], c.n[3], c.eq[3], c.n[4],
                c.eq[4],
            ))
            .unwrap();
            csv.flush().unwrap();
        })
        .fast_interaction(true)
        .build()?
        .solve()?;

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

#[test]
fn scattering_massless() -> Result<(), Box<dyn error::Error>> {
    let mut model = EmptyModel::default();
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
    let csv = create_csv("scattering_massless.csv", 4)?;
    let (n, na) = SolverBuilder::new()
        .model(model)
        .beta_range(BETA_START, BETA_END)
        .logger(move |c, dn, _dna| {
            let mut csv = csv.write().unwrap();
            if c.n.iter().any(|v| v.is_nan()) {
                csv.flush().unwrap();
                panic!("NaN at step {}.{}", c.step, c.substep);
            };
            csv.serialize((
                c.step, c.beta, dn[1], c.n[1], c.eq[1], c.n[2], c.eq[2], c.n[3], c.eq[3], c.n[4],
                c.eq[4],
            ))
            .unwrap();
            csv.flush().unwrap();
        })
        .initial_densities([(1, n0[1]), (2, n0[2]), (3, n0[3]), (4, n0[4])])
        .fast_interaction(true)
        .build()?
        .solve()?;

    approx_eq(n[1] * n[2], n[3] * n[4], 2.0, 0.0)?;

    assert_eq!(na0, na);

    Ok(())
}

#[test]
fn scattering_massless_eq_1122() -> Result<(), Box<dyn error::Error>> {
    let mut model = EmptyModel::default();
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
    let csv = create_csv("scattering_massless_eq_1122.csv", 4)?;
    let (n, na) = SolverBuilder::new()
        .model(model)
        .beta_range(BETA_START, BETA_END)
        .logger(move |c, dn, _dna| {
            let mut csv = csv.write().unwrap();
            if c.n.iter().any(|v| v.is_nan()) {
                csv.flush().unwrap();
                panic!("NaN at step {}.{}", c.step, c.substep);
            };
            csv.serialize((
                c.step, c.beta, dn[1], c.n[1], c.eq[1], c.n[2], c.eq[2], c.n[3], c.eq[3], c.n[4],
                c.eq[4],
            ))
            .unwrap();
            csv.flush().unwrap();
        })
        .initial_densities([(1, n0[1]), (2, n0[2]), (3, n0[3]), (4, n0[4])])
        .in_equilibrium([2])
        .fast_interaction(true)
        .build()?
        .solve()?;

    approx_eq(n[1] * n[2], n[3] * n[4], 2.0, 0.0)?;

    assert_eq!(na0, na);

    Ok(())
}

#[test]
fn scattering_massless_eq_1234() -> Result<(), Box<dyn error::Error>> {
    let mut model = EmptyModel::default();
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
    let csv = create_csv("scattering_massless_eq_1234.csv", 4)?;
    let (n, na) = SolverBuilder::new()
        .model(model)
        .beta_range(BETA_START, BETA_END)
        .logger(move |c, dn, _dna| {
            let mut csv = csv.write().unwrap();
            if c.n.iter().any(|v| v.is_nan()) {
                csv.flush().unwrap();
                panic!("NaN at step {}.{}", c.step, c.substep);
            };
            csv.serialize((
                c.step, c.beta, dn[1], c.n[1], c.eq[1], c.n[2], c.eq[2], c.n[3], c.eq[3], c.n[4],
                c.eq[4],
            ))
            .unwrap();
            csv.flush().unwrap();
        })
        .initial_densities([(1, n0[1]), (2, n0[2]), (3, n0[3]), (4, n0[4])])
        .in_equilibrium([3, 4])
        .fast_interaction(true)
        .build()?
        .solve()?;

    approx_eq(n[1] * n[2], n[3] * n[4], 2.0, 0.0)?;

    assert_eq!(na0, na);

    Ok(())
}

#[test]
fn scattering_m000() -> Result<(), Box<dyn error::Error>> {
    let mut model = EmptyModel::default();
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
    let csv = create_csv("scattering_m000.csv", 4)?;
    let (_n, na) = SolverBuilder::new()
        .model(model)
        .beta_range(BETA_START, BETA_END)
        .logger(move |c, dn, _dna| {
            let mut csv = csv.write().unwrap();
            if c.n.iter().any(|v| v.is_nan()) {
                csv.flush().unwrap();
                panic!("NaN at step {}.{}", c.step, c.substep);
            };
            csv.serialize((
                c.step, c.beta, dn[1], c.n[1], c.eq[1], c.n[2], c.eq[2], c.n[3], c.eq[3], c.n[4],
                c.eq[4],
            ))
            .unwrap();
            csv.flush().unwrap();
        })
        .fast_interaction(true)
        .build()?
        .solve()?;

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
fn scattering_m0m0() -> Result<(), Box<dyn error::Error>> {
    let mut model = EmptyModel::default();
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
    let csv = create_csv("scattering_m0m0.csv", 4)?;
    let (n, na) = SolverBuilder::new()
        .model(model)
        .beta_range(BETA_START, BETA_END)
        .logger(move |c, dn, _dna| {
            let mut csv = csv.write().unwrap();
            if c.n.iter().any(|v| v.is_nan()) {
                csv.flush().unwrap();
                panic!("NaN at step {}.{}", c.step, c.substep);
            };
            csv.serialize((
                c.step, c.beta, dn[1], c.n[1], c.eq[1], c.n[2], c.eq[2], c.n[3], c.eq[3], c.n[4],
                c.eq[4],
            ))
            .unwrap();
            csv.flush().unwrap();
        })
        .fast_interaction(true)
        .build()?
        .solve()?;

    // Check initial number densities
    approx_eq(n0[1], 1.0, 4.0, 1e-50)?;
    approx_eq(n0[2], 1.0, 4.0, 1e-50)?;
    approx_eq(n0[3], 1.0, 4.0, 1e-50)?;
    approx_eq(n0[4], 1.0, 4.0, 1e-50)?;

    // Check final number densities
    approx_eq(n[1], 0.05074, 4.0, 1e-4)?;
    approx_eq(n[1], n[2], 4.0, 1e-10)?;
    approx_eq(n[3], 1.9493, 4.0, 1e-4)?;
    approx_eq(n[3], n[4], 4.0, 1e-50)?;

    assert_eq!(na0, na);

    Ok(())
}

#[test]
fn scattering_mm00() -> Result<(), Box<dyn error::Error>> {
    let mut model = EmptyModel::default();
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
    let csv = create_csv("scattering_mm00.csv", 4)?;
    let (n, na) = SolverBuilder::new()
        .model(model)
        .beta_range(BETA_START, BETA_END)
        .logger(move |c, dn, _dna| {
            let mut csv = csv.write().unwrap();
            if c.n.iter().any(|v| v.is_nan()) {
                csv.flush().unwrap();
                panic!("NaN at step {}.{}", c.step, c.substep);
            };
            csv.serialize((
                c.step, c.beta, dn[1], c.n[1], c.eq[1], c.n[2], c.eq[2], c.n[3], c.eq[3], c.n[4],
                c.eq[4],
            ))
            .unwrap();
            csv.flush().unwrap();
        })
        .fast_interaction(true)
        .build()?
        .solve()?;

    // Check initial number densities
    approx_eq(n0[1], 1.0, 4.0, 1e-50)?;
    approx_eq(n0[2], 1.0, 4.0, 1e-50)?;
    approx_eq(n0[3], 1.0, 4.0, 1e-50)?;
    approx_eq(n0[4], 1.0, 4.0, 1e-50)?;

    // Check final number densities
    approx_eq(n[1], 0.276, 4.0, 1e-4)?;
    approx_eq(n[1], n[2], 4.0, 1e-10)?;
    approx_eq(n[3], 1.724, 4.0, 1e-4)?;
    approx_eq(n[3], n[4], 4.0, 1e-50)?;

    assert_eq!(na0, na);

    Ok(())
}

#[test]
fn scattering_mmmm() -> Result<(), Box<dyn error::Error>> {
    let mut model = EmptyModel::default();
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
    let csv = create_csv("scattering_mmmm.csv", 4)?;
    let (n, na) = SolverBuilder::new()
        .model(model)
        .beta_range(BETA_START, BETA_END)
        .logger(move |c, dn, _dna| {
            let mut csv = csv.write().unwrap();
            if c.n.iter().any(|v| v.is_nan()) {
                csv.flush().unwrap();
                panic!("NaN at step {}.{}", c.step, c.substep);
            };
            csv.serialize((
                c.step, c.beta, dn[1], c.n[1], c.eq[1], c.n[2], c.eq[2], c.n[3], c.eq[3], c.n[4],
                c.eq[4],
            ))
            .unwrap();
            csv.flush().unwrap();
        })
        .initial_densities([(1, 1.3), (2, 1.1), (3, 0.9), (4, 0.6)])
        .fast_interaction(true)
        .build()?
        .solve()?;

    // Check final number densities
    approx_eq(n[1] * n[2], n[3] * n[4], 2.0, 1e-5)?;

    assert_eq!(na0, na);

    Ok(())
}

#[test]
fn scattering_mm00_eq() -> Result<(), Box<dyn error::Error>> {
    let mut model = EmptyModel::default();
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
    let csv = create_csv("scattering_mm00_eq.csv", 4)?;
    let (n, na) = SolverBuilder::new()
        .model(model)
        .beta_range(BETA_START, BETA_END)
        .logger(move |c, dn, _dna| {
            let mut csv = csv.write().unwrap();
            if c.n.iter().any(|v| v.is_nan()) {
                csv.flush().unwrap();
                panic!("NaN at step {}.{}", c.step, c.substep);
            };
            csv.serialize((
                c.step, c.beta, dn[1], c.n[1], c.eq[1], c.n[2], c.eq[2], c.n[3], c.eq[3], c.n[4],
                c.eq[4],
            ))
            .unwrap();
            csv.flush().unwrap();
        })
        .initial_densities([(1, 1.3), (2, 1.1)])
        .in_equilibrium([3, 4])
        .fast_interaction(true)
        .build()?
        .solve()?;

    // Check initial number densities
    approx_eq(n0[1], 1.0, 4.0, 1e-50)?;
    approx_eq(n0[2], 1.0, 4.0, 1e-50)?;
    approx_eq(n0[3], 1.0, 4.0, 1e-50)?;
    approx_eq(n0[4], 1.0, 4.0, 1e-50)?;

    // Check final number densities
    approx_eq(n[1], 0.3772, 4.0, 1e-10)?;
    approx_eq(n[2], 0.1772, 4.0, 1e-10)?;
    approx_eq(n[3], 1.0, 4.0, 1e-10)?;
    approx_eq(n[4], 1.0, 4.0, 1e-10)?;

    assert_eq!(na0, na);

    Ok(())
}
