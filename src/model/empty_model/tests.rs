use crate::{
    model::{
        interaction,
        interaction::{InteractionParticles, RateDensity},
        EmptyModel, Interaction, Model, Particle,
    },
    solver::{Context, SolverBuilder},
    utilities::test::approx_eq,
    utilities::test::setup_logging,
};
use std::{env::temp_dir, error, fs, path::Path, sync::RwLock};

const BETA_START: f64 = 1e-10;
const BETA_END: f64 = 1e1;

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
        rate.symmetric = 1.0;
        Some(rate)
    }
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

/// First of the two coupled interactions:
///
/// ```math
/// \ddfrac{n_1}{\beta} = - n_2 \\
/// \ddfrac{n_2}{\beta} = n_1
/// ```
#[derive(Debug)]
pub struct CoupledRate1 {
    particles: InteractionParticles,
}

impl CoupledRate1 {
    pub fn new() -> Self {
        let particles = InteractionParticles::new(&[], &[1]);
        Self { particles }
    }
}

impl Interaction<EmptyModel> for CoupledRate1 {
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
pub struct CoupledRate2 {
    particles: InteractionParticles,
}

impl CoupledRate2 {
    pub fn new() -> Self {
        let particles = InteractionParticles::new(&[], &[2]);
        Self { particles }
    }
}

impl Interaction<EmptyModel> for CoupledRate2 {
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

fn create_csv<P: AsRef<Path>>(
    p: P,
    n: usize,
) -> Result<RwLock<csv::Writer<fs::File>>, Box<dyn error::Error>> {
    let mut csv = csv::Writer::from_path(temp_dir().join(p))?;
    csv.write_field("beta")?;
    for i in 1..=n {
        csv.write_field(format!("{}", i))?;
        csv.write_field(format!("({})", i))?;
    }
    csv.write_record(None::<&[u8]>)?;
    Ok(RwLock::new(csv))
}

#[test]
fn no_interaction() -> Result<(), Box<dyn error::Error>> {
    setup_logging(3);

    let mut model = EmptyModel::default();
    model.push_particle(Particle::new(0, 1e5, 1e1));
    model.push_particle(Particle::new(0, 1e3, 1e1));

    // Get the initial conditions
    model.set_beta(BETA_START);
    let c = model.as_context();
    let (n0, na0) = (c.eq, c.na);

    let (n, na) = SolverBuilder::new()
        .model(model)
        .beta_range(BETA_START, 1e1)
        .build()?
        .solve();

    // Check that number densities have no changed.
    assert_eq!(n0, n);
    assert_eq!(na0, na);

    Ok(())
}

/// Test a decay with a unit squared amplitude and the provided interaction.
#[test]
fn unit_amplitude() -> Result<(), Box<dyn error::Error>> {
    setup_logging(1);

    let mut model = EmptyModel::default();
    model.extend_particles(
        [1e5, 1e2, 1e1]
            .iter()
            .map(|&m| Particle::new(0, m, m / 100.0)),
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
        .logger(move |c| {
            let mut csv = csv.write().unwrap();
            if c.n.iter().any(|v| v.is_nan()) {
                csv.flush().unwrap();
                panic!("NaN at step {}", c.step);
            };
            csv.serialize([c.beta, c.n[1], c.eq[1], c.n[2], c.eq[2], c.n[3], c.eq[3]])
                .unwrap();
        })
        .build()?
        .solve();

    // Check initial number densities
    approx_eq(n0[1], 1.0, 8.0, 1e-50)?;
    approx_eq(n0[2], 1.0, 8.0, 1e-50)?;
    approx_eq(n0[3], 1.0, 8.0, 1e-50)?;

    // Check final number densities
    approx_eq(n[1], 0.0, 8.0, 1e-10)?;
    approx_eq(n[2], 2.0, 8.0, 1e-50)?;
    approx_eq(n[3], 2.0, 8.0, 1e-50)?;

    assert_eq!(na0, na);

    Ok(())
}

/// Test a decay with a unit squared amplitude, with 1 -> 2, 3.
#[test]
fn unit_gamma() -> Result<(), Box<dyn error::Error>> {
    setup_logging(1);

    let mut model = EmptyModel::default();

    model.extend_particles(
        [1e5, 1e2, 1e1]
            .iter()
            .map(|&m| Particle::new(0, m, m / 100.0)),
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
        .logger(move |c| {
            let mut csv = csv.write().unwrap();
            if c.n.iter().any(|v| v.is_nan()) {
                csv.flush().unwrap();
                panic!("NaN at step {}", c.step);
            };
            csv.serialize([c.beta, c.n[1], c.eq[1], c.n[2], c.eq[2], c.n[3], c.eq[3]])
                .unwrap();
        })
        .build()?
        .solve();

    // Check initial number densities
    approx_eq(n0[1], 1.0, 8.0, 1e-50)?;
    approx_eq(n0[2], 1.0, 8.0, 1e-50)?;
    approx_eq(n0[3], 1.0, 8.0, 1e-50)?;

    // Check final number densities
    approx_eq(n[1], 0.0, 8.0, 1e-9)?;
    approx_eq(n[2], 2.0, 8.0, 1e-50)?;
    approx_eq(n[3], 2.0, 8.0, 1e-50)?;

    assert_eq!(na0, na);

    Ok(())
}

/// Test a decay with a unit squared amplitude, with 1 -> 2, 3.
#[test]
fn unit_rate() -> Result<(), Box<dyn error::Error>> {
    setup_logging(1);

    let mut model = EmptyModel::default();

    model.extend_particles(
        [1e5, 1e2, 1e1]
            .iter()
            .map(|&m| Particle::new(0, m, m / 100.0)),
    );
    model.push_interaction(UnitRate::new());

    // Run the solver
    let csv = create_csv("unit_rate.csv", 3)?;
    let (n, _na) = SolverBuilder::new()
        .model(model)
        .beta_range(BETA_START, BETA_END)
        .logger(move |c| {
            let mut csv = csv.write().unwrap();
            if c.n.iter().any(|v| v.is_nan()) {
                csv.flush().unwrap();
                panic!("NaN at step {}", c.step);
            };
            csv.serialize([c.beta, c.n[1], c.eq[1], c.n[2], c.eq[2], c.n[3], c.eq[3]])
                .unwrap();
        })
        .build()?
        .solve();

    // Check final number densities
    approx_eq(n[1], 1.0 - (BETA_END - BETA_START), 8.0, 1e-10)?;
    approx_eq(n[2], 1.0 + (BETA_END - BETA_START), 8.0, 1e-50)?;
    approx_eq(n[3], 1.0 + (BETA_END - BETA_START), 8.0, 1e-50)?;

    Ok(())
}

/// Test a decay with a unit squared amplitude, with 1 -> 2, 3.
#[test]
fn unit_adj_rate() -> Result<(), Box<dyn error::Error>> {
    let mut model = EmptyModel::default();

    model.extend_particles(
        [1e5, 1e2, 1e1]
            .iter()
            .map(|&m| Particle::new(0, m, m / 100.0)),
    );
    model.push_interaction(UnitAdjRate::new());

    // Run the solver
    let csv = create_csv("unit_adj_rate.csv", 3)?;
    let (n, _na) = SolverBuilder::new()
        .model(model)
        .beta_range(BETA_START, BETA_END)
        .logger(move |c| {
            let mut csv = csv.write().unwrap();
            if c.n.iter().any(|v| v.is_nan()) {
                csv.flush().unwrap();
                panic!("NaN at step {}", c.step);
            };
            csv.serialize([c.beta, c.n[1], c.eq[1], c.n[2], c.eq[2], c.n[3], c.eq[3]])
                .unwrap();
        })
        .build()?
        .solve();

    // Check final number densities
    approx_eq(n[1], 1.0 - BETA_END, 4.0, 1e-9)?;
    approx_eq(n[2], 1.0 + BETA_END, 4.0, 1e-50)?;
    approx_eq(n[3], 1.0 + BETA_END, 4.0, 1e-50)?;

    Ok(())
}

/// Coupled rate to produce sin and cosine
#[test]
fn coupled_rate() -> Result<(), Box<dyn error::Error>> {
    let mut model = EmptyModel::default();

    model.extend_particles([1e5, 1e2].iter().map(|&m| Particle::new(0, m, m / 100.0)));
    model.push_interaction(CoupledRate1::new());
    model.push_interaction(CoupledRate2::new());

    // Run the solver
    let csv = create_csv("coupled_rate.csv", 3)?;
    let (n, _na) = SolverBuilder::new()
        .model(model)
        .beta_range(BETA_START, BETA_END)
        .logger(move |c| {
            let mut csv = csv.write().unwrap();
            if c.n.iter().any(|v| v.is_nan()) {
                csv.flush().unwrap();
                panic!("NaN at step {}", c.step);
            };
            csv.serialize([c.beta, c.n[1], c.eq[1], c.n[2], c.eq[2], 0.0, 0.0])
                .unwrap();
        })
        .initial_densities(vec![(1, 0.0), (2, 1.0)])
        .build()?
        .solve();

    // Check final number densities
    approx_eq(n[1], BETA_END.sin(), 3.0, 1e-10)?;
    approx_eq(n[2], BETA_END.cos(), 3.0, 1e-10)?;

    Ok(())
}