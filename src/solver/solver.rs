use crate::{
    model::{Model, Particle},
    solver::{options::StepPrecision, Context},
    statistic::{Statistic, Statistics},
    utilities::spline::rec_geomspace,
};
use ndarray::{prelude::*, Zip};
use rayon::prelude::*;
use std::{error, fmt, iter::FromIterator};

/// Error type returned by the solver builder in case there is an error.
#[derive(Debug)]
pub enum Error {
    /// The initial number densities are invalid.
    InvalidInitialDensities,
    /// The initial asymmetries are invalid.
    InvalidInitialAsymmetries,
    /// The number of particles held in equilibrium exceeds the number of
    /// particles in the model.
    TooManyInEquilibrium,
    /// The number of particles with no asymmetry exceeds the number of
    /// particles in the model.
    TooManyNoAsymmetry,
    /// The underlying model has not been specified
    UndefinedModel,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidInitialDensities => write!(f, "initial number densities are invalid"),
            Error::InvalidInitialAsymmetries => {
                write!(f, "initial number density asymmetries are invalid")
            }
            Error::TooManyInEquilibrium => {
                write!(f, "too many particles held in equilibrium for the model")
            }
            Error::TooManyNoAsymmetry => {
                write!(f, "too many particles without asymmetry for the model")
            }
            Error::UndefinedModel => write!(f, "underlying model is not defined"),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}

/// Boltzmann solver builder
pub struct SolverBuilder<M: Model> {
    model: Option<M>,
    initial_densities: Option<Array1<f64>>,
    initial_asymmetries: Option<Array1<f64>>,
    beta_range: (f64, f64),
    in_equilibrium: Vec<usize>,
    no_asymmetry: Vec<usize>,
    logger: Box<dyn Fn(&Context<M>)>,
    step_precision: StepPrecision,
    error_tolerance: f64,
}

/// Boltzmann solver
pub struct Solver<M: Model> {
    model: M,
    initial_densities: Array1<f64>,
    initial_asymmetries: Array1<f64>,
    beta_range: (f64, f64),
    in_equilibrium: Vec<usize>,
    no_asymmetry: Vec<usize>,
    logger: Box<dyn Fn(&Context<M>)>,
    step_precision: StepPrecision,
    error_tolerance: f64,
}

impl<M: Model> SolverBuilder<M> {
    /// Creates a new builder for the Boltzmann solver.
    ///
    /// The default range of temperatures span 1 GeV through to 10^{20} GeV, and
    /// it uses normalization by default.
    ///
    /// Most of the method for the builder are intended to be chained one after
    /// the other.
    ///
    /// ```
    /// use boltzmann_solver::prelude::*;
    /// use boltzmann_solver::model::StandardModel;
    ///
    /// let mut solver_builder: SolverBuilder<StandardModel> = SolverBuilder::new()
    ///     // .logger(..)
    ///     // .initial_densities(..)
    ///     .beta_range(1e-10, 1e-6);
    ///
    /// let solver = solver_builder.build();
    /// ```
    pub fn new() -> Self {
        Self {
            model: None,
            initial_densities: None,
            initial_asymmetries: None,
            beta_range: (1e-20, 1e0),
            in_equilibrium: Vec::new(),
            no_asymmetry: Vec::new(),
            logger: Box::new(|_| {}),
            step_precision: StepPrecision::default(),
            error_tolerance: 1e-4,
        }
    }

    /// Set a model function.
    ///
    /// Add a function that can modify the model before it is used by the
    /// context.  This is particularly useful if there is a baseline model with
    /// fixed parameters and only certain parameters are changed.
    pub fn model(mut self, model: M) -> Self {
        self.model = Some(model);
        self
    }

    /// Specify initial number densities explicitly.
    ///
    /// If unspecified, all number densities are assumed to be in equilibrium to
    /// begin with.
    ///
    /// The list of number densities must be in the same order as the particles
    /// in the model.
    pub fn initial_densities(mut self, n: Array1<f64>) -> Self {
        self.initial_densities = Some(n);
        self
    }

    /// Specify initial number density asymmetries explicitly.
    ///
    /// If unspecified, all asymmetries are assumed to be 0 to begin with.
    ///
    /// The list of number densities must be in the same order as the particles
    /// in the model.
    pub fn initial_asymmetries(mut self, na: Array1<f64>) -> Self {
        self.initial_asymmetries = Some(na);
        self
    }

    /// Set the range of inverse temperature values over which the solution is
    /// calculated.
    ///
    /// Inverse temperature must be provided in units of GeV^{-1}.
    ///
    /// This function has a convenience alternative called
    /// [`SolverBuilder::temperature_range`] allowing for the limits to be specified as
    /// temperature in the units of GeV.
    ///
    /// # Panics
    ///
    /// Panics if the starting value is larger than the final value.
    pub fn beta_range(mut self, start: f64, end: f64) -> Self {
        assert!(
            start < end,
            "The initial β must be smaller than the final β value."
        );
        self.beta_range = (start, end);
        self
    }

    /// Set the range of temperature values over which the phase space is
    /// evolved.
    ///
    /// Temperature must be provided in units of GeV.
    ///
    /// This function is a convenience alternative to [`Solver::beta_range`].
    ///
    /// # Panics
    ///
    /// Panics if the starting value is smaller than the final value.
    pub fn temperature_range(mut self, start: f64, end: f64) -> Self {
        assert!(
            start > end,
            "The initial temperature must be larger than the final temperature."
        );
        self.beta_range = (start.recip(), end.recip());
        self
    }

    /// Specify the particles which must remain in equilibrium.
    ///
    /// Particles which are held in equilibrium will never have their number
    /// density deviate from equilibrium, and the corresponding asymmetry never
    /// deviates from 0.
    ///
    /// These particles are specified by their index.
    pub fn in_equilibrium<I>(mut self, eq: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        self.in_equilibrium = eq.into_iter().collect();
        self
    }

    /// Specify the particles which never develop an asymmetry.
    ///
    /// These particles will never develop an asymmetry between their particle
    /// and antiparticle densities.  This does not prevent the number densities
    /// of the particle and antiparticles from deviating from their equilibrium.
    ///
    /// These particles are specified by their index.
    pub fn no_asymmetry<I>(mut self, na: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        self.no_asymmetry = na.into_iter().collect();
        self
    }

    /// Set the logger.
    ///
    /// The logger provides some insight into the numerical integration 'black
    /// box'.  Specifically, it is run at the start of each integration step and
    /// has access to the current value as a `&Array1`, the change from this
    /// step as a `&Solution`, and the current `Context` at the start of the
    /// integration step.  As a result, for the first step, the solution will be
    /// equal to the initial conditions.
    ///
    /// This is useful if one wants to track the evolution of the solutions and
    /// log these in a CSV file.
    pub fn logger<F: 'static>(mut self, f: F) -> Self
    where
        F: Fn(&Context<M>),
    {
        self.logger = Box::new(f);
        self
    }

    /// Specify how large or small the step size is allowed to become.
    ///
    /// The evolution of number densities are discretized in steps of \\(h\\)
    /// such that \\(\beta_{i+1} = \beta_{i} + h\\).  The algorithm will
    /// determine automatically the optimal step size \\(h\\) such that the
    /// error is deemed acceptable; however, one may wish to override this to
    /// prevent step sizes which are either too large or too small.
    ///
    /// The step precision sets the range of allowed values of \\(h\\) in
    /// proportion to the current value of \\(\beta\\):
    /// \\begin{equation}
    ///   p_\text{min} \beta < h < p_\text{max} \beta
    /// \\end{equation}
    ///
    /// The default values are `min = 1e-10` and `max = 1.0`.
    ///
    /// The relative step precision has a higher priority on the step size than
    /// the error.  That is, the step size will never be less than
    /// \\(p_\text{min} \beta\\) even if this results in a larger local error
    /// than desired.
    ///
    /// # Panic
    ///
    /// This will panic if `min >= max`.
    pub fn step_precision(mut self, min: f64, max: f64) -> Self {
        assert!(
            min < max,
            "Minimum step precision must be smaller than the maximum step precision."
        );
        self.step_precision = StepPrecision { min, max };
        self
    }

    /// Specify the local error tolerance.
    ///
    /// The algorithm will adjust the evolution step size such that the local
    /// error remains less than the specified the error tolerance.
    ///
    /// Note that the error is only ever estimated and thus may occasionally be
    /// inaccurate.  Furthermore, the [`Solver::step_precision`] takes
    /// precedence and thus if a large minimum step precision is requested, the
    /// local error may be larger than the error tolerance.
    pub fn error_tolerance(mut self, tol: f64) -> Self {
        assert!(tol > 0.0, "The tolerance must be greater than 0.");
        self.error_tolerance = tol;
        self
    }

    /// Build the Boltzmann solver.
    pub fn build(self) -> Result<Solver<M>, Error> {
        let mut model = self.model.ok_or(Error::UndefinedModel)?;
        model.set_beta(self.beta_range.0);
        let particles = model.particles();
        let beta_range = self.beta_range;

        // Get the initial conditions and check for their validity.  If they
        // weren't specified, assume everything to be in equilibrium.
        let initial_densities = self.initial_densities.unwrap_or_else(|| {
            Array1::from_iter(
                particles
                    .iter()
                    .map(|p| p.normalized_number_density(0.0, beta_range.0)),
            )
        });
        if initial_densities.len() != particles.len() {
            log::error!(
                "Initial densities is not the same length as the number of particles in the model."
            );
            return Err(Error::InvalidInitialDensities);
        } else if initial_densities
            .iter()
            .any(|v| !v.is_finite() || v.is_nan())
        {
            log::error!(
                "Some of initial densities are not finite or NaN:\n{:.3e}",
                initial_densities
            );
            return Err(Error::InvalidInitialDensities);
        }

        let initial_asymmetries = self
            .initial_asymmetries
            .unwrap_or_else(|| Array1::zeros(particles.len()));
        if initial_asymmetries.len() != particles.len() {
            log::error!("Initial asymmetries is not the same length as the number of particles in the model.");
            return Err(Error::InvalidInitialAsymmetries);
        } else if initial_asymmetries
            .iter()
            .any(|v| !v.is_finite() || v.is_nan())
        {
            log::error!(
                "Some of initial asymmetries are not finite or NaN:\n{:.3e}",
                initial_asymmetries
            );
            return Err(Error::InvalidInitialAsymmetries);
        }

        // Make sure that there aren't too many particles held in equilibrium
        if self.in_equilibrium.len() > particles.len() {
            log::error!("There are more particles held in equilibrium ({}) than particles in the model ({}).",
                        self.in_equilibrium.len(),
                        particles.len());
            return Err(Error::TooManyInEquilibrium);
        }
        if self.no_asymmetry.len() > particles.len() {
            log::error!(
                "There are more particles with 0 asymmetry ({}) than particles in the model ({}).",
                self.no_asymmetry.len(),
                particles.len()
            );
            return Err(Error::TooManyNoAsymmetry);
        }

        Ok(Solver {
            model,
            initial_densities,
            initial_asymmetries,
            beta_range,
            in_equilibrium: self.in_equilibrium,
            no_asymmetry: self.no_asymmetry,
            logger: self.logger,
            step_precision: self.step_precision,
            error_tolerance: self.error_tolerance,
        })
    }
}

impl<M: Model + Sync> Default for SolverBuilder<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: Model + Sync> Solver<M> {
    /// Precompute certain things so that the integration can happen faster.
    fn precompute(&mut self, n: u32) {
        log::info!("Pre-computing γ...");
        let zero = Array1::zeros(0);
        for (i, &beta) in vec![
            0.98 * self.beta_range.0,
            0.99 * self.beta_range.0,
            1.01 * self.beta_range.1,
            1.02 * self.beta_range.1,
        ]
        .iter()
        .chain(&rec_geomspace(self.beta_range.0, self.beta_range.1, n))
        .enumerate()
        {
            if i % 64 == 0 {
                log::debug!("Precomputing step {} / {}", i, 2usize.pow(n) + 4);
            } else {
                log::trace!("Precomputing step {} / {}", i, 2usize.pow(n) + 4);
            }
            self.model.set_beta(beta);
            // We could use `self.model.as_context()`; however, this allocated
            // new zero arrays every time.
            let c = self.context(0, 1.0, beta, &zero, &zero);

            self.model
                .interactions()
                .par_iter()
                .filter(|interaction| interaction.is_four_particle())
                .for_each(|interaction| {
                    interaction.gamma(&c);
                });
        }
    }

    /// Evolve the initial conditions by solving the PDEs.
    ///
    /// Note that this is not a true PDE solver, but instead converts the PDE
    /// into an ODE in time (or inverse temperature in this case), with the
    /// derivative in energy being calculated solely from the previous
    /// time-step.
    #[allow(clippy::cognitive_complexity)]
    pub fn solve(&mut self) -> (Array1<f64>, Array1<f64>) {
        // Precompute `gamma` for interactions that support it.
        self.precompute(10);

        use super::tableau::rk87::*;

        // Initialize all the variables that will be used in the integration
        let mut n = self.initial_densities.clone();
        let mut dn = Array1::zeros(n.dim());
        let mut dn_err = Array1::zeros(n.dim());
        let mut na = self.initial_asymmetries.clone();
        let mut dna = Array1::zeros(na.dim());
        let mut dna_err = Array1::zeros(na.dim());

        let mut k: [Array1<f64>; RK_S];
        let mut ka: [Array1<f64>; RK_S];
        unsafe {
            k = std::mem::MaybeUninit::uninit().assume_init();
            for ki in &mut k[..] {
                std::ptr::write(ki, Array1::zeros(n.dim()));
            }
            ka = std::mem::MaybeUninit::uninit().assume_init();
            for ki in &mut ka[..] {
                std::ptr::write(ki, Array1::zeros(n.dim()));
            }
        };

        let mut step = 0;
        let mut evals = 0;
        let mut beta = self.beta_range.0;
        let mut h = beta * f64::sqrt(self.step_precision.min * self.step_precision.max);
        let mut advance;

        while beta < self.beta_range.1 {
            step += 1;
            advance = false;
            dn.fill(0.0);
            dna.fill(0.0);
            dn_err.fill(0.0);
            dna_err.fill(0.0);

            // Ensure that h is within the desired range of step sizes.
            let h_on_beta = h / beta;
            if h_on_beta > self.step_precision.max {
                h = beta * self.step_precision.max;
                log::trace!("Step size too large, decreased h to {:.3e}", h);
            } else if h_on_beta < self.step_precision.min {
                h = beta * self.step_precision.min;
                log::debug!("Step size too small, increased h to {:.3e}", h);
                // Irrespective of the local error, if we're at the minimum step
                // size we will be integrating this step.
                advance = true;
            }

            // Log the progress of the integration
            if step % 100 == 0 {
                log::info!("Step {}, β = {:.4e}", step, beta);
            // log::info!("n = {:.3e}", n);
            } else if step % 10 == 0 {
                log::debug!("Step {}, , β = {:.4e}", step, beta);
            // log::debug!("n = {:.3e}", n);
            } else {
                log::trace!("Step {}, β = {:.4e}, h = {:.4e}", step, beta, h);
                // log::trace!("n = {:.3e}", n);
            }

            for i in 0..RK_S {
                evals += 1;

                // Compute the sub-step values
                let beta_i = beta + RK_C[i] * h;
                let ai = RK_A[i];
                let ni = (0..i).fold(n.clone(), |total, j| total + ai[j] * &k[j]);
                let nai = (0..i).fold(na.clone(), |total, j| total + ai[j] * &ka[j]);
                self.model.set_beta(beta_i);
                let ci = self.context(step, h, beta_i, &ni, &nai);

                // Compute k[i] and ka[i] from each interaction
                let (ki, kai) = self
                    .model
                    .interactions()
                    .par_iter()
                    .fold(
                        || (Array1::zeros(n.dim()), Array1::zeros(na.dim())),
                        |(dn, dna), interaction| interaction.change(dn, dna, &ci),
                    )
                    .reduce(
                        || (Array1::zeros(n.dim()), Array1::zeros(na.dim())),
                        |(dn, dna), (dni, dnai)| (dn + dni, dna + dnai),
                    );
                k[i] = ki;
                ka[i] = kai;

                // Set changes to zero for those particles in equilibrium
                for &eq in &self.in_equilibrium {
                    k[i][eq] = 0.0;
                    ka[i][eq] = 0.0;
                }
                for &eq in &self.no_asymmetry {
                    ka[i][eq] = 0.0;
                }

                let bi = RK_B[i];
                let ei = RK_E[i];
                Zip::from(&mut dn)
                    .and(&mut dn_err)
                    .and(&k[i])
                    .and(&mut dna)
                    .and(&mut dna_err)
                    .and(&ka[i])
                    .apply(|dn, dn_err, &ki, dna, dna_err, &kai| {
                        *dn += bi * ki;
                        *dn_err += ei * ki;
                        *dna += bi * kai;
                        *dna_err += ei * kai;
                    });
            }

            self.model.set_beta(beta);
            let c = self.context(step, h, beta, &n, &na);

            // Adjust dn for those particles in equilibrium
            for &eq in &self.in_equilibrium {
                dn[eq] = c.eq[eq] - n[eq];
                dna[eq] = -na[eq];
            }
            for &eq in &self.no_asymmetry {
                dna[eq] = -na[eq];
            }

            log::trace!("     dn = {:>10.3e}", dn);
            log::trace!("    dna = {:>10.3e}", dna);
            log::trace!(" dn_err = {:>10.3e}", dn_err);
            log::trace!("dna_err = {:>10.3e}", dna_err);

            // Get the local error using L∞-norm
            let err = dn_err
                .iter()
                .chain(dna_err.iter())
                .fold(0f64, |e, v| e.max(v.abs()));

            // If the error is within the tolerance, we'll be advancing the
            // iteration step
            if err < self.error_tolerance {
                advance = true;
            } else {
                log::trace!("Error is not within tolerance.");
            }

            // Compute the change in step size based on the current error and
            // correspondingly adjust the step size
            let delta = if err == 0.0 {
                10.0
            } else {
                0.9 * (self.error_tolerance / err).powf(1.0 / f64::from(RK_ORDER + 1))
            };
            let mut h_est = h * delta;

            // Update n and beta
            if advance {
                // Advance n and beta
                n += &dn;
                na += &dna;
                beta += h;

                (*self.logger)(&c);
            } else {
                log::trace!("Discarding integration step.");
            }

            // Adjust final integration step if needed
            if beta + h_est > self.beta_range.1 {
                log::trace!("Fixing overshoot of last integration step.");
                h_est = self.beta_range.1 - beta;
            }

            h = h_est;
        }

        log::info!("Number of integration steps: {}", step);
        log::info!("Number of evaluations: {}", evals);

        (n, na)
    }
}

impl<M: Model> Solver<M> {
    /// Generate the context at a given beta to pass to the logger/interaction
    /// functions.
    fn context(
        &self,
        step: u64,
        step_size: f64,
        beta: f64,
        n: &Array1<f64>,
        na: &Array1<f64>,
    ) -> Context<M> {
        let hubble_rate = self.model.hubble_rate(beta);
        let normalization =
            (hubble_rate * beta * Statistic::BoseEinstein.massless_number_density(0.0, beta))
                .recip();
        Context {
            step,
            step_size,
            beta,
            hubble_rate,
            normalization,
            eq: equilibrium_number_densities(self.model.particles(), beta),
            n: n.clone(),
            na: na.clone(),
            model: &self.model,
        }
    }
}

/// Create an array containing the equilibrium number densities of the model's
/// particles at the specified temperature.
///
/// All particles species are assumed to be in thermal equilibrium at this
/// energy, with the distribution following either the Bose–Einstein or
/// Fermi–Dirac distribution as determined by their spin.
///
/// The `normalized` flag
fn equilibrium_number_densities<'a, I>(particles: I, beta: f64) -> Array1<f64>
where
    I: IntoIterator<Item = &'a Particle>,
{
    Array1::from_iter(
        particles
            .into_iter()
            .map(|p| p.normalized_number_density(0.0, beta)),
    )
}
