use crate::{
    model::{interaction::Interaction, ModelInteractions, ParticleData},
    solver::{
        options::{ErrorTolerance, StepPrecision},
        Context, LoggerFn, Solver,
    },
};
use ndarray::prelude::*;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    error, fmt,
};

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
    /// Duplicate itneraction
    DuplicateInteractions,
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
            Error::DuplicateInteractions => write!(f, "duplicate interaction"),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}

/// Boltzmann solver builder
#[allow(clippy::module_name_repetitions)]
pub struct SolverBuilder<M> {
    /// Inner model
    pub model: Option<M>,
    /// Initial number densities
    initial_densities: HashMap<usize, f64>,
    /// Initial number density asymmetries
    initial_asymmetries: HashMap<usize, f64>,
    /// Range of beta values to integrate over
    beta_range: (f64, f64),
    /// Particles which are not allowed to deviate from equilibrium
    in_equilibrium: HashSet<usize>,
    /// Particles which are not allowed to have any asymmetry
    no_asymmetry: HashSet<usize>,
    /// Logger used at each step
    logger: LoggerFn<M>,
    /// Step precision (i.e. step size) allowed
    step_precision: StepPrecision,
    /// Local error allowed to determine step size as absolute and relative.
    error_tolerance: ErrorTolerance,
    /// The number of points to sample to build the spline, or 0 of disabling the precompute.
    precompute: usize,
    /// Whether fast interactions are enabled
    fast_interactions: bool,
    /// Whether to abort the integration if the step size becomes too small
    abort_when_inaccurate: bool,
}

impl<M> SolverBuilder<M> {
    /// Creates a new builder for the Boltzmann solver.
    ///
    /// The default range of temperatures span 1 GeV through to 10`$^{20}$` GeV, and
    /// it uses normalization by default.
    ///
    /// Most of the method for the builder are intended to be chained one after
    /// the other.
    ///
    /// ```rust
    /// use boltzmann_solver::prelude::*;
    /// use boltzmann_solver::model::Standard;
    ///
    /// let mut solver_builder: SolverBuilder<Standard> = SolverBuilder::new()
    ///     // .logger(..)
    ///     // .initial_densities(..)
    ///     .beta_range(1e-10, 1e-6);
    ///
    /// // let solver = solver_builder.build();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            model: None,
            initial_densities: HashMap::new(),
            initial_asymmetries: HashMap::new(),
            beta_range: (1e-20, 1e0),
            in_equilibrium: HashSet::new(),
            no_asymmetry: HashSet::new(),
            logger: Box::new(|_, _, _| {}),
            step_precision: StepPrecision::default(),
            error_tolerance: ErrorTolerance::default(),
            precompute: 1,
            fast_interactions: true,
            abort_when_inaccurate: false,
        }
    }

    /// Set a model function.
    ///
    /// Add a function that can modify the model before it is used by the
    /// context.  This is particularly useful if there is a baseline model with
    /// fixed parameters and only certain parameters are changed.
    #[must_use]
    pub fn model(mut self, model: M) -> Self {
        self.model = Some(model);
        self
    }

    /// Specify initial number densities explicitly.
    ///
    /// The specification is given as an iterator of `(usize, f64)` tuples where
    /// the first entry is the index of the particle whose equilibrium is
    /// specified.
    ///
    /// If unspecified, all number densities are assumed to be in equilibrium to
    /// begin with.
    ///
    /// Repeated initial conditions are combined together with any duplicates
    /// overiding previous calls.
    #[must_use]
    pub fn initial_densities<I>(mut self, n: I) -> Self
    where
        I: IntoIterator<Item = (usize, f64)>,
    {
        self.initial_densities.extend(n);
        self
    }

    /// Specify initial number density asymmetries explicitly.
    ///
    /// The specification is given as an iterator of `(usize, f64)` tuples where
    /// the first entry is the index of the particle whose asymmetry is
    /// specified.
    ///
    /// If unspecified, all asymmetries are assumed to be 0 to begin with.
    ///
    /// Repeated initial conditions are combined together with any duplicates
    /// overiding previous calls.
    #[must_use]
    pub fn initial_asymmetries<I>(mut self, na: I) -> Self
    where
        I: IntoIterator<Item = (usize, f64)>,
    {
        self.initial_asymmetries.extend(na);
        self
    }

    /// Set the range of inverse temperature values over which the solution is
    /// calculated.
    ///
    /// Inverse temperature must be provided in units of GeV`$^{-1}$`.
    ///
    /// This function has a convenience alternative called
    /// [`SolverBuilder::temperature_range`] allowing for the limits to be
    /// specified as temperature in the units of GeV.
    ///
    /// # Panics
    ///
    /// Panics if the starting value is larger than the final value.
    #[must_use]
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
    #[must_use]
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
    /// density deviate from equilibrium.  The corresponding asymmetry may
    /// deviate from 0 unless this is also pegged by
    /// [`SolverBuilder::no_asymmetry`].
    ///
    /// These particles are specified by their index.
    ///
    /// Multiple calls of this function will combine the results together.
    #[must_use]
    pub fn in_equilibrium<I>(mut self, eq: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        self.in_equilibrium.extend(eq);
        self
    }

    /// Specify the particles which never develop an asymmetry.
    ///
    /// These particles will never develop an asymmetry between their particle
    /// and antiparticle densities.  This does not prevent the number densities
    /// of the particle and antiparticles from deviating from their equilibrium.
    ///
    /// These particles are specified by their index.
    ///
    /// Multiple calls of this function will combine the results together.
    #[must_use]
    pub fn no_asymmetry<I>(mut self, na: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        self.no_asymmetry.extend(na);
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
    #[must_use]
    pub fn logger<F>(mut self, f: F) -> Self
    where
        F: Fn(&Context<M>, &Array1<f64>, &Array1<f64>) + 'static,
    {
        self.logger = Box::new(f);
        self
    }

    /// Specify how large or small the step size is allowed to become.
    ///
    /// The evolution of number densities are discretized in steps of `$h$` such
    /// that `$\beta_{i+1} = \beta_{i} + h$`.  The algorithm will determine
    /// automatically the optimal step size `$h$` such that the error is deemed
    /// acceptable; however, one may wish to override this to prevent step sizes
    /// which are either too large or too small.
    ///
    /// The step precision sets the range of allowed values of `$h$` in
    /// proportion to the current value of `$\beta$`:
    /// ```math
    /// p_\text{min} \beta < h < p_\text{max} \beta
    /// ```
    ///
    /// The relative step precision has a higher priority on the step size than
    /// the error.  That is, the step size will never be less than
    /// `$p_\text{min} \beta$` even if this results in a larger local error than
    /// desired.
    ///
    /// The minimum value for `$p_\text{min}$` is automatically adjusted such
    /// that the integration will keep increasing, albeit extremely slowly.
    ///
    /// # Panics
    ///
    /// This will panic if `$p_\text{min} \geq p_\text{max}$` or if
    /// `$p_\text{min}$` is negative.
    #[must_use]
    pub fn step_precision(mut self, min: f64, max: f64) -> Self {
        assert!(
            min < max,
            "Minimum step precision must be smaller than the maximum step precision."
        );
        assert!(min >= 0.0, "Minimum step precision cannot be negative.");
        self.step_precision = StepPrecision {
            min: min.max(f64::EPSILON),
            max,
        };
        self
    }

    /// Specify the local error tolerance.
    ///
    /// The algorithm will adjust the evolution step size such that the local
    /// error remains less than the specified the error tolerance. Specifically,
    /// given absolute and relative tolerances `$\varepsilon_\text{abs}$` and
    /// `$\varepsilon_\text{rel}$`, the local error be less than
    ///
    /// ```math
    /// \text{error} \leq \varepsilon_\text{abs} + \varepsilon_\text{rel} \cdot y
    /// ```
    ///
    /// where `$y$` is the relevant value that is being adjusted.
    ///
    /// Note that the error is only ever estimated and thus may occasionally be
    /// inaccurate.  Furthermore, the [`Solver::step_precision`] takes
    /// precedence and thus if a large minimum step precision is requested, the
    /// local error may be larger than the error tolerance.
    ///
    /// # Panics
    ///
    /// Panics if the tolerance is less than or equal to 0.
    #[must_use]
    pub fn error_tolerance(mut self, abs: f64, rel: f64) -> Self {
        assert!(abs > 0.0, "The absolute tolerance must be greater than 0.");
        assert!(rel > 0.0, "The relative tolerance must be greater than 0.");
        self.error_tolerance = ErrorTolerance { abs, rel };
        self
    }

    /// Upon calling [`SolverBuilder::build`], set whether interactions should
    /// be precomputed.
    ///
    /// If a number greater than 0 is provided, this sets the number of data
    /// points to compute in order to produce the spline.  Note that there is a
    /// lower bound of
    ///
    /// ```math
    /// 4 \left\lceil \log_{2} \frac{\beta_\text{end}}{\beta_\text{start}} \right\rceil
    /// ```
    ///
    /// for the number of points computed (which is approximately 13 points per
    /// decade), thus a value of 1 will use the default.
    ///
    /// If set to 0, the precomputation is disabled entirely.  Instead,
    /// computation are stored as they are computed which will still speed up
    /// cases where an integration step is repeated with a smaller step size,
    /// but will not otherwise help much.
    ///
    /// By default, interactions are precomputed.
    #[must_use]
    pub fn precompute(mut self, v: usize) -> Self {
        self.precompute = v;
        self
    }

    /// Specify whether to use fast interactions.
    ///
    /// # Warning
    ///
    /// This feature is currently experimental and may not produce reliable results.
    #[must_use]
    pub fn fast_interaction(mut self, v: bool) -> Self {
        self.fast_interactions = v;
        self
    }

    /// Specify whether local inaccuracies cause the integration to abort.
    #[must_use]
    pub fn abort_when_inaccurate(mut self, v: bool) -> Self {
        self.abort_when_inaccurate = v;
        self
    }

    /// Check the validity of the initial densities, making sure we have the
    /// right number of initial conditions and they are all finite.
    fn generate_initial_densities<'a, I>(
        beta: f64,
        particles: &[ParticleData],
        initial_densities: I,
    ) -> Result<Array1<f64>, Error>
    where
        I: IntoIterator<Item = (&'a usize, &'a f64)>,
    {
        let mut n: Array1<_> = particles
            .iter()
            .map(|p| p.normalized_number_density(beta, 0.0))
            .collect();

        for (&i, &ni) in initial_densities {
            if let Some(v) = n.get_mut(i) {
                *v = ni;
            } else {
                log::error!("Out of bounds initial density.");
                return Err(Error::InvalidInitialDensities);
            }
        }

        if n.iter().any(|v| !v.is_finite() || v.is_nan()) {
            log::error!("Some of initial densities are infinite or NaN:\n{:.3e}", n);
            return Err(Error::InvalidInitialDensities);
        }

        Ok(n)
    }

    /// Check the validity of the initial asymmetries, making sure we have the
    /// right number of initial conditions and they are all finite.
    fn generate_initial_asymmetries<'a, I>(
        particles: &[ParticleData],
        initial_asymmetries: I,
    ) -> Result<Array1<f64>, Error>
    where
        I: IntoIterator<Item = (&'a usize, &'a f64)>,
    {
        let mut na = Array1::zeros(particles.len());

        for (&i, &nai) in initial_asymmetries {
            if let Some(v) = na.get_mut(i) {
                *v = nai;
            } else {
                log::error!("Out of bounds initial asymmetry.");
                return Err(Error::InvalidInitialAsymmetries);
            }
        }

        if na.iter().any(|v| !v.is_finite() || v.is_nan()) {
            log::error!("Some of initial densities are infinite or NaN:\n{:.3e}", na);
            return Err(Error::InvalidInitialAsymmetries);
        }

        Ok(na)
    }
}

impl<M> SolverBuilder<M>
where
    M: ModelInteractions,
{
    /// Precompute the interaction rates.
    #[allow(dead_code)]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    #[cfg(not(feature = "parallel"))]
    fn do_precompute(model: &mut M, beta_range: (f64, f64), mut subdivs: usize) {
        subdivs = subdivs.max(4 * f64::log2(beta_range.1 / beta_range.0).ceil() as usize);

        for (i, beta) in // rec_geomspace(beta_range.0, beta_range.1, subdivs)
            Array1::geomspace(beta_range.0, beta_range.1, subdivs)
                .unwrap()
                .into_iter()
                .enumerate()
        {
            if i % 50 == 0 {
                log::info!("Precomputing step {:>5} / {}", i, subdivs);
            } else {
                log::trace!("Precomputing step {:>5} / {}", i, subdivs);
            }

            model.set_beta(beta);
            let c = model.as_context();

            for interaction in model.interactions() {
                interaction.gamma(&c, false);
            }
        }
        log::info!("Precomputation complete");
    }

    /// Precompute the interaction rates.
    #[allow(dead_code)]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    #[cfg(feature = "parallel")]
    fn do_precompute(model: &mut M, beta_range: (f64, f64), mut subdivs: usize) {
        subdivs = subdivs.max(4 * f64::log2(beta_range.1 / beta_range.0).ceil() as usize);

        for (i, beta) in // rec_geomspace(beta_range.0, beta_range.1, subdivs)
            Array1::geomspace(beta_range.0, beta_range.1, subdivs)
                .unwrap()
                .into_iter()
                .enumerate()
        {
            if i % 50 == 0 {
                log::info!("Precomputing step {:>5} / {}", i, subdivs);
            } else {
                log::debug!("Precomputing step {:>5} / {}", i, subdivs);
            }

            model.set_beta(beta);
            let c = model.as_context();

            model.interactions().par_iter().for_each(|interaction| {
                interaction.gamma(&c, false);
            });
        }

        log::info!("Precomputation complete");
    }

    /// Build the Boltzmann solver.
    ///
    /// # Errors
    ///
    /// This will produce an error if some of the configurations options are
    /// deemed to be invalid.
    pub fn build(mut self) -> Result<Solver<M>, Error> {
        let mut model = self.model.ok_or(Error::UndefinedModel)?;
        model.set_beta(self.beta_range.0);
        let particles = model.particles();
        let beta_range = self.beta_range;

        // If no initial densities were given, all particles are assumed to be
        // in equilibrium.
        let initial_densities =
            Self::generate_initial_densities(beta_range.0, particles, &self.initial_densities)?;

        // If no initial asymmetries were given, all particles are assumed to be
        // in equilibrium with no asymmetry.
        let initial_asymmetries =
            Self::generate_initial_asymmetries(particles, &self.initial_asymmetries)?;

        // Make sure that there aren't too many particles held in equilibrium or
        // forbidden from developing any asymmetry.  We also sort and remove
        // duplicates.  For the asymmetries, any particle which is its own
        // antiparticle is also prevented from developing an asymmetry.
        if self.in_equilibrium.len() > particles.len() {
            log::error!("There are more particles held in equilibrium ({}) than particles in the model ({}).",
                        self.in_equilibrium.len(),
                        particles.len());
            return Err(Error::TooManyInEquilibrium);
        }
        let mut in_equilibrium: Vec<usize> = self.in_equilibrium.drain().collect();
        in_equilibrium.sort_unstable();

        self.no_asymmetry
            .extend(model.particles().iter().enumerate().filter_map(|(i, p)| {
                if p.own_antiparticle {
                    Some(i)
                } else {
                    None
                }
            }));
        if self.no_asymmetry.len() > particles.len() {
            log::error!(
                "There are more particles with no asymmetry ({}) than particles in the model ({}).",
                self.no_asymmetry.len(),
                particles.len()
            );
            return Err(Error::TooManyNoAsymmetry);
        }
        let mut no_asymmetry: Vec<usize> = self.no_asymmetry.drain().collect();
        no_asymmetry.sort_unstable();

        // Collect the number of interactions within the model.
        let mut interactions: HashSet<_> = model
            .interactions()
            .iter()
            .map(Interaction::particles)
            .collect();
        if interactions.len() != model.interactions().len() {
            interactions.clear();
            let mut duplicates: Vec<_> = model
                .interactions()
                .iter()
                .filter_map(|i| {
                    let particles = i.particles();
                    if interactions.insert(particles) {
                        None
                    } else {
                        Some(
                            particles
                                .display(&model)
                                .unwrap_or_else(|_| particles.short_display()),
                        )
                    }
                })
                .collect();
            duplicates.sort();
            log::error!("There are multiple interactions with the same set of incoming / outgoing particles. \
            The amplitudes of these interactions should be combined instead of combining the squared ampliutes. \
            The interactions were:\n • {}", duplicates.join("\n • "));
            return Err(Error::DuplicateInteractions);
        }

        // Run the precomputations so that the solver can run multiple times
        // later.
        if self.precompute > 0 {
            Self::do_precompute(&mut model, beta_range, self.precompute);
        }

        Ok(Solver {
            model,
            particles_next: Vec::new(),
            initial_densities,
            initial_asymmetries,
            beta_range,
            in_equilibrium,
            no_asymmetry,
            logger: self.logger,
            step_precision: self.step_precision,
            error_tolerance: self.error_tolerance,
            use_fast_interactions: self.fast_interactions,
            is_inaccurate: false,
            abort_when_inaccurate: self.abort_when_inaccurate,
            is_precomputed: self.precompute > 0,
        })
    }
}

impl<M> Default for SolverBuilder<M> {
    fn default() -> Self {
        Self::new()
    }
}
