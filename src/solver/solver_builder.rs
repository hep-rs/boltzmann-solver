use crate::{
    model::{interaction::Interaction, ModelInteractions, Particle},
    solver::{options::StepPrecision, Context, Solver},
    utilities::spline::rec_geomspace,
};
use ndarray::prelude::*;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::{error, fmt};

// Number of recursive subdivisions in beta range (so there are 2^N
// subdivisions).
const PRECOMPUTE_SUBDIV: u32 = 12;

/// Error type returned by the solver builder in case there is an error.
#[derive(Debug)]
pub enum Error {
    /// One or more initial density is specified multiple times.
    DuplicateInitialDensities,
    /// The initial number densities are invalid.
    InvalidInitialDensities,
    /// One or more initial density asymmetry is specified multiple times.
    DuplicateInitialAsymmetries,
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
            Error::DuplicateInitialDensities => {
                write!(f, "one or more initial density is specified multiple times")
            }
            Error::InvalidInitialDensities => write!(f, "initial number densities are invalid"),
            Error::DuplicateInitialAsymmetries => write!(
                f,
                "one or more initial density asymmetry is specified multiple times"
            ),
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
#[allow(clippy::module_name_repetitions)]
pub struct SolverBuilder<M> {
    model: Option<M>,
    initial_densities: Vec<(usize, f64)>,
    initial_asymmetries: Vec<(usize, f64)>,
    beta_range: (f64, f64),
    in_equilibrium: Vec<usize>,
    no_asymmetry: Vec<usize>,
    logger: Box<dyn Fn(&Context<M>)>,
    step_precision: StepPrecision,
    error_tolerance: f64,
    precompute: bool,
}

impl<M> SolverBuilder<M> {
    /// Creates a new builder for the Boltzmann solver.
    ///
    /// The default range of temperatures span 1 GeV through to 10^{20} GeV, and
    /// it uses normalization by default.
    ///
    /// Most of the method for the builder are intended to be chained one after
    /// the other.
    ///
    /// ```rust
    /// use boltzmann_solver::prelude::*;
    /// use boltzmann_solver::model::StandardModel;
    ///
    /// let mut solver_builder: SolverBuilder<StandardModel> = SolverBuilder::new()
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
            initial_densities: Vec::new(),
            initial_asymmetries: Vec::new(),
            beta_range: (1e-20, 1e0),
            in_equilibrium: Vec::new(),
            no_asymmetry: Vec::new(),
            logger: Box::new(|_| {}),
            step_precision: StepPrecision::default(),
            error_tolerance: 1e-4,
            precompute: true,
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
    /// The specification is given as an iterator of `(usize, f64)` tuples where
    /// the first entry is the index of the particle whose equilibrium is
    /// specified.
    ///
    /// If unspecified, all number densities are assumed to be in equilibrium to
    /// begin with.
    ///
    /// Repeated initial conditions will result in an error.
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
    /// The list of number density asymmetries must be in the same order as the
    /// particles in the model.
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
    ///
    /// Multiple calls of this function will combine the results together, and
    /// any (accidental) duplications are removed by [`SolverBuilder::build`].
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
    /// Multiple calls of this function will combine the results together, and
    /// any (accidental) duplications are removed by [`SolverBuilder::build`].
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
    pub fn logger<F>(mut self, f: F) -> Self
    where
        F: Fn(&Context<M>) + 'static,
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
    /// The default values are `min = 1e-10` and `max = 1.0`.
    ///
    /// The relative step precision has a higher priority on the step size than
    /// the error.  That is, the step size will never be less than
    /// `$p_\text{min} \beta$` even if this results in a larger local error than
    /// desired.
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

    /// Upon calling [`SolverBuilder::build`], set whether interactions should
    /// be precomputed.
    ///
    /// By default, interactions are precomputed.
    pub fn precompute(mut self, v: bool) -> Self {
        self.precompute = v;
        self
    }

    /// Check the validity of the initial densities, making sure we have the
    /// right number of initial conditions and they are all finite.
    fn generate_initial_densities<I>(
        beta: f64,
        particles: &[Particle],
        initial_densities: I,
    ) -> Result<Array1<f64>, Error>
    where
        I: IntoIterator<Item = (usize, f64)>,
    {
        let mut n: Array1<_> = particles
            .iter()
            .map(|p| p.normalized_number_density(0.0, beta))
            .collect();

        for (i, ni) in initial_densities {
            if let Some(v) = n.get_mut(i) {
                *v = ni;
            } else {
                log::error!("Out of bounds initial density.");
                return Err(Error::InvalidInitialDensities);
            }
        }

        if n.iter().any(|v| !v.is_finite() || v.is_nan()) {
            log::error!(
                "Some of initial densities are not finite or NaN:\n{:.3e}",
                n
            );
            return Err(Error::InvalidInitialDensities);
        }

        Ok(n)
    }

    /// Check the validity of the initial asymmetries, making sure we have the
    /// right number of initial conditions and they are all finite.
    fn generate_initial_asymmetries<I>(
        particles: &[Particle],
        initial_asymmetries: I,
    ) -> Result<Array1<f64>, Error>
    where
        I: IntoIterator<Item = (usize, f64)>,
    {
        let mut na = Array1::zeros(particles.len());

        for (i, nai) in initial_asymmetries {
            if let Some(v) = na.get_mut(i) {
                *v = nai;
            } else {
                log::error!("Out of bounds initial asymmetry.");
                return Err(Error::InvalidInitialAsymmetries);
            }
        }

        if na.iter().any(|v| !v.is_finite() || v.is_nan()) {
            log::error!(
                "Some of initial densities are not finite or NaN:\n{:.3e}",
                na
            );
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
    #[cfg(not(feature = "parallel"))]
    fn do_precompute(model: &mut M, beta_range: (f64, f64)) {
        log::info!("Pre-computing γ...");
        for (i, &beta) in vec![
            0.98 * beta_range.0,
            0.99 * beta_range.0,
            1.01 * beta_range.1,
            1.02 * beta_range.1,
        ]
        .iter()
        .chain(&rec_geomspace(
            beta_range.0,
            beta_range.1,
            PRECOMPUTE_SUBDIV,
        ))
        .enumerate()
        {
            if i % 1024 == 3 {
                log::info!(
                    "Precomputing step {} / {}",
                    i - 3,
                    2_usize.pow(PRECOMPUTE_SUBDIV)
                );
            } else if i % 64 == 3 {
                log::debug!(
                    "Precomputing step {} / {}",
                    i - 3,
                    2_usize.pow(PRECOMPUTE_SUBDIV)
                );
            } else if i >= 3 {
                log::trace!(
                    "Precomputing step {} / {}",
                    i - 3,
                    2_usize.pow(PRECOMPUTE_SUBDIV)
                );
            }

            model.set_beta(beta);
            let c = model.as_context();

            for interaction in model.interactions() {
                interaction.gamma(&c);
            }
        }
    }

    /// Precompute the interaction rates.
    #[cfg(feature = "parallel")]
    fn do_precompute(model: &mut M, beta_range: (f64, f64)) {
        log::info!("Pre-computing γ...");
        for (i, &beta) in vec![
            0.98 * beta_range.0,
            0.99 * beta_range.0,
            1.01 * beta_range.1,
            1.02 * beta_range.1,
        ]
        .iter()
        .chain(&rec_geomspace(
            beta_range.0,
            beta_range.1,
            PRECOMPUTE_SUBDIV,
        ))
        .enumerate()
        {
            if i % 1024 == 3 {
                log::info!(
                    "Precomputing step {} / {}",
                    i - 3,
                    2_usize.pow(PRECOMPUTE_SUBDIV)
                );
            } else if i % 64 == 3 {
                log::debug!(
                    "Precomputing step {} / {}",
                    i - 3,
                    2_usize.pow(PRECOMPUTE_SUBDIV)
                );
            } else if i >= 3 {
                log::trace!(
                    "Precomputing step {} / {}",
                    i - 3,
                    2_usize.pow(PRECOMPUTE_SUBDIV)
                );
            }

            model.set_beta(beta);
            let c = model.as_context();

            model.interactions().par_iter().for_each(|interaction| {
                interaction.gamma(&c, false);
            });
        }
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

        let initial_densities_len = self.initial_densities.len();
        self.initial_densities.sort_unstable_by_key(|t| t.0);
        self.initial_densities.dedup_by_key(|t| t.0);
        if initial_densities_len != self.initial_densities.len() {
            log::error!("At least one initial number density was specified twice.");
            return Err(Error::DuplicateInitialDensities);
        }
        let initial_densities = Self::generate_initial_densities(
            beta_range.0,
            &particles,
            self.initial_densities.iter().cloned(),
        )?;

        // If no initial asymmetries were given, all particles are assumed to be
        // in equilibrium with no asymmetry.
        let initial_asymmetries_len = self.initial_asymmetries.len();
        self.initial_asymmetries.sort_unstable_by_key(|t| t.0);
        self.initial_asymmetries.dedup_by_key(|t| t.0);
        if initial_asymmetries_len != self.initial_asymmetries.len() {
            log::error!("At least one initial number density was specified twice.");
            return Err(Error::DuplicateInitialAsymmetries);
        }
        let initial_asymmetries = Self::generate_initial_asymmetries(
            &particles,
            self.initial_asymmetries.iter().cloned(),
        )?;

        // Make sure that there aren't too many particles held in equilibrium or
        // forbidden from developing any asymmetry.  We also sort and remove
        // duplicates.  For the asymmetries, any particle which is its own
        // antiparticle is also prevented from developing an asymmetry.
        self.in_equilibrium.sort_unstable();
        self.in_equilibrium.dedup();
        if self.in_equilibrium.len() > particles.len() {
            log::error!("There are more particles held in equilibrium ({}) than particles in the model ({}).",
                        self.in_equilibrium.len(),
                        particles.len());
            return Err(Error::TooManyInEquilibrium);
        }

        self.no_asymmetry
            .extend(model.particles().iter().enumerate().filter_map(|(i, p)| {
                if p.own_antiparticle {
                    Some(i)
                } else {
                    None
                }
            }));
        self.no_asymmetry.sort_unstable();
        self.no_asymmetry.dedup();
        if self.no_asymmetry.len() > particles.len() {
            log::error!(
                "There are more particles with 0 asymmetry ({}) than particles in the model ({}).",
                self.no_asymmetry.len(),
                particles.len()
            );
            return Err(Error::TooManyNoAsymmetry);
        }

        // Run the precomputations so that the solver can run multiple times
        // later.
        if self.precompute {
            Self::do_precompute(&mut model, beta_range);
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

impl<M> Default for SolverBuilder<M> {
    fn default() -> Self {
        Self::new()
    }
}
