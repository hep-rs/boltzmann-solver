//! Solvers for the Boltzmann equation, or sets of Boltzmann equations using
//! arbitrary precision floating point numbers.

pub mod number_density;
mod options;
// pub mod phase_space;
mod tableau;

pub use self::options::InitialCondition;
pub(crate) use self::options::{StepChange, StepPrecision};

use crate::{particle::Particle, universe::Universe};
use ndarray::{array, prelude::*};
use rug::Float;

pub(crate) const DEFAULT_WORKING_PRECISION: u32 = 100;

/// Contains all the information relevant to a particular model, including
/// masses, widths and couplings.  All these attributes can be dependent on the
/// inverse temperature \\(\beta\\).
pub trait Model {
    /// Instantiate a new instance of the model parameters with the values
    /// calculated at the inverse temperature \\(\beta\\).
    fn new(beta: &Float) -> Self;

    /// Return list of particles in the model.
    fn particles(&self) -> &Array1<Particle>;
}

/// An empty model containing no couplings, masses, etc.  This is can be used
/// for very simple implementations of the Boltzmann solver.
pub struct EmptyModel {
    particles: Array1<Particle>,
}

impl Model for EmptyModel {
    fn new(_: &Float) -> Self {
        EmptyModel {
            particles: array![],
        }
    }

    fn particles(&self) -> &Array1<Particle> {
        &self.particles
    }
}

/// Common interface for the Boltzmann equation solvers.
pub trait Solver {
    /// The final solution by the solver.  This will typically be an array for
    /// values with the ordering corresponding to the way `add_particle` was
    /// invoked.
    type Solution;

    /// Context containing relevant information precomputed by the solver which
    /// can be used in the calculation of the interactions.
    ///
    /// This is used in order to avoid running possibly time-consuming functions
    /// for each interaction.
    type Context;

    /// Create a new instance of the solver.
    ///
    /// In general, the solver is instantiated as follows:
    ///
    /// ```ignore
    /// let solver = Solver::new()
    ///              .temperate_range(1e10, 1e3)
    ///              // Other settings
    ///              .initialize();
    /// ```
    fn new() -> Self;

    /// Set the range of inverse temperature values over which the phase space
    /// is evolved.
    ///
    /// Inverse temperature must be provided in units of GeV^{-1}.
    ///
    /// This function has a convenience alternative called
    /// [`Solver::temperature_range`] allowing for the limits to be
    /// specified as temperature in the units of GeV.
    ///
    /// # Panics
    ///
    /// Panics if the starting value is larger than the final value.
    fn beta_range(self, start: f64, end: f64) -> Self;

    /// Set the range of temperature values over which the phase space is
    /// evolved.
    ///
    /// Temperature must be provided in units of GeV.
    ///
    /// This function is a convenience alternative to
    /// [`Solver::beta_range`].
    ///
    /// # Panics
    ///
    /// Panics if the starting value is smaller than the final value.
    fn temperature_range(self, start: f64, end: f64) -> Self;

    /// Specify the granularity of the way time evolution is done.
    ///
    /// The time evolution is done with a step size of \\(h\\) such that
    /// \\(\beta \to \beta + h\\) in the next step.  As the range of \\(\beta\\)
    /// spans several orders of magnitude, the step size must be adjusted during
    /// the time evolution.  This is done by estimating the error at each step
    /// and if it falls below a particular threshold, the step size is increased
    /// multiplicatively by \\(h \to h \times \Delta_{+}\\).  Similarly, if the
    /// estimated error becomes too large, the step size is decreases
    /// multiplicatively by \\(h \to h \times \Delta_{-}\\).
    ///
    /// The value of \\(\Delta_{+}\\) is specified by `increase` and the value
    /// of \\(\Delta_{-}\\) is specified by `decrease`.  They are `1.1` and
    /// `0.5` respectively by default.
    ///
    /// # Panic
    ///
    /// This will panic if the increase factor is not greater than `1.0` or if
    /// the decrease factor is not less than `1.0`.
    fn step_change(self, increase: f64, decrease: f64) -> Self;

    /// Specify what the relative size of the step size can be.
    ///
    /// The time evolution is done with a step size of \\(h\\) such that
    /// \\(\beta \to \beta + h\\) is the next step.  In general, one wants \\(h
    /// < \beta\\) when solving the Boltzmann equations.  The values of `min`
    /// and `max` specify how big \\(h\\) can be relative to \\(\beta\\):
    ///
    /// \\begin{equation}
    ///   p_\text{min} \beta < h < p_\text{max} \beta
    /// \\end{equation}
    ///
    /// The default values are `min = 1e-6` and `max = 1e-2`.
    ///
    /// The relative step precision has a higher priority on the step size than
    /// the error.  That is, the step size will never be less than
    /// \\(p_\text{min} \beta\\) even if this results in a larger local error
    /// than desired.
    ///
    /// # Panic
    ///
    /// This will panic if `min >= max`.
    fn step_precision(self, min: f64, max: f64) -> Self;

    /// Specify the local error tolerance.
    ///
    /// If the error deviates too far from the specified tolerance, the
    /// integration step size is adjusted accordingly.
    fn error_tolerance(self, tol: f64) -> Self;

    /// Specify initial conditions for the number densities.
    fn initial_conditions(self, cond: Vec<Float>) -> Self;

    /// Initialize the phase space solver.
    fn initialize(self) -> Self;

    /// Add an interaction.
    ///
    /// The interaction is a functional of the solution at a particular inverse
    /// temperature.  The function is of the following form:
    ///
    /// ```ignore
    /// f(sum: Self::Solution, densities: &Self::Solution, beta: f64) -> Self::Solution
    /// ```
    ///
    /// The first argument, `sum`, contains the sum of all interactions and the
    /// second argument, `densities`, contains the values of the various
    /// number densities at the specified `beta`.  The `sum` is moved into the
    /// function and is expected to be returned.  For example:
    ///
    /// ```ignore
    /// fn f(sum: mut Array1<f64>, densities: Array1<f64>, beta: f64) -> Array1<f64> {
    ///     sum[0] += - densities[1] * beta;
    ///     sum[1] += - densities[0] * beta;
    ///     sum
    /// }
    /// ```
    fn add_interaction<F: 'static>(&mut self, int: F) -> &mut Self
    where
        F: Fn(Self::Solution, &Self::Solution, &Self::Context) -> Self::Solution;

    /// Set the logger.
    ///
    /// The logger provides some insight into the numerical integration 'black
    /// box'.  Specifically, it is run at the start of each integration step and
    /// has access to the current value as a `&Solution`, the change from this
    /// step as a `&Solution`, and the current `Context` at the start of the
    /// integration step.  As a result, for the first step, the solution will be
    /// equal to the initial conditions.
    ///
    /// This is useful if one wants to track the evolution of the solutions and
    /// log these in a CSV file.
    fn set_logger<F: 'static>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&Self::Solution, &Self::Solution, &Self::Context);

    /// Evolve the initial conditions by solving the PDEs.
    ///
    /// Note that this is not a true PDE solver, but instead converts the PDE
    /// into an ODE in time (or inverse temperature in this case), with the
    /// derivative in energy being calculated solely from the previous
    /// time-step.
    fn solve<U: Universe>(&self, universe: &U) -> Self::Solution;
}
