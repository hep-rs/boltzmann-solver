//! Solvers for the Boltzmann equation, or sets of Boltzmann equations

use particle::Particle;
use universe::Universe;

pub mod number_density;
pub mod phase_space;

struct StepChange {
    increase: f64,
    decrease: f64,
}

struct ErrorTolerance {
    upper: f64,
    lower: f64,
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

    /// Specify the local error tolerance.
    ///
    /// At each step, the solver will compare approximates the integration using
    /// both the Euler and Runge-Kutta methods.  The error is then calculated as:
    ///
    /// \\begin{equation}
    ///   \mathrm{Error} = \left\lvert \frac{(\Delta y)_{\mathrm{Euler}}}{(\Delta y)_{\mathrm{RK}}} - 1 \right\rvert
    /// \\end{equation}
    ///
    /// If the error is greater than the upper bound specified, the integration
    /// step size is made smaller.  Conversely, if the error is smaller than the
    /// lower bound, the integration step is increased.
    fn error_tolerance(self, upper: f64, lower: f64) -> Self;

    /// Initialize the phase space solver.
    fn initialize(self) -> Self;

    /// Add a particle species.
    ///
    /// The initial conditions for this particle are generated assuming the
    /// particle to be in thermal and chemical equilibrium at the initial
    /// temperature.
    ///
    /// If the particle and anti-particle are to be treated separately, the two
    /// species have to be added.
    fn add_particle(&mut self, p: Particle);

    /// Add a multiple particles from a vector or slice.
    fn add_particles<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = Particle>;

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

    /// Evolve the initial conditions by solving the PDEs.
    ///
    /// Note that this is not a true PDE solver, but instead converts the PDE
    /// into an ODE in time (or inverse temperature in this case), with the
    /// derivative in energy being calculated solely from the previous
    /// time-step.
    fn solve<U: Universe>(&self, universe: &U) -> Self::Solution;
}
