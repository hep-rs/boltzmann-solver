//! Solver for the number density evolution given by integrating the Boltzmann
//! equation using arbitrary precision.
//!
//! See the documentation for the [regular (non arbitrary-precision)
//! solver](crate::solver::number_density) to see what assumptions
//! are made.

use super::{
    EmptyModel, InitialCondition, Model, Solver, StepChange, StepPrecision,
    DEFAULT_WORKING_PRECISION,
};
use crate::{particle::Particle, universe::Universe};
use log::{debug, info};
use ndarray::{prelude::*, FoldWhile, Zip};
use rug::{ops::*, Float};
use std::iter;

/// Context provided containing pre-computed values which might be useful when
/// evaluating interactions.
pub struct Context<M: Model> {
    /// Evaluation step
    pub step: u64,
    /// Inverse temperature in GeV^{-1}
    pub beta: Float,
    /// Current step size
    pub step_size: Float,
    /// Hubble rate, in GeV
    pub hubble_rate: f64,
    /// Equilibrium number densities for the particles, normalized to the
    /// equilibrium number density for a massless boson with \\(g = 1\\).  This
    /// is provided in the same order as specified to the solver
    pub eq_n: Array1<f64>,
    /// Model data
    pub model: M,
    /// Working precision
    pub working_precision: u32,
}

/// Boltzmann equation solver for the number density.
///
/// All number densities are normalized to that of a massless boson with a
/// single degree of freedom (\\(g = 1\\)).  As a result of this convention,
/// \\(n_\gamma = 2\\) as the photon has two degrees of freedom.
pub struct NumberDensitySolver<M: Model> {
    initialized: bool,
    beta_range: (Float, Float),
    particles: Vec<Particle>,
    initial_conditions: Vec<Float>,
    #[allow(clippy::type_complexity)]
    interactions: Vec<
        Box<
            Fn(
                <Self as Solver>::Solution,
                &<Self as Solver>::Solution,
                &<Self as Solver>::Context,
            ) -> <Self as Solver>::Solution,
        >,
    >,
    #[allow(clippy::type_complexity)]
    logger: Box<
        Fn(&<Self as Solver>::Solution, &<Self as Solver>::Solution, &<Self as Solver>::Context),
    >,
    step_change: StepChange,
    step_precision: StepPrecision,
    error_tolerance: Float,
    threshold_number_density: Float,
    working_precision: u32,
}

impl<M: Model> Solver for NumberDensitySolver<M> {
    /// The solution is a one-dimensional array of number densities for each
    /// particle species (or aggregated number density in the case of
    /// \\(n_{\mathsc{b-l}}\\)), in the same order as [`Solver::add_particle`]
    /// is invoked.
    type Solution = Array1<Float>;

    type Context = Context<M>;

    /// Create a new instance of the number density solver.
    ///
    /// The default range of temperatures span 1 GeV through to 10^{20} GeV.
    fn new() -> Self {
        Self {
            initialized: false,
            beta_range: (
                Float::with_val(DEFAULT_WORKING_PRECISION, 1e-20),
                Float::with_val(DEFAULT_WORKING_PRECISION, 1e0),
            ),
            particles: Vec::with_capacity(20),
            initial_conditions: Vec::with_capacity(20),
            interactions: Vec::with_capacity(100),
            logger: Box::new(|_, _, _| {}),
            step_change: StepChange::default(),
            step_precision: StepPrecision::default(),
            error_tolerance: Float::with_val(DEFAULT_WORKING_PRECISION, 1e-4),
            threshold_number_density: Float::with_val(DEFAULT_WORKING_PRECISION, 0.0),
            working_precision: 100,
        }
    }

    /// Set the range of inverse temperature values over which the phase space
    /// is evolved.
    ///
    /// Inverse temperature must be provided in units of GeV^{-1}.
    ///
    /// This function has a convenience alternative called
    /// [`Solver::temperature_range`] allowing for the limits to be specified as
    /// temperature in the units of GeV.
    ///
    /// # Panics
    ///
    /// Panics if the starting value is larger than the final value.
    fn beta_range(mut self, start: f64, end: f64) -> Self {
        assert!(
            start < end,
            "The initial β must be smaller than the final β value."
        );
        self.beta_range = (
            Float::with_val(self.working_precision, start),
            Float::with_val(self.working_precision, end),
        );
        self
    }

    /// Set the range of temperature values over which the phase space is
    /// evolved.
    ///
    /// Temperature must be provided in units of GeV.
    ///
    /// This function is a convenience alternative to
    /// [`NumberDensitySolver::beta_range`].
    ///
    /// # Panics
    ///
    /// Panics if the starting value is smaller than the final value.
    fn temperature_range(mut self, start: f64, end: f64) -> Self {
        assert!(
            start > end,
            "The initial temperature must be larger than the final temperature."
        );
        self.beta_range = (
            Float::with_val(self.working_precision, start).recip(),
            Float::with_val(self.working_precision, end).recip(),
        );
        self
    }

    fn step_change(mut self, increase: f64, decrease: f64) -> Self {
        assert!(
            increase > 1.0,
            "The multiplicative factor to increase the step size must be greater
            than 1."
        );
        assert!(
            decrease < 1.0,
            "The multiplicative factor to decrease the step size must be greater
            than 1."
        );
        self.step_change = StepChange {
            increase: Float::with_val(self.working_precision, increase),
            decrease: Float::with_val(self.working_precision, decrease),
        };
        self
    }

    fn step_precision(mut self, min: f64, max: f64) -> Self {
        assert!(
            min < max,
            "Minimum step precision must be smaller than the maximum step precision."
        );
        self.step_precision = StepPrecision {
            min: Float::with_val(self.working_precision, min),
            max: Float::with_val(self.working_precision, max),
        };
        self
    }

    fn error_tolerance(mut self, tol: f64) -> Self {
        assert!(tol > 0.0, "The tolerance must be greater than 0.");
        self.error_tolerance = Float::with_val(self.working_precision, tol);
        self
    }

    fn initialize(mut self) -> Self {
        self.initialized = true;

        self
    }

    fn add_particle(&mut self, s: Particle, initial_condition: InitialCondition) {
        let v = match initial_condition {
            InitialCondition::Equilibrium(mu) => Float::with_val(
                self.working_precision,
                s.normalized_number_density(mu, self.beta_range.0.to_f64()),
            ),
            InitialCondition::Fixed(n) => n,
            InitialCondition::Zero => Float::with_val(self.working_precision, 0.0),
        };
        self.initial_conditions.push(v);

        self.particles.push(s);
    }

    fn add_particles<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = Particle>,
    {
        self.particles.extend(iter);
    }

    fn add_interaction<F: 'static>(&mut self, f: F) -> &mut Self
    where
        F: Fn(Self::Solution, &Self::Solution, &Self::Context) -> Self::Solution,
    {
        self.interactions.push(Box::new(f));
        self
    }

    fn set_logger<F: 'static>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&Self::Solution, &Self::Solution, &Self::Context),
    {
        self.logger = Box::new(f);
        self
    }

    #[allow(clippy::many_single_char_names)]
    fn solve<U>(&self, universe: &U) -> Self::Solution
    where
        U: Universe,
    {
        use super::tableau::dp87::*;

        assert!(
            self.initialized,
            "The phase space solver has to be initialized first with the `initialize()` method."
        );

        let zeros = Self::Solution::from_iter(
            iter::repeat(Float::with_val(self.working_precision, 0.0))
                .take(self.initial_conditions.len()),
        );

        // Initialize all the variables that will be used in the integration
        let mut n = Self::Solution::from_iter(
            self.initial_conditions
                .iter()
                .map(|v| Float::with_val(self.working_precision, v)),
        );
        let mut dn = [zeros.clone(), zeros.clone()];

        let mut k: [Self::Solution; RK_S];
        unsafe {
            k = std::mem::uninitialized();
            for ki in &mut k[..] {
                std::ptr::write(ki, zeros.clone());
            }
        };

        let mut step = 0;
        let mut beta = self.beta_range.0.clone();
        let mut h = Float::with_val(self.working_precision, &beta * &self.step_precision.min);

        // Create the initial context and log the initial conditions
        let mut c = self.context(step, beta.clone(), universe, h.clone());
        (*self.logger)(&n, &dn[0], &c);

        while beta < self.beta_range.1 {
            step += 1;
            let mut advance = false;

            // Compute each k[i]
            for i in 0..RK_S {
                let beta_i = &beta + h.clone() * RK_C[i];
                let ci = self.context(step, beta_i, universe, h.clone());
                let ai = RK_A[i];
                let mut dni = (0..i).fold(zeros.clone(), |total, j| total + &k[j] * ai[j]);
                let ni = self.n_plus_dn(n.clone(), &mut dni, &ci);
                k[i] = self
                    .interactions
                    .iter()
                    .fold(zeros.clone(), |s, f| f(s, &ni, &ci));
                k[i].mapv_inplace(|v| v * &h);
            }

            // Calculate the two estimates
            dn[0] = (0..RK_S).fold(zeros.clone(), |total, i| total + &k[i] * RK_B[0][i]);
            dn[1] = (0..RK_S).fold(zeros.clone(), |total, i| total + &k[i] * RK_B[1][i]);

            // Get the error between the estimates
            let err = Zip::from(&dn[0])
                .and(&dn[1])
                .fold_while(Float::with_val(self.working_precision, 0.0), |e, a, b| {
                    let v = (a.clone() - b).abs();
                    FoldWhile::Continue(e.max(&v))
                })
                .into_inner()
                / &h;

            if &err < &self.error_tolerance {
                advance = true;
            }

            // Compute the change in step size based on the current error And
            // correspondingly adjust the step size
            let mut h_est = if err.is_zero() {
                h.clone() * &self.step_change.increase
            } else {
                let delta: Float =
                    0.9 * (&self.error_tolerance / err).pow(1.0 / f64::from(RK_ORDER + 1));

                &h * delta.clamp(&self.step_change.decrease, &self.step_change.increase)
            };

            // Prevent h from getting too small or too big in proportion to the
            // current value of beta.  Also advance the integration irrespective
            // of the local error if we reach the maximum or minimum step size.
            let max_step: Float = beta.clone() * &self.step_precision.max;
            if h_est > max_step {
                h_est = max_step;
                debug!(
                    "Step {:}, β = {:.4e} -> Step size too large, decreased h to {:.3e}",
                    step, beta, h_est
                );
                advance = true;
            } else {
                let min_step: Float = beta.clone() * &self.step_precision.min;
                if h_est < min_step {
                    h = min_step;
                    debug!(
                        "Step {:}, β = {:.4e} -> Step size too small, increased h to {:.3e}",
                        step, beta, h
                    );
                    advance = true;
                }
            }

            // Check if the error is within the tolerance, or we are advancing
            // irrespective of the local error
            if advance {
                c = self.context(step, beta.clone(), universe, h.clone());

                // Advance n and beta
                n = self.n_plus_dn(n, &mut dn[0], &c);
                beta += h;

                // Run the logger now
                (*self.logger)(&n, &dn[0], &c);
            }

            // Adjust final integration step if needed
            let next_beta: Float = beta.clone() + &h_est;
            if next_beta > self.beta_range.1 {
                h_est = &self.beta_range.1 - beta.clone();
            }

            h = h_est;
        }

        info!("Number of evaluations: {}", step);

        n
    }
}

impl<'a> Default for NumberDensitySolver<EmptyModel> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, M: Model> NumberDensitySolver<M> {
    /// Set the threshold number density to count as 0.
    ///
    /// Any number density whose absolute value is less than the threshold will
    /// be treated as being exactly zero.  This applies to both calculated
    /// number densities as well as equilibrium number densities.  Furthermore,
    /// this also applies to 'abstract' number densities such as \\(B-L\\).
    ///
    /// This is by default set to `0.0`.
    ///
    /// # Panics
    ///
    /// Panics if the threshold is negative.
    pub fn threshold_number_density(mut self, threshold: f64) -> Self {
        assert!(
            threshold >= 0.0,
            "Threshold number density must be a non-negative number."
        );
        self.threshold_number_density = Float::with_val(self.working_precision, threshold);
        self
    }

    /// Create an array containing the initial conditions for all the particle
    /// species.
    ///
    /// All particles species are assumed to be in thermal equilibrium at this
    /// energy, with the distribution following either the Bose–Einstein or
    /// Fermi–Dirac distribution as determined by their spin.
    fn equilibrium_number_densities(&self, beta: f64) -> Array1<f64> {
        Array1::from_iter(self.particles.iter().map(|p| {
            let v = p.normalized_number_density(0.0, beta);
            if v.abs() < self.threshold_number_density {
                0.0
            } else {
                v
            }
        }))
    }

    /// Generate the context at a given beta to pass to the logger/interaction
    /// functions.
    fn context<U: Universe>(
        &self,
        step: u64,
        beta: Float,
        universe: &U,
        step_size: Float,
    ) -> Context<M> {
        let beta_f64 = beta.to_f64();
        let model = M::new(&beta);
        Context {
            step,
            beta,
            step_size,
            hubble_rate: universe.hubble_rate(beta_f64),
            eq_n: self.equilibrium_number_densities(beta_f64),
            model,
            working_precision: self.working_precision,
        }
    }

    /// Add `dn` to `n`, but set the result to the equilibrium number density if
    /// the change overshoots it.
    ///
    /// If there is a strong process causing a particular number density to go
    /// towards equilibrium, the iteration step may overshoot the equilibrium
    /// point; and in the case where the process is *very* strong, it is
    /// possible the overshooting is so bad that it generates an even larger
    /// (opposite signed) number density.
    ///
    /// To avoid this, we set the number density to exactly the equilibrium
    /// number density whenever this might occur forcing an evaluation with the
    /// equilibrium number density.
    fn n_plus_dn(
        &self,
        mut n: <Self as Solver>::Solution,
        dn: &mut <Self as Solver>::Solution,
        c: &<Self as Solver>::Context,
    ) -> <Self as Solver>::Solution {
        Zip::from(&mut n).and(dn).and(&c.eq_n).apply(|n, dn, eq_n| {
            let delta_1: Float = n.clone() - eq_n;
            let delta_2: Float = delta_1.clone() + &*dn;

            if delta_1.is_sign_positive() != delta_2.is_sign_positive() {
                *dn = n.clone() - eq_n;
                *n = Float::with_val(self.working_precision, eq_n);
            } else {
                *n += &*dn;
            }

            if *n.as_abs() < self.threshold_number_density {
                *dn = n.clone();
                dn.neg_assign();
                *n = Float::with_val(self.working_precision, 0.0)
            };
        });
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::universe::StandardModel;
    use crate::utilities::test::*;

    /// The most trivial example with a single particle and no interactions.
    #[test]
    fn no_interaction() {
        let phi = Particle::new("φ".to_string(), 0, 1e3);
        let mut solver = NumberDensitySolver::default()
            .temperature_range(1e20, 1e-10)
            .initialize();

        solver.add_particle(phi, InitialCondition::Equilibrium(0.0));

        let sol = solver.solve(&StandardModel::new());
        approx_eq(sol[0].to_f64(), 1.0, 8.0, 0.0);
    }
}
