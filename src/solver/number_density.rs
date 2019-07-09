//! Solver for the number density evolution given by integrating the Boltzmann
//! equation.
//!
//! Solving the Boltzmann equation in generality without making any assumption
//! about the phase space distribution \\(f\\) (other than specifying some
//! initial condition) is a difficult problem.  Furthermore, within the context
//! of baryogenesis and leptogenesis, we are only interested in the number
//! density (or more specifically, the difference in the number densities or a
//! particle and its corresponding antiparticle).
//!
//! # Assumptions
//!
//! A couple of assumptions can be made to simplify the Boltzmann equation so
//! that the number densities can be computed directly.
//!
//! - *Assume kinetic equilibrium.* If the rate at which a particle species is
//!   scattering is sufficiently fast, phase space distribution of this species
//!   will rapidly converge converge onto either the Bose–Einstein or
//!   Fermi–Dirac distributions:
//!
//!   \\begin{equation}
//!     f_{\textsc{BE}} = \frac{1}{\exp[(E - \mu) \beta] - 1}, \qquad
//!     f_{\textsc{FD}} = \frac{1}{\exp[(E - \mu) \beta] + 1}.
//!   \\end{equation}
//!
//!   For a particular that remains in kinetic equilibrium, the evolution of its
//!   phase space is entirely described by the evolution of \\(\mu\\) in time.
//!
//! - *Assume \\(\beta \gg E - \mu\\).* In the limit that the temperature is
//!   much less than \\(E - \mu\\) (or equivalently, that the inverse
//!   temperature \\(\beta\\) is greater than \\(E - \mu\\)), both the
//!   Fermi–Dirac and Bose–Einstein approach the Maxwell–Boltzmann distribution,
//!
//!   \\begin{equation}
//!     f_{\textsc{MB}} = \exp[-(E - \mu) \beta].
//!   \\end{equation}
//!
//!   This simplifies the expression for the number density to
//!
//!   \\begin{equation}
//!     n = \frac{g m^2 K_2(m \beta)}{2 \pi^2 \beta} e^{\mu \beta} = n^{(0)} e^{\mu \beta},
//!   \\end{equation}
//!
//!   where \\(n^{(0)}\\) is the equilibrium number density when \\(\mu = 0\\).
//!   This also allows for the equilibrium phase space distribution to be
//!   expressed in terms of the \\(\mu = 0\\) distribution:
//!
//!   \\begin{equation}
//!     f_{\textsc{MB}} = e^{\mu \beta} f_{\textsc{MB}}^{(0)} = \frac{n}{n^{(0)}} f_{\textsc{MB}}^{(0)}.
//!   \\end{equation}
//!
//!   Furthermore, the assumption that \\(\beta \gg E - \mu\\) implies that
//!   \\(f_{\textsc{BE}}, f_{\textsc{FD}} \ll 1\\).  Consequently, the Pauli
//!   suppression and Bose enhancement factors in the collision term can all be
//!   neglected resulting in:
//!
//!   \\begin{equation}
//!     \begin{aligned}
//!       g_{a_1} \int \vt C[f_{a_{1}}] \frac{\dd \vt p_{a_1}}{(2\pi)^3}
//!         &= - \int_{\vt a}^{\vt b}
//!            \abs{\mathcal M(\vt a \to \vt b)}^2 \left( \prod_{i \in \vt a} f_i \right)
//!            - \abs{\mathcal M(\vt b \to \vt a)}^2 \left( \prod_{i \in \vt b} f_i \right) \\\\
//!         &= - \int_{\vt a}^{\vt b}
//!            \abs{\mathcal M(\vt a \to \vt b)}^2 \left( \prod_{i \in \vt a} \frac{n_i}{n_i^{(0)}} f_i^{(0)} \right)
//!             - \abs{\mathcal M(\vt b \to \vt a)}^2 \left( \prod_{i \in \vt b} \frac{n_i}{n_i^{(0)}} f_i^{(0)} \right).
//!     \end{aligned}
//!   \\end{equation}
//!
//!   This is then commonly expressed as,
//!
//!   \\begin{equation}
//!       g_{a_1} \int \vt C[f_{a_{1}}] \frac{\dd \vt p_{a_1}}{(2\pi)^3}
//!         = - \left( \prod_{i \in \vt a} \frac{n_i}{n_i^{(0)}} \right) \gamma(\vt a \to \vt b)
//!           + \left( \prod_{i \in \vt b} \frac{n_i}{n_i^{(0)}} \right) \gamma(\vt b \to \vt a),
//!   \\end{equation}
//!
//!   where we have introduced the interaction density
//!
//!   \\begin{equation}
//!     \gamma(\vt a \to \vt b)
//!       = \int_{\vt a}^{\vt b}
//!         \abs{\mathcal M(\vt a \to \vt b)}^2 \left( \prod_{i \in \vt a} f^{(0)}_i \right).
//!   \\end{equation}
//!
//! # Interactions
//!
//! The majority of interactions can be accounted for by looking at simply \\(1
//! \leftrightarrow 2\\) decays/inverse decays and \\(2 \leftrightarrow 2\\)
//! scatterings.  In the former case, the phase space integration can be done
//! entirely analytically and is presented below; whilst in the latter case, the
//! phase space integration can be simplified to just two integrals.
//!
//! ## Decays and Inverse Decays
//!
//! Considering the interaction \\(a \leftrightarrow b + c\\), the reaction is
//!
//! \\begin{equation}
//!   \begin{aligned}
//!     g_a \int \vt C\[f_a\] \frac{\dd \vt p_a}{(2\pi)^3}
//!       &= - \frac{n_a}{n^{(0)}_a} \gamma(a \to bc) + \frac{n_b n_c}{n^{(0)}_b n^{(0)}_c} \gamma(bc \to a)
//!   \end{aligned}
//! \\end{equation}
//!
//! In every case, the squared matrix element will be completely independent of
//! all initial and final state momenta (even \\(\varepsilon \cdot p\\) terms
//! vanish after averaging over both initial and final spins).
//!
//! ### Decay Term
//!
//! For the decay, the integration over the final state momenta is trivial as
//! the only final-state-momentum dependence appears within the Dirac delta
//! function.  Consequently, we need only integrate over the initial state
//! momentum:
//!
//! \\begin{equation}
//!   \begin{aligned}
//!     \gamma(a \to bc)
//!       &= - \abs{\mathcal M(a \to bc)}^2 \int_{a}^{b,c} f^{(0)}_a \\\\
//!       &= - \frac{\abs{\mathcal M(a \to bc)}^2}{16 \pi} \frac{m_a K_1(m_a \beta)}{2 \pi^2 \beta}
//!   \end{aligned}
//! \\end{equation}
//!
//! Combined with the number density scaling:
//!
//! \\begin{equation}
//!   \frac{n_a}{n^{(0)}_a} \gamma(a \to bc)
//!     = n_a \frac{\abs{\mathcal M(a \to bc)}^2}{16 \pi m_a} \frac{K_1(m_a \beta)}{K_2(m_a \beta)}
//! \\end{equation}
//!
//! where the analytic expression for \\(n^{(0)}_a\\) was used to introduce the
//! second Bessel function.
//!
//! The ratio of Bessel functions is generally referred to as the *time-dilation
//! factor* for the decaying particle.  When \\(m \beta \gg 1\\), the ratio of
//! Bessel functions approaches 1, and when \\(m \beta \ll 1\\), the ratio
//! approaches \\(0\\).
//!
//! If the final state particles are essentially massless in comparison to the
//! decaying particle, then the decay rate in the rest frame of the particle of
//! \\(\Gamma_\text{rest} = \abs{\mathcal M}^2 / 16 \pi m_a\\) and the above
//! expression can be simplified to
//!
//! \\begin{equation}
//!   \frac{n_a}{n^{(0)}\_a} \gamma(a \to bc)
//!   = n_a \Gamma_\text{rest} \frac{K_1(m_a \beta)}{K_2(m_a \beta)}.
//! \\end{equation}
//!
//! ### Inverse Decay
//!
//! The inverse decay rate is given by
//!
//! \\begin{equation}
//!   \gamma(bc \to a)
//!     = \abs{\mathcal M(bc \to a)}^2 \int_{a}^{b,c} f^{(0)}_b f^{(0)}_c
//! \\end{equation}
//!
//! The Dirac delta enforces that \\(E_a = E_b + E_c\\) which implies that
//! \\(f^{(0)}_a = f^{(0)}_b f^{(0)}_c\\) thereby making the integral identical
//! to the decay scenario:
//!
//! \\begin{equation}
//!   \gamma(bc \to a)
//!     = \frac{\abs{\mathcal M(bc \to a)}^2}{16 \pi} \frac{m_a K_1(m_a \beta)}{2 \pi^2 \beta}
//!     = n^{(0)}_a \frac{\abs{\mathcal M(bc \to a)}^2}{16 \pi m_a} \frac{K_1(m_a \beta)}{K_2(m_a \beta)}
//! \\end{equation}
//!
//! The full expression including the number density scaling becomes:
//!
//! \\begin{equation}
//!   \frac{n_b n_c}{n^{(0)}_b n^{(0)}_c} \gamma(bc \to a)
//!     = \frac{n_b n_c}{n^{(0)}_b n^{(0)}_c} n^{(0)}_a \frac{\abs{\mathcal M(bc \to a)}^2}{16 \pi m_a} \frac{K_1(m_a \beta)}{K_2(m_a \beta)}
//! \\end{equation}
//!
//! ### Combined Decay and Inverse Decay
//!
//! If only the tree-level contributions are considered for the decay and
//! inverse decays, then both squared amplitudes will be in fact identical and
//! thus \\(\gamma(ab \to c) \equiv \gamma(c \to ab)\\).  As a result, the decay
//! and inverse decay only differ by a scaling factor to take into account the
//! initial state particles.
//!
//! In particular, we can define an alternative reaction rate,
//!
//! \\begin{equation}
//!   \tilde \gamma(a \to bc) = \frac{\abs{\mathcal{M}(a \to bc)}^2}{16 \pi m_a} \frac{K_1(m_a \beta)}{K_2(m_a \beta)},
//! \\end{equation}
//!
//! which allows for the overall \\(1 \leftrightarrow 2\\) reaction rate to be expressed as:
//!
//! \\begin{equation}
//!   \frac{n_a}{n_a^{(0)}} \gamma(a \to bc) - \frac{n_b n_c}{n_b^{(0)} n_c^{(0)}} \gamma(bc \to a)
//!   = \left[ n_a - \frac{n_b n_c}{n_b^{(0)} n_c^{(0)}} n_a^{(0)} \right] \tilde \gamma(a \to bc)
//! \\end{equation}
//!
//! provided the forward and backward rates are equal.
//!
//! ## Two Body Scattering
//!
//! The two-to-two scattering \\(ab \to cd\\) reaction density is given by
//!
//! \\begin{equation}
//!   \gamma(ab \to cd) = \int_{a,b}^{c,d} \abs{\mathcal M(ab \to cd)}^2 f_a^{(0)} f_b^{(0)}
//! \\end{equation}
//!
//! The two initial-state phase space integrals can be reduced to a simple
//! integral over the centre-of-mass \\(s\\):
//!
//! \\begin{equation}
//!   \gamma(ab \to cd) = \frac{1}{8 \pi^3 \beta} \int \hat \sigma_{ab}^{cd}(s) \sqrt{s} K_1(\sqrt{s} \beta) \dd s
//! \\end{equation}
//!
//! where \\(\hat \sigma(s)\\) is the reduced cross-section:
//!
//! \\begin{equation}
//!   \hat \sigma_{ab}^{cd}(s) = \frac{g_a g_b g_c g_d}{64 \pi^2 s} \int \abs{\mathcal M(ab \to cd)}^2 \dd t
//! \\end{equation}
//!
//! in which \\(t\\) is the usual Mandelstam variable.
//!
//! As a result, the full \\(2 \to 2\\) cross-section can be expressed as
//!
//! \\begin{equation}
//!   \gamma(ab \to cd) = \frac{g_a g_b g_c g_d}{512 \pi^5 \beta} \int \abs{\mathcal M(ab \to cd)}^2 \frac{K_1(\sqrt s \beta)}{\sqrt s} \dd s \dd t
//! \\end{equation}
//!
//! ### Real Intermediate State
//!
//! When considering both \\(1 \leftrightarrow 2\\) and \\(2 \leftrightarrow
//! 2\\) scattering processes, there is a double counting issue that arises from
//! having a real intermediate state (RIS) in \\(2 \leftrightarrow 2\\)
//! interactions.
//!
//! As a concrete example, one may have the processes \\(ab \leftrightarrow X\\)
//! and \\(X \leftrightarrow cd\\) and thus also \\(ab \leftrightarrow cd\\).
//! In computing the squared amplitude for the \\(2 \leftrightarrow 2\\)
//! process, one needs to subtract the RIS:
//!
//! \\begin{equation}
//!   \abs{\mathcal M(ab \leftrightarrow cd)}^2 = \abs{\mathcal M_\text{full}(ab \leftrightarrow cd)}^2 - \abs{\mathcal M_\textsc{RIS}(ab \leftrightarrow cd)}^2
//! \\end{equation}
//!
//! In the case of a single scalar RIS, the RIS-subtracted amplitude is given by
//!
//! \\begin{align}
//!   \abs{\mathcal M_\textsc{RIS}(ab \to cd)}
//!     &= \frac{\pi}{m_X \Gamma_X} \delta(s - m_X^2) \theta(\sqrt{s}) \\\\
//!     &\quad \Big[
//!         \abs{\mathcal M(ab \to X)}^2 \abs{\mathcal M(X \to cd)}^2
//!         + \abs{\mathcal M(ab \to \overline X)}^2 \abs{\mathcal M(\overline X \to cd)}^2
//!       \Big].
//! \\end{align}
//!
//! Care must be taken for fermions as the spinorial structure prevents a simple
//! factorization in separate squared amplitude.  Furthermore if there are
//! multiple intermediate states, the mixing between these states must also be
//! taken into account.

use super::{EmptyModel, Model, Solver, StepChange, StepPrecision};
use crate::universe::Universe;
use log::{debug, info};
use ndarray::{array, prelude::*, FoldWhile, Zip};
use rayon::prelude::*;

/// Context provided containing pre-computed values which might be useful when
/// evaluating interactions.
pub struct Context<M: Model> {
    /// Current evaluation step
    pub step: u64,
    /// Current step size
    pub step_size: f64,
    /// Inverse temperature in GeV\\(^{-1}\\)
    pub beta: f64,
    /// Hubble rate, in GeV
    pub hubble_rate: f64,
    /// Equilibrium number densities for the particles, normalized to the
    /// equilibrium number density for a massless boson with \\(g = 1\\).  The
    /// particle species are provided in the same order as when specified to the
    /// solver
    pub eq_n: Array1<f64>,
    /// Model data
    pub model: M,
}

/// Boltzmann equation solver for the number density.
///
/// All number densities are normalized to that of a massless boson with a
/// single degree of freedom (\\(g = 1\\)).  As a result of this convention,
/// \\(n_\gamma = 2\\) as the photon has two degrees of freedom.
pub struct NumberDensitySolver<M: Model + Sync> {
    initialized: bool,
    beta_range: (f64, f64),
    initial_conditions: Array1<f64>,
    #[allow(clippy::type_complexity)]
    interactions: Vec<
        Box<
            Fn(
                    &<Self as Solver>::Solution,
                    &<Self as Solver>::Context,
                ) -> <Self as Solver>::Solution
                + Sync,
        >,
    >,
    #[allow(clippy::type_complexity)]
    logger: Box<
        Fn(&<Self as Solver>::Solution, &<Self as Solver>::Solution, &<Self as Solver>::Context),
    >,
    model_fn: Box<Fn(f64) -> M>,
    step_change: StepChange,
    step_precision: StepPrecision,
    error_tolerance: f64,
    threshold_number_density: f64,
}

impl<M: Model + Sync> Solver for NumberDensitySolver<M> {
    /// The solution is a one-dimensional array of number densities for each
    /// particle species (or aggregated number density in the case of
    /// \\(n_{\mathsc{b-l}}\\)), in the same order as [`Solver::add_particle`]
    /// is invoked.
    type Solution = Array1<f64>;

    type Context = Context<M>;

    /// Create a new instance of the number density solver.
    ///
    /// The default range of temperatures span 1 GeV through to 10^{20} GeV.
    fn new() -> Self {
        #[allow(clippy::redundant_closure)]
        Self {
            initialized: false,
            beta_range: (1e-20, 1e0),
            initial_conditions: array![],
            interactions: Vec::with_capacity(100),
            logger: Box::new(|_, _, _| {}),
            model_fn: Box::new(|beta| M::new(beta)),
            step_change: StepChange::default(),
            step_precision: StepPrecision::default(),
            error_tolerance: 1e-4,
            threshold_number_density: 0.0,
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
        self.beta_range = (start, end);
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
        self.beta_range = (start.recip(), end.recip());
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
        self.step_change = StepChange { increase, decrease };
        self
    }

    fn step_precision(mut self, min: f64, max: f64) -> Self {
        assert!(
            min < max,
            "Minimum step precision must be smaller than the maximum step precision."
        );
        self.step_precision = StepPrecision { min, max };
        self
    }

    fn error_tolerance(mut self, tol: f64) -> Self {
        assert!(tol > 0.0, "The tolerance must be greater than 0.");
        self.error_tolerance = tol;
        self
    }

    fn initial_conditions(mut self, v: Vec<f64>) -> Self {
        self.initial_conditions = Array1::from_vec(v);

        self
    }

    fn initialize(mut self) -> Self {
        let model = (self.model_fn)(1e-3);
        let particles = model.particles();
        assert_eq!(
            self.initial_conditions.len(),
            particles.len(),
            "The number of particles in the model ({}) is different to the number of initial conditions ({}).",
            particles.len(),
            self.initial_conditions.len(),
        );

        self.initialized = true;

        self
    }

    fn add_interaction<F: 'static>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&Self::Solution, &Self::Context) -> Self::Solution + Sync,
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

        // Initialize all the variables that will be used in the integration
        let mut n = self.initial_conditions.clone();
        let mut dn = [
            Self::Solution::zeros(n.dim()),
            Self::Solution::zeros(n.dim()),
        ];

        let mut k: [Self::Solution; RK_S];
        unsafe {
            k = std::mem::uninitialized();
            for ki in &mut k[..] {
                std::ptr::write(ki, Self::Solution::zeros(n.dim()));
            }
        };

        let mut step = 0;
        let mut beta = self.beta_range.0;
        let mut h = beta * self.step_precision.min;

        // Create the initial context and log the initial conditions
        let mut c = self.context(step, h, beta, universe);
        (*self.logger)(&n, &dn[0], &c);

        while beta < self.beta_range.1 {
            step += 1;
            let mut advance = false;
            debug!("Step {:}, β = {:.4e}", step, beta);

            // Compute each k[i]
            for i in 0..RK_S {
                let beta_i = beta + RK_C[i] * h;
                let ci = self.context(step, h, beta_i, universe);
                let ai = RK_A[i];
                let ni = (0..i).fold(n.clone(), |total, j| total + ai[j] * &k[j]);
                k[i] = h * self
                    .interactions
                    .par_iter()
                    .map(|f| f(&ni, &ci))
                    .reduce(|| Self::Solution::zeros(n.dim()), |sum, a| sum + a);
                // Apply the `n_plus_dn` check here (even though we are
                // discarding `ni`) to place a limit on `k[i]`.
                self.n_plus_dn(ni, &mut k[i], &ci);
            }

            // Calculate the two estimates for dn
            dn[0] = (0..RK_S).fold(Self::Solution::zeros(n.dim()), |total, i| {
                total + RK_B[0][i] * &k[i]
            });
            dn[1] = (0..RK_S).fold(Self::Solution::zeros(n.dim()), |total, i| {
                total + RK_B[1][i] * &k[i]
            });

            // Get the error between the estimates
            let err = Zip::from(&dn[0])
                .and(&dn[1])
                .fold_while(0.0f64, |e, a, b| {
                    let v = (a - b).abs();
                    FoldWhile::Continue(e.max(v))
                })
                .into_inner()
                / h;

            // If the error is within the tolerance, we'll be advancing the
            // iteration step
            if err < self.error_tolerance {
                advance = true;
            }

            // Compute the change in step size based on the current error And
            // correspondingly adjust the step size
            let mut h_est = if err == 0.0 {
                h * self.step_change.increase
            } else {
                let delta = 0.9 * (self.error_tolerance / err).powf(1.0 / f64::from(RK_ORDER + 1));

                h * if delta < self.step_change.decrease {
                    self.step_change.decrease
                } else if delta > self.step_change.increase {
                    self.step_change.increase
                } else {
                    delta
                }
            };

            // Prevent h from getting too small or too big in proportion to the
            // current value of beta.  Also advance the integration irrespective
            // of the local error if we reach the maximum or minimum step size.
            if h_est > beta * self.step_precision.max {
                h_est = beta * self.step_precision.max;
                debug!("Step size too large, decreased h to {:.3e}", h_est);
                advance = true;
            } else if h_est < beta * self.step_precision.min {
                h_est = beta * self.step_precision.min;
                debug!("Step size too small, increased h to {:.3e}", h_est);
                advance = true;
            }

            // Check if the error is within the tolerance, or we are advancing
            // irrespective of the local error
            if advance {
                c = self.context(step, h, beta, universe);

                // Advance n and beta
                n = self.n_plus_dn(n, &mut dn[0], &c);
                beta += h;

                // Run the logger now
                (*self.logger)(&n, &dn[0], &c);
            }

            // Adjust final integration step if needed
            if beta + h_est > self.beta_range.1 {
                debug!("Fixing overshoot of last integration step.");
                h_est = self.beta_range.1 - beta;
            }

            h = h_est;
        }

        info!("Number of integration steps: {}", step);

        n
    }
}

impl Default for NumberDensitySolver<EmptyModel> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: Model + Sync> NumberDensitySolver<M> {
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
        self.threshold_number_density = threshold;
        self
    }

    /// Create an array containing the initial conditions for all the particle
    /// species.
    ///
    /// All particles species are assumed to be in thermal equilibrium at this
    /// energy, with the distribution following either the Bose–Einstein or
    /// Fermi–Dirac distribution as determined by their spin.
    fn equilibrium_number_densities(&self, beta: f64, model: &M) -> Array1<f64> {
        Array1::from_iter(
            model
                .particles()
                .iter()
                .map(|p| p.normalized_number_density(0.0, beta)),
        )
    }

    /// Set a model function.
    ///
    /// Add a function that can modify the model before it is used by the
    /// context.  This is particularly useful if there is a baseline model with
    /// fixed parameters and only certain parameters are changed.
    pub fn model_fn<F: 'static>(&mut self, f: F) -> &mut Self
    where
        F: Fn(f64) -> M,
    {
        self.model_fn = Box::new(f);
        self
    }

    /// Generate the context at a given beta to pass to the logger/interaction
    /// functions.
    fn context<U: Universe>(
        &self,
        step: u64,
        step_size: f64,
        beta: f64,
        universe: &U,
    ) -> Context<M> {
        let model = (self.model_fn)(beta);
        Context {
            step,
            step_size,
            beta,
            hubble_rate: universe.hubble_rate(beta),
            eq_n: self.equilibrium_number_densities(beta, &model),
            model,
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
            let new_n = *n + *dn;

            if (*n > *eq_n) ^ (new_n > *eq_n) {
                *dn = eq_n - *n;
                *n = *eq_n;
            } else {
                *n = new_n;
            }

            if n.abs() < self.threshold_number_density {
                debug!("n going below threshold number density, setting to zero.",);
                *dn = -*n;
                *n = 0.0;
            }
        });

        n
    }
}
