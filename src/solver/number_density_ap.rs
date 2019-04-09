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

use super::{EmptyModel, InitialCondition, Model, Solver, StepChange, StepPrecision};
use crate::{particle::Particle, universe::Universe};
use log::{debug, info};
use ndarray::{prelude::*, FoldWhile, Zip};
use rug::{ops::*, Float};
use std::{cmp::Ordering, iter};

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
    beta_range: (f64, f64),
    particles: Vec<Particle>,
    initial_conditions: Vec<f64>,
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
    error_tolerance: f64,
    threshold_number_density: f64,
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
            beta_range: (1e-20, 1e0),
            particles: Vec::with_capacity(20),
            initial_conditions: Vec::with_capacity(20),
            interactions: Vec::with_capacity(100),
            logger: Box::new(|_, _, _| {}),
            step_change: StepChange::default(),
            step_precision: StepPrecision::default(),
            error_tolerance: 1e-4,
            threshold_number_density: 0.0,
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

    fn initialize(mut self) -> Self {
        self.initialized = true;

        self
    }

    fn add_particle(&mut self, s: Particle, initial_condition: InitialCondition) {
        match initial_condition {
            InitialCondition::Equilibrium(mu) => self
                .initial_conditions
                .push(s.normalized_number_density(mu, self.beta_range.0)),
            InitialCondition::Fixed(n) => self.initial_conditions.push(n),
            InitialCondition::Zero => self.initial_conditions.push(0.0),
        }

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
        use crate::solver::tableau::dp87::*;

        assert!(
            self.initialized,
            "The phase space solver has to be initialized first with the `initialize()` method."
        );

        // Initialize all the variables that will be used in the integration
        let mut n = Self::Solution::from_iter(
            self.initial_conditions
                .iter()
                .map(|v| Float::with_val(self.working_precision, v)),
        );
        let zero = Self::Solution::from_iter(
            iter::repeat(Float::with_val(self.working_precision, 0.0)).take(n.dim()),
        );
        let mut dn = [zero.clone(), zero.clone()];

        let mut k: [Self::Solution; RK_S];
        unsafe {
            k = std::mem::uninitialized();
            for ki in &mut k[..] {
                std::ptr::write(ki, zero.clone());
            }
        };

        let mut step = 0;
        let mut advanced: bool;
        let mut beta = Float::with_val(self.working_precision, self.beta_range.0);
        let mut h = Float::with_val(self.working_precision, &beta * &self.step_precision.min);

        // Create the initial context and log the initial conditions
        let mut c = self.context(step, beta.clone(), universe, h.clone());
        (*self.logger)(&n, &dn[0], &c);

        while &beta < &self.beta_range.1 {
            step += 1;
            advanced = false;

            // Compute each k[i]
            for i in 0..RK_S {
                let mut beta_i = h.clone() * RK_C[i];
                beta_i += &beta;
                let ci = self.context(step, beta_i, universe, h.clone());
                let ai = RK_A[i];
                let mut dni = (0..i).fold(zero.clone(), |mut total, j| {
                    Zip::from(&mut total)
                        .and(&k[i])
                        .apply(|t, k| *t += Float::with_val(self.working_precision, ai[j]) * k);
                    total
                });
                let ni = self.n_plus_dn(n.clone(), &mut dni, &ci);
                k[i] = self
                    .interactions
                    .iter()
                    .fold(zero.clone(), |s, f| f(s, &ni, &ci));
                k[i].mapv_inplace(|v| v * &h);
            }

            // Calculate the two estimates
            dn[0] = (0..RK_S).fold(zero.clone(), |mut total, i| {
                Zip::from(&mut total)
                    .and(&k[i])
                    .apply(|t, k| *t += Float::with_val(self.working_precision, RK_B[0][i]) * k);
                total
            });
            dn[1] = (0..RK_S).fold(zero.clone(), |mut total, i| {
                Zip::from(&mut total)
                    .and(&k[i])
                    .apply(|t, k| *t += Float::with_val(self.working_precision, RK_B[1][i]) * k);
                total
            });

            // Get the error between the estimates
            let err = Zip::from(&dn[0])
                .and(&dn[1])
                .fold_while(Float::with_val(self.working_precision, 0.0), |e, a, b| {
                    let v = Float::with_val(self.working_precision, a - b).abs();
                    match e.cmp_abs(&v) {
                        Some(Ordering::Less) => FoldWhile::Continue(v),
                        _ => FoldWhile::Continue(e),
                    }
                })
                .into_inner()
                / &h;

            // If the error is within the tolerance, add the result
            if err < self.error_tolerance {
                c = self.context(step, beta.clone(), universe, h.clone());
                n = self.advance(n, &mut dn[0], &mut beta, &h, &c);
                advanced = true;
            }

            // Compute the change in step size based on the current error And
            // correspondingly adjust the step size
            if err.is_zero() {
                h *= self.step_change.increase;
            } else {
                let delta = 0.9 * (&self.error_tolerance / err).pow(1.0 / f64::from(RK_ORDER + 1));
                // debug!("Step {:}, β = {:.4e} -> δ = {:<10.3e}", step, beta, delta);

                if delta < self.step_change.decrease {
                    h *= self.step_change.decrease;
                } else if delta > self.step_change.increase {
                    h *= self.step_change.increase;
                } else {
                    h *= &delta;
                }
            }

            // Prevent h from getting too small or too big in proportion to the
            // current value of beta.  Also advance the integration irrespective
            // of the local error if we reach the maximum or minimum step size.
            let min_step =
                Float::with_val(self.working_precision, &beta * &self.step_precision.min);
            let max_step =
                Float::with_val(self.working_precision, &beta * &self.step_precision.max);
            if h > max_step {
                h = max_step;
                debug!(
                    "Step {:}, β = {:.4e} -> Step size too large, decreased h to {:.3e}",
                    step, beta, h
                );

                if !advanced {
                    c = self.context(step, beta.clone(), universe, h.clone());
                    n = self.advance(n, &mut dn[0], &mut beta, &h, &c);
                }
            } else if h < min_step {
                h = min_step;
                debug!(
                    "Step {:}, β = {:.4e} -> Step size too small, increased h to {:.3e}",
                    step, beta, h
                );

                if !advanced {
                    c = self.context(step, beta.clone(), universe, h.clone());
                    n = self.advance(n, &mut dn[0], &mut beta, &h, &c);
                }
            }
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
        self.threshold_number_density = threshold;
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

    /// Advance `beta` and `n`, returning the new `n`.
    #[inline]
    fn advance(
        &self,
        mut n: <Self as Solver>::Solution,
        dn: &mut <Self as Solver>::Solution,
        beta: &mut Float,
        h: &Float,
        c: &<Self as Solver>::Context,
    ) -> <Self as Solver>::Solution {
        // Advance n and beta
        n = self.n_plus_dn(n, dn, c);
        *beta += h;

        // Run the logger now
        (*self.logger)(&n, dn, c);

        n
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
        Context {
            step,
            beta,
            step_size,
            hubble_rate: universe.hubble_rate(beta_f64),
            eq_n: self.equilibrium_number_densities(beta_f64),
            model: M::new(beta_f64),
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
