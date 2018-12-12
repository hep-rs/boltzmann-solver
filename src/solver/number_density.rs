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
// //! - *Assume \\(\mathcal{CP}\\) symmetry.* If \\(\mathcal{CP}\\) symmetry is
// //!   assumed, then \\(\abs{\mathcal M(\vt a \to \vt b)}^2 \equiv \abs{\mathcal
// //!   M(\vt b \to \vt a)}^2\\).  This simplification allows for the exponentials
// //!   of \\(\mu\\) to be taken out of the integral entirely:
// //!
// //!   \\begin{equation}
// //!     g_{a_1} \int \vt C[f_{a_{1}}] \frac{\dd \vt p_{a_1}}{(2\pi)^3}
// //!       = - \left[ e^{ \beta \sum_{\vt a} \mu_i } - e^{ \beta \sum_{\vt b} \mu_i } \right]
// //!            \int_{\vt a}^{\vt b} \abs{\mathcal M(\vt a \to \vt b)}^2 e^{ - \beta \sum_{\vt a} E_i }.
// //!   \\end{equation}
// //!
// //!   The remaining integrand is then independent of time and can be
// //!   pre-calculated.  In the case of a \\(2 \leftrightarrow 2\\) interaction,
// //!   this integral is related to the thermally averaged cross-section
// //!   \\(\angles{\sigma v_\text{rel}}\\).
// //!
// //!   Solving the Boltzmann equation is generally required within the context of
// //!   baryogenesis and leptogenesis where the assumption of \\(\mathcal{CP}\\)
// //!   symmetry is evidently not correct.  In such cases, it is convenient to
// //!   define the parameter \\(\epsilon\\) to account for all of the
// //!   \\(\mathcal{CP}\\) asymmetry as.  That is:
// //!
// //!   \\begin{equation}
// //!     \abs{\mathcal M(\vt a \to \vt b)}^2 = (1 + \epsilon) \abs{\mathcal{M^{(0)}(\vt a \to \vt b)}}^2, \qquad
// //!     \abs{\mathcal M(\vt b \to \vt a)}^2 = (1 - \epsilon) \abs{\mathcal{M^{(0)}(\vt a \to \vt b)}}^2,
// //!   \\end{equation}
// //!
// //!   where \\(\abs{\mathcal{M^{(0)}}(\vt a \to \vt b)}^2\\) is the
// //!   \\(\mathcal{CP}\\)-symmetric squared amplitude.  With \\(\epsilon\\)
// //!   defined as above, the collision term becomes:
// //!
// //!   \\begin{equation}
// //!     \begin{aligned}
// //!       g_{a_1} \int \vt C[f_{a_{1}}] \frac{\dd \vt p_{a_1}}{(2\pi)^3}
// //!         &= - \left[ e^{ \beta \sum_{\vt a} \mu_i } - e^{ \beta \sum_{\vt b} \mu_i } \right] \times
// //!              \int_{\vt a}^{\vt b} \abs{\mathcal M^{(0)}(\vt a \to \vt b)}^2 e^{ - \beta \sum_{\vt a} E_i } \\\\
// //!         &\quad - \left[ e^{ \beta \sum_{\vt a} \mu_i } + e^{ \beta \sum_{\vt b} \mu_i } \right] \times
// //!              \int_{\vt a}^{\vt b} \epsilon \abs{\mathcal M^{(0)}(\vt a \to \vt b)}^2 e^{ - \beta \sum_{\vt a} E_i }.
// //!     \\end{aligned}
// //!   \\end{equation}
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

// //!
// //! ## Two-to-\\(n\\) Scattering
// //!
// //! In \\(2 \to n\\) scattering, the initial state phase space integration can
// //! be done as before resulting in the same expression as above featuring the
// //! reduced cross-section; however, simplifying the final state phase space
// //! integration will depend on the nature of the process.  The general form of
// //! the reduced cross-section is:
// //!
// //! \\begin{equation}
// //!   \hat \sigma_{ab}^{\vt c}(s) = \Phi_{2}(q_{ab}; p_a, p_b) \int \left( \prod_{i \in \vt c} \dd \Pi_i \right) (2 \pi)^4 \delta^4(q_{ab} - p_{\vt c}) \abs{\mathcal M(ab \to \vt c)}^2
// //! \\end{equation}

use super::{EmptyModel, ErrorTolerance, InitialCondition, Model, Solver, StepChange};
use ndarray::{prelude::*, FoldWhile, Zip};
use particle::Particle;
use universe::Universe;

/// Context provided containing pre-computed values which might be useful when
/// evaluating interactions.
pub struct Context<M: Model> {
    /// Evaluation step
    pub step: u64,
    /// Inverse temperature in GeV^{-1}
    pub beta: f64,
    /// Hubble rate, in GeV
    pub hubble_rate: f64,
    /// Equilibrium number densities for the particles, normalized to the
    /// equilibrium number density for a massless boson with \\(g = 1\\).  This
    /// is provided in the same order as specified to the solver
    pub eq_n: Array1<f64>,
    /// Model data
    pub model: M,
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
    #[cfg_attr(feature = "cargo-clippy", allow(type_complexity))]
    interactions: Vec<Box<Fn(Array1<f64>, &Array1<f64>, &Context<M>) -> Array1<f64>>>,
    logger: Box<Fn(&Array1<f64>, &Context<M>)>,
    step_change: StepChange,
    error_tolerance: ErrorTolerance,
}

impl<M: Model> Solver for NumberDensitySolver<M> {
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
        Self {
            initialized: false,
            beta_range: (1e-20, 1e0),
            particles: Vec::with_capacity(20),
            initial_conditions: Vec::with_capacity(20),
            interactions: Vec::with_capacity(100),
            logger: Box::new(|_, _| {}),
            step_change: StepChange {
                increase: 1.1,
                decrease: 0.5,
            },
            error_tolerance: ErrorTolerance {
                upper: 1e-2,
                lower: 1e-5,
            },
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

    fn error_tolerance(mut self, upper: f64, lower: f64) -> Self {
        assert!(
            upper > lower,
            "The upper error tolerance must be greater than the lower tolerance"
        );
        self.error_tolerance = ErrorTolerance { upper, lower };
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
        F: Fn(&Self::Solution, &Self::Context),
    {
        self.logger = Box::new(f);
        self
    }

    fn solve<U>(&self, universe: &U) -> Self::Solution
    where
        U: Universe,
    {
        assert!(
            self.initialized,
            "The phase space solver has to be initialized first with the `initialize()` method."
        );

        let mut n = Array1::from_vec(self.initial_conditions.clone());
        let mut beta = self.beta_range.0;
        let mut h = beta / 10.0;

        // Allocate variables which will be re-used each for loop
        let mut k1: Self::Solution;
        let mut k2: Self::Solution;
        let mut k3: Self::Solution;
        let mut k4: Self::Solution;
        let mut tmp: Self::Solution;

        let mut step = 0;
        while beta < self.beta_range.1 {
            step += 1;

            // Create the initial context
            let c = self.context(step, beta, universe);

            // Run the logger now
            (*self.logger)(&n, &c);

            // Standard Runge-Kutta integration.
            k1 = self
                .interactions
                .iter()
                .fold(Self::Solution::zeros(n.dim()), |s, f| f(s, &n, &c))
                * h;

            let c = self.context(step, beta + 0.5 * h, universe);
            tmp = &n + &(&k1 * 0.5);
            k2 = self
                .interactions
                .iter()
                .fold(Self::Solution::zeros(n.dim()), |s, f| f(s, &tmp, &c))
                * h;

            let c = self.context(step, beta + 0.5 * h, universe);
            tmp = &n + &(&k2 * 0.5);
            k3 = self
                .interactions
                .iter()
                .fold(Self::Solution::zeros(n.dim()), |s, f| f(s, &tmp, &c))
                * h;

            let c = self.context(step, beta + h, universe);
            let tmp = &n + &k3;
            k4 = self
                .interactions
                .iter()
                .fold(Self::Solution::zeros(n.dim()), |s, f| f(s, &tmp, &c))
                * h;

            // Calculate dn.  Note that we consume k2, k3 and k4 here.  We use
            // k1 by reference since we need it later to get the error estimate.
            let dn = (k2 * 2.0 + k3 * 2.0 + k4 + &k1) / 6.0;

            // Check the error on the RK method vs the Euler method.  If it is
            // small enough, increase the step size.  We use the maximum error
            // for any given element of `dn`.
            let err = Zip::from(&k1)
                .and(&dn)
                .fold_while(0.0, |e, k, d| {
                    let v = (d / k - 1.0).abs();
                    if v.is_finite() && v > e {
                        FoldWhile::Continue(v)
                    } else {
                        FoldWhile::Continue(e)
                    }
                })
                .into_inner();

            // Adjust the step size as needed based on the step size.
            if err < self.error_tolerance.lower {
                h *= self.step_change.increase;
                debug!(
                    "Step {:>7}, β = {:>9.2e} -> Increased h to {:.3e} (error was {:.3e})",
                    step, beta, h, err
                );
            } else if err > self.error_tolerance.upper {
                h *= self.step_change.decrease;
                debug!(
                    "Step {:>7}, β = {:>9.2e} -> Decreased h to {:.3e} (error was {:.3e})",
                    step, beta, h, err
                );

                // Prevent h from getting too small that it might make
                // integration take too long.  Use the result regardless even
                // though it is bigger than desired error.
                if beta / h > 1e5 {
                    warn!(
                        "Step {:>7}, β = {:>9.2e} -> Step size getting too small (β / h = {:.1e}).",
                        step, beta, beta / h
                    );

                    while beta / h > 1e5 {
                        h *= self.step_change.increase;
                    }

                    n += &dn;
                    beta += h;
                }

                continue;
            }

            n += &dn;
            beta += h;
        }

        info!("Number of evaluations: {}", step);

        n
    }
}

impl Default for NumberDensitySolver<EmptyModel> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: Model> NumberDensitySolver<M> {
    /// Create an array containing the initial conditions for all the particle
    /// species.
    ///
    /// All particles species are assumed to be in thermal equilibrium at this
    /// energy, with the distribution following either the Bose–Einstein or
    /// Fermi–Dirac distribution as determined by their spin.
    fn equilibrium_number_densities(&self, beta: f64) -> Array1<f64> {
        Array1::from_iter(
            self.particles
                .iter()
                .map(|p| p.normalized_number_density(0.0, beta)),
        )
    }

    fn context<U: Universe>(&self, step: u64, beta: f64, universe: &U) -> Context<M> {
        Context {
            step,
            beta,
            hubble_rate: universe.hubble_rate(beta),
            eq_n: self.equilibrium_number_densities(beta),
            model: M::new(beta),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use universe::StandardModel;
    use utilities::test::*;

    /// The most trivial example with a single particle and no interactions.
    #[test]
    fn no_interaction() {
        let phi = Particle::new(0, 1e3);
        let mut solver = NumberDensitySolver::default()
            .temperature_range(1e20, 1e-10)
            .initialize();

        solver.add_particle(phi, InitialCondition::Equilibrium(0.0));

        let sol = solver.solve(&StandardModel::new());
        approx_eq(sol[0], 1.0, 8.0, 0.0);
    }
}
