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
//!
//! # Temperature Evolution
//!
//! Although it is most intuitive to describe the interaction rates in terms of
//! time, most quantities depend on the temperature of the Universe at a
//! particular time.  We therefore make the change of variables from \\(t\\) to
//! \\(\beta\\) which introduces a factor of \\(H(\beta) \beta\\):
//!
//! \\begin{equation}
//!   \pfrac{n}{t} + 3 H n \equiv H \beta \pfrac{n}{\beta} + 3 H n.
//! \\end{equation}
//!
//! As a result, the actual change in the number density, it becomes
//!
//! \\begin{equation}
//!   \pfrac{n}{\beta} = \frac{1}{H \beta} \left[\vt C\[n\] - 3 H n\right]
//! \\end{equation}
//!
//! and one must only input \\(\vt C\[n\]\\) in the interaction.
//!
//! # Normalized Number Density
//!
//! The number densities themselves can be often quite large (especially in the
//! early Universe), and often are compared to other number densities; as a
//! result, they are often normalized to either the photon number density
//! \\(n_\gamma\\) or the entropy density \\(s\\).  This library uses a
//! equilibrium number density of a single massless bosonic degree of freedom
//! (and thus differs from using the photon number density by a factor of 2).
//!
//! When dealing with number densities, the Liouville operator is:
//!
//! \\begin{equation}
//!   \pfrac{n}{t} + 3 H n = \vt C\[n\]
//! \\end{equation}
//!
//! where \\(\vt C\[n\]\\) is the change in the number density.  If we now
//! define
//!
//! \\begin{equation}
//!   Y \defeq \frac{n}{n_{\text{eq}}},
//! \\end{equation}
//!
//! then the change in this normalized number density is:
//!
//! \\begin{equation}
//!   \pfrac{Y}{t} = \frac{1}{n_{\text{eq}}} \vt C\[n\]
//! \\end{equation}
//!
//! with \\(n_{\text{eq}}\\) having the simple analytic form \\(3 \zeta(3) / 4
//! \pi^2 \beta^3\\).  Furthermore, make the change of variable from time
//! \\(t\\) to inverse temperature \\(\beta\\), we get:
//!
//! \\begin{equation}
//!   \pfrac{Y}{\beta} = \frac{1}{H \beta n_{\text{eq}}} \vt C\[n\].
//! \\end{equation}
//!
//! As with the non-normalized number density calculations, only \\(\vt
//! C\[n\]\\) must be inputted in the interaction.

mod interaction;

use crate::{
    solver::{Model, StepPrecision},
    statistic::{Statistic, Statistics},
    universe::Universe,
};
pub use interaction::{Interacting, Interaction};
use ndarray::{array, prelude::*, Zip};
use rayon::prelude::*;

/// Context provided containing pre-computed values which might be useful when
/// evaluating interactions.
pub struct Context<M: Model + Sync> {
    /// Current evaluation step
    pub step: u64,
    /// Current step size
    pub step_size: f64,
    /// Inverse temperature in GeV\\(^{-1}\\)
    pub beta: f64,
    /// Hubble rate, in GeV
    pub hubble_rate: f64,
    /// Normalization factor, which may be either
    /// \\begin{equation}
    ///   \frac{1}{H \beta} \quad \text{or} \quad \frac{1}{H \beta n_1}
    /// \\end{equation}
    /// depending on whether it is the plain number density or normalized number
    /// density (respectively), and where \\(n_1\\) is the number density of a
    /// single massless bosonic degree of freedom.
    pub normalization: f64,
    /// Equilibrium number densities for the particles.
    pub eq: Array1<f64>,
    /// Model data
    pub model: M,
}

/// Boltzmann solver builder
pub struct SolverBuilder<M: Model + Sync> {
    normalized: bool,
    model: Box<dyn Fn(f64) -> M>,
    initial_conditions: Array1<f64>,
    beta_range: (f64, f64),
    #[allow(clippy::type_complexity)]
    interactions: Vec<Box<dyn Fn(&Array1<f64>, &Context<M>) -> Vec<Interaction> + Sync>>,
    equilibrium: Vec<usize>,
    #[allow(clippy::type_complexity)]
    logger: Box<dyn Fn(&Array1<f64>, &Array1<f64>, &Context<M>)>,
    step_precision: StepPrecision,
    error_tolerance: f64,
    threshold_number_density: f64,
}

/// Boltzmann solver
pub struct Solver<M: Model + Sync> {
    normalized: bool,
    model: Box<dyn Fn(f64) -> M>,
    initial_conditions: Array1<f64>,
    beta_range: (f64, f64),
    equilibrium: Vec<usize>,
    #[allow(clippy::type_complexity)]
    interactions: Vec<Box<dyn Fn(&Array1<f64>, &Context<M>) -> Vec<Interaction> + Sync>>,
    #[allow(clippy::type_complexity)]
    logger: Box<dyn Fn(&Array1<f64>, &Array1<f64>, &Context<M>)>,
    step_precision: StepPrecision,
    error_tolerance: f64,
    threshold_number_density: f64,
}

impl<M: Model + Sync> SolverBuilder<M> {
    /// Creates a new builder for the Boltzmann solver.
    ///
    /// The default range of temperatures span 1 GeV through to 10^{20} GeV, and
    /// it uses normalization by default.
    ///
    /// Most of the method for the builder are intended to be chained one after
    /// the other.  The two notable exceptions which use the builder by
    /// reference are [`SolverBuilder::add_interaction`] and
    /// [`SolverBuilder::logger`].
    ///
    /// ```
    /// let mut solver_builder = SolverBuilder::new()
    ///     .initial_conditions(&[1.2, 1.3])
    ///     .beta_range(1e-10, 1e-6)
    ///     .equilibrium(&[0]);
    /// // builder.add_interaction(..);
    /// // builder.logger(..);
    /// let solver = solver_builder.build();
    /// ```
    pub fn new() -> Self {
        Self {
            normalized: true,
            model: Box::new(|beta| M::new(beta)),
            initial_conditions: array![],
            beta_range: (1e-20, 1e0),
            equilibrium: Vec::with_capacity(128),
            interactions: Vec::with_capacity(128),
            logger: Box::new(|_, _, _| {}),
            step_precision: StepPrecision::default(),
            error_tolerance: 1e-4,
            threshold_number_density: 0.0,
        }
    }

    /// Specify whether the solver is dealing with normalized quantities or not.
    ///
    /// If normalized, the number densities are normalized to the number density
    /// of a single massless bosonic degree of freedom.
    pub fn normalized(mut self, val: bool) -> Self {
        self.normalized = val;
        self
    }

    /// Set a model function.
    ///
    /// Add a function that can modify the model before it is used by the
    /// context.  This is particularly useful if there is a baseline model with
    /// fixed parameters and only certain parameters are changed.
    pub fn model<F: 'static>(mut self, f: F) -> Self
    where
        F: Fn(f64) -> M,
    {
        self.model = Box::new(f);
        self
    }

    /// Specify initial conditions explicitly for the number densities.
    ///
    /// The list of number densities must be in the same order as the particles
    /// in the model.
    pub fn initial_conditions<I>(mut self, v: I) -> Self
    where
        I: IntoIterator<Item = f64>,
    {
        self.initial_conditions = Array1::from_iter(v);

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
    /// These particles are specified by index, with these particles remaining
    /// in equilibrium no matter what the interactions are.
    pub fn equilibrium<I>(mut self, eq: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        self.equilibrium = eq.into_iter().collect();
        self
    }

    /// Add an interaction.
    ///
    /// The interaction is a functional of the solution at a particular inverse
    /// temperature.  The function is of the following form:
    ///
    /// ```ignore
    /// f(n: &Array1<f64>, c: &Context) -> Interaction
    /// ```
    ///
    /// The first argument contains the values of the various
    /// number densities calculated so far, with the context being given in
    /// the second argument.
    ///
    /// The returned value is an instance of `Interaction`.
    pub fn add_interaction<F>(&mut self, f: F)
    where
        F: 'static + Fn(&Array1<f64>, &Context<M>) -> Vec<Interaction> + Sync,
    {
        self.interactions.push(Box::new(f));
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
    pub fn logger<F: 'static>(&mut self, f: F)
    where
        F: Fn(&Array1<f64>, &Array1<f64>, &Context<M>),
    {
        self.logger = Box::new(f);
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

    /// Set the threshold number density to count as 0.
    ///
    /// Any number density whose absolute value is less than the threshold will
    /// be treated as being exactly zero.  This applies to both calculated
    /// number densities as well as equilibrium number densities.  Furthermore,
    /// this also applies to 'abstract' number densities such as \\(B-L\\).
    ///
    /// This is by default set to `1e-20` for normalized number density, or
    /// `1e-20 * n` for non-normalized number densities, where `n` is the number
    /// density of a massless bosonic degree of freedom.
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

    /// Build the Boltzmann solver.
    pub fn build(self) -> Solver<M> {
        let model = (self.model)((self.beta_range.0 * self.beta_range.1).sqrt());
        let particles = model.particles();
        assert_eq!(
            self.initial_conditions.len(),
            particles.len(),
            "The number of particles in the model ({}) is different to the number of initial conditions ({}).",
            particles.len(),
            self.initial_conditions.len(),
        );

        Solver {
            normalized: self.normalized,
            initial_conditions: self.initial_conditions,
            beta_range: self.beta_range,
            model: self.model,
            interactions: self.interactions,
            equilibrium: self.equilibrium,
            logger: self.logger,
            step_precision: self.step_precision,
            error_tolerance: self.error_tolerance,
            threshold_number_density: self.threshold_number_density,
        }
    }
}

impl<M: Model + Sync> Default for SolverBuilder<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: Model + Sync> Solver<M> {
    /// Evolve the initial conditions by solving the PDEs.
    ///
    /// Note that this is not a true PDE solver, but instead converts the PDE
    /// into an ODE in time (or inverse temperature in this case), with the
    /// derivative in energy being calculated solely from the previous
    /// time-step.
    #[allow(clippy::cognitive_complexity)]
    pub fn solve<U>(&self, universe: &U) -> Array1<f64>
    where
        U: Universe,
    {
        use super::tableau::rk76::*;

        // Initialize all the variables that will be used in the integration
        let mut n = self.initial_conditions.clone();
        let mut dn = Array1::zeros(n.dim());
        let mut dn_err = Array1::zeros(n.dim());

        let mut k: [Array1<f64>; RK_S];
        unsafe {
            k = std::mem::MaybeUninit::uninit().assume_init();
            for ki in &mut k[..] {
                std::ptr::write(ki, Array1::zeros(n.dim()));
            }
        };

        let mut step = 0;
        let mut beta = self.beta_range.0;
        let mut h = beta * self.step_precision.min;
        let mut advance;

        let mut c = self.context(step, h, beta, universe);
        (*self.logger)(&n, &dn, &c);

        while beta < self.beta_range.1 {
            step += 1;
            advance = false;
            c = self.context(step, h, beta, universe);
            dn.fill(0.0);
            dn_err.fill(0.0);

            // Ensure that h is within the desired range of step sizes.
            let h_on_beta = h / beta;
            if h_on_beta > self.step_precision.max {
                h = beta * self.step_precision.max;
                log::trace!("Step size too large, decreased h to {:.3e}", h);
            } else if h_on_beta < self.step_precision.min {
                h = beta * self.step_precision.min;
                log::debug!("Step size too small, increased h to {:.3e}", h);
                // Irrespective of the local error, if we're at the minimum step
                // size we'll be integrating this step.
                advance = true;
            }

            // Log the progress of the integration
            if step % 1000 == 0 {
                log::info!("Step {}, β = {:.4e}", step, beta);
                log::info!("n = {:.3e}", n);
            } else if step % 100 == 0 {
                log::debug!("Step {}, , β = {:.4e}", step, beta);
                log::debug!("n = {:.3e}", n);
            } else {
                log::trace!("Step {}, β = {:.4e}, h = {:.4e}", step, beta, h);
                log::trace!("n = {:.3e}", n);
            }

            for i in 0..RK_S {
                // Compute the sub-step values
                let beta_i = beta + RK_C[i] * h;
                let ai = RK_A[i];
                let ni = (0..i).fold(n.clone(), |total, j| total + ai[j] * &k[j]);
                let ci = self.context(step, h, beta_i, universe);
                log::trace!(" n[{:0>2}] = {:>10.3e}", i, ni);
                log::trace!("eq[{:0>2}] = {:>10.3e}", i, ci.eq);

                // Compute k[i] from each interaction
                let ki = &mut k[i];
                *ki = if self.normalized {
                    self.interactions
                        .par_iter()
                        .fold(
                            || Array1::zeros(n.dim()),
                            |mut dn, f| {
                                for mut interaction in f(&ni, &ci) {
                                    interaction *= ci.normalization * h;
                                    dn = interaction.dn(dn, &ni, &ci);
                                }
                                dn
                            },
                        )
                        .reduce(|| Array1::zeros(n.dim()), |dn, dni| dn + dni)
                } else {
                    unimplemented!()
                };
                log::trace!(" k[{:0>2}] = {:>10.3e}", i, ki);

                // Set changes to zero for those particles in equilibrium
                for &eq in &self.equilibrium {
                    ki[eq] = 0.0;
                }

                let bi = RK_B[i];
                let ei = RK_E[i];
                Zip::from(&mut dn)
                    .and(&mut dn_err)
                    .and(ki)
                    .apply(|dn, dn_err, &mut ki| {
                        *dn += bi * ki;
                        *dn_err += ei * ki;
                    })
            }

            // Adjust dn for those particles in equilibrium
            for &eq in &self.equilibrium {
                dn[eq] = c.eq[eq] - n[eq];
            }

            log::trace!("    dn = {:>10.3e}", dn);
            log::trace!("dn_err = {:>10.3e}", dn_err);

            // Get the local error using L∞-norm
            let err = dn_err.iter().fold(0f64, |e, v| e.max(v.abs()));
            log::trace!("Error = {:.3e}", err);

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
            log::trace!("Δ = {:.3e}", delta);
            let mut h_est = h * delta;

            // Update n and beta
            if advance {
                // Advance n and beta
                n += &dn;
                beta += h;

                (*self.logger)(&n, &dn, &c);
            }

            // Adjust final integration step if needed
            if beta + h_est > self.beta_range.1 {
                log::trace!("Fixing overshoot of last integration step.");
                h_est = self.beta_range.1 - beta;
            }

            h = h_est;
        }

        log::info!("Number of integration steps: {}", step);

        n
    }
}

impl<M: Model + Sync> Solver<M> {
    fn apply_threshold(&self, n: &mut Array1<f64>) {
        if self.normalized {
            for n in n {
                if n.abs() < self.threshold_number_density {
                    *n = 0.0
                }
            }
        } else {
            unimplemented!()
        }
    }

    /// Create an array containing the initial conditions for all the particle
    /// species.
    ///
    /// All particles species are assumed to be in thermal equilibrium at this
    /// energy, with the distribution following either the Bose–Einstein or
    /// Fermi–Dirac distribution as determined by their spin.
    fn equilibrium_number_densities(&self, beta: f64, model: &M) -> Array1<f64> {
        let mut eq = if self.normalized {
            Array1::from_iter(
                model
                    .particles()
                    .iter()
                    .map(|p| p.normalized_number_density(0.0, beta)),
            )
        } else {
            unimplemented!()
        };
        self.apply_threshold(&mut eq);
        eq
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
        let model = (self.model)(beta);
        let hubble_rate = universe.hubble_rate(beta);
        let normalization = if self.normalized {
            let n = Statistic::BoseEinstein.massless_number_density(0.0, beta);
            (hubble_rate * beta * n).recip()
        } else {
            (hubble_rate * beta).recip()
        };
        Context {
            step,
            step_size,
            beta,
            hubble_rate,
            normalization,
            eq: self.equilibrium_number_densities(beta, &model),
            model,
        }
    }
}
