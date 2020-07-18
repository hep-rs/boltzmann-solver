//! Solver for the number density evolution given by integrating the Boltzmann
//! equation.
//!
//! Solving the Boltzmann equation in generality without making any assumption
//! about the phase space distribution `$f$` (other than specifying some initial
//! condition) is a difficult problem.  Furthermore, within the context of
//! baryogenesis and leptogenesis, we are only interested in the number density
//! (or more specifically, the difference in the number densities or a particle
//! and its corresponding antiparticle).
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
//!   ```math
//!   f_{\textsc{BE}} = \frac{1}{\exp[(E - \mu) \beta] - 1}, \qquad
//!   f_{\textsc{FD}} = \frac{1}{\exp[(E - \mu) \beta] + 1}.
//!   ```
//!
//!   For a particular that remains in kinetic equilibrium, the evolution of its
//!   phase space is entirely described by the evolution of `$\mu$` in time.
//!
//! - *Assume `$\beta \gg E - \mu$`.* In the limit that the temperature is much
//!   less than `$E - \mu$` (or equivalently, that the inverse temperature
//!   `$\beta$` is greater than `$E - \mu$`), both the Fermi–Dirac and
//!   Bose–Einstein approach the Maxwell–Boltzmann distribution,
//!
//!   ```math
//!   f_{\textsc{MB}} = \exp[-(E - \mu) \beta].
//!   ```
//!
//!   This simplifies the expression for the number density to
//!
//!   ```math
//!   n = \frac{g m^2 K_2(m \beta)}{2 \pi^2 \beta} e^{\mu \beta} = n^{(0)} e^{\mu \beta},
//!   ```
//!
//!   where `$n^{(0)}$` is the equilibrium number density when `$\mu = 0$`.
//!   This also allows for the equilibrium phase space distribution to be
//!   expressed in terms of the `$\mu = 0$` distribution:
//!
//!   ```math
//!   f_{\textsc{MB}} = e^{\mu \beta} f_{\textsc{MB}}^{(0)} = \frac{n}{n^{(0)}} f_{\textsc{MB}}^{(0)}.
//!   ```
//!
//!   Furthermore, the assumption that `$\beta \gg E - \mu$` implies that
//!   `$f_{\textsc{BE}}, f_{\textsc{FD}} \ll 1$`.  Consequently, the Pauli
//!   suppression and Bose enhancement factors in the collision term can all be
//!   neglected resulting in:
//!
//!   ```math
//!   \begin{aligned}
//!     g_{a_1} \int \vt C[f_{a_{1}}] \frac{\dd \vt p_{a_1}}{(2\pi)^3}
//!       &= - \int_{\vt a}^{\vt b}
//!          \abs{\mathcal M(\vt a \to \vt b)}^2 \left( \prod_{i \in \vt a} f_i \right)
//!          - \abs{\mathcal M(\vt b \to \vt a)}^2 \left( \prod_{i \in \vt b} f_i \right) \\\\
//!       &= - \int_{\vt a}^{\vt b}
//!          \abs{\mathcal M(\vt a \to \vt b)}^2 \left( \prod_{i \in \vt a} \frac{n_i}{n_i^{(0)}} f_i^{(0)} \right)
//!           - \abs{\mathcal M(\vt b \to \vt a)}^2 \left( \prod_{i \in \vt b} \frac{n_i}{n_i^{(0)}} f_i^{(0)} \right).
//!   \end{aligned}
//!   ```
//!
//!   This is then commonly expressed as,
//!
//!   ```math
//!   g_{a_1} \int \vt C[f_{a_{1}}] \frac{\dd \vt p_{a_1}}{(2\pi)^3}
//!     = - \left( \prod_{i \in \vt a} \frac{n_i}{n_i^{(0)}} \right) \gamma(\vt a \to \vt b)
//!       + \left( \prod_{i \in \vt b} \frac{n_i}{n_i^{(0)}} \right) \gamma(\vt b \to \vt a),
//!   ```
//!
//!   where we have introduced the interaction density
//!
//!   ```math
//!   \gamma(\vt a \to \vt b)
//!     = \int_{\vt a}^{\vt b}
//!       \abs{\mathcal M(\vt a \to \vt b)}^2 \left( \prod_{i \in \vt a} f^{(0)}_i \right).
//!   ```
//!
//! # Interactions
//!
//! The majority of interactions can be accounted for by looking at simply `$`
//! \leftrightarrow 2$` decays/inverse decays and `$2 \leftrightarrow 2$`
//! scatterings.  In the former case, the phase space integration can be done
//! entirely analytically and is presented below; whilst in the latter case, the
//! phase space integration can be simplified to just two integrals.
//!
//! ## Decays and Inverse Decays
//!
//! Considering the interaction `$a \leftrightarrow b + c$`, the reaction is
//!
//! ```math
//! \begin{aligned}
//!   g_a \int \vt C\[f_a\] \frac{\dd \vt p_a}{(2\pi)^3}
//!     &= - \frac{n_a}{n^{(0)}_a} \gamma(a \to bc) + \frac{n_b n_c}{n^{(0)}_b n^{(0)}_c} \gamma(bc \to a)
//! \end{aligned}
//! ```
//!
//! In every case, the squared matrix element will be completely independent of
//! all initial and final state momenta (even `$\varepsilon \cdot p$` terms
//! vanish after averaging over both initial and final spins).
//!
//! ### Decay Term
//!
//! For the decay, the integration over the final state momenta is trivial as
//! the only final-state-momentum dependence appears within the Dirac delta
//! function.  Consequently, we need only integrate over the initial state
//! momentum:
//!
//! ```math
//! \begin{aligned}
//!   \gamma(a \to bc)
//!     &= - \abs{\mathcal M(a \to bc)}^2 \int_{a}^{b,c} f^{(0)}_a \\\\
//!     &= - \frac{\abs{\mathcal M(a \to bc)}^2}{16 \pi} \frac{m_a K_1(m_a \beta)}{2 \pi^2 \beta}
//! \end{aligned}
//! ```
//!
//! Combined with the number density scaling:
//!
//! ```math
//! \frac{n_a}{n^{(0)}_a} \gamma(a \to bc)
//!   = n_a \frac{\abs{\mathcal M(a \to bc)}^2}{16 \pi m_a} \frac{K_1(m_a \beta)}{K_2(m_a \beta)}
//! ```
//!
//! where the analytic expression for `$n^{(0)}_a$` was used to introduce the
//! second Bessel function.
//!
//! The ratio of Bessel functions is generally referred to as the *time-dilation
//! factor* for the decaying particle.  When `$m \beta \gg 1$`, the ratio of
//! Bessel functions approaches 1, and when `$m \beta \ll 1$`, the ratio
//! approaches `$0$`.
//!
//! If the final state particles are essentially massless in comparison to the
//! decaying particle, then the decay rate in the rest frame of the particle of
//! `$\Gamma_\text{rest} = \abs{\mathcal M}^2 / 16 \pi m_a$` and the above
//! expression can be simplified to
//!
//! ```math
//! \frac{n_a}{n^{(0)}\_a} \gamma(a \to bc)
//! = n_a \Gamma_\text{rest} \frac{K_1(m_a \beta)}{K_2(m_a \beta)}.
//! ```
//!
//! ### Inverse Decay
//!
//! The inverse decay rate is given by
//!
//! ```math
//! \gamma(bc \to a)
//!   = \abs{\mathcal M(bc \to a)}^2 \int_{a}^{b,c} f^{(0)}_b f^{(0)}_c
//! ```
//!
//! The Dirac delta enforces that `$E_a = E_b + E_c$` which implies that
//! `$f^{(0)}_a = f^{(0)}_b f^{(0)}_c$` thereby making the integral identical to
//! the decay scenario:
//!
//! ```math
//! \gamma(bc \to a)
//!   = \frac{\abs{\mathcal M(bc \to a)}^2}{16 \pi} \frac{m_a K_1(m_a \beta)}{2 \pi^2 \beta}
//!   = n^{(0)}_a \frac{\abs{\mathcal M(bc \to a)}^2}{16 \pi m_a} \frac{K_1(m_a \beta)}{K_2(m_a \beta)}
//! ```
//!
//! The full expression including the number density scaling becomes:
//!
//! ```math
//! \frac{n_b n_c}{n^{(0)}_b n^{(0)}_c} \gamma(bc \to a)
//!   = \frac{n_b n_c}{n^{(0)}_b n^{(0)}_c} n^{(0)}_a \frac{\abs{\mathcal M(bc \to a)}^2}{16 \pi m_a} \frac{K_1(m_a \beta)}{K_2(m_a \beta)}
//! ```
//!
//! ### Combined Decay and Inverse Decay
//!
//! If only the tree-level contributions are considered for the decay and
//! inverse decays, then both squared amplitudes will be in fact identical and
//! thus `$\gamma(ab \to c) \equiv \gamma(c \to ab)$`.  As a result, the decay
//! and inverse decay only differ by a scaling factor to take into account the
//! initial state particles.
//!
//! In particular, we can define an alternative reaction rate,
//!
//! ```math
//! \tilde \gamma(a \to bc) = \frac{\abs{\mathcal{M}(a \to bc)}^2}{16 \pi m_a} \frac{K_1(m_a \beta)}{K_2(m_a \beta)},
//! ```
//!
//! which allows for the overall `$` \leftrightarrow 2$` reaction rate to be
//! expressed as:
//!
//! ```math
//! \frac{n_a}{n_a^{(0)}} \gamma(a \to bc) - \frac{n_b n_c}{n_b^{(0)} n_c^{(0)}} \gamma(bc \to a)
//! = \left[ n_a - \frac{n_b n_c}{n_b^{(0)} n_c^{(0)}} n_a^{(0)} \right] \tilde \gamma(a \to bc)
//! ```
//!
//! provided the forward and backward rates are equal.
//!
//! ## Two Body Scattering
//!
//! The two-to-two scattering `$ab \to cd$` reaction density is given by
//!
//! ```math
//! \gamma(ab \to cd) = \int_{a,b}^{c,d} \abs{\mathcal M(ab \to cd)}^2 f_a^{(0)} f_b^{(0)}
//! ```
//!
//! The two initial-state phase space integrals can be reduced to a simple
//! integral over the centre-of-mass `$s$`:
//!
//! ```math
//! \gamma(ab \to cd) = \frac{1}{8 \pi^3 \beta} \int \hat \sigma_{ab}^{cd}(s) \sqrt{s} K_1(\sqrt{s} \beta) \dd s
//! ```
//!
//! where `$\hat \sigma(s)$` is the reduced cross-section:
//!
//! ```math
//! \hat \sigma_{ab}^{cd}(s) = \frac{g_a g_b g_c g_d}{64 \pi^2 s} \int \abs{\mathcal M(ab \to cd)}^2 \dd t
//! ```
//!
//! in which `$t$` is the usual Mandelstam variable.
//!
//! As a result, the full `$2 \to 2$` cross-section can be expressed as
//!
//! ```math
//! \gamma(ab \to cd) = \frac{g_a g_b g_c g_d}{512 \pi^5 \beta} \int \abs{\mathcal M(ab \to cd)}^2 \frac{K_1(\sqrt s \beta)}{\sqrt s} \dd s \dd t
//! ```
//!
//! ### Real Intermediate State
//!
//! When considering both `$` \leftrightarrow 2$` and `$2 \leftrightarrow 2$`
//! scattering processes, there is a double counting issue that arises from
//! having a real intermediate state (RIS) in `$2 \leftrightarrow 2$`
//! interactions.
//!
//! As a concrete example, one may have the processes `$ab \leftrightarrow X$`
//! and `$X \leftrightarrow cd$` and thus also `$ab \leftrightarrow cd$`.  In
//! computing the squared amplitude for the `$2 \leftrightarrow 2$` process, one
//! needs to subtract the RIS:
//!
//! ```math
//! \abs{\mathcal M(ab \leftrightarrow cd)}^2 = \abs{\mathcal M_\text{full}(ab \leftrightarrow cd)}^2 - \abs{\mathcal M_\textsc{RIS}(ab \leftrightarrow cd)}^2
//! ```
//!
//! In the case of a single scalar RIS, the RIS-subtracted amplitude is given by
//!
//! ```math
//! \begin{aligned}
//!   \abs{\mathcal M_\textsc{RIS}(ab \to cd)}
//!   &= \frac{\pi}{m_X \Gamma_X} \delta(s - m_X^2) \theta(\sqrt{s}) \\
//!   &\quad \Big[
//!       \abs{\mathcal M(ab \to X)}^2 \abs{\mathcal M(X \to cd)}^2
//!     + \abs{\mathcal M(ab \to \overline X)}^2 \abs{\mathcal M(\overline X \to cd)}^2
//!   \Big].
//! \end{aligned}
//! ```
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
//! particular time.  We therefore make the change of variables from `$t$` to
//! `$\beta$` which introduces a factor of `$H(\beta) \beta$`:
//!
//! ```math
//! \pfrac{n}{t} + 3 H n \equiv H \beta \pfrac{n}{\beta} + 3 H n.
//! ```
//!
//! As a result, the actual change in the number density, it becomes
//!
//! ```math
//! \pfrac{n}{\beta} = \frac{1}{H \beta} \left[\vt C\[n\] - 3 H n\right]
//! ```
//!
//! and one must only input `$\vt C\[n\]$` in the interaction.
//!
//! # Normalized Number Density
//!
//! The number densities themselves can be often quite large (especially in the
//! early Universe), and often are compared to other number densities; as a
//! result, they are often normalized to either the photon number density
//! `$n_\gamma$` or the entropy density `$s$`.  This library uses a equilibrium
//! number density of a single massless bosonic degree of freedom (and thus
//! differs from using the photon number density by a factor of 2).
//!
//! When dealing with number densities, the Liouville operator is:
//!
//! ```math
//! \pfrac{n}{t} + 3 H n = \vt C\[n\]
//! ```
//!
//! where `$\vt C\[n\]$` is the change in the number density.  If we now define
//!
//! ```math
//! Y \defeq \frac{n}{n_{\text{eq}}},
//! ```
//!
//! then the change in this normalized number density is:
//!
//! ```math
//! \pfrac{Y}{t} = \frac{1}{n_{\text{eq}}} \vt C\[n\]
//! ```
//!
//! with `$n_{\text{eq}}$` having the simple analytic form `$3 \zeta(3) / 4
//! \pi^2 \beta^3$`.  Furthermore, make the change of variable from time `$t$`
//! to inverse temperature `$\beta$`, we get:
//!
//! ```math
//! \pfrac{Y}{\beta} = \frac{1}{H \beta n_{\text{eq}}} \vt C\[n\].
//! ```
//!
//! As with the non-normalized number density calculations, only `$\vt C\[n\]$`
//! must be inputted in the interaction.

mod context;
mod options;
mod solver_builder;
mod tableau;

pub use context::Context;
pub use solver_builder::SolverBuilder;

use crate::{
    model::{interaction::Interaction, Model, ModelInteractions, Particle},
    solver::options::StepPrecision,
    statistic::{Statistic, Statistics},
};
use ndarray::{prelude::*, Zip};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::{error, fmt, mem, ptr};

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

/// Workspace of functions to reuse during the integration
struct Workspace {
    /// Number density
    n: Array1<f64>,
    /// Number density change
    dn: Array1<f64>,
    /// Number density local error estimate
    dn_err: Array1<f64>,
    /// Number density asymmetry
    na: Array1<f64>,
    /// Number density asymmetry change
    dna: Array1<f64>,
    /// Number density asymmetry local error estimate
    dna_err: Array1<f64>,

    /// k array for the number density
    k: [Array1<f64>; tableau::RK_S],
    /// k array for the number density asymmetry
    ka: [Array1<f64>; tableau::RK_S],
}

/// Workspace of variables allocated once and then reused during the integration.
impl Workspace {
    fn new(initial_densities: &Array1<f64>, initial_asymmetries: &Array1<f64>) -> Self {
        let dim = initial_densities.dim();

        let mut k: [Array1<f64>; tableau::RK_S];
        let mut ka: [Array1<f64>; tableau::RK_S];
        unsafe {
            k = mem::MaybeUninit::uninit().assume_init();
            for ki in &mut k[..] {
                ptr::write(ki, Array1::zeros(dim));
            }
            ka = mem::MaybeUninit::uninit().assume_init();
            for ki in &mut ka[..] {
                ptr::write(ki, Array1::zeros(dim));
            }
        };

        Self {
            n: initial_densities.clone(),
            dn: Array1::zeros(dim),
            dn_err: Array1::zeros(dim),
            na: initial_asymmetries.clone(),
            dna: Array1::zeros(dim),
            dna_err: Array1::zeros(dim),

            k,
            ka,
        }
    }

    /// Clear the workspace for the next step, filling the changes to 0 and the
    /// error estimates to 0.
    fn clear_step(&mut self) {
        self.dn.fill(0.0);
        self.dn_err.fill(0.0);
        self.dna.fill(0.0);
        self.dna_err.fill(0.0);
    }

    /// Compute and update the changes to number density from k and a given step
    /// of the Runge-Kutta integration.
    fn compute_dn(&mut self, i: usize) {
        let bi = tableau::RK_B[i];
        let ei = tableau::RK_E[i];
        Zip::from(&mut self.dn)
            .and(&mut self.dn_err)
            .and(&self.k[i])
            .and(&mut self.dna)
            .and(&mut self.dna_err)
            .and(&self.ka[i])
            .apply(|dn, dn_err, &ki, dna, dna_err, &kai| {
                *dn += bi * ki;
                *dn_err += ei * ki;
                *dna += bi * kai;
                *dna_err += ei * kai;
            });
    }

    /// Get the local error using L∞-norm
    fn local_error(&self) -> f64 {
        self.dn_err
            .iter()
            .chain(self.dna_err.iter())
            .fold(0_f64, |e, v| e.max(v.abs()))
    }

    /// Advance the integration by apply the computed changes to far to the number densities.
    fn advance(&mut self) {
        self.n += &self.dn;
        self.na += &self.dna;
    }

    fn result(self) -> (Array1<f64>, Array1<f64>) {
        (self.n, self.na)
    }
}

/// Boltzmann solver
pub struct Solver<M> {
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

impl<M> Solver<M>
where
    M: ModelInteractions,
{
    #[cfg(not(feature = "parallel"))]
    fn compute_ki(&self, ki: &mut Array1<f64>, kai: &mut Array1<f64>, ci: &Context<M>) {
        ki.fill(0.0);
        kai.fill(0.0);
        for interaction in self.model.interactions() {
            interaction.change(ki, kai, &ci);
        }
    }

    #[allow(clippy::similar_names)]
    #[cfg(feature = "parallel")]
    fn compute_ki(&self, ki: &mut Array1<f64>, kai: &mut Array1<f64>, ci: &Context<M>) {
        let dim = ki.dim();
        ki.fill(0.0);
        kai.fill(0.0);

        let (new_ki, new_kai) = self
            .model
            .interactions()
            .par_iter()
            .fold(
                || (Array1::zeros(dim), Array1::zeros(dim)),
                |(mut dn, mut dna), interaction| {
                    interaction.change(&mut dn, &mut dna, &ci);
                    (dn, dna)
                },
            )
            .reduce(
                || (Array1::zeros(dim), Array1::zeros(dim)),
                |(dn, dna), (dni, dnai)| (dn + dni, dna + dnai),
            );

        *ki = new_ki;
        *kai = new_kai;
    }

    // Adjust dn and/or dna for those particles held in equilibrium and/or
    // without asymmetry.
    fn fix_equilibrium(&self, c: &Context<M>, workspace: &mut Workspace) {
        for &eq in &self.in_equilibrium {
            workspace.dn[eq] = c.eq[eq] - workspace.n[eq];
            workspace.dna[eq] = -workspace.na[eq];
        }
        for &eq in &self.no_asymmetry {
            workspace.dna[eq] = -workspace.na[eq];
        }
    }

    /// Evolve the initial conditions by solving the PDEs.
    ///
    /// Note that this is not a true PDE solver, but instead converts the PDE
    /// into an ODE in time (or inverse temperature in this case), with the
    /// derivative in energy being calculated solely from the previous
    /// time-step.
    pub fn solve(&mut self) -> (Array1<f64>, Array1<f64>) {
        use tableau::{RK_A, RK_C, RK_ORDER, RK_S};

        // Initialize all the variables that will be used in the integration
        let mut workspace = Workspace::new(&self.initial_densities, &self.initial_asymmetries);

        let mut step = 0;
        let mut steps_discarded = 0_u64;
        let mut evals = 0_u64;
        let mut beta = self.beta_range.0;
        let mut h = beta * f64::sqrt(self.step_precision.min * self.step_precision.max);
        let mut advance;

        // Run logger for 0th step
        {
            self.model.set_beta(beta);
            let c = self.context(step, h, beta, &workspace.n, &workspace.na);
            (*self.logger)(&c);
        }

        while beta < self.beta_range.1 {
            step += 1;
            advance = false;
            workspace.clear_step();

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
            } else if step % 10 == 0 {
                log::debug!("Step {}, β = {:.4e}", step, beta);
            } else {
                log::trace!("Step {}, β = {:.4e}, h = {:.4e}", step, beta, h);
            }
            log::trace!("      n = {:<+10.3e}", workspace.n);
            log::trace!("     na = {:<+10.3e}", workspace.na);

            for i in 0..RK_S {
                // DEBUG
                // log::trace!("i = {}", i);
                evals += 1;

                // Compute the sub-step values
                let beta_i = beta + RK_C[i] * h;
                let ai = RK_A[i];
                let ni = (0..i).fold(workspace.n.clone(), |total, j| {
                    total + ai[j] * &workspace.k[j]
                });
                let nai = (0..i).fold(workspace.na.clone(), |total, j| {
                    total + ai[j] * &workspace.ka[j]
                });
                self.model.set_beta(beta_i);
                let ci = self.context(step, h, beta_i, &ni, &nai);

                // Compute k[i] and ka[i] from each interaction
                self.compute_ki(&mut workspace.k[i], &mut workspace.ka[i], &ci);

                // Set changes to zero for those particles in equilibrium
                for &eq in &self.in_equilibrium {
                    workspace.k[i][eq] = 0.0;
                    workspace.ka[i][eq] = 0.0;
                }
                for &eq in &self.no_asymmetry {
                    workspace.ka[i][eq] = 0.0;
                }

                workspace.compute_dn(i)
            }

            self.model.set_beta(beta);
            let c = self.context(step, h, beta, &workspace.n, &workspace.na);

            self.fix_equilibrium(&c, &mut workspace);

            let err = workspace.local_error();

            // If the error is within the tolerance, we'll be advancing the
            // iteration step
            if err < self.error_tolerance {
                advance = true;
                log::trace!(" dn = {:<+10.3e}", workspace.dn);
                log::trace!("dna = {:<+10.3e}", workspace.dna);
            } else if log::log_enabled!(log::Level::Trace) {
                log::trace!(
                    "Error is not within tolerance ({:e} > {:e}).",
                    err,
                    self.error_tolerance
                );
                log::trace!(" n_err = {:<+10.3e}", workspace.dn_err);
                log::trace!("na_err = {:<+10.3e}", workspace.dna_err);
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
                workspace.advance();
                beta += h;

                (*self.logger)(&c);
            } else {
                steps_discarded += 1;
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
        log::info!("Number of integration steps discarded: {}", steps_discarded);
        log::info!("Number of evaluations: {}", evals);

        workspace.result()
    }

    /// Compute the interaction rates.
    ///
    /// The number of logarithmic steps in beta is specified by `n`, with the
    /// range of `$\beta$` values being taken from the solver.  If `normalize`
    /// is true, the interaction rate is divided by `$n_1 H \beta$`, where
    /// `$n_1$` is the equilibrium number density of a single bosonic degree of
    /// freedom, `$H$` is the Hubble rate and `$\beta$` is the inverse
    /// temperature.
    ///
    /// The interactions rates are returned as two dimensional array with the
    /// first index indexing values of beta, and the second index corresponding
    /// to the index of the interaction, as returned by [`interactions`].  The
    /// second index is offset by one with the first index being for beta
    /// itself.
    ///
    /// The entries of the returned array as `Option<f64>` in order to
    /// distinguish cases where the rate is not computed due to being unphysical
    /// from cases where it is 0.
    ///
    pub fn gammas(&mut self, size: usize, normalize: bool) -> (Vec<String>, Array2<Option<f64>>) {
        let mut gammas = Array2::from_elem((size, self.model.interactions().len() + 1), None);
        let n = Array1::zeros(self.model.particles().len());
        let na = Array1::zeros(n.dim());

        for (i, &beta) in Array1::geomspace(self.beta_range.0, self.beta_range.1, size)
            .unwrap()
            .into_iter()
            .enumerate()
        {
            self.model.set_beta(beta);
            gammas[[i, 0]] = Some(beta);
            let mut c = self.context(0, 1.0, beta, &n, &na);
            c.n = c.eq.clone();
            let normalization = if normalize { c.normalization } else { 1.0 };

            #[cfg(not(feature = "parallel"))]
            let values: Vec<_> = self
                .model
                .interactions()
                .iter()
                .enumerate()
                .map(|(j, interaction)| (j, interaction.gamma(&c, true).map(|v| v * normalization)))
                .collect();

            #[cfg(feature = "parallel")]
            let values: Vec<_> = self
                .model
                .interactions()
                .par_iter()
                .enumerate()
                .map(|(j, interaction)| (j, interaction.gamma(&c, true).map(|v| v * normalization)))
                .collect();

            for (j, v) in values {
                gammas[[i, j + 1]] = v;
            }
        }

        let mut names = vec!["beta".to_string()];
        names.extend(self.model.interactions().iter().map(|interaction| {
            let ptcls = interaction.particles();
            ptcls
                .display(&self.model)
                .unwrap_or_else(|_| format!("{}", ptcls))
        }));

        (names, gammas)
    }

    /// Compute the asymmetric interaction rates.
    ///
    /// The arguments and returned values are identical to [`gammas`].
    pub fn asymmetries(
        &mut self,
        size: usize,
        normalize: bool,
    ) -> (Vec<String>, Array2<Option<f64>>) {
        let mut gammas = Array2::from_elem((size, self.model.interactions().len() + 1), None);
        let n = Array1::zeros(self.model.particles().len());
        let na = Array1::zeros(n.dim());

        for (i, &beta) in Array1::geomspace(self.beta_range.0, self.beta_range.1, size)
            .unwrap()
            .into_iter()
            .enumerate()
        {
            self.model.set_beta(beta);
            gammas[[i, 0]] = Some(beta);
            let mut c = self.context(0, 1.0, beta, &n, &na);
            c.n = c.eq.clone();
            let normalization = if normalize { c.normalization } else { 1.0 };

            #[cfg(not(feature = "parallel"))]
            let values: Vec<_> = self
                .model
                .interactions()
                .iter()
                .enumerate()
                .map(|(j, interaction)| {
                    (
                        j,
                        interaction.asymmetry(&c, true).map(|v| v * normalization),
                    )
                })
                .collect();

            #[cfg(feature = "parallel")]
            let values: Vec<_> = self
                .model
                .interactions()
                .par_iter()
                .enumerate()
                .map(|(j, interaction)| {
                    (
                        j,
                        interaction.asymmetry(&c, true).map(|v| v * normalization),
                    )
                })
                .collect();

            for (j, v) in values {
                gammas[[i, j + 1]] = v;
            }
        }

        let mut names = vec!["beta".to_string()];
        names.extend(self.model.interactions().iter().map(|interaction| {
            let ptcls = interaction.particles();
            ptcls
                .display(&self.model)
                .unwrap_or_else(|_| format!("{}", ptcls))
        }));

        (names, gammas)
    }
}

impl<M> Solver<M>
where
    M: Model,
{
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
    particles
        .into_iter()
        .map(|p| p.normalized_number_density(0.0, beta))
        .collect()
}
