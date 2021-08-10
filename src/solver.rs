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
//!   f_{\text{BE}} = \frac{1}{\exp[(E - \mu) \beta] - 1}, \qquad
//!   f_{\text{FD}} = \frac{1}{\exp[(E - \mu) \beta] + 1}.
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
//!   f_{\text{MB}} = \exp[-(E - \mu) \beta].
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
//!   f_{\text{MB}} = e^{\mu \beta} f_{\text{MB}}^{(0)} = \frac{n}{n^{(0)}} f_{\text{MB}}^{(0)}.
//!   ```
//!
//!   Furthermore, the assumption that `$\beta \gg E - \mu$` implies that
//!   `$f_{\text{BE}}, f_{\text{FD}} \ll 1$`.  Consequently, the Pauli
//!   suppression and Bose enhancement factors in the collision term can all be
//!   neglected resulting in:
//!
//!   ```math
//!   \begin{aligned}
//!     g_{a_1} \int \vt C[f_{a_{1}}] \frac{\dd \vt p_{a_1}}{(2\pi)^3}
//!       &= - \int_{\vt a}^{\vt b}
//!          \abs{\scM(\vt a \to \vt b)}^2 \left( \prod_{i \in \vt a} f_i \right)
//!          - \abs{\scM(\vt b \to \vt a)}^2 \left( \prod_{i \in \vt b} f_i \right) \\\\
//!       &= - \int_{\vt a}^{\vt b}
//!          \abs{\scM(\vt a \to \vt b)}^2 \left( \prod_{i \in \vt a} \frac{n_i}{n_i^{(0)}} f_i^{(0)} \right)
//!           - \abs{\scM(\vt b \to \vt a)}^2 \left( \prod_{i \in \vt b} \frac{n_i}{n_i^{(0)}} f_i^{(0)} \right).
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
//!       \abs{\scM(\vt a \to \vt b)}^2 \left( \prod_{i \in \vt a} f^{(0)}_i \right).
//!   ```
//!
//! # Interactions
//!
//! The majority of interactions can be accounted for by looking at simply `$1
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
//!   g_a \int \vt C[f_a] \frac{\dd \vt p_a}{(2\pi)^3}
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
//!     &= - \abs{\scM(a \to bc)}^2 \int_{a}^{b,c} f^{(0)}_a \\\\
//!     &= - \frac{\abs{\scM(a \to bc)}^2}{16 \pi} \frac{m_a K_1(m_a \beta)}{2 \pi^2 \beta}
//! \end{aligned}
//! ```
//!
//! Combined with the number density scaling:
//!
//! ```math
//! \frac{n_a}{n^{(0)}_a} \gamma(a \to bc)
//!   = n_a \frac{\abs{\scM(a \to bc)}^2}{16 \pi m_a} \frac{K_1(m_a \beta)}{K_2(m_a \beta)}
//! ```
//!
//! where the analytic expression for `$n^{(0)}_a$` was used to introduce the
//! second Bessel function.
//!
//! The ratio of Bessel functions is generally referred to as the *time-dilation
//! factor* for the decaying particle.  When `$m \beta \gg 1$`, the ratio of
//! Bessel functions approaches 1, and when `$m \beta \ll 1$`, the ratio
//! approaches 0.
//!
//! If the final state particles are essentially massless in comparison to the
//! decaying particle, then the decay rate in the rest frame of the particle of
//! `$\Gamma_\text{rest} = \abs{\scM}^2 / 16 \pi m_a$` and the above expression
//! can be simplified to
//!
//! ```math
//! \frac{n_a}{n^{(0)}_a} \gamma(a \to bc)
//! = n_a \Gamma_\text{rest} \frac{K_1(m_a \beta)}{K_2(m_a \beta)}.
//! ```
//!
//! ### Inverse Decay
//!
//! The inverse decay rate is given by
//!
//! ```math
//! \gamma(bc \to a)
//!   = \abs{\scM(bc \to a)}^2 \int_{a}^{b,c} f^{(0)}_b f^{(0)}_c
//! ```
//!
//! The Dirac delta enforces that `$E_a = E_b + E_c$` which implies that
//! `$f^{(0)}_a = f^{(0)}_b f^{(0)}_c$` thereby making the integral identical to
//! the decay scenario:
//!
//! ```math
//! \gamma(bc \to a)
//!   = \frac{\abs{\scM(bc \to a)}^2}{16 \pi} \frac{m_a K_1(m_a \beta)}{2 \pi^2 \beta}
//!   = n^{(0)}_a \frac{\abs{\scM(bc \to a)}^2}{16 \pi m_a} \frac{K_1(m_a \beta)}{K_2(m_a \beta)}
//! ```
//!
//! The full expression including the number density scaling becomes:
//!
//! ```math
//! \frac{n_b n_c}{n^{(0)}_b n^{(0)}_c} \gamma(bc \to a)
//!   = \frac{n_b n_c}{n^{(0)}_b n^{(0)}_c} n^{(0)}_a \frac{\abs{\scM(bc \to a)}^2}{16 \pi m_a} \frac{K_1(m_a \beta)}{K_2(m_a \beta)}
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
//! \tilde \gamma(a \to bc) = \frac{\abs{\scM(a \to bc)}^2}{16 \pi m_a} \frac{K_1(m_a \beta)}{K_2(m_a \beta)},
//! ```
//!
//! which allows for the overall `$1 \leftrightarrow 2$` reaction rate to be
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
//! \gamma(ab \to cd) = \int_{a,b}^{c,d} \abs{\scM(ab \to cd)}^2 f_a^{(0)} f_b^{(0)}
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
//! \hat \sigma_{ab}^{cd}(s) = \frac{g_a g_b g_c g_d}{64 \pi^2 s} \int \abs{\scM(ab \to cd)}^2 \dd t
//! ```
//!
//! in which `$t$` is the usual Mandelstam variable.
//!
//! As a result, the full `$2 \to 2$` cross-section can be expressed as
//!
//! ```math
//! \gamma(ab \to cd) = \frac{g_a g_b g_c g_d}{512 \pi^5 \beta} \int \abs{\scM(ab \to cd)}^2 \frac{K_1(\sqrt s \beta)}{\sqrt s} \dd s \dd t
//! ```
//!
//! ### Real Intermediate State
//!
//! When considering both `$1 \leftrightarrow 2$` and `$2 \leftrightarrow 2$`
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
//! \abs{\scM(ab \leftrightarrow cd)}^2 = \abs{\scM_\text{full}(ab \leftrightarrow cd)}^2
//! - \abs{\scM_\text{RIS}(ab \leftrightarrow cd)}^2
//! ```
//!
//! In the case of a single scalar RIS, the RIS-subtracted amplitude is given by
//!
//! ```math
//! \begin{aligned}
//!   \abs{\scM_\text{RIS}(ab \to cd)}
//!   &= \frac{\pi}{m_X \Gamma_X} \delta(s - m_X^2) \theta(\sqrt{s}) \\
//!   &\quad \Big[
//!       \abs{\scM(ab \to X)}^2 \abs{\scM(X \to cd)}^2
//!     + \abs{\scM(ab \to \overline X)}^2 \abs{\scM(\overline X \to cd)}^2
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
//! \pfrac{n}{\beta} = \frac{1}{H \beta} \left[\vt C[n] - 3 H n\right]
//! ```
//!
//! and one must only input `$\vt C[n]$` in the interaction.
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
//! \pfrac{n}{t} + 3 H n = \vt C[n]
//! ```
//!
//! where `$\vt C[n]$` is the change in the number density.  If we now define
//!
//! ```math
//! Y \defeq \frac{n}{n_{\text{eq}}},
//! ```
//!
//! then the change in this normalized number density is:
//!
//! ```math
//! \pfrac{Y}{t} = \frac{1}{n_{\text{eq}}} \vt C[n]
//! ```
//!
//! with `$n_{\text{eq}}$` having the simple analytic form `$3 \zeta(3) / 4
//! \pi^2 \beta^3$`.  Furthermore, make the change of variable from time `$t$`
//! to inverse temperature `$\beta$`, we get:
//!
//! ```math
//! \pfrac{Y}{\beta} = \frac{1}{H \beta n_{\text{eq}}} \vt C[n].
//! ```
//!
//! As with the non-normalized number density calculations, only `$\vt C[n]$`
//! must be inputted in the interaction.

mod builder;
mod context;
mod options;
mod tableau;

pub use builder::SolverBuilder;
pub use context::Context;

use crate::{
    model::interaction::InteractionParticles,
    model::{
        interaction::FastInteractionResult, interaction::Interaction, Model, ModelInteractions,
        Particle,
    },
    solver::options::StepPrecision,
    statistic::{Statistic, Statistics},
};
use ndarray::{prelude::*, Zip};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::{
    collections::HashSet, convert::TryFrom, error, fmt, io::Write, mem, ops, ptr, sync::RwLock,
};

/// Error type returned by the solver in case there is an error during
/// integration.
///
/// The underlying number density and number density asymmetry arrays can be
/// obtained with [`Error::into_inner`].
#[derive(Debug)]
pub enum Error {
    /// Result is inaccurate.
    ///
    /// If [`SolverBuilder::abort_when_inaccurate`] was set to true, the number
    /// densities returned
    Inaccurate((Array1<f64>, Array1<f64>)),
    /// One of more number density was NAN.
    NAN((Array1<f64>, Array1<f64>)),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Inaccurate(_) => write!(f, "Result my be inaccurate"),
            Error::NAN(_) => write!(f, "NAN number density encountered"),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}

impl Error {
    /// Unwrap the error and return the underlying number density and number
    /// density asymmetry arrays respectively.
    #[must_use]
    pub fn into_inner(self) -> (Array1<f64>, Array1<f64>) {
        match self {
            Error::Inaccurate(v) | Error::NAN(v) => v,
        }
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

        #[allow(clippy::uninit_assumed_init)]
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
            .for_each(|dn, dn_err, &ki, dna, dna_err, &kai| {
                *dn += bi * ki;
                *dn_err += ei * ki;
                *dna += bi * kai;
                *dna_err += ei * kai;
            });
    }

    /// Get the local error using L∞-norm.  Any NaN values in the changes result
    /// in the local error also being NaN.
    fn local_error(&self) -> f64 {
        self.dn_err
            .iter()
            .chain(&self.dna_err)
            // .zip(self.dn.iter().chain(&self.dna))
            .fold(0_f64, |result, &error| {
                if result.is_nan() || error.is_nan() {
                    f64::NAN
                } else {
                    result.max((error).abs())
                }
            })
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

impl ops::AddAssign<&FastInteractionResult> for Workspace {
    fn add_assign(&mut self, rhs: &FastInteractionResult) {
        self.dn += &rhs.dn;
        self.dna += &rhs.dna;
    }
}

type LoggerFn<M> = Box<dyn Fn(&Context<M>, &Array1<f64>, &Array1<f64>)>;

/// Boltzmann solver
pub struct Solver<M> {
    model: M,
    particles_next: Vec<Particle>,
    initial_densities: Array1<f64>,
    initial_asymmetries: Array1<f64>,
    beta_range: (f64, f64),
    /// The indices of particles forced in equilibrium (irrespective of
    /// interactions) **must** be ordered.
    in_equilibrium: Vec<usize>,
    /// The indices of particles forced to have no asymmetry (irrespective of
    /// interactions) **must** be ordered.
    no_asymmetry: Vec<usize>,
    logger: LoggerFn<M>,
    step_precision: StepPrecision,
    error_tolerance: f64,
    fast_interactions: bool,
    inaccurate: bool,
    abort_when_inaccurate: bool,
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
            interaction.change(ki, kai, ci);
        }
    }

    #[allow(clippy::similar_names)]
    #[cfg(feature = "parallel")]
    fn compute_ki(&self, ki: &mut Array1<f64>, kai: &mut Array1<f64>, ci: &Context<M>) {
        let dim = ki.dim();

        let (new_ki, new_kai) = self
            .model
            .interactions()
            .par_iter()
            .fold(
                || (Array1::zeros(dim), Array1::zeros(dim)),
                |(mut dn, mut dna), interaction| {
                    interaction.change(&mut dn, &mut dna, ci);
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

    /// Adjust `dn` and/or `dna` for to account for:
    ///
    /// - Particle which are held in equilibrium, irrespective if any
    ///   interaction going on.
    /// - Particles which have no asymmetry, irrespective if any interaction
    ///   going on.
    /// - Handle and equilibriate fast interaction (if applicable).
    #[allow(clippy::similar_names)]
    fn fix_change(&self, c: &Context<M>, dn: &mut Array1<f64>, dna: &mut Array1<f64>) {
        // For the fast interaction to converge, we need local mutable copies of
        // `n` and `na` which already incorporate the changes computed from the
        // regular integration.
        let (mut n, mut na) = (&c.n + &*dn, &c.na + &*dna);

        let mut iterations = 0_usize;
        loop {
            // Go through all fast interactions and apply the change to the
            // number densities.  We also store the deltas applied to determine
            // convergence.
            let deltas = c
                .fast_interactions
                .as_ref()
                .map_or_else(Vec::new, |fast_interactions| {
                    fast_interactions
                        .read()
                        .unwrap()
                        .iter()
                        .map(|interaction| {
                            // Fix equilibrium before each fast interaction
                            for &p in &self.in_equilibrium {
                                n[p] = c.eqn[p];
                            }
                            for &p in &self.no_asymmetry {
                                na[p] = 0.0;
                            }

                            let result = interaction.fast_interaction(
                                &n,
                                &na,
                                &c.eqn,
                                c.in_equilibrium,
                                c.no_asymmetry,
                            );

                            n += &result.dn;
                            na += &result.dna;

                            (result.symmetric_delta, result.asymmetric_delta)
                        })
                        .collect()
                });

            log::trace!("Iteration {}:\nδ = {:?}", iterations, deltas);

            // Fix equilibrium one last time
            for &p in &self.in_equilibrium {
                n[p] = c.eqn[p];
            }
            for &p in &self.no_asymmetry {
                na[p] = 0.0;
            }

            let equilibrium = deltas.is_empty()
                || deltas.iter().all(|(symmetric_delta, asymmetric_delta)| {
                    symmetric_delta.abs() < 1e-8 && asymmetric_delta.abs() < 1e-12
                });
            if equilibrium {
                break;
            }

            iterations += 1;
            if iterations >= 50 {
                log::error!(
                    "[{}.{:02}|{:>9.3e}] Unable to converge fast interactions after {} iterations.",
                    c.step,
                    c.substep,
                    c.beta,
                    iterations
                );
                panic!();
            }
        }

        *dn = n - &c.n;
        *dna = na - &c.na;
    }

    /// Compute the adjustment to the step size based on the local error.
    ///
    /// Given a local error estimate of `$\tilde\varepsilon$` and target local
    /// error `$\varepsilon$`, the new step size should be:
    ///
    /// ```math
    /// h_{\text{new}} = h \times \underbrace{S \sqrt[p + 1]{\frac{\varepsilon}{\tilde \varepsilon}}}_{\Delta}
    /// ```
    ///
    /// where `$S \in [0, 1]$` is a safety factor to purposefully underestimate
    /// the result.  The values of `\Delta` are bounded so as to avoid too
    /// step size adjustments.
    fn delta(&self, err: f64) -> f64 {
        use tableau::RK_ORDER;

        let delta = 0.8 * (self.error_tolerance / err).powf(1.0 / f64::from(RK_ORDER + 1));

        if delta.is_nan() {
            0.1
        } else {
            delta.clamp(0.1, 2.0)
        }
    }

    /// Ensure that h is within the desired range of step sizes.
    ///
    /// This function returns a [`Result`] with:
    ///
    /// - `Ok(h)` indicating that the step size was either unchanged or made
    ///   smaller and integration can proceed with no worry about the error
    ///   estimate.
    /// - `Err(h)` indicating that the step size was increased.  The local error
    ///   estimate requires a smaller step size than the minimal value allowed
    ///   and thus the result of the integration may be inaccurate.
    fn adjust_h(&mut self, step: usize, mut h: f64, beta: f64) -> Result<f64, f64> {
        let h_on_beta = h / beta;

        if h_on_beta > self.step_precision.max {
            h = beta * self.step_precision.max;
            log::trace!(
                "[{}|{:.3e}] Step size too large, decreased h to {:.3e}",
                step,
                beta,
                h
            );

            Ok(h)
        } else if h_on_beta < self.step_precision.min {
            h = beta * self.step_precision.min;
            log::debug!(
                "[{}|{:.3e}] Step size too small, increased h to {:.3e}",
                step,
                beta,
                h
            );

            // Give a warning (if not done already).
            if !self.inaccurate {
                log::warn!(
                    "[{}|{:.3e}] Step size too small at β.  Result may not be accurate",
                    step,
                    beta,
                );
                self.inaccurate = true;
            }

            Err(h)

            // // Stop integration or force going ahead if the local error is too large.
            // if self.abort_when_inaccurate {
            //     log::warn!(
            //         "[{}|{:.3e}] Aborting integration due to inaccuracy.",
            //         step,
            //         beta,
            //     );
            //     break;
            // }
            // advance = true;
        } else {
            Ok(h)
        }
    }

    /// Evolve the initial conditions by solving the PDEs.
    ///
    /// Note that this is not a true PDE solver, but instead converts the PDE
    /// into an ODE in time (or inverse temperature in this case), with the
    /// derivative in energy being calculated solely from the previous
    /// time-step.
    ///
    /// # Errors
    ///
    /// If the time evolution encounters an error, the function returns an
    /// error.  The final number density (despite the error) can still be
    /// obtained by using [`Error::into_inner`], though this may not be at the
    /// expected final temperature as some errors abort the integration.
    #[allow(clippy::too_many_lines)]
    pub fn solve(&mut self) -> Result<(Array1<f64>, Array1<f64>), Error> {
        use tableau::{RK_A, RK_C, RK_S};

        const BETA_LOG_STEP: f64 = 1.0;

        // Initialize all the variables that will be used in the integration
        let mut workspace = Workspace::new(&self.initial_densities, &self.initial_asymmetries);

        let mut step = 0_usize;
        let mut steps_discarded = 0_usize;
        let mut evals = 0_usize;
        let mut beta = self.beta_range.0;
        let mut h = beta
            * f64::sqrt(
                self.step_precision.min.max(self.step_precision.max / 1e6)
                    * self.step_precision.max,
            );
        let mut beta_logging = beta * BETA_LOG_STEP;

        // Run logger for 0th step
        {
            self.model.set_beta(beta);
            let c = self.context(
                step,
                None,
                h,
                (beta, beta + h),
                &workspace.n,
                &workspace.na,
                None,
            );
            (*self.logger)(&c, &workspace.dn, &workspace.dna);
        }

        while beta < self.beta_range.1 {
            step += 1;
            workspace.clear_step();

            // Compute next step particles
            self.model.set_beta(beta + h);
            self.particles_next = self.model.particles().to_vec();

            // Log the progress of the integration
            if step % 100 == 0 {
                log::info!("Step {}, β = {:.4e}, h = {:.4e}", step, beta, h);
            } else if step % 10 == 0 {
                log::debug!("Step {}, β = {:.4e}, h = {:.4e}", step, beta, h);
            } else {
                log::trace!("Step {}, β = {:.4e}, h = {:.4e}", step, beta, h);
            }
            // log::trace!("[{}|{:.3e}]       n = {:<+10.3e}", step, beta, workspace.n);
            // log::trace!("[{}|{:.3e}]      na = {:<+10.3e}", step, beta, workspace.na);

            // Initialize the fast interactions if they are being used
            let mut fast_interactions = if self.fast_interactions {
                Some(RwLock::new(HashSet::new()))
            } else {
                None
            };

            for i in 0..RK_S {
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
                let ci = self.context(
                    step,
                    Some(i),
                    h,
                    (beta_i, beta + h),
                    &ni,
                    &nai,
                    fast_interactions,
                );

                // Compute k[i] and ka[i] from each interaction
                self.compute_ki(&mut workspace.k[i], &mut workspace.ka[i], &ci);

                fast_interactions = ci.into_fast_interactions();
                workspace.compute_dn(i);
            }

            self.model.set_beta(beta);
            let c = self.context(
                step,
                None,
                h,
                (beta, beta + h),
                &workspace.n,
                &workspace.na,
                fast_interactions,
            );
            // log::trace!("[{}|{:.3e}]      eq = {:<+10.3e}", step, beta, c.eq);
            self.fix_change(&c, &mut workspace.dn, &mut workspace.dna);

            #[allow(clippy::cast_precision_loss)]
            let err = workspace.local_error() / self.model.interactions().len() as f64;

            // If the error is within the tolerance, we'll be advancing the
            // iteration step
            if err < self.error_tolerance {
                if beta > beta_logging || beta >= self.beta_range.1 {
                    (*self.logger)(&c, &workspace.dn, &workspace.dna);
                    beta_logging = beta * BETA_LOG_STEP;
                }

                workspace.advance();
                beta += h;
            } else if log::log_enabled!(log::Level::Trace) {
                log::trace!(
                    "[{}|{:.3e}] Error is not within tolerance ({:e} > {:e}).",
                    step,
                    beta,
                    err,
                    self.error_tolerance
                );
                steps_discarded += 1;
            } else {
                steps_discarded += 1;
            }

            // Adjust the step size based on the error
            h *= self.delta(err);
            h = match self.adjust_h(step, h, beta) {
                Ok(h) => h,
                Err(h) => {
                    if self.abort_when_inaccurate {
                        return Err(Error::Inaccurate(workspace.result()));
                    }

                    workspace.advance();
                    beta += h;

                    h
                }
            };

            // Adjust final integration step if needed
            if beta + h > self.beta_range.1 {
                log::trace!(
                    "[{}|{:.3e}] Fixing overshoot of last integration step.",
                    step,
                    beta
                );
                h = self.beta_range.1 - beta;
            }
        }

        log::info!("Number of integration steps: {}", step);
        log::info!("Number of integration steps discarded: {}", steps_discarded);
        log::info!("Number of evaluations: {}", evals);

        if self.inaccurate {
            Err(Error::Inaccurate(workspace.result()))
        } else {
            Ok(workspace.result())
        }
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
    /// The interactions rates are returned as two dimensional array with
    /// `data[[i, j]]` giving the interaction rate for the `$j$`th interaction
    /// at the `$i$`th `$\beta$` step.  The interaction index corresponds to the
    /// index of the interaction as returned by
    /// [`ModelInteractions::interactions`], offset by 1 such that the 0th value
    /// is for `$\beta$` itself.
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
            .unwrap_or(ndarray::array![])
            .iter()
            .enumerate()
        {
            if log::log_enabled!(log::Level::Info) {
                print!("Step {:>5} / {}\r", i, size);
                std::io::stdout().flush().unwrap_or(());
            }

            self.model.set_beta(beta);
            gammas[[i, 0]] = Some(beta);
            let mut c = self.context(0, None, 1.0, (beta, beta), &n, &na, None);
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
    /// The arguments and returned values are identical to [`Solver::gammas`].
    pub fn asymmetries(
        &mut self,
        size: usize,
        normalize: bool,
    ) -> (Vec<String>, Array2<Option<f64>>) {
        let mut gammas = Array2::from_elem((size, self.model.interactions().len() + 1), None);
        let n = Array1::zeros(self.model.particles().len());
        let na = Array1::zeros(n.dim());

        for (i, &beta) in Array1::geomspace(self.beta_range.0, self.beta_range.1, size)
            .unwrap_or_else(|| Array1::zeros(10))
            .iter()
            .enumerate()
        {
            self.model.set_beta(beta);
            gammas[[i, 0]] = Some(beta);
            let mut c = self.context(0, None, 1.0, (beta, beta), &n, &na, None);
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
                        interaction.delta_gamma(&c, true).map(|v| v * normalization),
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
                        interaction.delta_gamma(&c, true).map(|v| v * normalization),
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
    #[allow(clippy::too_many_arguments)]
    fn context(
        &self,
        step: usize,
        substep: Option<usize>,
        step_size: f64,
        (beta, next_beta): (f64, f64),
        n: &Array1<f64>,
        na: &Array1<f64>,
        fast_interactions: Option<RwLock<HashSet<InteractionParticles>>>,
    ) -> Context<M> {
        let hubble_rate = self.model.hubble_rate(beta);
        let normalization =
            (hubble_rate * beta * Statistic::BoseEinstein.number_density(0.0, 0.0, beta)).recip();
        Context {
            step,
            substep: substep.map_or(-1, |v| TryFrom::try_from(v).unwrap()),
            step_size,
            beta,
            hubble_rate,
            normalization,
            eq: equilibrium_number_densities(self.model.particles(), beta),
            eqn: equilibrium_number_densities(&self.particles_next, next_beta),
            n: n.clone(),
            na: na.clone(),
            model: &self.model,
            fast_interactions,
            in_equilibrium: &self.in_equilibrium,
            no_asymmetry: &self.no_asymmetry,
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
