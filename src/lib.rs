//! `boltzmann-solver` is a library allowing for Boltzmann equation in the
//! context of particle physics / early cosmology.  It provides functionalities
//! to solve Boltzmann equation in the case where a single species is out of
//! equilibrium, as well as functionalities to solve the Boltzmann equations
//! more general when multiple species are all out of equilibrium.
//!
//! **This library is still undergoing development.**
//!
//! [![Crates.io](https://img.shields.io/crates/v/hep-boltzmann-solver.svg)](https://crates.io/crates/hep-boltzmann-solver)
//! [![Travis](https://img.shields.io/travis/hep-rs/boltzmann-solver/master.svg)](https://travis-ci.org/hep-rs/boltzmann-solver)
//! [![Codecov](https://img.shields.io/codecov/c/github/hep-rs/boltzmann-solver/master.svg)](https://codecov.io/gh/hep-rs/boltzmann-solver)
//!
//! Licensed under [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html).
//!
//! The documentation makes use of [MathJax](https://www.mathjax.org/) in order
//! to display mathematics.  A version of the documentation with MathJax
//! integrated is available [here](https://hep.rs/boltzmann-solver/boltzmann_solver).
//!
//! # Introduction
//!
//! Within the context of cosmology, the Boltzmann equation is a differential
//! equation which describes the evolution of the phase-space distribution of a
//! species of particles.  In full generality, it is given by
//!
//! \\begin{equation}
//!     \vt L[f] = \vt C[f]
//! \\end{equation}
//!
//! where \\(\vt L\\) is the Liouville operator and \\(\vt C\\) is the collision
//! operator.
//!
//! ### Liouville Operator
//!
//! In general, the phase space distribution dependent on position, momentum and
//! time \\(f(\vt x, \vt p, t)\\); however, assuming the early Universe to be
//! isotropic and homogeneous it can be expressed as a function of just energy
//! and time, \\(f(E, t)\\).  In the FLRW metric, the Liouville operator
//! simplifies to only be dependent the time-like derivatives:
//!
//! \\begin{equation}
//!     \vt L[f] \stackrel{\mathrm{\tiny FLRW}}{=} \left[ E \pfrac{}{t} - H (E^2 - m^2) \pfrac{}{E} \right] f,
//! \\end{equation}
//!
//! where \\(H \defeq \dot a / a\\) is Hubble's constant.
//!
//! ### Collision Operator
//!
//! In the absence of any collisions, the Boltzmann equation ensures that the
//! phase space distribution is conserved.  More explicitly, integrating the
//! phase space, the Liouville operator becomes
//!
//! \\begin{equation}
//!     g \int \vt L[f] \frac{\dd^3 \vt p}{(2 \pi)^3 E} = \pfrac{n}{t} + 3 H n = \frac{1}{a^3} \pfrac{(n a^3)}{t}
//! \\end{equation}
//!
//! which ensures that the overall number of particles is conserved under the
//! scaling by the FLRW metric.  Note that the number density is related to the
//! phase space distribution through
//!
//! \\begin{equation}
//!     n \defeq g \int f \frac{\dd^3 \vt p}{(2 \pi)^3} = \frac{g}{2 \pi^2} \int_m^\infty f \sqrt{E^2 - m^2} E \dd E
//! \\end{equation}
//!
//! where \\(g\\) is the number of internal degrees of freedom of the particle.
//!
//! The collision term in the Boltzmann equation describes the changes to the
//! phase space distribution that arise from collisions—either through the
//! scattering of other particles, or creation/annihilation of the species in
//! question.  The integrated collision term for a particular process \\(a_1 +
//! a_2 + \cdots \leftrightarrow b_1 + b_2 + \cdots\\) (which we will denote as
//! \\(\vt a \leftrightarrow \vt b\\)) is:
//!
//! \\begin{equation}
//!   \begin{aligned}
//!     g_{a_1} \int \vt C[f_{a_{1}}] \frac{\dd^3 \vt p_{a_1}}{(2\pi)^3 E_{a_1}}
//!       &= - \int \left( \prod_{\vt a, \vt b} \dd \Pi_i \right)
//!                 (2 \pi)^4 \delta(p_{\vt a} - p_{\vt b}) \\\\
//!       &\quad \times \Bigl[ \abs{\mathcal M(\vt a | \vt b)}^2 \left( \prod_{\vt a} f_i \right) \left(\prod_{\vt b} 1 \pm f_i \right)
//!                          - \abs{\mathcal M(\vt b | \vt a)}^2 \left( \prod_{\vt b} f_i \right) \left(\prod_{\vt a} 1 \pm f_i \right) \Bigr],
//!   \end{aligned}
//! \\end{equation}
//!
//! where \\(p_{\vt a} \defeq p_{a_1} + p_{a_2} + \cdots\\) (and similarly for
//! \\(p_{\vt b}\\)); \\(\abs{\mathcal M(\vt a | \vt b)}^2\\) is the squared
//! amplitude going from initial state \\(\vt a\\) to \\(\vt b\\) and is *summed
//! over* all *internal degrees of freedom*; and
//!
//! \\begin{equation}
//!   \dd \Pi_i \defeq \frac{\dd^3 \vt p_i}{(2 \pi)^3 E_i}
//!             \equiv E_i \sqrt{E_i^2 - m_i^2} \frac{\dd E_i \dd \Omega_i}{(2 \pi)^3}.
//! \\end{equation}
//!
//! The factors \\(1 \pm f\\) account for Pauli suppression and Bose enhancement
//! in the transition \\(\vt a \leftrightarrow \vt b\\), where \\(1 + f\\) is
//! used for Bose–Einstein statistics, and \\(1 - f\\) is used for Fermi–Dirac
//! statistics.
//!
//! The above collision term accounts for a single interaction and in general a
//! sum over all possible interactions must be done.
//!
//! ## Simplifying Assumptions
//!
//! Solving the Boltzmann equation in generality without making any assumption
//! about the phase space distribution \\(f\\) (other than specifying some
//! initial condition) is a difficult problem.  A couple of assumptions can be
//! made to simplify the Boltzmann equation above.
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
//!   Fermi–Dirac and Bose–Einstein approach the Maxwell–Boltzmann distribution:
//!
//!   \\begin{equation}
//!     f_{\textsc{MB}} = \exp[-(E - \mu) \beta]
//!   \\end{equation}
//!
//!   Furthermore, the assumption that \\(\beta \gg E - \mu\\) implies that
//!   \\(f_{\textsc{BE}}, f_{\textsc{FD}} \ll 1\\).  Consequently, the Pauli
//!   suppression and Bose enhancement factors in the collision term can all be
//!   neglected resulting in:
//!
//!   \\begin{equation}
//!     \begin{aligned}
//!       g_{a_1} \int \vt C[f_{a_{1}}] \frac{\dd^3 \vt p_{a_1}}{(2\pi)^3 E_{a_1}}
//!         &= - \int \left( \prod_{\vt a, \vt b} \dd \Pi_i \right)
//!                   (2 \pi)^4 \delta(p_{\vt a} - p_{\vt b}) \\\\
//!         &\quad \times \Biggl[ \abs{\mathcal M(\vt a | \vt b)}^2 \left( \prod_{\vt a} f_i \right)
//!                             - \abs{\mathcal M(\vt b | \vt a)}^2 \left( \prod_{\vt b} f_i \right) \Biggr],
//!     \end{aligned}
//!   \\end{equation}
//!
//!   Combined with the assumption that all species are in kinetic equilibrium,
//!   then the collision term can be further simplified to:
//!
//!   \\begin{equation}
//!     \begin{aligned}
//!       g_{a_1} \int \vt C[f_{a_{1}}] \frac{\dd^3 \vt p_{a_1}}{(2\pi)^3 E_{a_1}}
//!         &= - \int \left( \prod_{\vt a, \vt b} \dd \Pi_i \right)
//!                   (2 \pi)^4 \delta(p_{\vt a} - p_{\vt b}) \\\\
//!         &\quad \times \Biggl[ \abs{\mathcal M(\vt a | \vt b)}^2 e^{ \beta \sum_{\vt a} \mu_i }
//!                             - \abs{\mathcal M(\vt b | \vt a)}^2 e^{ \beta \sum_{\vt b} \mu_i } \Biggr]
//!                       e^{ - \beta \sum_{\vt a} E_i },
//!     \end{aligned}
//!   \\end{equation}
//!
//!   where energy conservation (as imposed by the Dirac delta) is used to
//!   equate \\(\sum_{\vt a} E_i = \sum_{\vt b} E_i\\).
//!
//!   It should be noted that this assumption also simplifies greatly the
//!   expression for the number density.  In particular, we obtain:
//!
//!  \\begin{equation}
//!    n = e^{\mu \beta} \frac{m^2 K_2(m \beta)}{2 \pi^2 \beta} = e^{\mu \beta} n^{(0)}
//!  \\end{equation}
//!
//! - *Assume \\(\mathcal{CP}\\) symmetry.* If \\(\mathcal{CP}\\) symmetry is
//!   assumed, then \\(\abs{\mathcal M(\vt a | \vt b)}^2 \equiv \abs{\mathcal
//!   M(\vt b | \vt a)}^2\\).  This simplification allows for the exponentials
//!   of \\(\mu\\) to be taken out of the integral entirely:
//!
//!   \\begin{equation}
//!     g_{a_1} \int \vt C[f_{a_{1}}] \frac{\dd^3 \vt p_{a_1}}{(2\pi)^3 E_{a_1}}
//!       = - \left[ e^{ \beta \sum_{\vt a} \mu_i } - e^{ \beta \sum_{\vt b} \mu_i } \right]
//!            \int \left( \prod_{\vt a, \vt b} \dd \Pi_i \right) (2 \pi)^4 \delta(p_{\vt a} - p_{\vt b})
//!            \abs{\mathcal M(\vt a | \vt b)}^2 e^{ - \beta \sum_{\vt a} E_i }.
//!   \\end{equation}
//!
//!   The remaining integrand is then independent of time and can be
//!   pre-calculated.  In the case of a \\(2 \leftrightarrow 2\\) interaction,
//!   this integral is related to the thermally averaged cross-section
//!   \\(\angles{\sigma v_\text{rel}}\\).
//!
//!   Solving the Boltzmann equation is generally required within the context of
//!   baryogenesis and leptogenesis where the assumption of \\(\mathcal{CP}\\)
//!   symmetry is evidently not correct.  In such cases, it is convenient to
//!   define the parameter \\(\epsilon\\) to account for all of the
//!   \\(\mathcal{CP}\\) asymmetry as.  That is:
//!
//!   \\begin{equation}
//!     \abs{\mathcal M(\vt a | \vt b)}^2 = (1 + \epsilon) \abs{\mathcal{M^{(0)}(\vt a | \vt b)}}^2, \qquad
//!     \abs{\mathcal M(\vt b | \vt a)}^2 = (1 - \epsilon) \abs{\mathcal{M^{(0)}(\vt a | \vt b)}}^2,
//!   \\end{equation}
//!
//!   where \\(\abs{\mathcal{M^{(0)}}(\vt a | \vt b)}^2\\) is the
//!   \\(\mathcal{CP}\\)-symmetric squared amplitude.  With \\(\epsilon\\)
//!   defined as above, the collision term becomes:
//!
//!   \\begin{equation}
//!     \begin{aligned}
//!       g_{a_1} \int \vt C[f_{a_{1}}] \frac{\dd^3 \vt p_{a_1}}{(2\pi)^3 E_{a_1}}
//!         &= - \left(
//!                 \left[ e^{ \beta \sum_{\vt a} \mu_i } - e^{ \beta \sum_{\vt b} \mu_i } \right]
//!                 + \epsilon \left[ e^{ \beta \sum_{\vt a} \mu_i } + e^{ \beta \sum_{\vt b} \mu_i } \right]
//!              \right) \\\\
//!         &\quad \times
//!              \int \left( \prod_{\vt a, \vt b} \dd \Pi_i \right) (2 \pi)^4 \delta(p_{\vt a} - p_{\vt b})
//!              \abs{\mathcal M^{(0)}(\vt a | \vt b)}^2 e^{ - \beta \sum_{\vt a} E_i }.
//!     \\end{aligned}
//!   \\end{equation}

#![cfg_attr(feature = "nightly", feature(test))]
#![cfg_attr(feature = "nightly", feature(iterator_step_by))]
#![cfg_attr(feature = "cargo-clippy", allow(unreadable_literal))]
#![cfg_attr(feature = "strict", deny(missing_docs))]
#![cfg_attr(feature = "strict", deny(warnings))]

#[macro_use]
extern crate log;
extern crate quadrature;
extern crate special_functions;

#[cfg(test)]
extern crate csv;

#[cfg(feature = "nightly")]
extern crate test;

macro_rules! debug_assert_warn {
    ($cond:expr, $($arg:tt)+) => (
        if cfg!(debug_assertions) && ($cond) {
            warn!($($arg,)*);
        }
    )
}

pub mod constants;
pub mod statistic;
pub mod universe;

// pub mod common;
pub(crate) mod utilities;
