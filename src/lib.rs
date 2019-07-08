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
//!   \vt L\[f\] = \vt C\[f\]
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
//!   \vt L\[f\] \stackrel{\mathrm{\tiny FLRW}}{=} \left[ E \pfrac{}{t} - H (E^2 - m^2) \pfrac{}{E} \right] f,
//! \\end{equation}
//!
//! where \\(H \defeq \dot a / a\\) is Hubble's constant.  Unless there is a
//! sudden increase in entropy due to the out-of-equilibrium decay of a
//! particle, the entropy per comoving volume remains constant: \\(\dd(s a^3) =
//! 0\\).  Furthermore,
//!
//! In the absence of any collisions, the Boltzmann equation ensures that the
//! phase space distribution is conserved.  This implies that after integrating
//! the phase space, the number density per comoving volume remains constant if
//! there are no collisions.  This is evident after integrating the Liouville
//! operator over the phase space:
//!
//! \\begin{equation}
//!   g \int \vt L\[f\] \frac{\dd^3 \vt p}{(2 \pi)^3} = \pfrac{n}{t} + 3 H n = \frac{1}{a^3} \pfrac{(n a^3)}{t}.
//! \\end{equation}
//!
//! The number density is related to the phase space distribution through
//!
//! \\begin{equation}
//!   n \defeq g \int f \frac{\dd^3 \vt p}{(2 \pi)^3} = \frac{g}{2 \pi^2} \int_m^\infty f \sqrt{E^2 - m^2} E \dd E,
//! \\end{equation}
//!
//! where \\(g\\) is the number of internal degrees of freedom of the particle.
//!
//! ### Collision Operator
//!
//! The collision term in the Boltzmann equation describes the changes to the
//! phase space distribution that arise from collisions—either through the
//! scattering of other particles, or creation/annihilation of the species in
//! question.  The integrated collision term for a particular process \\(a_1 +
//! \cdots + a_n \leftrightarrow b_1 + \cdots + b_m\\) (which we will denote as
//! \\(\vt a \leftrightarrow \vt b\\)) is:
//!
//! \\begin{equation}
//!   \begin{aligned}
//!     g_{a_1} \int \vt C[f_{a_1}] \frac{\dd^3 \vt p_{a_1}}{(2\pi)^3}
//!       &= - \int_{\vt a}^{\vt b}
//!          \abs{\mathcal M(\vt a | \vt b)}^2 \left( \prod_{\vt a} f_i \right) \left(\prod_{\vt b} 1 \pm f_i \right)
//!                          - \abs{\mathcal M(\vt b | \vt a)}^2 \left( \prod_{\vt b} f_i \right) \left(\prod_{\vt a} 1 \pm f_i \right),
//!   \end{aligned}
//! \\end{equation}
//!
//! where \\(\abs{\mathcal M(\vt a | \vt b)}^2\\) is the squared amplitude going
//! from initial state \\(\vt a\\) to \\(\vt b\\) and is *averaged over* all
//! *internal degrees of freedom*, and the integration is done of the
//! Lorentz-invariant phase space
//!
//! \\begin{equation}
//!   \int_{\vt a}^{\vt b} \defeq
//!      \int \dd \Pi_{a_1} \dots \dd \Pi_{a_n} \dd \Pi_{b_1} \dots \dd \Pi_{b_m}
//!      (2 \pi)^4 \delta^4(p_{a_1} + \cdots + p_{a_n} - p_{b_1} - \cdots - p_{b_m})
//! \\end{equation}
//!
//! in which
//!
//! \\begin{equation}
//!   \dd \Pi_i \defeq \frac{g_i \dd^3 \vt p_i}{(2 \pi)^3 2 E_i}.
//! \\end{equation}
//!
//! Note that as the squared amplitude is averaged over all internal degrees of
//! freedom, they will cancel out with the \\(g_i\\) factors in each \\(\dd
//! \Pi_i\\) term; as a result, some authors omit \\(g_i\\) from the definition
//! of \\(\dd \Pi_i\\).
//!
//! The factors \\(1 \pm f\\) account for Pauli suppression and Bose enhancement
//! in the transition \\(\vt a \leftrightarrow \vt b\\), where \\(1 + f\\) is
//! used for Bose–Einstein statistics as it increases the probability of a
//! transition into a highly occupied state; and \\(1 - f\\) is used for
//! Fermi–Dirac statistics as it decreases the probability of a transition into
//! a highly occupied state.
//!
//! The above collision term accounts for a single interaction and in general a
//! sum over all possible interactions must be done.

// Enable feature(test) on nightly builds to make use of the `test` crate.
#![cfg_attr(feature = "nightly", feature(test))]

macro_rules! debug_assert_warn {
    ($cond:expr, $($arg:tt)+) => (
        if cfg!(debug_assertions) && ($cond) {
            log::warn!($($arg,)*);
        }
    )
}

pub mod constants;
pub mod particle;
pub mod solver;
pub mod statistic;
pub mod universe;
pub mod utilities;
