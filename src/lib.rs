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
//!       &= - \int \left( \prod_{\vt a, \vt b} \dd \Pi_i \right) \times (2 \pi)^4 \delta(\vt p_{\vt a} - \vt p_{\vt b}) \\\\
//!       &\quad \times \Bigl[ \abs{\mathcal M(\vt a | \vt b)}^2 f_{a_1} f_{a_2} \cdots (1 \pm f_{b_1}) (1 \pm f_{b_2}) \cdots \\\\
//!       &\qquad - \abs{\mathcal M(\vt b | \vt a)}^2 f_{b_1} f_{b_2} \cdots (1 \pm f_{a_1}) (1 \pm f_{a_2}) \cdots \Bigr],
//!   \end{aligned}
//! \\end{equation}
//!
//! where \\(\dd \Pi_i \defeq \dd^3 \vt p_i / (2 \pi)^3 E_i\\); \\(\vt p_{\vt a}
//! \defeq \vt p_{a_1} + \vt p_{a_2} + \cdots\\) (and similarly for \\(\vt
//! p_{\vt b}\\)), \\(\abs{\mathcal M(\vt a | \vt b)}^2\\) is the squared matrix
//! element going from initial state \\(\vt a\\) to \\(\vt b\\) and is *summed
//! over* all *internal degrees of freedom*.

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
mod standard_model;
mod statistic;
mod universe;

pub use standard_model::StandardModel;
pub use statistic::Statistic;
pub use universe::{SingleSpecies, Universe};

// pub mod common;
pub(crate) mod utilities;