// Boltzmann-solver
// Copyright (C) 2020  JP-Ellis <josh@jpells.me>

// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version.

// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
// details.

// You should have received a copy of the GNU General Public License along with
// this program.  If not, see <https://www.gnu.org/licenses/>.

//! `boltzmann-solver` is a library allowing for Boltzmann equation in the
//! context of particle physics / early cosmology.  It provides functionalities
//! to solve Boltzmann equation in the case where a single species is out of
//! equilibrium, as well as functionalities to solve the Boltzmann equations
//! more general when multiple species are all out of equilibrium.
//!
//! **This library is still undergoing development.**
//!
//! [![crates.io](https://img.shields.io/crates/v/hep-boltmzann-solver.svg)](https://crates.io/crates/hep-boltzmann-solver)
//! [![crates.io](https://img.shields.io/crates/d/hep-boltmzann-solver.svg)](https://crates.io/crates/hep-boltmzann-solver)
//! [![Codecov
//! branch](https://img.shields.io/codecov/c/github/hep-rs/boltzmann-solver/master)](https://codecov.io/gh/hep-rs/boltzmann-solver)
//! [![Build
//! Status](https://img.shields.io/github/workflow/status/hep-rs/boltzmann-solver/Rust/master.svg)](https://github.com/hep-rs/boltzmann-solver/actions)
//!
//! Licensed under [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html).
//!
//! # Introduction
//!
//! Within the context of cosmology, the Boltzmann equation is a differential
//! equation which describes the evolution of the phase-space distribution of a
//! species of particles.  In full generality, it is given by
//!
//! ```math
//! \vt L[f] = \vt C[f]
//! ```
//!
//! where `$\vt L$` is the Liouville operator and `$\vt C$` is the collision
//! operator.
//!
//! ### Liouville Operator
//!
//! In general, the phase space distribution dependent on position, momentum and
//! time `$f(\vt x, \vt p, t)$`; however, assuming the early Universe to be
//! isotropic and homogeneous it can be expressed as a function of just energy
//! and time, `$f(E, t)$`.  In the FLRW metric, the Liouville operator
//! simplifies to only be dependent the time-like derivatives:
//!
//! ```math
//! \vt L[f] \stackrel{\mathrm{\tiny FLRW}}{=}
//! \left[ E \pfrac{}{t} - H (E^2 - m^2) \pfrac{}{E} \right] f,
//! ```
//!
//! where `$H \defeq \dot a / a$` is Hubble's constant.  Unless there is a
//! sudden increase in entropy due to the out-of-equilibrium decay of a
//! particle, the entropy per comoving volume remains constant: `$\dd(s a^3) =
//! 0$`.  Furthermore,
//!
//! In the absence of any collisions, the Boltzmann equation ensures that the
//! phase space distribution is conserved.  This implies that after integrating
//! the phase space, the number density per comoving volume remains constant if
//! there are no collisions.  This is evident after integrating the Liouville
//! operator over the phase space:
//!
//! ```math
//! g \int \vt L[f] \frac{\dd^3 \vt p}{(2 \pi)^3}
//! = \pfrac{n}{t} + 3 H n = \frac{1}{a^3} \pfrac{(n a^3)}{t}.
//! ```
//!
//! The number density is related to the phase space distribution through
//!
//! ```math
//!   n \defeq g \int f \frac{\dd^3 \vt p}{(2 \pi)^3}
//!   = \frac{g}{2 \pi^2} \int_m^\infty f \sqrt{E^2 - m^2} E \dd E,
//! ```
//!
//! where `$g$` is the number of internal degrees of freedom of the particle.
//!
//! ### Collision Operator
//!
//! The collision term in the Boltzmann equation describes the changes to the
//! phase space distribution that arise from collisions—either through the
//! scattering of other particles, or creation/annihilation of the species in
//! question.  The integrated collision term for a particular process `$a_1 +
//! \cdots + a_n \leftrightarrow b_1 + \cdots + b_m$` (which we will denote as
//! `$\vt a \leftrightarrow \vt b$`) is:
//!
//! ```math
//! \begin{aligned}
//!   g_{a_1} \int \vt C[f_{a_1}] \frac{\dd^3 \vt p_{a_1}}{(2\pi)^3}
//!   &= - \int_{\vt a}^{\vt b}
//!      \abs{\scM(\vt a | \vt b)}^2 \left( \prod_{\vt a} f_i \right) \left(\prod_{\vt b} 1 \pm f_i \right)
//!      - \abs{\scM(\vt b | \vt a)}^2 \left( \prod_{\vt b} f_i \right) \left(\prod_{\vt a} 1 \pm f_i \right),
//! \end{aligned}
//! ```
//!
//! where `$\abs{\scM(\vt a | \vt b)}^2$` is the squared amplitude going
//! from initial state `$\vt a$` to `$\vt b$` and is *averaged over* all
//! *internal degrees of freedom*, and the integration is done of the
//! Lorentz-invariant phase space
//!
//! ```math
//! \int_{\vt a}^{\vt b} \defeq
//!      \int \dd \Pi_{a_1} \dots \dd \Pi_{a_n} \dd \Pi_{b_1} \dots \dd \Pi_{b_m}
//!      (2 \pi)^4 \delta^4(p_{a_1} + \cdots + p_{a_n} - p_{b_1} - \cdots - p_{b_m})
//! ```
//!
//! in which
//!
//! ```math
//!   \dd \Pi_i \defeq \frac{g_i \dd^3 \vt p_i}{(2 \pi)^3 2 E_i}.
//! ```
//!
//! Note that as the squared amplitude is averaged over all internal degrees of
//! freedom, they will cancel out with the `$g_i$` factors in each `$\dd \Pi_i$`
//! term; as a result, some authors omit `$g_i$` from the definition of `$\dd
//! \Pi_i$`.
//!
//! The factors `$\pm f$` account for Pauli suppression and Bose enhancement in
//! the transition `$\vt a \leftrightarrow \vt b$`, where `$+f$` is used for
//! Bose–Einstein statistics as it increases the probability of a transition
//! into a highly occupied state; and `$-f$` is used for Fermi–Dirac statistics
//! as it decreases the probability of a transition into a highly occupied
//! state.
//!
//! The above collision term accounts for a single interaction and in general a
//! sum over all possible interactions must be done.

// Enable feature(test) on nightly builds to make use of the `test` crate.
#![cfg_attr(feature = "nightly", feature(test))]
// Configure warnings
#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::non_ascii_literal)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
// #![allow(clippy::cast_precision_loss)] // Check this swarning occasionally
#![allow(clippy::doc_markdown)] // Clippy incorrectly thinkgs that 'GeV' refers to code.

#[cfg(feature = "nightly")]
extern crate test;

#[cfg(feature = "blas")]
extern crate blas_src;

macro_rules! debug_assert_warn {
    ($cond:expr, $($arg:tt)+) => (
        if cfg!(debug_assertions) && ($cond) {
            log::warn!($($arg,)*);
        }
    )
}

pub mod constants;
pub mod model;
pub mod prelude;
pub mod solver;
pub mod statistic;
pub mod utilities;

pub use crate::model::standard::data as sm_data;
pub use special_functions::particle_physics::pave_absorptive as pave;
