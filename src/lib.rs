//! `boltzmann-solver` is a library allowing for Boltzmann equation in the
//! context of particle physics / early cosmology.  It provides functionalities
//! to solve Boltzmann equation in the case where a single species is out of
//! equilibrium, as well as functionalities to solve the Boltzmann equations
//! more general when multiple species are all out of equilibrium.

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
