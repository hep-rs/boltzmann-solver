#![cfg_attr(feature = "nightly", feature(test))]
#![cfg_attr(feature = "cargo-clippy", allow(unreadable_literal))]

#[macro_use]
extern crate log;
extern crate quadrature;
extern crate rgsl;

#[cfg(feature = "nightly")]
extern crate test;

pub mod common;
pub(crate) mod utilities;
pub mod prelude;
