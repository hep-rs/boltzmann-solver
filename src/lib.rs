#![cfg_attr(feature = "nightly", feature(test))]
#![cfg_attr(feature = "cargo-clippy", allow(unreadable_literal))]

#[macro_use]
extern crate log;
extern crate quadrature;

#[cfg(feature = "nightly")]
extern crate test;

macro_rules! debug_assert_warn {
    ($cond:expr, $($arg:tt)+) => (
        if cfg!(debug_assertions) && ($cond) {
            warn!($($arg,)*);
        }
    )
}

pub mod common;
pub(crate) mod utilities;
pub mod prelude;
