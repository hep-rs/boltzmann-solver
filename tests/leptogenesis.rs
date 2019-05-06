//! Test `boltzmann-solver` to calculate the baryon asymmetry generated through
//! leptogenesis in the standard type-I seesaw.

extern crate boltzmann_solver;
extern crate csv;
extern crate itertools;
extern crate ndarray;
extern crate num;
extern crate quadrature;
extern crate rgsl;
extern crate special_functions;

pub mod leptogenesis_sp;

#[cfg(feature = "arbitrary-precision")]
pub mod leptogenesis_ap;
