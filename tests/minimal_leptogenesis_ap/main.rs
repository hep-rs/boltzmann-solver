#![cfg(arbitrary_precision)]

extern crate boltzmann_solver;
extern crate csv;
extern crate itertools;
extern crate ndarray;
extern crate num;
extern crate quadrature;
extern crate rgsl;
extern crate rug;
extern crate special_functions;

mod model;
mod solve;

use boltzmann_solver::solver_ap::Model;
use model::VanillaLeptogenesisModel;
use rug::Float;

#[test]
fn minimal_leptogenesis() {
    // Setup the directory for CSV output
    ::std::fs::create_dir("/tmp/minimal_leptogenesis_ap/").unwrap_or(());

    let model = VanillaLeptogenesisModel::new(&Float::with_val(100, 1e-17));

    let sol = solve::solve(model);

    assert!(1e-10 < sol[0].to_f64().abs() && sol[0].to_f64().abs() < 1e-5);
    assert!(sol[1] < 1e-20);
}
