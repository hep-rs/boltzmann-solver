extern crate boltzmann_solver;
extern crate csv;
extern crate itertools;
extern crate ndarray;
extern crate num;
extern crate quadrature;
extern crate rgsl;
extern crate special_functions;

mod model;
mod solve;

use boltzmann_solver::solver::Model;
use model::VanillaLeptogenesisModel;

#[test]
fn minimal_leptogenesis() {
    // Setup the directory for CSV output
    ::std::fs::create_dir("/tmp/minimal_leptogenesis/").unwrap_or(());

    let mut model = VanillaLeptogenesisModel::new(1e-15);
    model.coupling.y_v = model.coupling.y_v * 1e-4;

    let sol = solve::solve(model);

    assert!(1e-10 < sol[0].abs() && sol[0].abs() < 1e-5);
    assert!(sol[1] < 1e-20);
}
