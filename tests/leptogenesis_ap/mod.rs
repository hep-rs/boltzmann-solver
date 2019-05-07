//! Use the standard precision Boltzmann solver with the vanilla leptogenesis
//! scenario.

pub mod interaction;
pub mod model;
pub mod solve;

use boltzmann_solver::solver_ap::Model;
use model::LeptogenesisModel;
use rug::Float;

/// Test a single fiducial data point
#[test]
pub fn run() {
    // Setup the directory for CSV output
    ::std::fs::create_dir("/tmp/leptogenesis_ap/").unwrap_or(());

    let model = LeptogenesisModel::new(&Float::with_val(100, 1e-17));

    let sol = solve::solve(model);

    assert!(1e-10 < sol[0].to_f64().abs() && sol[0].to_f64().abs() < 1e-5);
    assert!(sol[1] < 1e-20);
}
