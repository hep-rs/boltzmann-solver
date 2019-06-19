//! Use the standard precision Boltzmann solver with the vanilla leptogenesis
//! scenario.

pub mod interaction;
pub mod model;
pub mod solve;

use boltzmann_solver::solver_ap::Model;
use log::info;
use model::{p_i, LeptogenesisModel};

/// Test a single fiducial data point
#[test]
pub fn run() {
    super::setup_logging();

    let sol = solve::solve(|beta| LeptogenesisModel::new(beta));

    info!("Final number density: {:.3e}", sol);

    assert!(1e-10 < *sol[p_i("B-L", 0)].as_abs() && *sol[p_i("B-L", 0)].as_abs() < 1e-5);
    for i in 0..3 {
        assert!(sol[p_i("N", i)] < 1e-20);
    }
}
