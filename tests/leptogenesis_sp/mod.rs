//! Use the standard precision Boltzmann solver with the vanilla leptogenesis
//! scenario.

pub mod interaction;
pub mod model;
pub mod solve;

use boltzmann_solver::solver::Model;
use itertools::iproduct;
use log::info;
use model::{p_i, LeptogenesisModel};
use ndarray::prelude::*;
use rayon::prelude::*;
use std::sync::RwLock;

/// Test a single fiducial data point
#[test]
pub fn run() {
    crate::setup_logging();

    let sol = solve::solve(|beta| LeptogenesisModel::new(beta));

    info!("Final number density: {:.3e}", sol);

    assert!(1e-10 < sol[p_i("B-L", 0)].abs() && sol[p_i("B-L", 0)].abs() < 1e-5);
    for i in 0..3 {
        assert!(sol[p_i("N", i)] < 1e-20);
    }
}

/// Provide an example of a very simple scan over parameter space.
#[test]
pub fn scan() {
    crate::setup_logging();

    // Setup the directory for CSV output
    let output_dir = crate::output_dir().join("sp");
    let csv = RwLock::new(csv::Writer::from_path(output_dir.join("scan.csv")).unwrap());
    csv.write().unwrap().serialize(("y", "m", "B-L")).unwrap();

    let ym: Vec<_> = iproduct!(
        Array1::linspace(-8.0, 0.0, 2).into_iter(),
        Array1::linspace(6.0, 14.0, 2).into_iter()
    )
    .map(|(&y, &m)| (10.0f64.powf(y), 10.0f64.powf(m)))
    .collect();

    ym.into_par_iter().for_each(|(y, n0)| {
        let f = move |beta: f64| {
            let mut m = LeptogenesisModel::new(beta);
            m.coupling.y_v.mapv_inplace(|yi| yi / 1e-4 * y);
            m.particles[p_i("N", 0)].set_mass(n0);
            m.mass.n[0] = n0;
            m.mass2.n[0] = n0.powi(2);
            m
        };
        let sol = solve::solve(f);

        csv.write()
            .unwrap()
            .serialize((
                format!("{:.10e}", y),
                format!("{:.10e}", n0),
                format!("{:.10e}", sol[p_i("B-L", 0)]),
            ))
            .unwrap();
    });
}
