//! Use the standard precision Boltzmann solver with the vanilla leptogenesis
//! scenario.

pub mod interaction;
pub mod model;
pub mod solve;

use boltzmann_solver::solver::Model;
use itertools::iproduct;
use model::VanillaLeptogenesisModel;
use ndarray::prelude::*;
use rayon::prelude::*;
use std::sync::RwLock;

/// Test a single fiducial data point
#[test]
pub fn run() {
    // Setup the directory for CSV output
    ::std::fs::create_dir("/tmp/leptogenesis_sp/").unwrap_or(());

    let model = VanillaLeptogenesisModel::new(1e-17);

    let sol = solve::solve(model, |m| m);

    assert!(1e-10 < sol[0].abs() && sol[0].abs() < 1e-5);
    assert!(sol[1] < 1e-20);
}

/// Provide an example of a very simple scan over parameter space.
#[test]
pub fn scan() {
    // Setup the directory for CSV output
    ::std::fs::create_dir("/tmp/leptogenesis_sp/").unwrap_or(());

    let csv = RwLock::new(csv::Writer::from_path("/tmp/leptogenesis_sp/scan.csv").unwrap());
    csv.write().unwrap().serialize(("y", "m", "B-L")).unwrap();

    let ym: Vec<_> = iproduct!(
        Array1::linspace(-8.0, -4.0, 2).into_iter(),
        Array1::linspace(6.0, 14.0, 2).into_iter()
    )
    .map(|(&y, &m)| (10.0f64.powf(y), 10.0f64.powf(m)))
    .collect();

    ym.into_par_iter().for_each(|(y, n0)| {
        let model = VanillaLeptogenesisModel::new(1e-17);
        let f = move |mut m: VanillaLeptogenesisModel| {
            m.y_v.mapv_inplace(|yi| yi * y);
            m.m_n[0] = n0;
            m
        };
        let sol = solve::solve(model, f);

        csv.write()
            .unwrap()
            .serialize((
                format!("{:.10e}", y),
                format!("{:.10e}", n0),
                format!("{:.10e}", sol[0]),
            ))
            .unwrap();
    });
}
