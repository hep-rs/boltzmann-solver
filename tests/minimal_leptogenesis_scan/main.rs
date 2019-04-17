extern crate boltzmann_solver;
extern crate csv;
extern crate itertools;
extern crate ndarray;
extern crate num;
extern crate quadrature;
extern crate rayon;
extern crate rgsl;
extern crate special_functions;

mod model;
mod solve;

use boltzmann_solver::solver::Model;
use itertools::iproduct;
use model::VanillaLeptogenesisModel;
use ndarray::prelude::*;
use rayon::prelude::*;
use std::sync::RwLock;

#[test]
fn minimal_leptogenesis_scan() {
    // Setup the directory for CSV output
    ::std::fs::create_dir("/tmp/minimal_leptogenesis/").unwrap_or(());

    let csv = RwLock::new(csv::Writer::from_path("/tmp/minimal_leptogenesis/scan.csv").unwrap());
    csv.write().unwrap().serialize(("y", "m", "B-L")).unwrap();

    let ym: Vec<_> = iproduct!(
        Array1::linspace(-8.0, -4.0, 2).into_iter(),
        Array1::linspace(6.0, 14.0, 2).into_iter()
    )
    .map(|(&y, &m)| (10.0f64.powf(y), 10.0f64.powf(m)))
    .collect();

    ym.into_par_iter().for_each(|(y, m)| {
        let model = VanillaLeptogenesisModel::new(1e-17);
        let sol = solve::solve(model, y, m);

        csv.write()
            .unwrap()
            .serialize((
                format!("{:.10e}", y),
                format!("{:.10e}", m),
                format!("{:.10e}", sol[0]),
            ))
            .unwrap();
    });
}
