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
use ndarray::prelude::*;
use ndarray_parallel::prelude::*;

#[test]
fn minimal_leptogenesis_scan() {
    // Setup the directory for CSV output
    ::std::fs::create_dir("/tmp/minimal_leptogenesis/").unwrap_or(());

    let mut csv = csv::Writer::from_path("/tmp/minimal_leptogenesis/scan.csv").unwrap();
    csv.serialize(("y", "m", "B-L")).unwrap();

    for &y in &Array1::linspace(-8.0, -4.0, 2) {
        let y = 10.0f64.powf(y);

        let ms: Vec<_> = Array1::linspace(-4.0, 4.0, 3)
            .par_iter()
            .map(|&m| {
                let m = 10.0f64.powf(m);
                let model = VanillaLeptogenesisModel::new(1e-17);
                let sol = solve::solve(model, y, m);
                (m, sol[0])
            })
            .collect();

        for (m, bl) in &ms {
            csv.serialize((
                format!("{:.10e}", y),
                format!("{:.10e}", m),
                format!("{:.10e}", bl),
            ))
            .unwrap();
        }
    }
}
