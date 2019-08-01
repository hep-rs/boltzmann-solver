//! Test `boltzmann-solver` to calculate the baryon asymmetry generated through
//! leptogenesis in the standard type-I seesaw.

extern crate boltzmann_solver;
extern crate chrono;
extern crate csv;
extern crate fern;
extern crate itertools;
extern crate ndarray;
extern crate num;
extern crate quadrature;
extern crate rgsl;
extern crate special_functions;

use boltzmann_solver::solver::Model;
use fern::colors;
use itertools::iproduct;
use log::info;
use model::{p_i, LeptogenesisModel};
use ndarray::prelude::*;
use rayon::prelude::*;
use std::{env::temp_dir, fs, io, path::PathBuf, sync::RwLock};

pub mod interaction;
pub mod model;
pub mod solve;

/// Setup logging
fn setup_logging() -> Result<(), log::SetLoggerError> {
    let mut base_config = fern::Dispatch::new();

    let colors = colors::ColoredLevelConfig::new()
        .error(colors::Color::Red)
        .warn(colors::Color::Yellow)
        .info(colors::Color::Green)
        .debug(colors::Color::White)
        .trace(colors::Color::Black);

    let verbosity = 1;

    let lvl = match verbosity {
        0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Info,
        2 => log::LevelFilter::Debug,
        _3_or_more => log::LevelFilter::Trace,
    };
    base_config = base_config.level(lvl);

    let stderr_config = fern::Dispatch::new()
        .format(move |out, message, record| {
            out.finish(format_args!(
                "[{level}] {target} - {message}",
                target = record.target(),
                level = colors.color(record.level()),
                message = message
            ))
        })
        .chain(io::stderr());

    base_config.chain(stderr_config).apply()?;

    log::debug!("Verbosity set to Debug.");
    log::trace!("Verbosity set to Trace.");

    Ok(())
}

/// Output directory
fn output_dir() -> PathBuf {
    let mut dir = if false {
        temp_dir()
    } else {
        PathBuf::from("./output/")
    };
    dir.push("leptogenesis");
    if !dir.is_dir() {
        log::info!("Creating output directory: {}", dir.display());
    }

    match fs::create_dir_all(&dir) {
        Ok(()) => (),
        Err(e) => {
            log::error!("Unable to created directory: {}", e);
            panic!()
        }
    }

    dir
}
/// Test a single fiducial data point
#[test]
pub fn run() {
    crate::setup_logging();

    let sol = solve::solve(LeptogenesisModel::new);

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
    let output_dir = crate::output_dir();
    let csv = RwLock::new(csv::Writer::from_path(output_dir.join("scan.csv")).unwrap());
    csv.write().unwrap().serialize(("y", "m", "B-L")).unwrap();

    let ym: Vec<_> = iproduct!(
        Array1::linspace(-8.0, -4.0, 4).into_iter(),
        Array1::linspace(6.0, 14.0, 4).into_iter()
    )
    .map(|(&y, &m)| (10.0f64.powf(y), 10.0f64.powf(m)))
    .collect();

    ym.into_par_iter().for_each(|(y, mass_n)| {
        let f = move |beta: f64| {
            let mut m = LeptogenesisModel::new(beta);
            m.coupling.y_v.mapv_inplace(|yi| yi / 1e-4 * y);
            m.particles[p_i("N", 0)].set_mass(mass_n);
            m.mass.n[0] = mass_n;
            m.mass2.n[0] = mass_n.powi(2);
            m
        };
        let sol = solve::solve(f);

        csv.write()
            .unwrap()
            .serialize((
                format!("{:.10e}", y),
                format!("{:.10e}", mass_n),
                format!("{:.10e}", sol[p_i("B-L", 0)]),
            ))
            .unwrap();
    });
}
