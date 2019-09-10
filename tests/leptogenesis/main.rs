//! Test `boltzmann-solver` to calculate the baryon asymmetry generated through
//! leptogenesis in the standard type-I seesaw.

pub mod interaction;
pub mod model;

use boltzmann_solver::{
    solver::{number_density::SolverBuilder, Model},
    universe::StandardModel,
};
use fern::colors;
use itertools::iproduct;
use log::info;
use model::{p_i, LeptogenesisModel};
use ndarray::prelude::*;
use rayon::prelude::*;
use std::{cell::RefCell, env::temp_dir, fs, io, path::PathBuf, sync::RwLock};

/// Setup logging
fn setup_logging() {
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

    base_config.chain(stderr_config).apply().unwrap_or(());

    log::debug!("Verbosity set to Debug.");
    log::trace!("Verbosity set to Trace.");
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

    let sol = solve(LeptogenesisModel::new);

    info!("Final number density: {:.3e}", sol);

    assert!(1e-10 < sol[p_i("B-L", 0)].abs() && sol[p_i("B-L", 0)].abs() < 1e-5);
    for i in 0..3 {
        assert!(sol[p_i("N", i)] < 1e-9);
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
        Array1::linspace(-8.0, -2.0, 8).into_iter(),
        Array1::linspace(6.0, 14.0, 8).into_iter()
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
        let sol = solve(f);
        assert!(1e-20 < sol[p_i("B-L", 0)].abs() && sol[p_i("B-L", 0)].abs() < 1e-1);
        for i in 1..3 {
            assert!(sol[p_i("N", i)] < 1e-15);
        }

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

/// Solve the Boltzmann equations for the given model.
///
/// This routine sets up the solve, runs it and returns the final array of
/// number densities.
pub fn solve<F: 'static>(f: F) -> Array1<f64>
where
    F: Fn(f64) -> LeptogenesisModel,
{
    // Set up the universe in which we'll run the Boltzmann equations
    let universe = StandardModel::new();
    let beta_start = 1e-17;
    let beta_end = 1e-3;
    let model = f(beta_start);

    // Create the SolverBuilder and set base parameters
    let mut solver_builder: SolverBuilder<LeptogenesisModel> = SolverBuilder::new()
        .model(f)
        .equilibrium(vec![1])
        .beta_range(beta_start, beta_end)
        // .error_tolerance(1e-1)
        // .step_precision(1e-2, 5e-1)
        .initial_conditions(
            model
                .particles()
                .iter()
                .map(|p| p.normalized_number_density(0.0, beta_start)),
        );

    // Logging of number densities
    ////////////////////////////////////////////////////////////////////////////////
    let output_dir = crate::output_dir();
    let csv = RefCell::new(csv::Writer::from_path(output_dir.join("n.csv")).unwrap());
    {
        let mut csv = csv.borrow_mut();
        csv.write_field("step").unwrap();
        csv.write_field("beta").unwrap();

        for name in &model::NAMES {
            csv.write_field(name).unwrap();
            csv.write_field(format!("({})", name)).unwrap();
            csv.write_field(format!("Δ{}", name)).unwrap();
        }

        csv.write_record(None::<&[u8]>).unwrap();
    }

    solver_builder.logger(move |n, dn, c| {
        let mut csv = csv.borrow_mut();
        csv.write_field(format!("{}", c.step)).unwrap();
        csv.write_field(format!("{:.15e}", c.beta)).unwrap();

        for i in 0..n.len() {
            csv.write_field(format!("{:.3e}", n[i])).unwrap();
            csv.write_field(format!("{:.3e}", c.eq[i])).unwrap();
            csv.write_field(format!("{:.3e}", dn[i])).unwrap();
        }

        csv.write_record(None::<&[u8]>).unwrap();

        if n.iter().any(|n| !n.is_finite()) {
            panic!("Obtained a non-finite number.")
        }
    });

    // Interactions
    ////////////////////////////////////////////////////////////////////////////////

    interaction::n_el_h(&mut solver_builder);

    // Build and run the solver
    ////////////////////////////////////////////////////////////////////////////////

    let solver = solver_builder.build();

    solver.solve(&universe)
}
