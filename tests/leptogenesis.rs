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

use std::io;

/// Setup logging
fn setup_logging() {
    let mut base_config = fern::Dispatch::new();

    base_config = base_config
        .level(log::LevelFilter::Info)
        .level_for("overly-verbose-target", log::LevelFilter::Warn);

    let stderr_config = fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "{} {} {} - {}",
                chrono::Local::now().format("%H:%M:%S%.3f"),
                record.target(),
                record.level(),
                message
            ))
        })
        .chain(io::stderr());

    base_config.chain(stderr_config).apply().unwrap();
}

pub mod leptogenesis_sp;

#[cfg(feature = "arbitrary-precision")]
pub mod leptogenesis_ap;
