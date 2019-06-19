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

use std::{env::temp_dir, fs, io, path::PathBuf};

/// Setup logging
fn setup_logging() {
    let mut base_config = fern::Dispatch::new();

    base_config = base_config.level(log::LevelFilter::Info);

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

    base_config.chain(stderr_config).apply().unwrap_or(());
}

/// Output directory
fn output_dir() -> PathBuf {
    let mut dir = temp_dir();
    dir.push("leptogenesis");
    if !dir.is_dir() {
        log::info!("Creating output directory: {}", dir.display());
    }

    for subdir in &["sp", "ap"] {
        match fs::create_dir_all(dir.join(subdir)) {
            Ok(()) => (),
            Err(e) => {
                log::error!("Unable to created directory: {}", e);
                panic!()
            }
        }
    }

    dir
}

pub mod leptogenesis_sp;

#[cfg(feature = "arbitrary-precision")]
pub mod leptogenesis_ap;
