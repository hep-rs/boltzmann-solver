use boltzmann_solver::prelude::*;
use fern::colors;
use std::{
    env::temp_dir,
    fs, io,
    path::{Path, PathBuf},
};

/// Setup logging with the specified verbosity.  If unset, nothing is printed
/// during tests.
///
/// The verbosity options are:
/// - `0`: Warn level
/// - `1`: Info level
/// - `2`: Debug level
/// - `3` and above: Trace level
#[allow(dead_code)]
pub fn setup_logging(verbosity: usize) {
    let mut base_config = fern::Dispatch::new();

    let colors = colors::ColoredLevelConfig::new()
        .error(colors::Color::Red)
        .warn(colors::Color::Yellow)
        .info(colors::Color::Green)
        .debug(colors::Color::White)
        .trace(colors::Color::Black);

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

    match verbosity {
        0 => log::warn!("Verbosity set to Warn"),
        1 => log::info!("Verbosity set to Info."),
        2 => log::debug!("Verbosity set to Debug."),
        _3_or_more => log::trace!("Verbosity set to Trace."),
    }
}

/// Setup an output directory and return the corresponding [`PathBuf`].
///
/// The output directory is created in the system's temporary directory and
/// named `"boltzmann_solver"`, and the relevant `subdir` is dreated within
/// that.
pub fn output_dir<P: AsRef<Path>>(subdir: P) -> PathBuf {
    let mut dir = temp_dir();
    dir.push("boltzmann_solver");
    dir.push(subdir);
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

/// Box an interaction
#[cfg(feature = "parallel")]
pub fn into_interaction_box<I, M>(interaction: I) -> Box<dyn Interaction<M> + Sync>
where
    I: Interaction<M> + Sync + 'static,
    M: Model,
{
    Box::new(interaction)
}

/// Box an interaction
#[cfg(not(feature = "parallel"))]
pub fn into_interaction_box<I, M>(interaction: I) -> Box<dyn Interaction<M>>
where
    I: Interaction<M> + 'static,
    M: Model,
{
    Box::new(interaction)
}

/// Filter interactions based whether they involve first-generation particles
/// only or not.
pub fn one_generation<I, M>(interaction: &I) -> bool
where
    I: Interaction<M>,
    M: Model,
{
    let ptcl = interaction.particles_idx();
    ptcl.incoming.iter().chain(&ptcl.outgoing).all(|i| match i {
        1 | 2 | 3 | 4 | 5 | 8 | 11 | 14 | 17 | 20 => true,
        _ => false,
    })
}
