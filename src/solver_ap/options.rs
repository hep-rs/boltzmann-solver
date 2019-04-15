use super::DEFAULT_WORKING_PRECISION;
use rug::Float;

/// Determines how the step size is allowed changed, with the change being
/// proportional: `h *= increase` or `h *= decrease`.
pub(crate) struct StepChange {
    pub increase: Float,
    pub decrease: Float,
}

impl Default for StepChange {
    fn default() -> Self {
        StepChange {
            increase: Float::with_val(DEFAULT_WORKING_PRECISION, 4.0),
            decrease: Float::with_val(DEFAULT_WORKING_PRECISION, 0.1),
        }
    }
}

/// Determines the range of step sizes allowed: `min * beta < h < max * beta`.
pub(crate) struct StepPrecision {
    pub min: Float,
    pub max: Float,
}

impl Default for StepPrecision {
    fn default() -> Self {
        StepPrecision {
            min: Float::with_val(DEFAULT_WORKING_PRECISION, 1e-4),
            max: Float::with_val(DEFAULT_WORKING_PRECISION, 1e-1),
        }
    }
}

/// Initial conditions for a particle.
pub enum InitialCondition {
    /// The particle's initial density is its equilibrium number with the
    /// provided value of the chemical potential (in GeV)
    Equilibrium(f64),
    /// The particle's initial density begins with a fixed (arbitrary) value.
    Fixed(Float),
    /// The particle's initial abundance is zero.
    Zero,
}
