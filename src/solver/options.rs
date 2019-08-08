/// Determines how the step size is allowed changed, with the change being
/// proportional: `h *= increase` or `h *= decrease`.
pub(crate) struct StepChange {
    pub increase: f64,
    pub decrease: f64,
}

impl Default for StepChange {
    fn default() -> Self {
        StepChange {
            increase: 4.0,
            decrease: 0.1,
        }
    }
}

/// Determines the range of step sizes allowed: `min * beta < h < max * beta`.
pub(crate) struct StepPrecision {
    pub min: f64,
    pub max: f64,
}

impl Default for StepPrecision {
    fn default() -> Self {
        StepPrecision {
            min: 1e-4,
            max: 1e-0,
        }
    }
}

/// Initial conditions for a particle.
pub enum InitialCondition {
    /// The particle's initial density is its equilibrium number with the
    /// provided value of the chemical potential (in GeV)
    Equilibrium(f64),
    /// The particle's initial density begins with a fixed (arbitrary) value.
    Fixed(f64),
    /// The particle's initial abundance is zero.
    Zero,
}
