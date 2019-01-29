/// Determines how the step size is allowed changed, with the change being
/// proportional: `h *= increase` or `h *= decrease`.
pub(crate) struct StepChange {
    pub increase: f64,
    pub decrease: f64,
}

impl Default for StepChange {
    fn default() -> Self {
        StepChange {
            increase: 1.1,
            decrease: 0.5,
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
            min: 1e-6,
            max: 1e-1,
        }
    }
}

/// Specifies the allowed error tolerance.  If the local error tolerance falls
/// outside the specified range, the step size is accordingly adjusted.
pub(crate) struct ErrorTolerance {
    pub upper: f64,
    pub lower: f64,
}

impl Default for ErrorTolerance {
    fn default() -> Self {
        ErrorTolerance {
            upper: 1e-2,
            lower: 1e-5,
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
