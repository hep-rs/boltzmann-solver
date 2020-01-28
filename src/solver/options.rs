#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Determines the range of step sizes allowed: `min * beta < h < max * beta`.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub(crate) struct StepPrecision {
    pub min: f64,
    pub max: f64,
}

impl Default for StepPrecision {
    fn default() -> Self {
        StepPrecision {
            min: 1e-10,
            max: 1e-0,
        }
    }
}

impl std::fmt::Display for StepPrecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "StepPrecision {{ min: {}, max: {} }}",
            self.min, self.max
        )
    }
}
