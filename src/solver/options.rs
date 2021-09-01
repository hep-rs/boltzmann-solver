#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::{fmt, ops};

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
            min: 1e-4,
            max: 1e0,
        }
    }
}

impl fmt::Display for StepPrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StepPrecision {{ min: {}, max: {} }}",
            self.min, self.max
        )
    }
}

/// Determines the range of step sizes allowed: `min * beta < h < max * beta`.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub(crate) struct ErrorTolerance {
    pub abs: f64,
    pub rel: f64,
}

impl ErrorTolerance {
    /// Compute the maximum tolerance given a value that is being compared:
    ///
    /// ```math
    /// \varepsilon_\text{max} = \max[\varepsilon_\text{abs}, \varepsilon_\text{rel} * \abs{y}]
    /// ```
    pub(crate) fn max_tolerance(&self, y: f64) -> f64 {
        self.abs.max(y.abs() * self.rel)
    }
}

impl Default for ErrorTolerance {
    fn default() -> Self {
        ErrorTolerance {
            abs: 1e-10,
            rel: 1e-4,
        }
    }
}

impl ops::Mul<f64> for ErrorTolerance {
    type Output = Self;

    fn mul(self, other: f64) -> Self {
        ErrorTolerance {
            abs: self.abs * other,
            rel: self.rel * other,
        }
    }
}

impl ops::Div<f64> for ErrorTolerance {
    type Output = Self;

    fn div(self, other: f64) -> Self {
        ErrorTolerance {
            abs: self.abs / other,
            rel: self.rel / other,
        }
    }
}

impl fmt::Display for ErrorTolerance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ErrorTolerance {{ abs: {}, rel: {} }}",
            self.abs, self.rel
        )
    }
}
