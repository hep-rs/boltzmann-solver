#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Partial width from a particle.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct RateDensity {
    /// Forward rate for the number density
    pub forward: f64,
    /// Backward rate for the number density
    pub backward: f64,
    /// Forward rate for the number density asymmetry
    pub asymmetric_forward: f64,
    /// Backward rate for the number density asymmetry
    pub asymmetric_backward: f64,
}

impl RateDensity {
    /// Return the net forward rate for the number density.
    pub fn net_rate(&self) -> f64 {
        self.forward - self.backward
    }

    /// Return the net forward rate for the number density asymmetry.
    pub fn net_asymmetric_rate(&self) -> f64 {
        self.asymmetric_forward - self.asymmetric_backward
    }
}

impl std::fmt::Display for RateDensity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RateDensity {{ forward: {:e}, backward: {:e}, ... }}",
            self.forward, self.backward
        )
    }
}

impl std::ops::Mul<f64> for RateDensity {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Self {
            forward: self.forward * rhs,
            backward: self.backward * rhs,
            asymmetric_forward: self.asymmetric_forward * rhs,
            asymmetric_backward: self.asymmetric_backward * rhs,
        }
    }
}

impl std::ops::MulAssign<f64> for RateDensity {
    fn mul_assign(&mut self, rhs: f64) {
        self.forward *= rhs;
        self.backward *= rhs;
        self.asymmetric_forward *= rhs;
        self.asymmetric_backward *= rhs;
    }
}
