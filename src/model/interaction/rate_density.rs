#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::{default, fmt, ops};

/// Rate density associate with an interaction.
///
/// The rate density is defined such that if it is positive, it converts
/// incoming particles into outgoing particles.  As a result, the change initial
/// state particles is proportional to the negative of the rates contained,
/// while the change for final state particles is proportional to the rates
/// themselves.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct RateDensity {
    /// Net symmetric rate
    pub symmetric: f64,
    /// Net asymmetric rate
    pub asymmetric: f64,
}

impl RateDensity {
    /// Create a new instanse with both rates being 0.
    #[must_use]
    pub fn zero() -> Self {
        RateDensity {
            symmetric: 0.0,
            asymmetric: 0.0,
        }
    }
}

impl default::Default for RateDensity {
    fn default() -> Self {
        Self::zero()
    }
}

impl fmt::Display for RateDensity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RateDensity {{ symmetric: {:e}, asymmetric: {:e} }}",
            self.symmetric, self.asymmetric
        )
    }
}

impl ops::Mul<f64> for RateDensity {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Self {
            symmetric: self.symmetric * rhs,
            asymmetric: self.asymmetric * rhs,
        }
    }
}

impl ops::Mul<RateDensity> for f64 {
    type Output = RateDensity;

    fn mul(self, rhs: RateDensity) -> RateDensity {
        rhs * self
    }
}

impl ops::MulAssign<f64> for RateDensity {
    fn mul_assign(&mut self, rhs: f64) {
        self.symmetric *= rhs;
        self.asymmetric *= rhs;
    }
}

impl ops::Div<f64> for RateDensity {
    type Output = Self;

    fn div(self, rhs: f64) -> Self {
        Self {
            symmetric: self.symmetric / rhs,
            asymmetric: self.asymmetric / rhs,
        }
    }
}

impl ops::DivAssign<f64> for RateDensity {
    fn div_assign(&mut self, rhs: f64) {
        self.symmetric /= rhs;
        self.asymmetric /= rhs;
    }
}
