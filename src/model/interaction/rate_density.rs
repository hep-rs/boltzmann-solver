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
///
/// The rate density can be multiplied / divided by a float which will multiply
/// / divide all three fields of the struct.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct RateDensity {
    /// Interaction rate
    ///
    /// This is the rate as returned by [`Interaction::gamma`], which may be
    /// scaled by the normalization factor and/or integration step size.
    ///
    /// This is used to help determine whether the interaction rate is deemed
    /// fast or not.
    pub gamma: f64,
    /// Interaction rate
    ///
    /// This is the rate as returned by [`Interaction::delta_gamma`], which may
    /// be scaled by the normalization factor and/or integration step size.
    ///
    /// This is used to help determine whether the interaction rate is deemed
    /// fast or not.
    ///
    /// A value of `None` indicates that the interaction has no CP asymmetry,
    /// whereas a value of `0` indicates that there is no change.
    pub delta_gamma: Option<f64>,
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
            gamma: 0.0,
            delta_gamma: None,
            symmetric: 0.0,
            asymmetric: 0.0,
        }
    }

    /// Compute the ratio of the asymmetric interaction rate to the symmetric
    /// interaction rate.
    #[must_use]
    pub fn gamma_ratio(&self) -> Option<f64> {
        self.delta_gamma.map(|delta_gamma| delta_gamma / self.gamma)
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
            "RateDensity {{ symmetric: {}, asymmetric: {} }}",
            self.symmetric, self.asymmetric
        )
    }
}

impl fmt::LowerExp for RateDensity {
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

    /// Multiplies the symmetric and asymmetric rates, but leaves `gamma` unchanged.
    fn mul(self, rhs: f64) -> Self {
        Self {
            gamma: self.gamma * rhs,
            delta_gamma: self.delta_gamma.map(|v| v * rhs),
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
        self.gamma *= rhs;
        self.delta_gamma = self.delta_gamma.map(|v| v * rhs);
        self.symmetric *= rhs;
        self.asymmetric *= rhs;
    }
}

impl ops::Div<f64> for RateDensity {
    type Output = Self;

    fn div(self, rhs: f64) -> Self {
        Self {
            gamma: self.gamma / rhs,
            delta_gamma: self.delta_gamma.map(|v| v / rhs),
            symmetric: self.symmetric / rhs,
            asymmetric: self.asymmetric / rhs,
        }
    }
}

impl ops::DivAssign<f64> for RateDensity {
    fn div_assign(&mut self, rhs: f64) {
        self.gamma /= rhs;
        self.delta_gamma = self.delta_gamma.map(|v| v / rhs);
        self.symmetric /= rhs;
        self.asymmetric /= rhs;
    }
}
