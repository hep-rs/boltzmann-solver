use super::{Propagator, Single};
use num::{Complex, Zero};
use std::ops;

/// Multiple propagators.
///
/// This isn't designed to be used on its own and instead allows for better
/// handling of multiple propagators.
#[derive(Debug)]
pub struct Multi {
    pub(crate) propagators: Vec<Single>,
    pub(crate) conjugated: bool,
}

impl Propagator for Multi {
    /// Complex conjugatet the propagator.
    #[must_use]
    fn conj(mut self) -> Self {
        self.ref_conj();
        self
    }

    fn ref_conj(&mut self) -> &mut Self {
        self.propagators.iter_mut().for_each(|p| {
            p.ref_conj();
        });
        self.conjugated = !self.conjugated;
        self
    }

    /// Propagator denominator for this particle:
    ///
    /// ```math
    /// P_{i}^{-1}(s) \defeq s - m_i^2 + i \theta(s) m_i \Gamma_i.
    /// ```
    #[must_use]
    fn eval(&self) -> Complex<f64> {
        self.propagators.iter().map(Single::eval).sum()
    }

    /// Compute the norm squared of the propagator.
    ///
    /// This is equivalent to having `p * p.conj()`, but does not require the
    /// allocation of a second propagator.
    #[must_use]
    fn norm_sqr(&self) -> f64 {
        let mut result = Complex::zero();

        for pi in &self.propagators {
            for pj in &self.propagators {
                result += pi.mul_conj(pj);
            }
        }

        // TODO: Remove this as this should not be required.
        debug_assert!(
            result.im < 2.0 * f64::EPSILON,
            "Imaginary part of norm_sqr was larger than expected."
        );

        result.re
    }
}

impl ops::Add<Self> for Multi {
    type Output = Self;

    fn add(mut self, mut rhs: Self) -> Self::Output {
        debug_assert!(
            self.conjugated == rhs.conjugated,
            "When adding two propagators, they must have the same conjugation."
        );
        self.propagators.append(&mut rhs.propagators);
        self
    }
}

impl ops::Add<Single> for Multi {
    type Output = Self;

    fn add(mut self, rhs: Single) -> Self::Output {
        debug_assert!(
            self.conjugated == rhs.conjugated,
            "When adding two propagators, they must have the same conjugation."
        );
        self.propagators.push(rhs);
        self
    }
}

impl ops::Add<Multi> for Single {
    type Output = Multi;

    fn add(self, rhs: Multi) -> Self::Output {
        rhs + self
    }
}

impl ops::Sub<Self> for Multi {
    type Output = Self;

    fn sub(mut self, mut rhs: Self) -> Self::Output {
        debug_assert!(
            self.conjugated == rhs.conjugated,
            "When adding two propagators, they must have the same conjugation."
        );
        self.propagators
            .extend(rhs.propagators.drain(..).map(ops::Neg::neg));
        self
    }
}

impl ops::Sub<Single> for Multi {
    type Output = Self;

    fn sub(mut self, rhs: Single) -> Self::Output {
        debug_assert!(
            self.conjugated == rhs.conjugated,
            "When adding two propagators, they must have the same conjugation."
        );
        self.propagators.push(-rhs);
        self
    }
}

impl ops::Sub<Multi> for Single {
    type Output = Multi;

    fn sub(self, mut rhs: Multi) -> Self::Output {
        rhs = -rhs;
        rhs.propagators.push(self);
        rhs
    }
}

impl ops::Mul<Self> for Multi {
    type Output = Complex<f64>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.conjugated != rhs.conjugated,
            "When multiplying two propagators, they must have the opposite conjugation."
        );

        let mut result = Complex::zero();

        for pi in &self.propagators {
            for pj in &rhs.propagators {
                result += pi * pj;
            }
        }

        result
    }
}

impl ops::Mul<Single> for Multi {
    type Output = Complex<f64>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Single) -> Self::Output {
        debug_assert!(
            self.conjugated != rhs.conjugated,
            "When multiplying two propagators, they must have the opposite conjugation."
        );

        let mut result = Complex::zero();
        for pi in &self.propagators {
            result += pi * &rhs;
        }
        result
    }
}

impl ops::Mul<Multi> for Single {
    type Output = Complex<f64>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Multi) -> Complex<f64> {
        rhs * self
    }
}

impl ops::Mul<f64> for Multi {
    type Output = Self;

    fn mul(mut self, rhs: f64) -> Self::Output {
        for p in &mut self.propagators {
            *p *= rhs;
        }
        self
    }
}

impl ops::Mul<Multi> for f64 {
    type Output = Multi;

    fn mul(self, mut rhs: Multi) -> Self::Output {
        for p in &mut rhs.propagators {
            *p *= self;
        }
        rhs
    }
}

impl ops::MulAssign<f64> for Multi {
    fn mul_assign(&mut self, rhs: f64) {
        for p in &mut self.propagators {
            *p *= rhs;
        }
    }
}

impl ops::Mul<Complex<f64>> for Multi {
    type Output = Self;

    fn mul(mut self, rhs: Complex<f64>) -> Self::Output {
        for p in &mut self.propagators {
            *p *= rhs;
        }
        self
    }
}

impl ops::Mul<Multi> for Complex<f64> {
    type Output = Multi;

    fn mul(self, mut rhs: Multi) -> Self::Output {
        for p in &mut rhs.propagators {
            *p *= self;
        }
        rhs
    }
}

impl ops::MulAssign<Complex<f64>> for Multi {
    fn mul_assign(&mut self, rhs: Complex<f64>) {
        for p in &mut self.propagators {
            *p *= rhs;
        }
    }
}

impl ops::Div<f64> for Multi {
    type Output = Self;

    fn div(mut self, rhs: f64) -> Self::Output {
        for p in &mut self.propagators {
            *p /= rhs;
        }
        self
    }
}

impl ops::DivAssign<f64> for Multi {
    fn div_assign(&mut self, rhs: f64) {
        for p in &mut self.propagators {
            *p /= rhs;
        }
    }
}

impl ops::Div<Complex<f64>> for Multi {
    type Output = Self;

    fn div(mut self, rhs: Complex<f64>) -> Self::Output {
        for p in &mut self.propagators {
            *p /= rhs;
        }
        self
    }
}

impl ops::DivAssign<Complex<f64>> for Multi {
    fn div_assign(&mut self, rhs: Complex<f64>) {
        for p in &mut self.propagators {
            *p /= rhs;
        }
    }
}

impl ops::Neg for Multi {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.propagators = self.propagators.drain(..).map(ops::Neg::neg).collect();
        self
    }
}
