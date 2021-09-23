use super::{Propagator, Single};
use num::{Complex, Zero};
use std::ops;

/// Multiple propagators.
///
/// This isn't designed to be used on its own and instead allows for better
/// handling of multiple propagators.
#[derive(Debug)]
pub struct Multi<'a> {
    pub(crate) propagators: Vec<Single<'a>>,
    pub(crate) conjugated: bool,
}

impl<'a> Propagator for Multi<'a> {
    /// Complex conjugatet the propagator.
    #[must_use]
    fn conj(mut self) -> Self {
        for p in &mut self.propagators {
            *p = p.conj();
        }
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

impl<'a> ops::Add<Self> for Multi<'a> {
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

impl<'a> ops::Add<Single<'a>> for Multi<'a> {
    type Output = Self;

    fn add(mut self, rhs: Single<'a>) -> Self::Output {
        debug_assert!(
            self.conjugated == rhs.conjugated,
            "When adding two propagators, they must have the same conjugation."
        );
        self.propagators.push(rhs);
        self
    }
}

impl<'a> ops::Add<Multi<'a>> for Single<'a> {
    type Output = Multi<'a>;

    fn add(self, rhs: Multi<'a>) -> Self::Output {
        rhs + self
    }
}

impl<'a> ops::Sub<Self> for Multi<'a> {
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

impl<'a> ops::Sub<Single<'a>> for Multi<'a> {
    type Output = Self;

    fn sub(mut self, rhs: Single<'a>) -> Self::Output {
        debug_assert!(
            self.conjugated == rhs.conjugated,
            "When adding two propagators, they must have the same conjugation."
        );
        self.propagators.push(-rhs);
        self
    }
}

impl<'a> ops::Sub<Multi<'a>> for Single<'a> {
    type Output = Multi<'a>;

    fn sub(self, mut rhs: Multi<'a>) -> Self::Output {
        rhs = -rhs;
        rhs.propagators.push(self);
        rhs
    }
}

impl<'a> ops::Mul<Self> for Multi<'a> {
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

impl<'a> ops::Mul<Single<'a>> for Multi<'a> {
    type Output = Complex<f64>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Single<'a>) -> Self::Output {
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

impl<'a> ops::Mul<Multi<'a>> for Single<'a> {
    type Output = Complex<f64>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Multi<'a>) -> Complex<f64> {
        rhs * self
    }
}

impl<'a> ops::Mul<f64> for Multi<'a> {
    type Output = Self;

    fn mul(mut self, rhs: f64) -> Self::Output {
        for p in &mut self.propagators {
            *p *= rhs;
        }
        self
    }
}

impl<'a> ops::Mul<Multi<'a>> for f64 {
    type Output = Multi<'a>;

    fn mul(self, mut rhs: Multi<'a>) -> Self::Output {
        for p in &mut rhs.propagators {
            *p *= self;
        }
        rhs
    }
}

impl<'a> ops::MulAssign<f64> for Multi<'a> {
    fn mul_assign(&mut self, rhs: f64) {
        for p in &mut self.propagators {
            *p *= rhs;
        }
    }
}

impl<'a> ops::Mul<Complex<f64>> for Multi<'a> {
    type Output = Self;

    fn mul(mut self, rhs: Complex<f64>) -> Self::Output {
        for p in &mut self.propagators {
            *p *= rhs;
        }
        self
    }
}

impl<'a> ops::Mul<Multi<'a>> for Complex<f64> {
    type Output = Multi<'a>;

    fn mul(self, mut rhs: Multi<'a>) -> Self::Output {
        for p in &mut rhs.propagators {
            *p *= self;
        }
        rhs
    }
}

impl<'a> ops::MulAssign<Complex<f64>> for Multi<'a> {
    fn mul_assign(&mut self, rhs: Complex<f64>) {
        for p in &mut self.propagators {
            *p *= rhs;
        }
    }
}

impl<'a> ops::Div<f64> for Multi<'a> {
    type Output = Self;

    fn div(mut self, rhs: f64) -> Self::Output {
        for p in &mut self.propagators {
            *p /= rhs;
        }
        self
    }
}

impl<'a> ops::DivAssign<f64> for Multi<'a> {
    fn div_assign(&mut self, rhs: f64) {
        for p in &mut self.propagators {
            *p /= rhs;
        }
    }
}

impl<'a> ops::Div<Complex<f64>> for Multi<'a> {
    type Output = Self;

    fn div(mut self, rhs: Complex<f64>) -> Self::Output {
        for p in &mut self.propagators {
            *p /= rhs;
        }
        self
    }
}

impl<'a> ops::DivAssign<Complex<f64>> for Multi<'a> {
    fn div_assign(&mut self, rhs: Complex<f64>) {
        for p in &mut self.propagators {
            *p /= rhs;
        }
    }
}

impl<'a> ops::Neg for Multi<'a> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        for p in &mut self.propagators {
            *p = -*p;
        }
        self
    }
}
