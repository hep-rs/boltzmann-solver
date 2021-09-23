use super::{Multi, Propagator};
use crate::prelude::Particle;
use num::{Complex, One};
use std::ops;

#[derive(Debug, Copy, Clone)]
pub struct Single<'a> {
    particle: &'a Particle,
    momentum: f64,
    numerator: Complex<f64>,
    pub(crate) conjugated: bool,
}

impl<'a> Single<'a> {
    /// Create a new propagator.
    pub(crate) fn new(p: &'a Particle, s: f64) -> Self {
        Single {
            particle: p,
            momentum: s,
            numerator: One::one(),
            conjugated: false,
        }
    }

    /// Evaluate the propagator denominator.
    fn denominator(&self) -> Complex<f64> {
        match (self.momentum > 0.0, self.conjugated) {
            (true, false) => Complex::new(
                self.momentum - self.particle.mass2,
                self.particle.mass * self.particle.width,
            ),
            (true, true) => Complex::new(
                self.momentum - self.particle.mass2,
                -self.particle.mass * self.particle.width,
            ),
            (false, _) => Complex::new(self.momentum - self.particle.mass2, 0.0),
        }
    }

    /// Evaluate the conjugated propagator denominator.
    ///
    /// This is meant to be used when one wants to conjugate the propagator
    /// without modifying the propagator itself.
    fn denominator_conj(&self) -> Complex<f64> {
        match (self.momentum > 0.0, !self.conjugated) {
            (true, false) => Complex::new(
                self.momentum - self.particle.mass2,
                self.particle.mass * self.particle.width,
            ),
            (true, true) => Complex::new(
                self.momentum - self.particle.mass2,
                -self.particle.mass * self.particle.width,
            ),
            (false, _) => Complex::new(self.momentum - self.particle.mass2, 0.0),
        }
    }

    /// Equivalent to `self * rhs.conj()` without the need to mutate the
    /// other.
    pub(crate) fn mul_conj(&self, rhs: &Self) -> Complex<f64> {
        debug_assert!(
                self.conjugated == rhs.conjugated,
                "Multiplying propagators with the same complex conjugation (after the automatic conjugation)."
            );

        #[allow(clippy::float_cmp)]
        if self.particle == rhs.particle && self.momentum == rhs.momentum {
            let mw2 = (self.particle.mass * self.particle.width).powi(2);
            let numerator = (self.momentum - self.particle.mass2).powi(2) - mw2;
            let denominator = (self.momentum - self.particle.mass2).powi(2) + mw2;

            self.numerator * rhs.numerator.conj() * numerator / denominator.powi(2)
        } else {
            self.numerator
                * rhs.numerator.conj()
                * (self.denominator() * rhs.denominator_conj()).finv()
        }
    }
}

impl<'a> Propagator for Single<'a> {
    fn conj(mut self) -> Self {
        self.numerator = self.numerator.conj();
        self.conjugated = !self.conjugated;
        self
    }

    fn eval(&self) -> Complex<f64> {
        self.numerator * self.denominator().finv()
    }

    fn norm_sqr(&self) -> f64 {
        let propagator_squared = if self.momentum > 0.0 {
            let mw2 = (self.particle.mass * self.particle.width).powi(2);
            let numerator = (self.momentum - self.particle.mass2).powi(2) - mw2;
            let denominator = (self.momentum - self.particle.mass2).powi(2) + mw2;

            numerator / denominator.powi(2)
        } else {
            (self.momentum - self.particle.mass2).powi(-2)
        };

        self.numerator.norm_sqr() * propagator_squared
    }
}

impl<'a> ops::Add<Self> for Single<'a> {
    type Output = Multi<'a>;

    /// Add two propagators, combining them into a [`Multi`].
    ///
    /// ## Panic
    ///
    /// The two propagators should have the same conjugation or this will panic.
    /// The check is only done on debug builds.
    fn add(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.conjugated == rhs.conjugated,
            "When adding two propagators, they must have the same conjugation."
        );

        Multi {
            propagators: vec![self, rhs],
            conjugated: self.conjugated,
        }
    }
}

impl<'a> ops::Sub<Self> for Single<'a> {
    type Output = Multi<'a>;

    /// Take the difference of two propagators, combining them into a
    /// [`Multi`].
    ///
    /// ## Panic
    ///
    /// The two propagators should have the same conjugation or this will panic.
    /// The check is only done on debug builds.
    fn sub(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.conjugated == rhs.conjugated,
            "When adding two propagators, they must have the same conjugation."
        );

        Multi {
            propagators: vec![self, -rhs],
            conjugated: self.conjugated,
        }
    }
}

impl<'a, 'x> ops::Mul<Self> for Single<'a> {
    type Output = Complex<f64>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.conjugated != rhs.conjugated,
            "When multiplying two propagators, they must have the opposite conjugation."
        );

        #[allow(clippy::float_cmp)]
        if self.particle == rhs.particle && self.momentum == rhs.momentum {
            let mw2 = (self.particle.mass * self.particle.width).powi(2);
            let numerator = (self.momentum - self.particle.mass2).powi(2) - mw2;
            let denominator = (self.momentum - self.particle.mass2).powi(2) + mw2;

            self.numerator * rhs.numerator * numerator / denominator.powi(2)
        } else {
            self.numerator * rhs.numerator * (self.denominator() * rhs.denominator()).finv()
        }
    }
}

impl<'a, 'x> ops::Mul<Self> for &'x Single<'a> {
    type Output = Complex<f64>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.conjugated != rhs.conjugated,
            "When multiplying two propagators, they must have the opposite conjugation."
        );

        #[allow(clippy::float_cmp)]
        if self.particle == rhs.particle && self.momentum == rhs.momentum {
            let mw2 = (self.particle.mass * self.particle.width).powi(2);
            let numerator = (self.momentum - self.particle.mass2).powi(2) - mw2;
            let denominator = (self.momentum - self.particle.mass2).powi(2) + mw2;

            self.numerator * rhs.numerator * numerator / denominator.powi(2)
        } else {
            self.numerator * rhs.numerator * (self.denominator() * rhs.denominator()).finv()
        }
    }
}

impl<'a> ops::Mul<f64> for Single<'a> {
    type Output = Self;

    fn mul(mut self, rhs: f64) -> Self::Output {
        self.numerator *= rhs;
        self
    }
}

impl<'a> ops::Mul<Single<'a>> for f64 {
    type Output = Single<'a>;

    fn mul(self, rhs: Single<'a>) -> Self::Output {
        rhs * self
    }
}

impl<'a> ops::MulAssign<f64> for Single<'a> {
    fn mul_assign(&mut self, rhs: f64) {
        self.numerator *= rhs;
    }
}

impl<'a> ops::Mul<Complex<f64>> for Single<'a> {
    type Output = Self;

    fn mul(mut self, rhs: Complex<f64>) -> Self::Output {
        self.numerator *= rhs;
        self
    }
}

impl<'a> ops::Mul<Single<'a>> for Complex<f64> {
    type Output = Single<'a>;

    fn mul(self, rhs: Single<'a>) -> Self::Output {
        rhs * self
    }
}

impl<'a> ops::MulAssign<Complex<f64>> for Single<'a> {
    fn mul_assign(&mut self, rhs: Complex<f64>) {
        self.numerator *= rhs;
    }
}

impl<'a> ops::Div<f64> for Single<'a> {
    type Output = Self;

    fn div(mut self, rhs: f64) -> Self {
        self.numerator /= rhs;
        self
    }
}

impl<'a> ops::DivAssign<f64> for Single<'a> {
    fn div_assign(&mut self, rhs: f64) {
        self.numerator /= rhs;
    }
}

impl<'a> ops::Div<Complex<f64>> for Single<'a> {
    type Output = Self;

    fn div(mut self, rhs: Complex<f64>) -> Self {
        self.numerator /= rhs;
        self
    }
}

impl<'a> ops::DivAssign<Complex<f64>> for Single<'a> {
    fn div_assign(&mut self, rhs: Complex<f64>) {
        self.numerator /= rhs;
    }
}

impl<'a> ops::Neg for Single<'a> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.numerator = -self.numerator;
        self
    }
}
