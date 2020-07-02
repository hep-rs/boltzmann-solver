//! Particle propagators

use crate::model::Particle;
use num::{Complex, One, Zero};
use std::ops;

/// Single particle propagator.
///
/// This isn't designed to be used on its own and instead allows for better
/// handling of multiple propagators.
#[derive(Debug)]
struct SinglePropagator<'a> {
    particle: &'a Particle,
    momentum: f64,
    numerator: Complex<f64>,
    conjugated: bool,
}

impl<'a> SinglePropagator<'a> {
    /// Create a new propagator.
    fn new(p: &'a Particle, s: f64) -> Self {
        SinglePropagator {
            particle: p,
            momentum: s,
            numerator: One::one(),
            conjugated: false,
        }
    }

    /// Complex conjugate the propagator
    fn conj(&mut self) -> &Self {
        self.numerator = self.numerator.conj();
        self.conjugated = !self.conjugated;
        self
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

    /// Evaluate the propagator on its own.
    fn eval(&self) -> Complex<f64> {
        self.numerator * self.denominator().finv()
    }
}

impl<'a> ops::MulAssign<f64> for SinglePropagator<'a> {
    fn mul_assign(&mut self, rhs: f64) {
        self.numerator *= rhs;
    }
}
impl<'a> ops::MulAssign<Complex<f64>> for SinglePropagator<'a> {
    fn mul_assign(&mut self, rhs: Complex<f64>) {
        self.numerator *= rhs;
    }
}

impl<'a> ops::Neg for SinglePropagator<'a> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.numerator = -self.numerator;
        self
    }
}

impl<'a, 'b, 'x, 'y> ops::Mul<&'y SinglePropagator<'b>> for &'x SinglePropagator<'a> {
    type Output = Complex<f64>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: &'y SinglePropagator<'b>) -> Complex<f64> {
        if self.conjugated == rhs.conjugated {
            panic!("Multiplying propagators with the same complex conjugation.");
        }

        #[allow(clippy::float_cmp)]
        if self.particle == rhs.particle && self.momentum == rhs.momentum {
            let mw2 = (self.particle.mass * self.particle.width).powi(2);
            let numerator = (self.momentum - self.particle.mass2).powi(2) - mw2;
            let denominator = (self.momentum - self.particle.mass2).powi(2) + mw2;

            self.numerator * rhs.numerator * Complex::new(numerator / denominator.powi(2), 0.0)
        } else {
            self.numerator * rhs.numerator * (self.denominator() * rhs.denominator()).finv()
        }
    }
}

/// Propagator for a particle.
///
/// This is defined to correctly handle real intermediate states.  Specifically,
/// the propagator is defined such that
///
/// \\begin{equation}
///   P_{i}^*(s) P_{j}(s) = \begin{cases}
///     \frac{(s - m_i^2)^2 - (m_i \Gamma_i)^2}
///          {[(s - m_i^2)^2 + (m_i \Gamma_i)^2]^2} & i = j \\
///     P_i^*(s) P_j(s) & i \neq j
///   \end{cases}
/// \\end{equation}
///
/// where
///
/// \\begin{equation}
///    P_{i}^{-1}(s) \defeq s - m_i^2 + i \theta(s) m_i \Gamma_i.
/// \\end{equation}
///
/// When implementing this, it is always expected that it will be multiplied by
/// complex conjugated propagator at which point the `Propagator` type will
/// produce a floating point.  For example:
///
/// ```rust
/// use boltzman_solver::{prelude::*, utilities::propagator};
///
/// let p1 = Particle::new(0, 125.0, 1e-3);
/// let p2 = Particle::new(1, 350.0, 1.0);
///
/// let m1 = 4.0 * propagator(&p1, 125.0) * propagator(&p1, 125.0).conj();
/// let m2 = 4.0 * propagator(&p1, 125.0) * propagator(&p2, 125.0).conj();
/// let m3 = 4.0 * propagator(&p2, 125.0) * propagator(&p2, 125.0).conj();
///
/// println!("m1 = {}", m1);
/// println!("m2 = {}", m2);
/// println!("m3 = {}", m3);
/// ```
#[derive(Debug)]
pub struct Propagator<'a> {
    propagators: Vec<SinglePropagator<'a>>,
}

impl<'a> Propagator<'a> {
    /// Create a new instance of a [`Propagator`].
    pub(crate) fn new(p: &'a Particle, s: f64) -> Self {
        Propagator {
            propagators: vec![SinglePropagator::new(p, s)],
        }
    }

    /// Complex conjugatet the propagator.
    #[must_use]
    pub fn conj(mut self) -> Self {
        for p in &mut self.propagators {
            p.conj();
        }
        self
    }

    /// Propagator denominator for this particle:
    ///
    /// \\begin{equation}
    ///    P_{i}^{-1}(s) \defeq s - m_i^2 + i \theta(s) m_i \Gamma_i.
    /// \\end{equation}
    #[must_use]
    pub fn eval(&self) -> Complex<f64> {
        self.propagators.iter().map(SinglePropagator::eval).sum()
    }
}

/// Add propagators together
impl<'a> ops::Add<Propagator<'a>> for Propagator<'a> {
    type Output = Self;

    fn add(mut self, mut other: Self) -> Self::Output {
        self.propagators.append(&mut other.propagators);
        self
    }
}

/// Subtract propagators from each other
impl<'a> ops::Sub<Self> for Propagator<'a> {
    type Output = Self;

    fn sub(mut self, mut other: Self) -> Self::Output {
        self.propagators
            .extend(other.propagators.drain(..).map(ops::Neg::neg));
        self
    }
}

/// Multiply propagators with floating points.
impl<'a> ops::Mul<f64> for Propagator<'a> {
    type Output = Self;

    fn mul(mut self, other: f64) -> Self {
        for p in &mut self.propagators {
            *p *= other;
        }
        self
    }
}
/// Multiply propagators with floating points.
impl<'a> ops::Mul<Propagator<'a>> for f64 {
    type Output = Propagator<'a>;

    fn mul(self, mut other: Propagator<'a>) -> Self::Output {
        for p in &mut other.propagators {
            *p *= self;
        }
        other
    }
}

/// Multiply propagators with floating points.
impl<'a> ops::Mul<Complex<f64>> for Propagator<'a> {
    type Output = Self;

    fn mul(mut self, other: Complex<f64>) -> Self {
        for p in &mut self.propagators {
            *p *= other;
        }
        self
    }
}
/// Multiply propagators with floating points.
impl<'a> ops::Mul<Propagator<'a>> for Complex<f64> {
    type Output = Propagator<'a>;

    fn mul(self, mut other: Propagator<'a>) -> Self::Output {
        for p in &mut other.propagators {
            *p *= self;
        }
        other
    }
}

/// Multiply propagators with each other
impl<'a> ops::Mul<Propagator<'a>> for Propagator<'a> {
    type Output = Complex<f64>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, other: Self) -> Complex<f64> {
        let mut result = Complex::zero();

        for pi in &self.propagators {
            for pj in &other.propagators {
                result += pi * pj
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use crate::{model::Particle, utilities::test::complex_approx_eq};
    use num::Complex;
    use std::error;

    #[test]
    fn eval() -> Result<(), Box<dyn error::Error>> {
        let p = Particle::new(0, 10.0, 1e-3);

        for &numerator in &[Complex::new(-3.45, 1.23), Complex::new(1.23, 3.45)] {
            complex_approx_eq(
                numerator * p.propagator(-100.0).eval(),
                numerator * Complex::new(-0.005, 0.0),
                8.0,
                1e-200,
            )?;

            complex_approx_eq(
                numerator * p.propagator(0.0).eval(),
                numerator * Complex::new(-0.01, 0.0),
                8.0,
                1e-200,
            )?;

            complex_approx_eq(
                numerator * p.propagator(100.0).eval(),
                numerator * Complex::new(0.0, -100.0),
                8.0,
                1e-200,
            )?;
        }

        Ok(())
    }

    #[test]
    fn add_sub() -> Result<(), Box<dyn error::Error>> {
        let p1 = Particle::new(0, 10.0, 1e-3);
        let p2 = Particle::new(1, 500.0, 2.0);

        for &s in &[-p1.mass2, -p2.mass2, -50.0, 0.0, 50.0, p1.mass2, p2.mass2] {
            complex_approx_eq(
                (p1.propagator(s) + p2.propagator(s)).eval(),
                p1.propagator(s).eval() + p2.propagator(s).eval(),
                8.0,
                1e-200,
            )?;
            complex_approx_eq(
                (p1.propagator(s) - p2.propagator(s)).eval(),
                p1.propagator(s).eval() - p2.propagator(s).eval(),
                8.0,
                1e-200,
            )?;
            complex_approx_eq(
                (p1.propagator(s) + 12.34 * p2.propagator(s)).eval(),
                p1.propagator(s).eval() + 12.34 * p2.propagator(s).eval(),
                8.0,
                1e-200,
            )?;
            complex_approx_eq(
                (p1.propagator(s) - 12.34 * p2.propagator(s)).eval(),
                p1.propagator(s).eval() - 12.34 * p2.propagator(s).eval(),
                8.0,
                1e-200,
            )?;
        }

        Ok(())
    }

    #[test]
    fn multiplication() -> Result<(), Box<dyn error::Error>> {
        let p1 = Particle::new(0, 10.0, 1e-3);
        let p2 = Particle::new(1, 500.0, 2.0);

        for &s in &[-p1.mass2, -p2.mass2, -50.0, 0.0, 50.0, p1.mass2, p2.mass2] {
            let m11 = p1.propagator(s) * p1.propagator(s).conj();
            let m12 = p1.propagator(s) * p2.propagator(s).conj();
            let m21 = p2.propagator(s) * p1.propagator(s).conj();
            let m22 = p2.propagator(s) * p2.propagator(s).conj();

            let m = (p1.propagator(s) + p2.propagator(s))
                * (p1.propagator(s) + p2.propagator(s)).conj();

            complex_approx_eq(m11 + m12 + m21 + m22, m, 8.0, 1e-200)?
        }

        Ok(())
    }
}
