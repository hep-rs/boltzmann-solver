//! Particle propagators

use num::Complex;

mod multi;
mod single;

pub use multi::Multi;
pub use single::Single;

/// Trait to handle particle proapgators.
///
/// This is defined to correctly handle real intermediate states.  Specifically,
/// the propagator is defined such that
///
/// ```math
/// P_{i}^*(s) P_{j}(s)
/// = \begin{cases}
///   \frac{(s - m_i^2)^2 - (m_i \Gamma_i)^2}
///        {[(s - m_i^2)^2 + (m_i \Gamma_i)^2]^2} & i = j \\
///   P_i^*(s) P_j(s) & i \neq j
/// \end{cases}
/// ```
///
/// where
///
/// ```math
/// P_{i}^{-1}(s) \defeq s - m_i^2 + i \theta(s) m_i \Gamma_i.
/// ```
///
/// When implementing this, it is always expected that it will be multiplied by
/// complex conjugated propagator at which point the `Propagator` type will
/// produce a floating point.  For example:
///
/// ```rust
/// use boltzmann_solver::prelude::*;
///
/// let p1 = Particle::new(0, 125.0, 1e-3);
/// let p2 = Particle::new(1, 350.0, 1.0);
///
/// let m11 = p1.propagator(125.0) * p1.propagator(125.0).conj();
/// let m12 = p1.propagator(125.0) * p2.propagator(125.0).conj();
/// let m22 = p2.propagator(125.0) * p2.propagator(125.0).conj();
///
/// let m = (p1.propagator(125.0) + p2.propagator(125.0)) * (p1.propagator(125.0) + p2.propagator(125.0)).conj();
///
/// assert!((m11 + m12 + m12.conj() + m22 - m).re.abs() < f64::EPSILON);
/// assert!((m11 + m12 + m12.conj() + m22 - m).im.abs() < f64::EPSILON);
/// ```
pub trait Propagator {
    /// Return the complex conjugate the propagator.
    fn conj(self) -> Self;

    /// Evaluate the propagator.
    ///
    /// This is generally not used explicity as the amplitude itself is rarely
    /// evaluated, only the squared amplitude.
    fn eval(&self) -> Complex<f64>;

    /// Compute the norm squared of the propagator.
    ///
    /// This is equivalent to having `p * p.conj()`, but does not require the
    /// allocation of a second propagator.
    #[must_use]
    fn norm_sqr(&self) -> f64;
}

#[cfg(test)]
mod tests {
    use crate::{
        model::{particle::Propagator, Particle},
        utilities::test::complex_approx_eq,
    };
    use num::Complex;
    use std::error;

    // TODO: These tests should be restructed to be more coherant and more
    // readable.
    //
    // Also have to test the following:
    // - [ ] Single::denominator (all variants)
    // - [ ] Single::denominator_conj
    // - [ ] Single::mul_conj
    // - [ ] Single::mul_assign<f64>
    // - [ ] Single::mul_assign<Complex<f64>>
    // - [ ] Single::add<Multi>
    // - [ ] Single::sub<Multi>
    // - [ ] Single::mul<Multi>
    // - [ ] Single::div<f64>
    // - [ ] Single::div_assign<f64>
    // - [ ] Single::div<Complex<f64>>
    // - [ ] Single::div_assign<Complex<f64>>
    // - [ ] Multi::norm_sqr
    // - [ ] Multi::add<Multi>
    // - [ ] Multi::sub<Multi>
    // - [ ] Multi::mul<Single>
    // - [ ] Multi::mul<f64>
    // - [ ] Multi::mul_assign<f64>
    // - [ ] Multi::mul<Complex<f64>>
    // - [ ] Multi::mul_assign<Complex<f64>>
    // - [ ] Multi::div<f64>
    // - [ ] Multi::div_assign<f64>
    // - [ ] Multi::div<Complex<f64>>
    // - [ ] Multi::div_assign<Complex<f64>>
    // - [ ] Multi::neg

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
        let p3 = Particle::new(1, 1000.0, 3.0);

        // Adding two propagators
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

            complex_approx_eq(
                (p1.propagator(s) + p2.propagator(s) + p3.propagator(s)).eval(),
                p1.propagator(s).eval() + p2.propagator(s).eval() + p3.propagator(s).eval(),
                8.0,
                1e-200,
            )?;
            complex_approx_eq(
                (p1.propagator(s) - p2.propagator(s) - p3.propagator(s)).eval(),
                p1.propagator(s).eval() - p2.propagator(s).eval() - p3.propagator(s).eval(),
                8.0,
                1e-200,
            )?;
            complex_approx_eq(
                (p1.propagator(s)
                    + 12.34 * p2.propagator(s)
                    + Complex::new(2.0, 3.0) * p3.propagator(s))
                .eval(),
                p1.propagator(s).eval()
                    + 12.34 * p2.propagator(s).eval()
                    + Complex::new(2.0, 3.0) * p3.propagator(s).eval(),
                8.0,
                1e-200,
            )?;
            complex_approx_eq(
                (p1.propagator(s)
                    - 12.34 * p2.propagator(s)
                    - Complex::new(2.0, 3.0) * p3.propagator(s))
                .eval(),
                p1.propagator(s).eval()
                    - 12.34 * p2.propagator(s).eval()
                    - Complex::new(2.0, 3.0) * p3.propagator(s).eval(),
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
            // Check that `$P_1^* P_2 = (P_2^* P_1)^*$`
            complex_approx_eq(
                p1.propagator(s) * p2.propagator(s).conj(),
                (p1.propagator(s).conj() * p2.propagator(s)).conj(),
                8.0,
                1e-200,
            )?;

            let m11 = p1.propagator(s) * p1.propagator(s).conj();
            let m12 = p1.propagator(s) * p2.propagator(s).conj();
            let m21 = p2.propagator(s) * p1.propagator(s).conj();
            let m22 = p2.propagator(s) * p2.propagator(s).conj();

            let m = (p1.propagator(s) + p2.propagator(s))
                * (p1.propagator(s) + p2.propagator(s)).conj();

            complex_approx_eq(m11 + m12 + m21 + m22, m, 8.0, 1e-200)?;
        }

        Ok(())
    }

    #[test]
    fn norm_sqr() -> Result<(), Box<dyn error::Error>> {
        let p1 = Particle::new(0, 10.0, 1e-3);
        let p2 = Particle::new(1, 500.0, 2.0);

        for &s in &[-p1.mass2, -p2.mass2, -50.0, 0.0, 50.0, p1.mass2, p2.mass2] {
            complex_approx_eq(
                p1.propagator(s).conj() * p1.propagator(s),
                Complex::new(p1.propagator(s).norm_sqr(), 0.0),
                6.0,
                1e-200,
            )?;
        }

        Ok(())
    }
    //
}
