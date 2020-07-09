//! Module of various useful miscellaneous functions.

// pub(crate) mod clenshaw_curtis;
// pub(crate) mod double_exponential;
pub mod spline;
#[cfg(test)]
pub(crate) mod test;

use crate::constants::PI_5;
use quadrature::{clenshaw_curtis, double_exponential};
use special_functions::bessel;
use std::f64;

pub use special_functions::particle_physics::{kallen_lambda, kallen_lambda_sqrt};

const INTEGRATION_PRECISION: f64 = 1e-10;

/// Return the minimum and maximum value of the Mandelstam variable `$t$` based
/// on the four particle **squared** masses `$m_1^2$`, `$m_2^2$`, `$m_3^2$` and
/// `$m_4^2$`, where particles 1 and 2 are initial state and particles 3 and 4
/// are the final state particles.
///
/// Explicitly, the values are:
///
/// ```math
/// \begin{aligned}
///   t_{\text{min}}
///   &= \frac{(m_1^2 - m_2^2 - m_3^3 + m_4^4)^2}{4 s}
///    - \frac{\lambda^{\frac{1}{2}}(s, m_1^2, m_2^2) \lambda^{\frac{1}{2}}(s, m_3^2, m_4^2)}{s} \\\\
/// t_{\text{max}}
///   &= \frac{(m_1^2 - m_2^2 - m_3^3 + m_4^4)^2}{4 s}
/// \end{aligned}
/// ```
#[must_use]
pub fn t_range(s: f64, m1: f64, m2: f64, m3: f64, m4: f64) -> (f64, f64) {
    debug_assert!(
        s * (1.0 + 5.0 * f64::EPSILON) >= m1 + m2,
        "s cannot be smaller than (m1² + m2²)."
    );
    debug_assert!(
        s * (1.0 + 5.0 * f64::EPSILON) >= m3 + m4,
        "s cannot be smaller than (m3² + m4²)."
    );

    if s == 0.0 {
        return (0.0, 0.0);
    }

    let baseline = (m1 - m2 - m3 + m4).powi(2) / (4.0 * s);
    let delta = kallen_lambda_sqrt(s, m1, m2) * kallen_lambda_sqrt(s, m3, m4) / s;

    (baseline - delta, baseline)
}

/// Integrate the amplitude with respect to the Mandelstam variable `$s$` and
/// `$t$`:
///
/// ```math
/// \int_{s_\text{min}}^{\infty} \int_{t_\text{min}}^{t_\text{max}}
///     \abs{\mathcal{M}(s, t)}^2 \frac{K_1(\sqrt{s} \beta)}{\sqrt{s}}
/// \dd t \dd s
/// ```
///
/// where `$s_\text{min} = \mathop{\text{max}}(m_1^2 + m_2^2, m_3^2 + m_4^2)$`
/// and `$t_{\text{min},\text{max}}$` are determined from [`t_range`].
///
/// The squared masses should be given.
#[must_use]
pub fn integrate_st<F>(amplitude: F, beta: f64, m1: f64, m2: f64, m3: f64, m4: f64) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    // The weighing by the bessel function means  that most of the integral's
    // non-zero values are near the lower bound of the s domain.  We use
    // Clenshaw-Curtis integration for the lower range, and then
    // double-exponential integration for the rest of the range.
    //
    // For values above of s above 8328.95/β², the factor of K₁(√s β)/√s
    // evaluates to 0 within machine precision.  So we use this as the upper
    // bound instead of infinity.

    let s_min = (m1 + m2).max(m3 + m4);
    let s_med = 10.0 * s_min;
    let s_max = (8328.95 / beta.powi(2)).max(10.0 * s_med);

    let s_integrand_0 = |s: f64| {
        // Combination of factors constant w.r.t. t
        let sqrt_s = s.sqrt();
        let s_factors = bessel::k1(sqrt_s * beta) / sqrt_s;

        let (t_min, t_max) = t_range(s, m1, m2, m3, m4);
        let t_integrand = |t: f64| amplitude(s, t) * s_factors;
        clenshaw_curtis::integrate(&t_integrand, t_min, t_max, INTEGRATION_PRECISION).integral
    };

    let s_integrand_1 = |ss: f64| {
        // Remap the interval [s_med, s_max] onto [0, 1].  The following
        // rescaling will cause the numerical integration to sample more heavily
        // from the lower part of the domain (unlike a simple linear rescaling).
        let delta = (s_max - s_med).recip() + 1.0;
        let s = ss / (delta - ss) + s_med;
        let dsdss = delta / (delta - ss).powi(2);

        // Remap the semi-infinite s interval onto [0, 1)
        // let s = ss / (1.0 - ss) + s_min;
        // let dsdss = (ss - 1.0).powi(-2);

        // Combination of factors constant w.r.t. t
        let sqrt_s = s.sqrt();
        let s_factors = bessel::k1(sqrt_s * beta) / sqrt_s * dsdss;

        let (t_min, t_max) = t_range(s, m1, m2, m3, m4);
        let t_integrand = |t: f64| amplitude(s, t) * s_factors;
        clenshaw_curtis::integrate(&t_integrand, t_min, t_max, INTEGRATION_PRECISION).integral
    };

    (clenshaw_curtis::integrate(&s_integrand_0, s_min, s_med, INTEGRATION_PRECISION).integral
        + double_exponential::integrate(&s_integrand_1, 0.0, 1.0, INTEGRATION_PRECISION).integral)
        / (512.0 * PI_5 * beta)
}

#[cfg(test)]
mod tests {
    use crate::utilities::test::approx_eq;
    use std::error;

    #[test]
    fn t_range() -> Result<(), Box<dyn error::Error>> {
        let data = vec![
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 10.0, -10.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 2.0, 0.5, 0.5],
            [2.0, 0.0, 0.0, 0.0, 10.0, -7.9, 0.1],
            [0.0, 2.0, 0.0, 0.0, 2.0, 0.5, 0.5],
            [0.0, 2.0, 0.0, 0.0, 10.0, -7.9, 0.1],
            [0.0, 0.0, 2.0, 0.0, 2.0, 0.5, 0.5],
            [0.0, 0.0, 2.0, 0.0, 10.0, -7.9, 0.1],
            [0.0, 0.0, 0.0, 2.0, 2.0, 0.5, 0.5],
            [0.0, 0.0, 0.0, 2.0, 10.0, -7.9, 0.1],
            [1.0, 2.0, 3.0, 4.0, 8.0, -3.533_323_506, 0.0],
            [1.0, 2.0, 3.0, 4.0, 20.0, -9.219_680_038, 0.0],
        ];

        for &[m1, m2, m3, m4, s, ea, eb] in &data {
            let (ta, tb) = super::t_range(s, m1, m2, m3, m4);
            approx_eq(ta, ea, 8.0, 0.0)?;
            approx_eq(tb, eb, 8.0, 0.0)?;
        }

        Ok(())
    }

    #[allow(clippy::shadow_unrelated)]
    #[test]
    fn integrate_st() -> Result<(), Box<dyn error::Error>> {
        const BETA: f64 = 1e-2;
        const M1: f64 = 1.0;
        const M2: f64 = 10.0;
        const M3: f64 = 2.0;
        const M4: f64 = 20.0;

        let m2 = |_, _| 1.0;
        approx_eq(
            super::integrate_st(m2, BETA, M1, M2, M3, M4),
            2_538.664_748_452_267,
            1.5,
            0.0,
        )?;

        let m2 = |s: f64, _| 1.0 / (s.powi(2) + 1.0);
        approx_eq(
            super::integrate_st(m2, BETA, M1, M2, M3, M4),
            0.001_196_687_768_601_514,
            1.5,
            0.0,
        )?;

        let m2 = |_, t: f64| 1.0 / (t.powi(2) + 1.0);
        approx_eq(
            super::integrate_st(m2, BETA, M1, M2, M3, M4),
            0.678_416_052_166_338_6,
            1.5,
            0.0,
        )?;

        let m2 = |s: f64, t: f64| (s * t) / (s + 1.0) / (t + 1.0);
        approx_eq(
            super::integrate_st(m2, BETA, M1, M2, M3, M4),
            2_540.321_286_335_379,
            1.5,
            0.0,
        )?;

        Ok(())
    }
}

#[cfg(all(test, feature = "nightly"))]
mod benches {
    use test::Bencher;

    const BETA: f64 = 1e-2;
    const M1: f64 = 1.0;
    const M2: f64 = 10.0;
    const M3: f64 = 2.0;
    const M4: f64 = 20.0;

    #[bench]
    fn integrate_st_const(b: &mut Bencher) {
        let m2 = |_, _| 1.0;
        b.iter(|| test::black_box(super::integrate_st(m2, BETA, M1, M2, M3, M4)));
    }

    #[bench]
    fn integrate_st_s_inv(b: &mut Bencher) {
        let m2 = |s: f64, _| 1.0 / (s.powi(2) + 1.0);
        b.iter(|| test::black_box(super::integrate_st(m2, BETA, M1, M2, M3, M4)));
    }

    #[bench]
    fn integrate_st_t_inv(b: &mut Bencher) {
        let m2 = |_, t: f64| 1.0 / (t.powi(2) + 1.0);
        b.iter(|| test::black_box(super::integrate_st(m2, BETA, M1, M2, M3, M4)));
    }

    #[bench]
    fn integrate_st_st(b: &mut Bencher) {
        let m2 = |s: f64, t: f64| (s * t) / (s + 1.0) / (t + 1.0);
        b.iter(|| test::black_box(super::integrate_st(m2, BETA, M1, M2, M3, M4)))
    }
}
