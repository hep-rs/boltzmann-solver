//! Module of various useful miscellaneous functions.

// pub(crate) mod clenshaw_curtis;
// pub(crate) mod double_exponential;
pub mod spline;
#[cfg(test)]
pub(crate) mod test;

use crate::{constants::PI_5, model::Particle};
use num_complex::Complex;
use quadrature::{clenshaw_curtis, double_exponential};
use special_functions::bessel;

const INTEGRATION_PRECISION: f64 = 1e-10;

/// Kallen lambda function:
///
/// \\begin{equation}
///   \lambda(a, b, c) = a^2 + b^2 + c^2 - 2ab - 2ac - 2bc
/// \\end{equation}
///
/// # Example
///
/// ```
/// use boltzmann_solver::utilities::kallen_lambda;
///
/// assert_eq!(kallen_lambda(5.0, 2.0, 0.5), 2.25);
/// assert_eq!(kallen_lambda(1.0, 1.0, 1.0), -3.0);
/// ```
pub fn kallen_lambda(a: f64, b: f64, c: f64) -> f64 {
    a.powi(2) + b.powi(2) + c.powi(2) - 2.0 * (a * b + a * c + b * c)
}

/// Square root of the Kallen lambda function:
///
/// \\begin{equation}
///   \lambda^{\frac{1}{2}}(a, b, c) = \sqrt{a^2 + b^2 + c^2 - 2ab - 2ac - 2bc}
/// \\end{equation}
///
/// This implementation is more precise than taking the square root of
/// [`kallen_lambda`] in cases where the arguments are span several orders of
/// magnitude.
///
/// # Example
///
/// ```
/// use boltzmann_solver::utilities::{kallen_lambda, kallen_lambda_sqrt};
///
/// assert!((kallen_lambda_sqrt(5.0, 2.0, 0.5) - 1.5).abs() < 1e-14);
/// assert!((kallen_lambda(5.0, 2.0, 0.5).sqrt() - kallen_lambda_sqrt(5.0, 2.0, 0.5)).abs() < 1e-14);
/// assert!((kallen_lambda_sqrt(1.0, 1.0, 1.0) - 3f64.sqrt()).abs() < 1e-14);
/// ```
///
/// # Warning
///
/// This function only returns the *absolute value* of the result.
pub fn kallen_lambda_sqrt(a: f64, b: f64, c: f64) -> f64 {
    let max = if a > b { a } else { b };
    let max = if max > c { max } else { c };

    max * kallen_lambda(1.0, b / max, c / max).abs().sqrt()
}

/// Return the minimum and maximum value of the Mandelstam variable \\(t\\)
/// based on the four particle **squared** masses \\(m_1^2\\), \\(m_2^2\\),
/// \\(m_3^2\\) and \\(m_4^2\\), where particles 1 and 2 are initial state and
/// particles 3 and 4 are the final state particles.
///
/// Explicitly, the values are:
///
/// \\begin{equation}\\begin{aligned}
///   t_{\text{min}} &= \frac{(m_1^2 - m_2^2 - m_3^3 + m_4^4)^2}{4 s}
///       - \frac{\lambda^{\frac{1}{2}}(s, m_1^2, m_2^2) \lambda^{\frac{1}{2}}(s, m_3^2, m_4^2)}{s} \\\\
///   t_{\text{max}} &= \frac{(m_1^2 - m_2^2 - m_3^3 + m_4^4)^2}{4 s}
/// \\end{aligned}\\end{equation}
pub fn t_range(s: f64, m1: f64, m2: f64, m3: f64, m4: f64) -> (f64, f64) {
    debug_assert!(s >= m1 + m2, "s must be greater than m1^2 + m2^2.");
    debug_assert!(s >= m3 + m4, "s must be greater than m3^2 + m4^2.");

    if s == 0.0 {
        return (0.0, 0.0);
    }

    let baseline = (m1 - m2 - m3 + m4).powi(2) / (4.0 * s);
    let delta = kallen_lambda_sqrt(s, m1, m2) * kallen_lambda_sqrt(s, m3, m4) / s;

    (baseline - delta, baseline)
}

/// Integrate the amplitude with respect to the Mandelstam variable \\(s\\) and
/// \\(t\\):
///
/// \\begin{equation}
///   \int_{s_\text{min}}^{\infty} \int_{t_\text{min}}^{t_\text{max}}
///       \abs{\mathcal{M}(s, t)}^2 \frac{K_1(\sqrt{s} \beta)}{\sqrt{s}}
///   \dd t \dd s
/// \\end{equation}
///
/// where \\(s_\text{min} = \mathop{\text{max}}(m_1^2 + m_2^2, m_3^2 + m_4^2)\\)
/// and \\(t_{\text{min},\text{max}}\\) are determined from [`t_range`].
///
/// The squared masses should be given.
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
    let s_max = 8328.95 / beta.powi(2);

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

/// Propagator with squared momentum `q2` involving particle `p`, defined as:
///
/// \\begin{equation}
///   \mathcal{P}_{p}(q^2) \defeq \frac{1}{q^2 - m_p^2 + i m_p \Gamma_p}
/// \\end{equation}
///
/// # To-do
///
/// Adjust the propagator to take into account particles in the thermal bath:
///
/// \\begin{equation}\begin{aligned}
///   S(q^2, m) &= \left[\frac{i}{q^2 - m^2 + i0} - 2 \pi n \delta(q^2 - m^2)] (\slashed p + m) \\\\
///   D(q^2, m) &= \left[\frac{i}{q^2 - m^2 + i0} + 2 \pi n \delta(q^2 - m^2)]
/// \\end{aligned}\\end{equation}
pub fn propagator(q2: f64, p: &Particle) -> Complex<f64> {
    // TODO: Implement Dirac delta?
    1.0 / (q2 - p.mass2 + Complex::i() * p.mass * p.width)
}

#[cfg(test)]
mod tests {
    use crate::utilities::test::*;

    #[test]
    fn t_range() {
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
            approx_eq(ta, ea, 8.0, 0.0);
            approx_eq(tb, eb, 8.0, 0.0);
        }
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn kallen_lambda() {
        let (a, b, c) = (1.0, 2.0, 3.0);
        assert_eq!(super::kallen_lambda(a, b, c), -8.0);

        assert_eq!(super::kallen_lambda(a, b, c), super::kallen_lambda(a, c, b));
        assert_eq!(super::kallen_lambda(a, b, c), super::kallen_lambda(b, a, c));
        assert_eq!(super::kallen_lambda(a, b, c), super::kallen_lambda(b, c, a));
        assert_eq!(super::kallen_lambda(a, b, c), super::kallen_lambda(c, a, b));
        assert_eq!(super::kallen_lambda(a, b, c), super::kallen_lambda(c, b, a));
    }

    #[test]
    fn integrate_st() {
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
        );

        let m2 = |s: f64, _| 1.0 / (s.powi(2) + 1.0);
        approx_eq(
            super::integrate_st(m2, BETA, M1, M2, M3, M4),
            0.001_196_687_768_601_514,
            1.5,
            0.0,
        );

        let m2 = |_, t: f64| 1.0 / (t.powi(2) + 1.0);
        approx_eq(
            super::integrate_st(m2, BETA, M1, M2, M3, M4),
            0.678_416_052_166_338_6,
            1.5,
            0.0,
        );

        let m2 = |s: f64, t: f64| (s * t) / (s + 1.0) / (t + 1.0);
        approx_eq(
            super::integrate_st(m2, BETA, M1, M2, M3, M4),
            2_540.321_286_335_379,
            1.5,
            0.0,
        );
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
