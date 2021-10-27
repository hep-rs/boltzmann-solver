use super::M_BETA_THRESHOLD;
use crate::{constants::PI_5, prelude::Particle};
use ndarray::Array1;
use quadrature::{clenshaw_curtis, double_exponential};
use special_functions::{bessel, particle_physics::kallen_lambda_sqrt};
use std::f64;

const INTEGRATION_PRECISION: f64 = 1e-10;

/// Return the minimum and maximum value of the Mandelstam variable `$t$` based
/// on the four particle masses with particles 1 and 2 being initial state and
/// particles 3 and 4 being the final state particles.
///
/// Explicitly, the range of `$t$` values are:
///
/// ```math
/// \begin{aligned}
///   t &= \frac{1}{2} \left[
///       (m_1^2 + m_2^2 + m_3^2 + m_4^2) - s - \frac{(m_1^2 - m_2^2)(m_3^2 - m_4^2)}{s}
///     \right] \\
///     &\quad + \frac{\lambda^{\frac{1}{2}}(s, m_1^2, m_2^2) \lambda^{\frac{1}{2}}(s, m_3^2, m_4^2)}{2 s} \cos \theta
/// \end{aligned}
/// ```
#[must_use]
pub fn t_range(s: f64, p1: &Particle, p2: &Particle, p3: &Particle, p4: &Particle) -> (f64, f64) {
    const THRESHOLD: f64 = 1e-3;

    // debug_assert!(
    //     s * (1.0 + 10.0 * f64::EPSILON) >= (p1.mass + p2.mass).powi(2),
    //     "s cannot be smaller than (m1 + m2)² (s = {:e}, (m1 + m2)² = {:e}).",
    //     s,
    //     (p1.mass + p2.mass).powi(2)
    // );
    // debug_assert!(
    //     s * (1.0 + 10.0 * f64::EPSILON) >= (p3.mass + p4.mass - 5.0 * f64::EPSILON).powi(2),
    //     "s cannot be smaller than (m3 + m4)² (s = {:e}, (m3 + m4)² = {:e}).",
    //     s,
    //     (p3.mass + p4.mass).powi(2)
    // );

    if s == 0.0 {
        return (0.0, 0.0);
    }

    let (m1, m2, m3, m4) = (p1.mass2, p2.mass2, p3.mass2, p4.mass2);
    let (x1, x2, x3, x4) = (m1 / s, m2 / s, m3 / s, m4 / s);

    let x_max = x1.max(x2).max(x3).max(x4);

    match (
        x1 / x_max < THRESHOLD,
        x2 / x_max < THRESHOLD,
        x3 / x_max < THRESHOLD,
        x4 / x_max < THRESHOLD,
    ) {
        // // 2 small values
        // (true, true, false, false) => {
        //     let sqrt34 = kallen_lambda_sqrt(1.0, x3, x4);
        //     (
        //         0.5 * s
        //             * (sqrt34 * (2.0 * x1 * x2 + x1 + x2 - 1.0)
        //                 + x3
        //                 + x2 * (x3 - x4 + 1.0)
        //                 + x4
        //                 + x1 * (-x3 + x4 + 1.0)
        //                 - 1.0),
        //         0.5 * s
        //             * (sqrt34 * (-x2 - x1 * (2.0 * x2 + 1.0) + 1.0)
        //                 + x3
        //                 + x2 * (x3 - x4 + 1.0)
        //                 + x4
        //                 + x1 * (-x3 + x4 + 1.0)
        //                 - 1.0),
        //     )
        // }
        // (true, false, true, false) => {
        //     let ox2 = 1.0 / (x2 - 1.0);
        //     let ox24 = ox2 / (x4 - 1.0);
        //     (
        //         s * (ox24 * ((x2 - 1.0).powi(2) - x1 * (x2 + 1.0)) * x3 + x2 * (x3 + 1.0)
        //             - ox2 * (x1 * (x2 * x3 + 1.0) + ((x1 - x2 + 2.0) * x2 - 1.0) * x4)
        //             - 1.0),
        //         s * (ox24 * (x1 * (x2 + 1.0) - (x2 - 1.0).powi(2)) * x3
        //             + x3
        //             + ox2 * (x1 * (x2 + x3) - x1 * x4)),
        //     )
        // }
        // (true, false, false, true) => {
        //     let ox2 = 1.0 / (x2 - 1.0);
        //     let ox23 = ox2 / (x3 - 1.0);
        //     (
        //         x2 + x3 - 1.0 + ox2 * x1 * (x3 - 1.0) + ox23 * (x2 * x2 - (x1 + 2.0) * x2 - x1 * x3 + 1.0) * x4,
        //         - ox23 * (x1 * x2 * x3 * x3) + ox3 * x2 * x3 * x3 +
        //     )
        // }
        // (false, true, true, false) => {
        //     (
        //         s * (

        //         ),
        //         s * ()
        //     )
        // }
        // (false, true, false, true) => {
        //     (
        //         s * (
        //             -x3 * x1 + x1 + x3 + x2 * (x3 * (x1 * (x3 - x4 - 1.0) - 1.0) - x4 + 1.0) / (x1 - 1.0) / (x3 - 1.0) + (x1 * x3 - 1.0) * x4 / (x3 - 1.0) - 1.0
        //         ),
        //         s * (
        //             x2 * (x1 - x3) * (x3 - 1.0) + (x1 * (-x1 + x2 + 1.0) + (x1 + x2 - 1.0) * x3) * x4
        //         ) / (x1 - 1.0) / (x3 - 1.0)
        //     )
        // }
        // (false, false, true, true) => {
        //     let sqrt12 = kallen_lambda_sqrt(1.0, x1, x2);
        //     (
        //         0.5 * s
        //             * (x3 + x2 * (x3 - x4 + 1.0) + x4 + x1 * (-x3 + x4 + 1.0) + sqrt12 * (2.0 * x3 * x4 + x3 + x4 - 1) - 1.0),
        //         0.5 * s
        //             * (x3 + x2 * (x3 - x4 + 1.0) + x4 + x1 * (-x3 + x4 + 1.0) + sqrt12 * (-x4 - x3 * (2.0 * x3 + 1.0) + 1.0) + 1.0),
        //     )
        // }
        // 3 small values
        // (true, true, true, false) => {
        //     let prefactor = x3 / (x4 - 1.0);
        //     (
        //         s * (x2 * x1 + x1 + x2 - (x1 + 1.0) * x2 * x4 + x4 - 1.0
        //             + prefactor * (x2 * (x4 + 1.0) * x1 + x1 + x2 * x4 - 1.0)),
        //         s * (x1 * (x2 * (x4 - 1.0) + x4)
        //             + prefactor * (-((x1 + 1.0) * x2) - x1 * (x2 + 1.0) * x4 + x4)),
        //     )
        // }
        // (true, true, false, true) => {
        //     let prefactor = x4 / (x3 - 1.0);
        //     (
        //         s * (x2 - x1 * (x2 + 1.0) * (x3 - 1.0) + x3 - 1.0
        //             + prefactor * (x1 * (x3 + 1.0) * x2 + x2 + x1 * x3 - 1.0)),
        //         s * (x2 * (x1 * (x3 - 1.0) + x3)
        //             + prefactor * (-x1 * (x2 + 1.0) - (x1 + 1.0) * x2 * x3 + x3)),
        //     )
        // }
        // (true, false, true, true) => {
        //     let prefactor = x1 / (x2 - 1.0);
        //     (
        //         s * (x2 + x3 - (x2 - 1.0) * (x3 + 1.0) * x4 - 1.0
        //             + prefactor * (x3 + (x3 + x2 * (x3 + 1.0)) * x4 - 1.0)),
        //         s * (x3 * (x2 + (x2 - 1.0) * x4)
        //             + prefactor * (-x3 * (x4 + 1.0) * x2 + x2 - (x3 + 1.0) * x4)),
        //     )
        // }
        // (false, true, true, true) => {
        //     let prefactor = x2 / (x1 - 1.0);
        //     (
        //         s * (-x3 * (x4 + 1.0) * x1 + x1 + x4 + x3 * (x4 + 1.0) - 1.0
        //             + prefactor * (x4 + x3 * (x4 + x1 * (x4 + 1.0)) - 1.0)),
        //         s * ((x1 + (x1 - 1.0) * x3) * x4
        //             + prefactor * (-((x3 + 1.0) * x4 * x1) + x1 - x3 * (x4 + 1.0))),
        //     )
        // }
        _ => {
            let baseline = x1 + x2 + x3 + x4 - 1.0 - (x1 - x2) * (x3 - x4);
            let sqrt12 = kallen_lambda_sqrt(1.0, x1, x2);
            let sqrt34 = kallen_lambda_sqrt(1.0, x3, x4);
            let cosine = sqrt12 * sqrt34;
            (0.5 * s * (baseline - cosine), 0.5 * s * (baseline + cosine))
        }
    }
}

/// Integrate the function of `$t$` over its range.
///
/// This function subdivides the integration domain to focus around the particle
/// masses where resonances are likely to appear.
#[must_use]
pub fn integrate_t<F>(
    integrand: F,
    s: f64,
    p1: &Particle,
    p2: &Particle,
    p3: &Particle,
    p4: &Particle,
) -> f64
where
    F: Fn(f64) -> f64,
{
    let (ta, tb) = t_range(s, p1, p2, p3, p4);
    let mut t_divs = vec![ta, tb];

    for m_squared in [p1, p2, p3, p4].iter().map(|p| p.mass2) {
        t_divs.extend(&[m_squared * 0.98, m_squared * 1.02]);
    }
    t_divs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    t_divs.dedup();
    t_divs.retain(|&t| ta <= t && t <= tb);

    let mut integral = 0.0;

    for (&t1, &t2) in t_divs[..t_divs.len() - 1].iter().zip(&t_divs[1..]) {
        integral += clenshaw_curtis::integrate(&integrand, t1, t2, INTEGRATION_PRECISION).integral;
    }

    integral
}

#[must_use]
fn s_divisions(beta: f64, p1: &Particle, p2: &Particle, p3: &Particle, p4: &Particle) -> Vec<f64> {
    // The weighing by the bessel function means  that most of the integral's
    // non-zero values are near the lower bound of the s domain.  We use
    // Clenshaw-Curtis integration for the lower range, and then
    // double-exponential integration for the rest of the range.

    let s_min = (p1.mass + p2.mass).max(p3.mass + p4.mass).powi(2);
    // For values above of √s β above 700, the factor of K₁(√s β)/√s β evaluates
    // to 0 within machine precision.  So we use this as the upper bound instead
    // of infinity.
    let s_max = (1e3 / beta.powi(2)).max(1e1 * s_min);

    let mut s_divs = Array1::geomspace(if s_min == 0.0 { 1e-50 } else { s_min }, s_max, 4)
        .unwrap()
        .into_raw_vec();
    s_divs[0] = s_min;
    s_divs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    s_divs
}

/// Integrate the amplitude with respect to the Mandelstam variable `$s$` and
/// `$t$`:
///
/// ```math
/// \int_{s_\text{min}}^{\infty} \int_{t_\text{min}}^{t_\text{max}}
///     \abs{\scM(s, t)}^2 \frac{K_1(\sqrt{s} \beta)}{\sqrt{s}}
/// \dd t \dd s
/// ```
///
/// where `$s_\text{min} = \mathop{\text{max}}((m_1 + m_2)^2, (m_3 + m_4)^2)$`
/// and `$t_{\text{min},\text{max}}$` are determined from [`t_range`].
#[must_use]
pub fn integrate_st<F>(
    amplitude: F,
    beta: f64,
    p1: &Particle,
    p2: &Particle,
    p3: &Particle,
    p4: &Particle,
) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    let s_divs = s_divisions(beta, p1, p2, p3, p4);
    let s_max = *s_divs.last().unwrap();

    // Integrand of `s` for regular Clenshaw-Curtis integration
    let s_integrand_cc = |s: f64| {
        // Combination of factors constant w.r.t. t
        let sqrt_s = s.sqrt();
        let s_factors = bessel::k1(sqrt_s * beta) / (sqrt_s * beta);

        // let (t_min, t_max) = t_range(s, m1, m2, m3, m4);
        let t_integrand = |t: f64| amplitude(s, t);
        integrate_t(t_integrand, s, p1, p2, p3, p4) * s_factors
        // clenshaw_curtis::integrate(&t_integrand, t_min, t_max, INTEGRATION_PRECISION).integral
    };

    // Integrand optimized for the last upper interval intended to be used with
    // a double exponential
    let s_integrand_de = |ss: f64| {
        // Remap the interval [s_med, s_max] onto [0, 1].  The following
        // rescaling will cause the numerical integration to sample more heavily
        // from the lower part of the domain (unlike a simple linear rescaling).
        // let delta = (s_max - s_med).recip() + 1.0;
        // let s = ss / (delta - ss) + s_med;
        // let dsdss = delta / (delta - ss).powi(2);

        // Remap the semi-infinite s interval onto [0, 1)
        let s = ss / (1.0 - ss) + s_max;
        let dsdss = (ss - 1.0).powi(-2);

        // Combination of factors constant w.r.t. t
        let sqrt_s = s.sqrt();
        let s_factors = bessel::k1(sqrt_s * beta) / (sqrt_s * beta) * dsdss;

        let t_integrand = |t: f64| amplitude(s, t);
        integrate_t(t_integrand, s, p1, p2, p3, p4) * s_factors
    };

    let mut integral = 0.0;

    for (&s1, &s2) in s_divs.iter().zip(&s_divs[1..]) {
        integral +=
            clenshaw_curtis::integrate(&s_integrand_cc, s1, s2, INTEGRATION_PRECISION).integral;
    }

    integral +=
        double_exponential::integrate(&s_integrand_de, 0.0, 1.0, INTEGRATION_PRECISION).integral;

    integral / (512.0 * PI_5)
}

/// Integrate the amplitude with respect to the Mandelstam variable `$s$` and
/// `$t$`, pre-divided by the number density of heavy particles.
///
/// A particle is deemed heavy when `$b \beta$` is greater than a particular
/// threshold.
#[must_use]
pub fn integrate_st_on_n<F>(
    amplitude: F,
    beta: f64,
    p1: &Particle,
    p2: &Particle,
    p3: &Particle,
    p4: &Particle,
) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    match (
        p1.mass * beta >= M_BETA_THRESHOLD,
        p2.mass * beta >= M_BETA_THRESHOLD,
        p3.mass * beta >= M_BETA_THRESHOLD,
        p4.mass * beta >= M_BETA_THRESHOLD,
    ) {
        // No heavy particle
        (false, false, false, false) => integrate_st(amplitude, beta, p1, p2, p3, p4),
        // One heavy particle
        (true, false, false, false) => integrate_st_on_n1(amplitude, beta, p1, p2, p3, p4, p1),
        (false, true, false, false) => integrate_st_on_n1(amplitude, beta, p1, p2, p3, p4, p2),
        (false, false, true, false) => integrate_st_on_n1(amplitude, beta, p1, p2, p3, p4, p3),
        (false, false, false, true) => integrate_st_on_n1(amplitude, beta, p1, p2, p3, p4, p4),
        // Two heavy particles
        (true, true, false, false) => integrate_st_on_n2(amplitude, beta, p1, p2, p3, p4, p1, p2),
        (true, false, true, false) => integrate_st_on_n2(amplitude, beta, p1, p2, p3, p4, p1, p3),
        (true, false, false, true) => integrate_st_on_n2(amplitude, beta, p1, p2, p3, p4, p1, p4),
        (false, true, true, false) => integrate_st_on_n2(amplitude, beta, p1, p2, p3, p4, p2, p3),
        (false, true, false, true) => integrate_st_on_n2(amplitude, beta, p1, p2, p3, p4, p2, p4),
        (false, false, true, true) => integrate_st_on_n2(amplitude, beta, p1, p2, p3, p4, p3, p4),
        // Three heavy particles
        (true, true, true, false) => {
            integrate_st_on_n3(amplitude, beta, p1, p2, p3, p4, p1, p2, p3)
        }
        (true, true, false, true) => {
            integrate_st_on_n3(amplitude, beta, p1, p2, p3, p4, p1, p2, p4)
        }
        (true, false, true, true) => {
            integrate_st_on_n3(amplitude, beta, p1, p2, p3, p4, p1, p3, p4)
        }
        (false, true, true, true) => {
            integrate_st_on_n3(amplitude, beta, p1, p2, p3, p4, p2, p3, p4)
        }
        // Four heavy particles
        (true, true, true, true) => integrate_st_on_n4(amplitude, beta, p1, p2, p3, p4),
    }
}

/// Integrate the over `$s$` and `$t$` divided by the heavy particle `h1`.
///
/// Uses the approximation
///
/// ```math
/// \frac{K_1(\sqrt s \beta)}{\sqrt s \beta} \frac{1}{n_a^{(0)}}
/// = \left( \sqrt{\frac{8}{\pi}} \zeta(3) \right) \frac{\sqrt{\pi / 2}}{g_a}
///   \frac{e^{\beta (m_a - \sqrt{s})}}{(\sqrt{m_a} \sqrt[4]{s} \beta)^3}
/// ```
#[must_use]
fn integrate_st_on_n1<F>(
    amplitude: F,
    beta: f64,
    p1: &Particle,
    p2: &Particle,
    p3: &Particle,
    p4: &Particle,
    h1: &Particle,
) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    let s_divs = s_divisions(beta, p1, p2, p3, p4);
    let s_max = *s_divs.last().unwrap();

    // Integrand of `s` for regular Clenshaw-Curtis integration
    let s_integrand_cc = |s: f64| {
        // Combination of factors constant w.r.t. t
        let sqrt_s = s.sqrt();
        // 2 ζ(3) ≅ 2.4041138063191885
        let s_factors = (2.404_113_806_319_188_5 / h1.degrees_of_freedom())
            * f64::exp((h1.mass - sqrt_s) * beta)
            / ((h1.mass * sqrt_s).sqrt() * beta).powi(3);

        let t_integrand = |t: f64| amplitude(s, t) * s_factors;
        integrate_t(t_integrand, s, p1, p2, p3, p4)
    };

    // Integrand optimized for the last upper interval intended to be used with
    // a double exponential
    let s_integrand_de = |ss: f64| {
        // Remap the interval [s_med, s_max] onto [0, 1].  The following
        // rescaling will cause the numerical integration to sample more heavily
        // from the lower part of the domain (unlike a simple linear rescaling).
        // let delta = (s_max - s_med).recip() + 1.0;
        // let s = ss / (delta - ss) + s_med;
        // let dsdss = delta / (delta - ss).powi(2);

        // Remap the semi-infinite s interval onto [0, 1)
        let s = ss / (1.0 - ss) + s_max;
        let dsdss = (ss - 1.0).powi(-2);

        // Combination of factors constant w.r.t. t
        let sqrt_s = s.sqrt();
        let s_factors = (2.404_113_806_319_188_5 / h1.degrees_of_freedom())
            * f64::exp((h1.mass - sqrt_s) * beta)
            / ((h1.mass * sqrt_s).sqrt() * beta).powi(3)
            * dsdss;

        let t_integrand = |t: f64| amplitude(s, t) * s_factors;
        integrate_t(t_integrand, s, p1, p2, p3, p4)
    };

    let mut integral = 0.0;

    for (&s1, &s2) in s_divs.iter().zip(&s_divs[1..]) {
        integral +=
            clenshaw_curtis::integrate(&s_integrand_cc, s1, s2, INTEGRATION_PRECISION).integral;
    }

    integral +=
        double_exponential::integrate(&s_integrand_de, 0.0, 1.0, INTEGRATION_PRECISION).integral;

    integral / (512.0 * PI_5)
}

/// Integrate the over `$s$` and `$t$` divided by the heavy particles `h1` and `h2`.
///
/// Uses the approximation
///
/// ```math
/// \frac{K_1(\sqrt s \beta)}{\sqrt s \beta} \frac{1}{n_a^{(0)} n_b^{(0)}}
/// = \left( \sqrt{\frac{8}{\pi}} \zeta(3) \right)^2 \frac{\sqrt{\pi / 2}}{g_a g_b}
///   \frac{e^{\beta (m_a + m_b - \sqrt{s})}}{\left(\sqrt{m_a m_b} \sqrt[4]{s} \beta^\frac{3}{2}\right)^3} \\
/// ```
#[allow(clippy::too_many_arguments)]
#[must_use]
fn integrate_st_on_n2<F>(
    amplitude: F,
    beta: f64,
    p1: &Particle,
    p2: &Particle,
    p3: &Particle,
    p4: &Particle,
    h1: &Particle,
    h2: &Particle,
) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    let s_divs = s_divisions(beta, p1, p2, p3, p4);
    let s_max = *s_divs.last().unwrap();

    // Integrand of `s` for regular Clenshaw-Curtis integration
    let s_integrand_cc = |s: f64| {
        // Combination of factors constant w.r.t. t
        let sqrt_s = s.sqrt();
        // 4 √(2 / π) ζ(3)² ≅ 4.611583817377448
        let s_factors = (4.611_583_817_377_448
            / (h1.degrees_of_freedom() * h2.degrees_of_freedom()))
            * f64::exp((h1.mass + h2.mass - sqrt_s) * beta)
            / ((h1.mass * h2.mass * sqrt_s * beta.powi(3)).sqrt()).powi(3);

        let t_integrand = |t: f64| amplitude(s, t) * s_factors;
        integrate_t(t_integrand, s, p1, p2, p3, p4)
    };

    // Integrand optimized for the last upper interval intended to be used with
    // a double exponential
    let s_integrand_de = |ss: f64| {
        // Remap the interval [s_med, s_max] onto [0, 1].  The following
        // rescaling will cause the numerical integration to sample more heavily
        // from the lower part of the domain (unlike a simple linear rescaling).
        // let delta = (s_max - s_med).recip() + 1.0;
        // let s = ss / (delta - ss) + s_med;
        // let dsdss = delta / (delta - ss).powi(2);

        // Remap the semi-infinite s interval onto [0, 1)
        let s = ss / (1.0 - ss) + s_max;
        let dsdss = (ss - 1.0).powi(-2);

        // Combination of factors constant w.r.t. t
        let sqrt_s = s.sqrt();
        let s_factors = (4.611_583_817_377_448
            / (h1.degrees_of_freedom() * h2.degrees_of_freedom()))
            * f64::exp((h1.mass + h2.mass - sqrt_s) * beta)
            / ((h1.mass * h2.mass * sqrt_s).sqrt() * beta.powf(1.5)).powi(3)
            * dsdss;

        let t_integrand = |t: f64| amplitude(s, t) * s_factors;
        integrate_t(t_integrand, s, p1, p2, p3, p4)
    };

    let mut integral = 0.0;

    for (&s1, &s2) in s_divs.iter().zip(&s_divs[1..]) {
        integral +=
            clenshaw_curtis::integrate(&s_integrand_cc, s1, s2, INTEGRATION_PRECISION).integral;
    }

    integral +=
        double_exponential::integrate(&s_integrand_de, 0.0, 1.0, INTEGRATION_PRECISION).integral;

    integral / (512.0 * PI_5)
}

/// Integrate the over `$s$` and `$t$` divided by the heavy particles `h1`, `h2`
/// and `h3`.
///
/// Uses the approximation
///
/// ```math
/// \frac{K_1(\sqrt s \beta)}{\sqrt s \beta} \frac{1}{n_a^{(0)} n_b^{(0)} n_c^{(0)}}
/// = \left( \sqrt{\frac{8}{\pi}} \zeta(3) \right)^3 \frac{\sqrt{\pi / 2}}{g_a g_b g_c}
///   \frac{e^{\beta (m_a + m_b + m_c - \sqrt{s})}}{\left(\sqrt{m_a m_b m_c} \sqrt[4]{s} \beta^2\right)^3} \\
/// ```
#[allow(clippy::too_many_arguments)]
#[must_use]
fn integrate_st_on_n3<F>(
    amplitude: F,
    beta: f64,
    p1: &Particle,
    p2: &Particle,
    p3: &Particle,
    p4: &Particle,
    h1: &Particle,
    h2: &Particle,
    h3: &Particle,
) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    let mut s_divs = s_divisions(beta, p1, p2, p3, p4);
    s_divs.push((h1.mass + h2.mass + h3.mass).powi(2));
    s_divs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let s_max = *s_divs.last().unwrap();

    // Integrand of `s` for regular Clenshaw-Curtis integration
    let s_integrand_cc = |s: f64| {
        // Combination of factors constant w.r.t. t
        let sqrt_s = s.sqrt();
        // 16 ζ(3)³ / π ≅ 8.845964466739566
        let s_factors = (8.845_964_466_739_566
            / (h1.degrees_of_freedom() * h2.degrees_of_freedom() * h3.degrees_of_freedom()))
            * f64::exp((h1.mass + h2.mass + h3.mass - sqrt_s) * beta)
            / ((h1.mass * h2.mass * h3.mass * sqrt_s).sqrt() * beta.powi(2)).powi(3);

        let t_integrand = |t: f64| amplitude(s, t) * s_factors;
        integrate_t(t_integrand, s, p1, p2, p3, p4)
    };

    // Integrand optimized for the last upper interval intended to be used with
    // a double exponential
    let s_integrand_de = |ss: f64| {
        // Remap the interval [s_med, s_max] onto [0, 1].  The following
        // rescaling will cause the numerical integration to sample more heavily
        // from the lower part of the domain (unlike a simple linear rescaling).
        // let delta = (s_max - s_med).recip() + 1.0;
        // let s = ss / (delta - ss) + s_med;
        // let dsdss = delta / (delta - ss).powi(2);

        // Remap the semi-infinite s interval onto [0, 1)
        let s = ss / (1.0 - ss) + s_max;
        let dsdss = (ss - 1.0).powi(-2);

        // Combination of factors constant w.r.t. t
        let sqrt_s = s.sqrt();
        let s_factors = (8.845_964_466_739_566
            / (h1.degrees_of_freedom() * h2.degrees_of_freedom() * h3.degrees_of_freedom()))
            * f64::exp((h1.mass + h2.mass + h3.mass - sqrt_s) * beta)
            / ((h1.mass * h2.mass * h3.mass * sqrt_s).sqrt() * beta.powi(2)).powi(3)
            * dsdss;

        let t_integrand = |t: f64| amplitude(s, t) * s_factors;
        integrate_t(t_integrand, s, p1, p2, p3, p4)
    };

    let mut integral = 0.0;

    for (&s1, &s2) in s_divs.iter().zip(&s_divs[1..]) {
        integral +=
            clenshaw_curtis::integrate(&s_integrand_cc, s1, s2, INTEGRATION_PRECISION).integral;
    }

    integral +=
        double_exponential::integrate(&s_integrand_de, 0.0, 1.0, INTEGRATION_PRECISION).integral;

    integral / (512.0 * PI_5)
}

/// Integrate the over `$s$` and `$t$` divided by all four particles.
///
/// Uses the approximation
///
/// ```math
/// \frac{K_1(\sqrt s \beta)}{\sqrt s \beta} \frac{1}{n_a^{(0)} n_b^{(0)} n_c^{(0)} n_d^{(0)}}
/// = \left( \sqrt{\frac{8}{\pi}} \zeta(3) \right)^4 \frac{\sqrt{\pi / 2}}{g_a g_b g_c g_d}
///   \frac{e^{\beta (m_a + m_b + m_c + m_d - \sqrt{s})}}{\left(\sqrt{m_a m_b m_c m_d} \sqrt[4]{s} \beta^\frac{5}{2} \right)^3}
/// ```
#[must_use]
fn integrate_st_on_n4<F>(
    amplitude: F,
    beta: f64,
    p1: &Particle,
    p2: &Particle,
    p3: &Particle,
    p4: &Particle,
) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    let mut s_divs = s_divisions(beta, p1, p2, p3, p4);
    s_divs.extend_from_slice(&[
        (p1.mass + p2.mass + p3.mass).powi(2),
        (p1.mass + p2.mass + p4.mass).powi(2),
        (p1.mass + p3.mass + p4.mass).powi(2),
        (p2.mass + p3.mass + p4.mass).powi(2),
        (p1.mass + p2.mass + p3.mass + p4.mass).powi(2),
    ]);
    s_divs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    s_divs.dedup();
    let s_max = *s_divs.last().unwrap();

    // Integrand of `s` for regular Clenshaw-Curtis integration
    let s_integrand_cc = |s: f64| {
        // Combination of factors constant w.r.t. t
        let sqrt_s = s.sqrt();
        // 32 √2 ζ(3)⁴ / π^(3/2) ≅ 16.968375821762574
        let s_factors = (16.968_375_821_762_574
            / (p1.degrees_of_freedom()
                * p2.degrees_of_freedom()
                * p3.degrees_of_freedom()
                * p4.degrees_of_freedom()))
            * f64::exp((p1.mass + p2.mass + p3.mass + p4.mass - sqrt_s) * beta)
            / ((p1.mass * p2.mass * p3.mass * p4.mass * sqrt_s * beta.powi(5)).sqrt()).powi(3);

        let t_integrand = |t: f64| amplitude(s, t) * s_factors;
        integrate_t(t_integrand, s, p1, p2, p3, p4)
    };

    // Integrand optimized for the last upper interval intended to be used with
    // a double exponential
    let s_integrand_de = |ss: f64| {
        // Remap the interval [s_med, s_max] onto [0, 1].  The following
        // rescaling will cause the numerical integration to sample more heavily
        // from the lower part of the domain (unlike a simple linear rescaling).
        // let delta = (s_max - s_med).recip() + 1.0;
        // let s = ss / (delta - ss) + s_med;
        // let dsdss = delta / (delta - ss).powi(2);

        // Remap the semi-infinite s interval onto [0, 1)
        let s = ss / (1.0 - ss) + s_max;
        let dsdss = (ss - 1.0).powi(-2);

        // Combination of factors constant w.r.t. t
        let sqrt_s = s.sqrt();
        let s_factors = (16.968_375_821_762_574
            / (p1.degrees_of_freedom()
                * p2.degrees_of_freedom()
                * p3.degrees_of_freedom()
                * p4.degrees_of_freedom()))
            * f64::exp((p1.mass + p2.mass + p3.mass + p4.mass - sqrt_s) * beta)
            / ((p1.mass * p2.mass * p3.mass * p4.mass * sqrt_s).sqrt() * beta.powf(2.5)).powi(3)
            * dsdss;

        let t_integrand = |t: f64| amplitude(s, t) * s_factors;
        integrate_t(t_integrand, s, p1, p2, p3, p4)
    };

    let mut integral = 0.0;

    for (&s1, &s2) in s_divs.iter().zip(&s_divs[1..]) {
        integral +=
            clenshaw_curtis::integrate(&s_integrand_cc, s1, s2, INTEGRATION_PRECISION).integral;
    }

    integral +=
        double_exponential::integrate(&s_integrand_de, 0.0, 1.0, INTEGRATION_PRECISION).integral;

    integral / (512.0 * PI_5)
}

#[cfg(test)]
mod tests {
    use crate::{prelude::Particle, utilities::test::approx_eq};
    use ndarray::Array1;
    use serde::{Deserialize, Serialize};
    use std::{env::temp_dir, error, fs, path::Path};

    /// Shorthand to create the CSV file in the appropriate directory and with
    /// headers.
    fn create_csv<P: AsRef<Path>>(p: P) -> Result<csv::Writer<fs::File>, Box<dyn error::Error>> {
        let dir = temp_dir()
            .join("boltzmann-solver")
            .join("interaction")
            .join("four_particle");
        fs::create_dir_all(&dir)?;

        let csv = csv::Writer::from_path(dir.join(p))?;

        Ok(csv)
    }

    /// Test Mandelstam t domain for all massless particles
    #[test]
    fn t_range_massless() -> Result<(), Box<dyn error::Error>> {
        let p = Particle::new(0, 0.0, 0.0);

        for &s in &Array1::geomspace(1e-10, 1e10, 1000).unwrap() {
            let (t1, t2) = super::t_range(s, &p, &p, &p, &p);
            approx_eq(t1, -s, 8.0, 0.0)?;
            approx_eq(t2, 0.0, 8.0, s)?;
        }

        Ok(())
    }

    /// Test Mandelstam t domain with a single mass scale (whereby the domain
    /// simplifies to a nice closed form).
    #[test]
    fn t_range_massive() -> Result<(), Box<dyn error::Error>> {
        let p0 = Particle::new(0, 0.0, 0.0);

        for &m in &Array1::geomspace(1e-10, 1e10, 100).unwrap() {
            let pm = Particle::new(0, m, 0.0);

            for &s in &Array1::geomspace(pm.mass2, pm.mass2 * 1e20, 1000).unwrap() {
                // Due to minor rounding issues, it's possible that s is small
                // than `pm.mass2`
                let s = s.max(pm.mass2);

                let (t1, t2) = super::t_range(s, &pm, &p0, &p0, &p0);
                approx_eq(t1, pm.mass2 - s, 8.0, s)?;
                approx_eq(t2, 0.0, 8.0, s)?;

                let (t1, t2) = super::t_range(s, &pm, &p0, &pm, &p0);
                approx_eq(t1, -(pm.mass2 - s).powi(2) / s, 8.0, s)?;
                approx_eq(t2, 0.0, 8.0, s)?;

                // Minimal value of s is larger in this case
                if s >= 4.0 * pm.mass2 {
                    let (t1, t2) = super::t_range(s, &pm, &pm, &p0, &p0);
                    approx_eq(
                        t1,
                        pm.mass2 - 0.5 * (s + f64::sqrt(s * (s - 4.0 * pm.mass2).abs())),
                        8.0,
                        s,
                    )?;
                    approx_eq(
                        t2,
                        pm.mass2 - 0.5 * (s - f64::sqrt(s * (s - 4.0 * pm.mass2).abs())),
                        8.0,
                        s,
                    )?;

                    let (t1, t2) = super::t_range(s, &pm, &pm, &pm, &pm);
                    approx_eq(t1, 4.0 * pm.mass2 - s, 8.0, s)?;
                    approx_eq(t2, 0.0, 8.0, s)?;
                }
            }
        }

        Ok(())
    }

    /// Test Mandelstam t domainfor a few random values.
    #[test]
    fn t_range_random() -> Result<(), Box<dyn error::Error>> {
        #[derive(Debug, Deserialize, Serialize)]
        struct Row {
            m1: f64,
            m2: f64,
            m3: f64,
            m4: f64,
            s: f64,
            t1: f64,
            t2: f64,
        }

        // Data generated randomly
        let mut csv = csv::Reader::from_reader(zstd::Decoder::new(fs::File::open(
            "tests/data/t_range_random.csv.zst",
        )?)?);

        for record in csv.deserialize() {
            let row: Row = record?;
            let p1 = Particle::new(0, row.m1, 0.0);
            let p2 = Particle::new(0, row.m2, 0.0);
            let p3 = Particle::new(0, row.m3, 0.0);
            let p4 = Particle::new(0, row.m4, 0.0);

            let (t1, t2) = super::t_range(row.s, &p1, &p2, &p3, &p4);
            approx_eq(row.t1, t1, 0.5, row.s / 1e12)?;
            approx_eq(row.t2, t2, 0.5, row.s / 1e12)?;

            // Simultaneous interchange of 1 ↔ 2 and 3 ↔ 4 leaves the t variable
            // unchanged
            let (t1, t2) = super::t_range(row.s, &p2, &p1, &p4, &p3);
            approx_eq(row.t1, t1, 0.5, row.s / 1e12)?;
            approx_eq(row.t2, t2, 0.5, row.s / 1e12)?;
        }

        Ok(())
    }

    #[test]
    fn integrate_t_massless() -> Result<(), Box<dyn error::Error>> {
        let p = Particle::new(0, 0.0, 0.0);

        for &s in &Array1::geomspace(1e-10, 1e10, 1000).unwrap() {
            approx_eq(super::integrate_t(|_t| 1.0, s, &p, &p, &p, &p), s, 8.0, 0.0)?;
        }

        for &s in &Array1::geomspace(1e-10, 1e10, 1000).unwrap() {
            approx_eq(
                super::integrate_t(|t| t, s, &p, &p, &p, &p),
                -s.powi(2) / 2.0,
                8.0,
                0.0,
            )?;
        }

        Ok(())
    }

    #[test]
    fn integrate_st_massless() -> Result<(), Box<dyn error::Error>> {
        #[derive(Debug, serde::Serialize)]
        struct Row {
            beta: f64,
            m0: f64,
            m1: f64,
            m2: f64,
        }

        let p = Particle::new(0, 0.0, 0.0);
        let mut csv = create_csv("integrate_st_massless.csv")?;

        let m0 = |_, _| 1.0;
        let m1 = |s, _| s;
        let m2 = |s: f64, _| s.powi(2);
        for &beta in &Array1::geomspace(1e-20, 1e20, 1000).unwrap() {
            let row = Row {
                beta,
                m0: super::integrate_st(m0, beta, &p, &p, &p, &p),
                m1: super::integrate_st(m1, beta, &p, &p, &p, &p),
                m2: super::integrate_st(m2, beta, &p, &p, &p, &p),
            };
            csv.serialize(row)?;

            // approx_eq(
            //     super::integrate_st(m0, beta, &p, &p, &p, &p),
            //     1.0 / (128.0 * PI_5 * beta.powi(4)),
            //     0.5,
            //     0.0,
            // )?;
            // approx_eq(
            //     super::integrate_st(m1, beta, &p, &p, &p, &p),
            //     1.0 / (16.0 * PI_5 * beta.powi(6)),
            //     4.0,
            //     0.0,
            // )?;
            // approx_eq(
            //     super::integrate_st(m2, beta, &p, &p, &p, &p),
            //     1.0 / (2.0 * PI_5 * beta.powi(8)),
            //     4.0,
            //     0.0,
            // )?;
        }

        // let m3 = |s: f64, _| 1.0 / (s.powi(2) + 1.0);
        // approx_eq(
        //     super::integrate_st(m3, BETA, &p, &p, &p, &p),
        //     5.671_609_064_313_106e-6,
        //     4.0,
        //     0.0,
        // )?;

        // let m4 = |_, t: f64| 1.0 / (t.powi(2) + 1.0);
        // approx_eq(
        //     super::integrate_st(m4, BETA, &p, &p, &p, &p),
        //     1.063_840_621_992_219e-5,
        //     4.0,
        //     0.0,
        // )?;

        Ok(())
    }

    #[allow(clippy::shadow_unrelated)]
    #[test]
    fn integrate_st_massive() -> Result<(), Box<dyn error::Error>> {
        const BETA: f64 = 1e-2;
        let p1 = Particle::new(0, 0.2, 0.0);
        let p2 = Particle::new(0, 0.5, 0.0);
        let p3 = Particle::new(0, 2.0, 0.0);
        let p4 = Particle::new(0, 5.0, 0.0);

        let m2 = |_, _| 1.0;
        approx_eq(
            super::integrate_st(m2, BETA, &p1, &p2, &p3, &p4),
            2539.02,
            3.0,
            1e-20,
        )?;

        let m2 = |s: f64, _| 1.0 / (s.powi(2) + 1.0);
        approx_eq(
            super::integrate_st(m2, BETA, &p1, &p2, &p3, &p4),
            0.000_802_015,
            3.0,
            1e-20,
        )?;

        let m2 = |_, t: f64| 1.0 / (t.powi(2) + 1.0);
        approx_eq(
            super::integrate_st(m2, BETA, &p1, &p2, &p3, &p4),
            0.439_373,
            1.5,
            1e-20,
        )?;

        Ok(())
    }

    #[test]
    fn integrate_st_on_n1() -> Result<(), Box<dyn error::Error>> {
        #[derive(Debug, serde::Serialize)]
        struct Row {
            beta: f64,
            n: f64,
            gamma: f64,
            gamma_on_n: f64,
            gamma_auto: f64,
            gamma_tilde: f64,
        }

        let p0 = Particle::new(0, 0.0, 0.0);
        let pm = Particle::new(0, 1e10, 0.0);

        let mut csv = create_csv("integrate_st_on_n1.csv")?;

        let m2 = |_, _| 1.0;
        for &beta in &Array1::geomspace(1e-12, 1e-6, 1000).unwrap() {
            let n = pm.normalized_number_density(beta, 0.0);
            let gamma = super::integrate_st(m2, beta, &pm, &p0, &p0, &p0);
            let row = Row {
                beta,
                n,
                gamma,
                gamma_on_n: gamma / n,
                gamma_auto: super::integrate_st_on_n(m2, beta, &pm, &p0, &p0, &p0),
                gamma_tilde: super::integrate_st_on_n1(m2, beta, &pm, &p0, &p0, &p0, &pm),
            };

            csv.serialize(row)?;
        }

        Ok(())
    }

    #[test]
    fn integrate_st_on_n2() -> Result<(), Box<dyn error::Error>> {
        #[derive(Debug, serde::Serialize)]
        struct Row {
            beta: f64,
            n: f64,
            gamma: f64,
            gamma_on_n: f64,
            gamma_auto: f64,
            gamma_tilde: f64,
        }

        let p0 = Particle::new(0, 0.0, 0.0);
        let pm = Particle::new(0, 1e10, 0.0);

        let mut csv = create_csv("integrate_st_on_n2.csv")?;

        let m2 = |_, _| 1.0;
        for &beta in &Array1::geomspace(1e-12, 1e-6, 1000).unwrap() {
            let n = pm.normalized_number_density(beta, 0.0);
            let gamma = super::integrate_st(m2, beta, &pm, &pm, &p0, &p0);
            let row = Row {
                beta,
                n,
                gamma,
                gamma_on_n: gamma / n.powi(2),
                gamma_auto: super::integrate_st_on_n(m2, beta, &pm, &pm, &p0, &p0),
                gamma_tilde: super::integrate_st_on_n2(m2, beta, &pm, &pm, &p0, &p0, &pm, &pm),
            };

            csv.serialize(row)?;
        }

        Ok(())
    }

    #[test]
    fn integrate_st_on_n3() -> Result<(), Box<dyn error::Error>> {
        #[derive(Debug, serde::Serialize)]
        struct Row {
            beta: f64,
            n: f64,
            gamma: f64,
            gamma_on_n: f64,
            gamma_auto: f64,
            gamma_tilde: f64,
        }

        let p0 = Particle::new(0, 0.0, 0.0);
        let pm = Particle::new(0, 1e10, 0.0);

        let mut csv = create_csv("integrate_st_on_n3.csv")?;

        let m2 = |_, _| 1.0;
        for &beta in &Array1::geomspace(1e-12, 1e-6, 1000).unwrap() {
            let n = pm.normalized_number_density(beta, 0.0);
            let gamma = super::integrate_st(m2, beta, &pm, &pm, &pm, &p0);
            let row = Row {
                beta,
                n,
                gamma,
                gamma_on_n: gamma / n.powi(3),
                gamma_auto: super::integrate_st_on_n(m2, beta, &pm, &pm, &pm, &p0),
                gamma_tilde: super::integrate_st_on_n3(m2, beta, &pm, &pm, &pm, &p0, &pm, &pm, &pm),
            };

            csv.serialize(row)?;
        }

        Ok(())
    }

    #[test]
    fn integrate_st_on_n4() -> Result<(), Box<dyn error::Error>> {
        #[derive(Debug, serde::Serialize)]
        struct Row {
            beta: f64,
            n: f64,
            gamma: f64,
            gamma_on_n: f64,
            gamma_auto: f64,
            gamma_tilde: f64,
        }

        let pm = Particle::new(0, 1e10, 0.0);

        let mut csv = create_csv("integrate_st_on_n4.csv")?;

        let m2 = |_, _| 1.0;
        for &beta in &Array1::geomspace(1e-12, 1e-6, 1000).unwrap() {
            let n = pm.normalized_number_density(beta, 0.0);
            let gamma = super::integrate_st(m2, beta, &pm, &pm, &pm, &pm);
            let row = Row {
                beta,
                n,
                gamma,
                gamma_on_n: gamma / n.powi(4),
                gamma_auto: super::integrate_st_on_n(m2, beta, &pm, &pm, &pm, &pm),
                gamma_tilde: super::integrate_st_on_n4(m2, beta, &pm, &pm, &pm, &pm),
            };

            csv.serialize(row)?;
        }

        Ok(())
    }
}
