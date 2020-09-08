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
/// on the four particle *squared* masses `$m_1^2$`, `$m_2^2$`, `$m_3^2$` and
/// `$m_4^2$`, where particles 1 and 2 are initial state and particles 3 and 4
/// are the final state particles.
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
pub fn t_range(s: f64, m1: f64, m2: f64, m3: f64, m4: f64) -> (f64, f64) {
    debug_assert!(
        s * (1.0 + 2.0 * f64::EPSILON) >= (m1 + m2).powi(2),
        "s cannot be smaller than (m1 + m2)² (s = {:e}, (m1 + m2)² = {:e}).",
        s,
        (m1 + m2).powi(2)
    );
    debug_assert!(
        s * (1.0 + 2.0 * f64::EPSILON) >= (m3 + m4).powi(2),
        "s cannot be smaller than (m3 + m4)² (s = {:e}, (m3 + m4)² = {:e}).",
        s,
        (m3 + m4).powi(2)
    );

    if s == 0.0 {
        return (0.0, 0.0);
    }

    // Deal with squared masses from now on
    let (m1, m2, m3, m4) = (m1.powi(2), m2.powi(2), m3.powi(2), m4.powi(2));

    let baseline = 0.5 * (m1 + m2 + m3 + m4 - s - (m1 - m2) * (m3 - m4) / s);
    let cosine = kallen_lambda_sqrt(s, m1, m2) * kallen_lambda_sqrt(s, m3, m4) / (2.0 * s);

    (baseline - cosine, baseline + cosine)
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

    let s_min = (m1 + m2).powi(2).max((m3 + m4).powi(2));
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

/// Create a recursively generated geometrically spaced interval between `start`
/// and `end`.
///
/// This is the analogous version of the recursively generated linearly spaced
/// interval [`rec_linspace`].
#[must_use]
pub fn rec_geomspace(start: f64, end: f64, recursions: u32) -> Vec<f64> {
    let mut v = Vec::with_capacity(2_usize.pow(recursions));

    v.push(start);
    v.push(end);

    let start = start.ln();
    let end = end.ln();

    let mut base = 2.0;
    for i in 2..2_u64.pow(recursions) {
        let i = i as f64;
        if i > base {
            base *= 2.0;
        }

        let b = 2.0 * i - base - 1.0;
        let a = base - b;

        v.push(((a * start + b * end) / base).exp())
    }
    v
}

/// Create a recursively generated linearly spaced interval spanning `start` and
/// `end` inclusively.
///
/// The resulting vector will have values in the following order:
///
/// ```math
/// \left[
///   a,
///   b,
///   \frac{a + b}{2},
///   \frac{3a + b}{4}, \frac{a + 3b}{4},
///   \frac{7a + b}{8}, \frac{5a + 3b}{8}, \frac{3a + 5b}{8}, \frac{a + 8b}{8},
///   \dots
/// \right]
/// ```
///
/// The number of recursions is determined by `recursions`.
#[must_use]
pub fn rec_linspace(start: f64, end: f64, recursions: u32) -> Vec<f64> {
    let mut v = Vec::with_capacity(2_usize.pow(recursions));

    v.push(start);
    v.push(end);

    let mut base = 2.0;
    for i in 2..2_u64.pow(recursions) {
        let i = i as f64;
        if i > base {
            base *= 2.0;
        }

        let b = 2.0 * i - base - 1.0;
        let a = base - b;

        v.push((a * start + b * end) / base)
    }
    v
}

#[cfg(test)]
mod tests {
    use crate::{constants::PI, utilities::test::approx_eq};
    use std::error;

    #[test]
    fn t_range() -> Result<(), Box<dyn error::Error>> {
        // Data generated randomly
        let mut csv = csv::Reader::from_reader("\
19.04885812196941,0.20017322767927473,2.360700597936626,1.1887668016292394,0.04742988454808271,-12.008565256859335,0.3979078853224429
14.529983396258071,0.112999458353636,0.15490206418326763,0.5964126836824825,0.016643444422671495,-14.137520271622149,0.0005600997147947595
4.8072503985349035,0.07628006372174885,0.7552382355730599,0.07278942540383221,2.086894319558819,-0.33486763454489543,-0.04659951490356354
83.50565208353986,0.013255043179495626,0.08692733978507598,1.3366290727167602,0.3020678650841765,-81.6181128211148,-0.001835088372004634
5.978519943765303,1.1148701583787173,0.06268616712376433,0.10001148063515142,0.13830774364126305,-4.703562770028053,0.002930631395400063
0.38856980461333546,0.01756230305294807,0.023201127408091014,0.03911586893225636,0.031015115489286177,-0.38522940690672086,-1.34624715969478e-6
0.30914531844697724,0.07097679356533206,0.06693357058697391,0.23835593629582447,0.22121662528505898,-0.18118637276407323,-0.012704997145410818
0.614863605759527,0.6446716328461544,0.11386398377926585,0.011796681448556475,0.08172426090005198,-0.14430650462897476,-0.030890119227830207
14.453093604343984,0.034198594626101454,0.5646698029247602,1.5540761510629382,0.21912119015923992,-11.661673063027923,0.04379803855304143
2.710305844048692,0.06092161390619277,1.0298337397924684,0.48644845475092463,0.09565239280767306,-1.401449839030379,0.08989817306128634".as_bytes()
        );

        for record in csv.deserialize() {
            let (s, m1, m2, m3, m4, ea, eb) = record?;
            let (ta, tb) = super::t_range(s, m1, m2, m3, m4);
            approx_eq(ta, ea, 8.0, 0.0)?;
            approx_eq(tb, eb, 8.0, 0.0)?;
        }

        Ok(())
    }

    #[allow(clippy::shadow_unrelated)]
    #[test]
    fn integrate_st_massless() -> Result<(), Box<dyn error::Error>> {
        const BETA: f64 = 1e0;
        const M1: f64 = 0.0;
        const M2: f64 = 0.0;
        const M3: f64 = 0.0;
        const M4: f64 = 0.0;

        let m2 = |_, _| 1.0;
        approx_eq(
            super::integrate_st(m2, BETA, M1, M2, M3, M4),
            1.0 / (128.0 * PI.powi(5)),
            4.0,
            0.0,
        )?;

        let m2 = |s: f64, _| 1.0 / (s.powi(2) + 1.0);
        approx_eq(
            super::integrate_st(m2, BETA, M1, M2, M3, M4),
            5.671_609_064_313_106e-6,
            4.0,
            0.0,
        )?;

        let m2 = |_, t: f64| 1.0 / (t.powi(2) + 1.0);
        approx_eq(
            super::integrate_st(m2, BETA, M1, M2, M3, M4),
            1.063_840_621_992_219e-5,
            4.0,
            0.0,
        )?;

        Ok(())
    }

    #[allow(clippy::shadow_unrelated)]
    #[test]
    fn integrate_st_massive() -> Result<(), Box<dyn error::Error>> {
        const BETA: f64 = 1e-2;
        const M1: f64 = 0.2;
        const M2: f64 = 0.5;
        const M3: f64 = 2.0;
        const M4: f64 = 5.0;

        let m2 = |_, _| 1.0;
        approx_eq(
            super::integrate_st(m2, BETA, M1, M2, M3, M4),
            2539.02,
            3.0,
            1e-20,
        )?;

        let m2 = |s: f64, _| 1.0 / (s.powi(2) + 1.0);
        approx_eq(
            super::integrate_st(m2, BETA, M1, M2, M3, M4),
            0.000_802_015,
            3.0,
            1e-20,
        )?;

        let m2 = |_, t: f64| 1.0 / (t.powi(2) + 1.0);
        approx_eq(
            super::integrate_st(m2, BETA, M1, M2, M3, M4),
            0.439_373,
            1.5,
            1e-20,
        )?;

        Ok(())
    }
}
