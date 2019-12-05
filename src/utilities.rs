//! Module of various useful miscellaneous functions.

pub mod spline;
#[cfg(test)]
pub(crate) mod test;

use crate::{constants::PI_5, model::Particle};
use num_complex::Complex;
use quadrature::integrate;
use special_functions::bessel;

/// Kallen lambda function:
///
/// \\begin{equation}
///   \lambda(a, b, c) = a^2 + b^2 + c^2 - 2ab - 2ac - 2bc
/// \\end{equation}
pub fn kallen_lambda(a: f64, b: f64, c: f64) -> f64 {
    a.powi(2) + b.powi(2) + c.powi(2) - 2.0 * (a * b + a * c + b * c)
}

/// Return the minimum and maximum value of the Mandelstam variable \\(t\\)
/// based on the four particle masses \\(m_1\\), \\(m_2\\), \\(m_3\\) and
/// \\(m_4\\), where particles 1 and 2 are initial state and particles 3 and 4
/// are the final state particles.
pub fn t_range(s: f64, m1: f64, m2: f64, m3: f64, m4: f64) -> (f64, f64) {
    debug_assert!(s >= m1 + m2, "s must be greater than m1^2 + m2^2.");
    debug_assert!(s >= m3 + m4, "s must be greater than m3^2 + m4^2.");

    if s == 0.0 {
        return (0.0, 0.0);
    }

    let baseline = (m1 - m2 - m3 + m4).powi(2) / (4.0 * s);
    let delta = kallen_lambda(s, m1, m2).abs().sqrt() * kallen_lambda(s, m3, m4).abs().sqrt() / s;
    // let t_const = 0.5 * ((m1 + m2 + m3 + m4) - s - (m1 - m2) * (m3 - m4) / s);
    // let t_cos = f64::sqrt(f64::abs(
    //     (s.powi(2) - 2.0 * s * (m1 + m2) + (m1 - m2).powi(2))
    //         * (s.powi(2) - 2.0 * s * (m3 + m4) + (m3 - m4).powi(2)),
    // )) / (2.0 * s);

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
    let s_min = (m1 + m2).max(m3 + m4);

    let s_integrand = |ss: f64| {
        // Remap the semi-infinite s interval onto [0, 1)
        let s = ss / (1.0 - ss) + s_min;
        let dsdss = (ss - 1.0).powi(-2);
        let sqrt_s = s.sqrt();

        // Combination of factors constant w.r.t. t
        let s_factors = bessel::k1(sqrt_s * beta) / sqrt_s * dsdss;

        // Remap the (potentially very large) t interval onto [0, 1]
        // This appears to be substantially slower in benchmarks
        // let (t_min, t_max) = t_range(s, m1, m2, m3, m4);
        // let delta = (t_max - t_min).recip() + 1.0;
        // let t_integrand = |tt: f64| {
        //     let t = tt / (delta - tt) + t_min;
        //     let dtdtt = delta / (delta - tt).powi(2);
        //     amplitude(s, t) * dtdtt * s_factors
        // };
        // integrate(t_integrand, 0.0, 1.0, 0.0).integral

        let (t_min, t_max) = t_range(s, m1, m2, m3, m4);
        let t_integrand = |t: f64| amplitude(s, t) * s_factors;
        integrate(t_integrand, t_min, t_max, 0.0).integral
    };

    integrate(s_integrand, 0.0, 1.0, 0.0).integral / (512.0 * PI_5 * beta)
}

/// Propagator with squared momentum `q2` involving particle `p`, defined as
pub fn propagator(q2: f64, p: &Particle) -> Complex<f64> {
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
            [1.0, 2.0, 3.0, 4.0, 8.0, -3.533323506, 0.0],
            [1.0, 2.0, 3.0, 4.0, 20.0, -9.219680038, 0.0],
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
        // |M|² = 1
        approx_eq(
            super::integrate_st(|_, _| 1.0, 0.01, 1.0, 10.0, 2.0, 20.0),
            2538.6650417543515,
            2.0,
            0.0,
        );

        // |M|² = 1 / (s^2 + 1)
        approx_eq(
            super::integrate_st(|s, _| 1.0 / (s.powi(2) + 1.0), 0.01, 1.0, 10.0, 2.0, 20.0),
            0.0011966876930957792,
            2.0,
            0.0,
        );

        // |M|² = 1 / (t^2 + 1)
        approx_eq(
            super::integrate_st(|_, t| 1.0 / (t.powi(2) + 1.0), 0.01, 1.0, 10.0, 2.0, 20.0),
            0.6784160555589841,
            2.0,
            0.0,
        );

        // |M|² = (s t) / [(s+1) (t+1)]
        approx_eq(
            super::integrate_st(
                |s, t| (s * t) / (s + 1.0) / (t + 1.0),
                0.01,
                1.0,
                10.0,
                2.0,
                20.0,
            ),
            2540.321286336532,
            1.9,
            0.0,
        );
    }
}

#[cfg(all(test, feature = "nightly"))]
mod benches {
    use test::Bencher;

    /// |M|² = 1
    #[bench]
    fn integrate_st_const(b: &mut Bencher) {
        b.iter(|| test::black_box(super::integrate_st(|_, _| 1.0, 0.01, 1.0, 10.0, 2.0, 20.0)));
    }

    /// |M|² = 1 / (s^2 + 1)
    #[bench]
    fn integrate_st_s_inv(b: &mut Bencher) {
        b.iter(|| {
            test::black_box(super::integrate_st(
                |s, _| 1.0 / (s.powi(2) + 1.0),
                0.01,
                1.0,
                10.0,
                2.0,
                20.0,
            ))
        });
    }

    /// |M|² = 1 / (t^2 + 1)
    #[bench]
    fn integrate_st_t_inv(b: &mut Bencher) {
        b.iter(|| {
            test::black_box(super::integrate_st(
                |_, t| 1.0 / (t.powi(2) + 1.0),
                0.01,
                1.0,
                10.0,
                2.0,
                20.0,
            ))
        });
    }

    /// |M|² = (s t) / [(s+1) (t+1)]
    #[bench]
    fn integrate_st_st(b: &mut Bencher) {
        b.iter(|| {
            test::black_box(super::integrate_st(
                |s, t| (s * t) / (s + 1.0) / (t + 1.0),
                0.01,
                1.0,
                10.0,
                2.0,
                20.0,
            ))
        })
    }
}
