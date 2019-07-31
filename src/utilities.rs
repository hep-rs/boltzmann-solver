//! Module of various useful miscellaneous functions.

use quadrature::integrate;
use special_functions::bessel;

const CHECKED_DIV_MAX: f64 = 1e3;

#[cfg(test)]
pub(crate) mod test;

/// Perform a 'checked' division for number densities.
///
/// As the number densities can become quite small (especially the equilibrium
/// number densities), computing `n / n_eq` can cause certain issues.  As a
/// result, this function provides a 'well-behaved' division that handles small
/// values of `n_eq`.
///
/// # Implementation
///
/// - If the numerator is 0, 0 is returned.
/// - If the denominator is 0, a maximum value is returned with the same sign as
///   the numerator.
/// - For all other values, the value returned is:
///   \\begin{equation}
///     M \tanh\left( \frac{1}{M} \frac{a}{b} \right)
///   \\end{equation}
///   where \\(M\\) is the maximum value allowed (typically set to 10).
#[inline]
pub fn checked_div(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        0.0
    } else if b == 0.0 {
        CHECKED_DIV_MAX.copysign(a)
    } else {
        // Use a sigmoid to have bounds on the division to prevent very small
        // denominators from blowing up.
        let v = a / b;
        if v.abs() > CHECKED_DIV_MAX {
            CHECKED_DIV_MAX.copysign(v)
        } else {
            v
        }
    }
}

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
///
/// The values are given by:
/// \\begin{equation}
///   t_{\text{min},\text{max}} = a \pm b
/// \\end{equation}
/// where
/// \\begin{equation}
///   \\begin{aligned}
///     a &= \frac{1}{2} \left( m_1^2 + m_2^2 + m_3^2 + m_4^2 - s - \frac{\left( m_1^2 - m_2^2 \right) \left( m_3^2 - m_4^2 \right)}{s} \right) \\\\
///     b &= \frac{\mathop{\text{sign}}\left(m_1^2 - m_2^2\right)}{2s}
///           \sqrt{m_2^2 + \left(s - m_1^2\right)^2 - 2 m_2^2 \left(s + m_1^2\right)}
///           \sqrt{m_4^2 + \left(s - m_3^2\right)^2 - 2 m_4^2 \left(s + m_3^2\right)}
///   \\end{aligned}
/// \\end{equation}
pub fn t_min_max(s: f64, m1: f64, m2: f64, m3: f64, m4: f64) -> (f64, f64) {
    let m1 = m1.powi(2);
    let m2 = m2.powi(2);
    let m3 = m3.powi(2);
    let m4 = m4.powi(2);

    debug_assert!(s >= m1 + m2, "s must be greater than m1^2 + m2^2.");
    debug_assert!(s >= m3 + m4, "s must be greater than m3^2 + m4^2.");

    if s == 0.0 {
        return (-0.0, 0.0);
    }

    let t_const = (m1 + m2 + m3 + m4 - s - (m1 - m2) * (m3 - m4) / s) / 2.0;
    let t_cos = ((m2.powi(2) + (s - m1).powi(2) - 2.0 * m2 * (s + m1))
        * (m4.powi(2) + (s - m3).powi(2) - 2.0 * m4 * (s + m3)))
        .abs()
        .sqrt()
        / (2.0 * s);

    (t_const - t_cos, t_const + t_cos)
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
/// and \\(t_{\text{min},\text{max}}\\) are determined from [`t_min_max`].
pub fn integrate_st<F>(amplitude: F, beta: f64, m1: f64, m2: f64, m3: f64, m4: f64) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    let s_min = if m1.powi(2) + m2.powi(2) > m3.powi(2) + m4.powi(4) {
        m1.powi(2) + m2.powi(2)
    } else {
        m3.powi(2) + m4.powi(4)
    };

    let s_integrand = |ss: f64| {
        let s = (1.0 - ss) / ss + s_min;
        let dsdss = ss.powi(-2);
        let sqrt_s = s.sqrt();

        let (t_min, t_max) = t_min_max(s, m1, m2, m3, m4);
        let t_integrand = |t: f64| amplitude(s, t);

        integrate(t_integrand, t_min, t_max, 0.0).integral * bessel::k1(sqrt_s * beta) / sqrt_s
            * dsdss
    };

    integrate(s_integrand, 0.0, 1.0, 0.0).integral
}

#[cfg(test)]
mod tests {
    use super::{t_min_max, CHECKED_DIV_MAX};
    use crate::utilities::test::*;
    use ndarray::prelude::*;

    #[allow(clippy::float_cmp)]
    #[test]
    fn checked_div() {
        assert_eq!(super::checked_div(0.0, 0.0), 0.0);

        assert_eq!(super::checked_div(1.0, 0.0), CHECKED_DIV_MAX);
        assert_eq!(super::checked_div(-1.0, 0.0), -CHECKED_DIV_MAX);

        assert_eq!(super::checked_div(0.0, 1.0), 0.0);
        assert_eq!(super::checked_div(0.0, -1.0), 0.0);

        approx_eq(super::checked_div(1.0, 2.0), 0.5, 2.0, 0.0);
        approx_eq(super::checked_div(-1.0, 2.0), -0.5, 2.0, 0.0);
        approx_eq(super::checked_div(1.0, -2.0), -0.5, 2.0, 0.0);
        approx_eq(super::checked_div(-1.0, -2.0), 0.5, 2.0, 0.0);

        assert_eq!(super::checked_div(1.0, 1e-5), CHECKED_DIV_MAX);
        assert_eq!(super::checked_div(-1.0, 1e-5), -CHECKED_DIV_MAX);
        assert_eq!(super::checked_div(1.0, -1e-5), -CHECKED_DIV_MAX);
        assert_eq!(super::checked_div(-1.0, -1e-5), CHECKED_DIV_MAX);
    }

    #[test]
    fn t_min_max_zero() {
        let (m1, m2, m3, m4): (f64, f64, f64, f64) = (0.0, 0.0, 0.0, 0.0);

        let s_min = m1.powi(2) + m2.powi(2);
        let ss = Array1::linspace(s_min, s_min + 1e3, 100);

        for &s in &ss {
            let (t_min, t_max) = t_min_max(s, m1, m2, m3, m4);

            approx_eq(t_min, -s, 8.0, 1e-8);
            approx_eq(t_max, 0.0, 8.0, 1e-8);
        }
    }

    #[test]
    fn t_min_max_one() {
        // m1 != 0
        let (m1, m2, m3, m4): (f64, f64, f64, f64) = (100.0, 0.0, 0.0, 0.0);

        let s_min = m1.powi(2) + m2.powi(2);
        let ss = Array1::linspace(s_min, s_min + 1e3, 100);

        for &s in &ss {
            let (t_min, t_max) = t_min_max(s, m1, m2, m3, m4);

            approx_eq(t_min, m1.powi(2) - s, 8.0, 1e-8);
            approx_eq(t_max, 0.0, 8.0, 1e-8);
        }

        // m2 != 0
        let (m1, m2, m3, m4): (f64, f64, f64, f64) = (0.0, 100.0, 0.0, 0.0);

        let s_min = m1.powi(2) + m2.powi(2);
        let ss = Array1::linspace(s_min, s_min + 1e3, 100);

        for &s in &ss {
            let (t_min, t_max) = t_min_max(s, m1, m2, m3, m4);

            approx_eq(t_min, m2.powi(2) - s, 8.0, 1e-8);
            approx_eq(t_max, 0.0, 8.0, 1e-8);
        }

        // m3 != 0
        let (m1, m2, m3, m4): (f64, f64, f64, f64) = (0.0, 0.0, 100.0, 0.0);

        let s_min = m3.powi(2) + m4.powi(2);
        let ss = Array1::linspace(s_min, s_min + 1e3, 100);

        for &s in &ss {
            let (t_min, t_max) = t_min_max(s, m1, m2, m3, m4);

            approx_eq(t_min, m3.powi(2) - s, 8.0, 1e-8);
            approx_eq(t_max, 0.0, 8.0, 1e-8);
        }

        // m4 != 0
        let (m1, m2, m3, m4): (f64, f64, f64, f64) = (0.0, 0.0, 0.0, 100.0);

        let s_min = m3.powi(2) + m4.powi(2);
        let ss = Array1::linspace(s_min, s_min + 1e3, 100);

        for &s in &ss {
            let (t_min, t_max) = t_min_max(s, m1, m2, m3, m4);

            approx_eq(t_min, m4.powi(2) - s, 8.0, 1e-8);
            approx_eq(t_max, 0.0, 8.0, 1e-8);
        }
    }

    #[test]
    fn t_min_max_multiple() {
        let (m1, m2, m3, m4): (f64, f64, f64, f64) = (10.0, 20.0, 30.0, 40.0);

        let s_min = m3.powi(2) + m4.powi(2);
        let ss = Array1::linspace(s_min, s_min + 1e3, 100);

        for &s in &ss {
            let (t_min, t_max) = t_min_max(s, m1, m2, m3, m4);

            assert!(t_min.is_finite() && t_max.is_finite());
            assert!(t_min <= t_max);
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
            super::integrate_st(|_, _| 1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            4.0,
            8.0,
            0.0,
        );

        // |M|² = sqrt(s)
        approx_eq(
            super::integrate_st(|s, _| s.sqrt(), 1.0, 0.0, 0.0, 0.0, 0.0),
            9.424_777_960_769_38,
            8.0,
            0.0,
        );

        // |M|² = sqrt(s) t
        approx_eq(
            super::integrate_st(|s, t| s.sqrt() * t, 1.0, 0.0, 0.0, 0.0, 0.0),
            -70.685_834_705_770_35,
            8.0,
            0.0,
        );
    }
}
