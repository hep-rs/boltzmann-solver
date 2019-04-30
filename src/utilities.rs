//! Module of various useful miscellaneous functions.

#[cfg(arbitrary_precision)]
use rug::Float;

#[cfg(test)]
pub(crate) mod test;

/// Perform a 'checked' division.
///
/// Computes the result `a / b` with the following special conditions:
///
/// - If `a == 0.0`, returns `0.0` irrespective of the value of `b`;
/// - If `b == 0.0`, returns `1.0` irrespective of the value of `a` (unless `a
///   == 0`);
#[inline]
pub fn checked_div(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        0.0
    } else if b == 0.0 {
        1.0
    } else {
        a / b
    }
}

/// Perform a 'checked' division.
///
/// Computes the result `a / b` with the following special conditions:
///
/// - If `a == 0.0`, returns `0.0` irrespective of the value of `b`;
/// - If `b == 0.0`, returns `1.0` irrespective of the value of `a` (unless `a
///   == 0`);
#[inline]
#[cfg(arbitrary_precision)]
pub fn checked_div_ap(a: &Float, b: &Float) -> Float {
    if a.is_zero() {
        Float::with_val(a.prec(), 0.0)
    } else if b.is_zero() {
        Float::with_val(a.prec(), 1.0)
    } else {
        Float::with_val(a.prec(), a / b)
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

#[cfg(test)]
mod tests {
    use super::{kallen_lambda, t_min_max};
    use crate::utilities::test::*;
    use ndarray::prelude::*;

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

    #[test]
    fn kallen() {
        let (a, b, c) = (1.0, 2.0, 3.0);
        assert_eq!(kallen_lambda(a, b, c), -8.0);

        assert_eq!(kallen_lambda(a, b, c), kallen_lambda(a, c, b));
        assert_eq!(kallen_lambda(a, b, c), kallen_lambda(b, a, c));
        assert_eq!(kallen_lambda(a, b, c), kallen_lambda(b, c, a));
        assert_eq!(kallen_lambda(a, b, c), kallen_lambda(c, a, b));
        assert_eq!(kallen_lambda(a, b, c), kallen_lambda(c, b, a));
    }
}
