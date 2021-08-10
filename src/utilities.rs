//! Module of various useful miscellaneous functions.

// pub(crate) mod clenshaw_curtis;
// pub(crate) mod double_exponential;
pub mod spline;
#[cfg(test)]
pub(crate) mod test;

use std::f64;

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
        #[allow(clippy::cast_precision_loss)]
        let i = i as f64;
        if i > base {
            base *= 2.0;
        }

        let b = 2.0 * i - base - 1.0;
        let a = base - b;

        v.push(((a * start + b * end) / base).exp());
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
        #[allow(clippy::cast_precision_loss)]
        let i = i as f64;
        if i > base {
            base *= 2.0;
        }

        let b = 2.0 * i - base - 1.0;
        let a = base - b;

        v.push((a * start + b * end) / base);
    }
    v
}
