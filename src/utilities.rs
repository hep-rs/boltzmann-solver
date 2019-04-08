//! Module of various useful miscellaneous functions.

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
pub fn checked_div_ap(a: &Float, b: &Float) -> Float {
    if a.is_zero() {
        Float::with_val(a.prec(), 0.0)
    } else if b.is_zero() {
        Float::with_val(a.prec(), 1.0)
    } else {
        Float::with_val(a.prec(), a / b)
    }
}
