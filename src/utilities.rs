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
