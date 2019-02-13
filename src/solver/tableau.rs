/// # Butcher tableaux
///
/// A Butcher tableau is as described at https://en.wikipedia.org/wiki/Butcher_tableau.
///
///
/// Note that the indexing is slightly different than usual convention (and uses 0-indexing).  In
/// particular, for an `n`th order solver:
///
/// - the Runge-Kutta matrix `a[i][j]` is an `(n-1)x(n-1)` matrix with non-zero entries
///   located when `i ≤ j`.
/// - the weights vector `b[i]` is an `n` vector.
/// - the nodes vector `c[i]` is an `n-1` vector and differs from literature where
///   the first (always zero) entry is omitted.  Combined with 0-indexing, this
///   means that `c[0]` corresponds to *c₂* in the literature.
///
/// Lastly, adaptive methods with two weights vectors will be stored as
/// `b[0][i]` and `b[1][i]`; with `b[0]` containing the lower-order weights, and
/// `b[1]` the higher order weights.
pub(crate) mod euler;
pub(crate) mod midpoint;
pub(crate) mod rk4;
pub(crate) mod rk4_38;
pub(crate) mod rkf45;
