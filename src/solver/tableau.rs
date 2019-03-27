//! # Butcher tableaux
//!
//! A Butcher tableau is as described at
//! https://en.wikipedia.org/wiki/Butcher_tableau.  Specifically, given
//! \\(y_n\\), the estimate for \\(y_{n+1}\\) is:
//!
//! \\begin{equation}
//!   y_{n+1} = y_n + \sum_{i = 1}^{s} b_i k_i
//! \\end{equation}
//!
//! where
//!
//! \\begin{align}
//!   k_1 &= h f(x_n, y_n), \\\\
//!   k_i &= h f\left(x_n + c_i h, y_n + \sum_{j=1}^{i-1} a_{ij} k_j \right),
//! \\end{align}
//!
//! and \\(a_{ij}\\), \\(b_i\\) and \\(c_i\\) are specified by the Butcher
//! tableau.  Note that although the above notation uses 1-indexing, the
//! parameters are defined in the submodules using 0-indexing.
//!
//! For a Runge-Kutta method of size \\(s\\), we have:
//! - the Runge-Kutta matrix `a[i][j]` is an `s×(s-1)` matrix with non-zero
//!   entries located when `i < j`;
//! - the weights vector `b[i]` is an `s` vector.  Note that for adaptive
//!   methods, `b` will instead be a `2×s` matrix with `b[0]` containing the
//!   weights for the estimate of order `RK_ORDER` and `b[1]` containing the
//!   weights for the estimate of order `RK_ORDER - 1`; and,
//! - the nodes vector `c[i]` is an `s` vector.

pub mod bs32;
pub mod ck54;
pub mod dp54;
pub mod dp87;
pub mod rkf54;

#[cfg(test)]
mod tests {
    use crate::utilities::test::*;
    use ndarray::{array, prelude::*, FoldWhile, Zip};
    use std::f64::consts::PI;

    const TWO_PI: f64 = 2.0 * PI;

    macro_rules! solve_ode {
        ( $name:ident, $method:ident ) => {
            pub(crate) fn $name<F>(
                mut x: Array1<f64>,
                mut t: f64,
                tf: f64,
                f: F,
            ) -> (f64, Array1<f64>)
            where
                F: Fn(f64, &Array1<f64>) -> Array1<f64>,
            {
                use super::$method::*;

                let mut dx = [Array1::zeros(x.dim()), Array1::zeros(x.dim())];
                let mut k: [Array1<f64>; RK_S];
                unsafe {
                    k = std::mem::uninitialized();
                    for ki in &mut k[..] {
                        std::ptr::write(ki, Array1::zeros(x.dim()));
                    }
                };

                let mut h = (tf - t) / 1e3;
                let h_min = (tf - t) / 1e5;
                let h_max = (tf - t) / 1e1;
                while t < tf {
                    // Computer each k[i]
                    for i in 0..RK_S {
                        let ti = t + RK_C[i] * h;
                        let ai = RK_A[i];
                        let xi = (0..i).fold(x.clone(), |total, j| total + ai[j] * &k[j]);
                        k[i] = h * f(ti, &xi);
                    }

                    // Compute the two estimates
                    dx[0] = (0..RK_S).fold(Array1::zeros(x.dim()), |total, i| {
                        total + &k[i] * RK_B[0][i]
                    });
                    dx[1] = (0..RK_S).fold(Array1::zeros(x.dim()), |total, i| {
                        total + &k[i] * RK_B[1][i]
                    });

                    // Get the error between the estimates
                    let err = Zip::from(&dx[0])
                        .and(&dx[1])
                        .fold_while(0.0, |e, a, b| {
                            let v = (a - b).abs();
                            if v > e {
                                FoldWhile::Continue(v)
                            } else {
                                FoldWhile::Continue(e)
                            }
                        })
                        .into_inner()
                        / h;

                    // If the error is within the tolerance, add the result
                    if err < 1e-5 {
                        x = x + &dx[0];
                        t += h;
                    }

                    // Compute the change in step size based on the current error
                    let delta = 0.9 * (1e-5 / err).powf(1.0 / f64::from(RK_ORDER + 1));

                    // And correspondingly adjust the error
                    h *= if delta < 0.8 {
                        0.8
                    } else if delta > 1.2 {
                        1.2
                    } else {
                        delta
                    };

                    // Make sure the error never becomes *too* big
                    if h < h_min {
                        h = h_min;
                    } else if h > h_max {
                        h = h_max;
                    };

                    // And treat the end specially
                    if t + h > tf {
                        h = tf - t;
                    }
                }

                (t, x)
            }
        };
    }

    solve_ode!(solve_ode_bs32, bs32);
    solve_ode!(solve_ode_ck54, ck54);
    solve_ode!(solve_ode_dp54, dp54);
    solve_ode!(solve_ode_dp87, dp87);
    solve_ode!(solve_ode_rkf54, rkf54);

    macro_rules! test_sine {
        ( $name:ident, $solver:ident, $prec:expr ) => {
            #[test]
            fn $name() {
                let f = |_t: f64, x: &Array1<f64>| -> Array1<f64> { array![-x[1], x[0]] };

                let x = array![1.0, 0.0];
                let t = 0.0;

                let (_, x) = $solver(x, t, TWO_PI, f);
                approx_eq(x[0], 1.0, $prec, 0.0);
                approx_eq(x[1], 0.0, $prec, 10.0f64.powf(-$prec));
            }
        };
    }

    test_sine!(test_sine_bs32, solve_ode_bs32, 4.0);
    test_sine!(test_sine_ck54, solve_ode_ck54, 4.0);
    test_sine!(test_sine_dp54, solve_ode_dp54, 4.0);
    test_sine!(test_sine_dp87, solve_ode_dp87, 4.0);
    test_sine!(test_sine_rkf54, solve_ode_rkf54, 4.0);

    macro_rules! test_lotka_volterra {
        ( $name:ident, $solver:ident, $prec:expr ) => {
            #[test]
            fn $name() {
                let f = |_t: f64, x: &Array1<f64>| -> Array1<f64> {
                    array![x[0] * (2.0 - x[1]), x[1] * (x[0] - 1.0)]
                };

                let x = array![1.0, 2.7];
                let t = 0.0;

                let (_, x) = $solver(x, t, 10.0, f);
                approx_eq(x[0], 0.622_374_063_518_922_9, $prec, 0.0);
                approx_eq(x[1], 2.115_331_268_162_712_8, $prec, 0.0);
            }
        };
    }

    test_lotka_volterra!(test_lotka_volterra_bs32, solve_ode_bs32, 4.0);
    test_lotka_volterra!(test_lotka_volterra_ck54, solve_ode_ck54, 4.0);
    test_lotka_volterra!(test_lotka_volterra_dp54, solve_ode_dp54, 4.0);
    test_lotka_volterra!(test_lotka_volterra_dp87, solve_ode_dp87, 4.0);
    test_lotka_volterra!(test_lotka_volterra_rkf54, solve_ode_rkf54, 4.0);
}

#[cfg(feature = "nightly")]
#[cfg(test)]
mod bench {
    use super::tests::{solve_ode_bs32, solve_ode_ck54, solve_ode_dp54, solve_ode_rkf54};
    use crate::utilities::test::*;
    use ndarray::{array, Array1};
    use std::f64::consts::PI;
    use test::Bencher;

    const TWO_PI: f64 = 2.0 * PI;

    macro_rules! bench_sine {
        ( $name:ident, $solver:ident ) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let f = |_t: f64, x: &Array1<f64>| -> Array1<f64> { array![-x[1], x[0]] };

                let h = TWO_PI / 1e5;
                b.iter(|| {
                    let x = array![1.0, 0.0];
                    let t = 0.0;

                    let (_, x) = $solver(x, t, TWO_PI, h, f);
                    approx_eq(x[0], 1.0, 1.0, 0.0);
                    approx_eq(x[1], h, 1.0, 0.0);
                });
            }
        };
    }

    bench_sine!(bench_sine_bs32, solve_ode_bs32);
    bench_sine!(bench_sine_ck54, solve_ode_ck54);
    bench_sine!(bench_sine_dp54, solve_ode_dp54);
    bench_sine!(bench_sine_rkf54, solve_ode_rkf54);

    macro_rules! bench_lotka_volterra {
        ( $name:ident, $solver:ident ) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let f = |_t: f64, x: &Array1<f64>| -> Array1<f64> {
                    array![x[0] * (2.0 - x[1]), x[1] * (x[0] - 1.0)]
                };

                let h = 10.0 / 1e5;
                b.iter(|| {
                    let x = array![1.0, 2.7];
                    let t = 0.0;

                    let (_, x) = $solver(x, t, 10.0, h, f);
                    approx_eq(x[0], 0.622_374_063_518_922_9, 1.0, 0.0);
                    approx_eq(x[1], 2.115_331_268_162_712_8, 1.0, 0.0);
                });
            }
        };
    }

    bench_lotka_volterra!(bench_lotka_volterra_bs32, solve_ode_bs32);
    bench_lotka_volterra!(bench_lotka_volterra_ck54, solve_ode_ck54);
    bench_lotka_volterra!(bench_lotka_volterra_dp54, solve_ode_dp54);
    bench_lotka_volterra!(bench_lotka_volterra_rkf54, solve_ode_rkf54);
}
