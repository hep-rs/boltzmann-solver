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
//!   k_1 &= h f(t_n, y_n), \\\\
//!   k_i &= h f\left(t_n + c_i h, y_n + \sum_{j=1}^{i-1} a_{ij} k_j \right),
//! \\end{align}
//!
//! and he local error is given by
//!
//! \\begin{equation}
//!   e = \sum_{i = 1}^{s} e_i k_i.
//! \\end{equation}
//!
//! Each \\(a_{ij}\\), \\(b_i\\) and \\(c_i\\) are specified by the Butcher
//! tableau, and \\(e_i = b_i - \hat b_i\\).  Note that although the above
//! notation uses 1-indexing, the parameters are defined in the submodules using
//! 0-indexing.

pub mod rk21;
pub mod rk32;
pub mod rk43;
pub mod rk54;
pub mod rk65;
pub mod rk76;
pub mod rk87;
pub mod rk98;

#[cfg(feature = "rk21")]
pub use rk21::{RK_A, RK_B, RK_C, RK_E, RK_ORDER, RK_S};
#[cfg(feature = "rk32")]
pub use rk32::{RK_A, RK_B, RK_C, RK_E, RK_ORDER, RK_S};
#[cfg(feature = "rk43")]
pub use rk43::{RK_A, RK_B, RK_C, RK_E, RK_ORDER, RK_S};
#[cfg(feature = "rk54")]
pub use rk54::{RK_A, RK_B, RK_C, RK_E, RK_ORDER, RK_S};
#[cfg(feature = "rk65")]
pub use rk65::{RK_A, RK_B, RK_C, RK_E, RK_ORDER, RK_S};
#[cfg(feature = "rk76")]
pub use rk76::{RK_A, RK_B, RK_C, RK_E, RK_ORDER, RK_S};
#[cfg(feature = "rk87")]
pub use rk87::{RK_A, RK_B, RK_C, RK_E, RK_ORDER, RK_S};
#[cfg(feature = "rk98")]
pub use rk98::{RK_A, RK_B, RK_C, RK_E, RK_ORDER, RK_S};

// Use Runge-Kutta 8(7) by default
#[cfg(all(
    not(feature = "rk21"),
    not(feature = "rk32"),
    not(feature = "rk43"),
    not(feature = "rk54"),
    not(feature = "rk65"),
    not(feature = "rk76"),
    not(feature = "rk87"),
    not(feature = "rk98")
))]
pub use rk87::{RK_A, RK_B, RK_C, RK_E, RK_ORDER, RK_S};

#[cfg(test)]
mod tests {
    use crate::utilities::test::approx_eq;
    use ndarray::{array, prelude::*};
    use std::{f64::consts::PI, mem, ptr};

    const TWO_PI: f64 = 2.0 * PI;

    macro_rules! solve_rk {
        ( $name:ident, $method:ident ) => {
            pub(crate) fn $name<F>(
                mut y: Array1<f64>,
                mut t: f64,
                tf: f64,
                mut f: F,
            ) -> (f64, Array1<f64>)
            where
                F: FnMut(f64, &Array1<f64>) -> Array1<f64>,
            {
                use super::$method::*;

                // Fixed parameters
                let tol = 1e-8;
                let delta_min = 0.1;
                let delta_max = 4.0;
                let h_min = (tf - t) / 1e14;
                let h_max = (tf - t) / 1e1;

                let mut dy;
                let mut dy_err;
                let mut k: [Array1<f64>; RK_S];
                unsafe {
                    k = mem::MaybeUninit::uninit().assume_init();
                    for ki in &mut k[..] {
                        ptr::write(ki, Array1::zeros(y.dim()));
                    }
                };

                let mut h = (tf - t) / 1e3;
                while t < tf {
                    let mut advance = false;

                    // Computer each k[i]
                    for i in 0..RK_S {
                        let ti = t + RK_C[i] * h;
                        let ai = RK_A[i];
                        let yi = (0..i).fold(y.clone(), |total, j| total + ai[j] * &k[j]);
                        k[i] = h * f(ti, &yi);
                    }

                    // Compute the two estimates
                    dy = (0..RK_S).fold(Array1::zeros(y.dim()), |total, i| total + &k[i] * RK_B[i]);
                    dy_err = (0..RK_S).fold(Array1::<f64>::zeros(y.dim()), |total, i| {
                        total + &k[i] * RK_E[i]
                    });

                    let err = dy_err.iter().fold(0_f64, |e, v| e.max(v.abs()));
                    // If the error is within the tolerance, add the result
                    if err < tol {
                        advance = true;
                    }

                    // Compute the change in step size based on the current error And
                    // correspondingly adjust the step size
                    let mut h_est = if err == 0.0 {
                        h * delta_max
                    } else {
                        let delta = 0.9 * (tol / err).powf(1.0 / f64::from(RK_ORDER + 1));

                        h * if delta < delta_min {
                            delta_min
                        } else if delta > delta_max {
                            delta_max
                        } else {
                            delta
                        }
                    };

                    // Prevent h from getting too small or too big in proportion to the
                    // current value of beta.  Also advance the integration irrespective
                    // of the local error if we reach the maximum or minimum step size.
                    if h_est > h_max {
                        h_est = h_max;
                        advance = true;
                    } else if h_est < h_min {
                        h_est = h_min;
                        advance = true;
                    }

                    // Check if the error is within the tolerance, or we are advancing
                    // irrespective of the local error
                    if advance {
                        y += &dy;
                        t += h;
                    } else {
                        log::trace!("Not incrementing step.");
                    }

                    // Adjust final step size if needed
                    if t + h_est > tf {
                        h_est = tf - t;
                    }

                    h = h_est;
                }

                (t, y)
            }
        };
    }

    solve_rk!(solve_rk21, rk21);
    solve_rk!(solve_rk32, rk32);
    solve_rk!(solve_rk43, rk43);
    solve_rk!(solve_rk54, rk54);
    solve_rk!(solve_rk65, rk65);
    solve_rk!(solve_rk76, rk76);
    solve_rk!(solve_rk87, rk87);
    solve_rk!(solve_rk98, rk98);

    macro_rules! test_sine {
        ( $name:ident, $solver:ident, $prec:expr ) => {
            #[test]
            fn $name() {
                let f = |_t: f64, x: &Array1<f64>| -> Array1<f64> { array![-x[1], x[0]] };

                let x = array![1.0, 0.0];
                let t = 0.0;

                let (_t, x) = $solver(x, t, TWO_PI, f);
                approx_eq(x[0], 1.0, $prec, 0.0);
                approx_eq(x[1], 0.0, $prec, 10_f64.powf(-$prec));
            }
        };
    }

    test_sine!(test_sine_rk21, solve_rk21, 6.0);
    test_sine!(test_sine_rk32, solve_rk32, 6.0);
    test_sine!(test_sine_rk43, solve_rk43, 6.0);
    test_sine!(test_sine_rk54, solve_rk54, 6.0);
    test_sine!(test_sine_rk65, solve_rk65, 6.0);
    test_sine!(test_sine_rk76, solve_rk76, 6.0);
    test_sine!(test_sine_rk87, solve_rk87, 6.0);
    test_sine!(test_sine_rk98, solve_rk98, 6.0);

    macro_rules! test_lotka_volterra {
        ( $name:ident, $solver:ident, $prec:expr ) => {
            #[test]
            fn $name() {
                let f = |_t: f64, x: &Array1<f64>| -> Array1<f64> {
                    array![x[0] * (2.0 - x[1]), x[1] * (x[0] - 1.0)]
                };

                let x = array![1.0, 2.7];
                let t = 0.0;

                let (_t, x) = $solver(x, t, 10.0, f);
                approx_eq(x[0], 0.622_374_063_518_922_9, $prec, 0.0);
                approx_eq(x[1], 2.115_331_268_162_712_8, $prec, 0.0);
            }
        };
    }

    test_lotka_volterra!(test_lotka_volterra_rk21, solve_rk21, 6.0);
    test_lotka_volterra!(test_lotka_volterra_rk32, solve_rk32, 6.0);
    test_lotka_volterra!(test_lotka_volterra_rk43, solve_rk43, 6.0);
    test_lotka_volterra!(test_lotka_volterra_rk54, solve_rk54, 6.0);
    test_lotka_volterra!(test_lotka_volterra_rk65, solve_rk65, 6.0);
    test_lotka_volterra!(test_lotka_volterra_rk76, solve_rk76, 6.0);
    test_lotka_volterra!(test_lotka_volterra_rk87, solve_rk87, 6.0);
    test_lotka_volterra!(test_lotka_volterra_rk98, solve_rk98, 6.0);
}

#[cfg(all(test, feature = "nightly"))]
mod benches {
    use super::tests::{
        solve_rk21, solve_rk32, solve_rk43, solve_rk54, solve_rk65, solve_rk76, solve_rk87,
        solve_rk98,
    };
    use ndarray::{array, Array1};
    use std::f64::consts::PI;
    use test::Bencher;

    const TWO_PI: f64 = 2.0 * PI;

    macro_rules! bench_sine {
        ( $name:ident, $solver:ident ) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let f = |_t: f64, x: &Array1<f64>| -> Array1<f64> { array![-x[1], x[0]] };

                b.iter(|| {
                    let x = array![1.0, 0.0];
                    let t = 0.0;

                    let r = $solver(x, t, TWO_PI, f);
                    test::black_box(r)
                });
            }
        };
    }

    bench_sine!(bench_sine_rk21, solve_rk21);
    bench_sine!(bench_sine_rk32, solve_rk32);
    bench_sine!(bench_sine_rk43, solve_rk43);
    bench_sine!(bench_sine_rk54, solve_rk54);
    bench_sine!(bench_sine_rk65, solve_rk65);
    bench_sine!(bench_sine_rk76, solve_rk76);
    bench_sine!(bench_sine_rk87, solve_rk87);
    bench_sine!(bench_sine_rk98, solve_rk98);

    macro_rules! bench_lotka_volterra {
        ( $name:ident, $solver:ident ) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let f = |_t: f64, x: &Array1<f64>| -> Array1<f64> {
                    array![x[0] * (2.0 - x[1]), x[1] * (x[0] - 1.0)]
                };

                b.iter(|| {
                    let x = array![1.0, 2.7];
                    let t = 0.0;

                    let r = $solver(x, t, 10.0, f);
                    test::black_box(r)
                });
            }
        };
    }

    bench_lotka_volterra!(bench_lotka_volterra_rk21, solve_rk21);
    bench_lotka_volterra!(bench_lotka_volterra_rk32, solve_rk32);
    bench_lotka_volterra!(bench_lotka_volterra_rk43, solve_rk43);
    bench_lotka_volterra!(bench_lotka_volterra_rk54, solve_rk54);
    bench_lotka_volterra!(bench_lotka_volterra_rk65, solve_rk65);
    bench_lotka_volterra!(bench_lotka_volterra_rk76, solve_rk76);
    bench_lotka_volterra!(bench_lotka_volterra_rk87, solve_rk87);
    bench_lotka_volterra!(bench_lotka_volterra_rk98, solve_rk98);
}
