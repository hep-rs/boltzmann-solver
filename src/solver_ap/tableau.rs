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
    // use crate::{solver_ap::DEFAULT_WORKING_PRECISION as DWP, utilities::test::*};
    use crate::utilities::test::*;
    use ndarray::{array, prelude::*, FoldWhile, Zip};
    use rug::{ops::*, Float};
    use std::{f64::consts::PI, iter};

    const DWP: u32 = 53;

    macro_rules! solve_ode {
        ( $name:ident, $method:ident ) => {
            pub(crate) fn $name<F>(
                mut x: Array1<Float>,
                mut t: Float,
                tf: Float,
                f: F,
            ) -> (Float, Array1<Float>)
            where
                F: Fn(Float, &Array1<Float>) -> Array1<Float>,
            {
                use super::$method::*;

                // Fixed parameters
                let tol = Float::with_val(DWP, 1e-5);
                let delta_min = Float::with_val(DWP, 0.1);
                let delta_max = Float::with_val(DWP, 4.0);
                let h_min: Float = (tf.clone() - &t) / 1e5;
                let h_max: Float = (tf.clone() - &t) / 1e1;

                let zeros =
                    Array1::from_iter(iter::repeat(Float::with_val(DWP, 0.0)).take(x.dim()));

                let mut dx = [zeros.clone(), zeros.clone()];
                let mut k: [Array1<Float>; RK_S];
                unsafe {
                    k = std::mem::uninitialized();
                    for ki in &mut k[..] {
                        std::ptr::write(ki, zeros.clone());
                    }
                };

                let mut h: Float = (tf.clone() - &t) / 1e3;
                while &t < &tf {
                    let mut advance = false;

                    // Computer each k[i]
                    for i in 0..RK_S {
                        let ti = &t + RK_C[i] * h.clone();
                        let ai = RK_A[i];
                        let xi = (0..i).fold(x.clone(), |total, j| total + &k[j] * ai[j]);
                        k[i] = f(ti, &xi);
                        k[i].mapv_inplace(|v| v * &h);
                    }

                    // Compute the two estimates
                    dx[0] = (0..RK_S).fold(zeros.clone(), |total, i| total + &k[i] * RK_B[0][i]);
                    dx[1] = (0..RK_S).fold(zeros.clone(), |total, i| total + &k[i] * RK_B[1][i]);

                    // Get the error between the estimates
                    let err = Zip::from(&dx[0])
                        .and(&dx[1])
                        .fold_while(Float::with_val(DWP, 0.0), |e, a, b| {
                            let v = (a.clone() - b).abs();
                            FoldWhile::Continue(e.max(&v))
                        })
                        .into_inner()
                        / &h;

                    // If the error is within the tolerance, add the result
                    if &err < &tol {
                        advance = true;
                    }

                    // Compute the change in step size based on the current
                    // error And correspondingly adjust the step size
                    let mut h_est = if err.is_zero() {
                        h.clone() * &delta_max
                    } else {
                        let delta: Float = 0.9 * (&tol / err).pow(1.0 / f64::from(RK_ORDER + 1));

                        h.clone()
                            * if delta < delta_min {
                                &delta_min
                            } else if delta > delta_max {
                                &delta_max
                            } else {
                                &delta
                            }
                    };

                    // Prevent h from getting too small or too big in proportion
                    // to the current value of beta.  Also advance the
                    // integration irrespective of the local error if we reach
                    // the maximum or minimum step size.
                    if h_est > h_max {
                        h_est = h_max.clone();
                        advance = true;
                    } else if h_est < h_min {
                        h_est = h_min.clone();
                        advance = true;
                    }

                    // Check if the error is within the tolerance, or we are
                    // advancing irrespective of the local error
                    if advance {
                        x += &dx[0];
                        t += h;
                    }

                    // Adjust final step size if needed
                    let next_t: Float = t.clone() + &h_est;
                    if next_t > tf {
                        h_est = tf.clone() - &t;
                    }

                    h = h_est;
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
                let f = |_t: Float, x: &Array1<Float>| -> Array1<Float> {
                    array![-x[1].clone(), x[0].clone()]
                };

                let x = array![Float::with_val(DWP, 1.0), Float::with_val(DWP, 0.0)];
                let t = Float::with_val(DWP, 0.0);

                let (_t, x) = $solver(x, t, Float::with_val(DWP, 2.0 * PI), f);
                approx_eq(x[0].to_f64(), 1.0, $prec, 0.0);
                approx_eq(x[1].to_f64(), 0.0, $prec, 10.0f64.powf(-$prec));
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
                let f = |_t: Float, x: &Array1<Float>| -> Array1<Float> {
                    array![&x[0] * (2.0 - x[1].clone()), &x[1] * (x[0].clone() - 1.0)]
                };

                let x = array![Float::with_val(DWP, 1.0), Float::with_val(DWP, 2.7)];
                let t = Float::with_val(DWP, 0.0);

                let (_t, x) = $solver(x, t, Float::with_val(DWP, 10.0), f);
                approx_eq(x[0].to_f64(), 0.622_374_063_518_922_9, $prec, 0.0);
                approx_eq(x[1].to_f64(), 2.115_331_268_162_712_8, $prec, 0.0);
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
    use super::tests::{
        solve_ode_bs32, solve_ode_ck54, solve_ode_dp54, solve_ode_dp87, solve_ode_rkf54,
    };
    use crate::{solver_ap::DEFAULT_WORKING_PRECISION as DWP, utilities::test::*};
    use ndarray::{array, Array1};
    use rug::Float;
    use std::f64::consts::PI;
    use test::Bencher;

    macro_rules! bench_sine {
        ( $name:ident, $solver:ident ) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let f = |_t: Float, x: &Array1<Float>| -> Array1<Float> {
                    array![-x[1].clone(), x[0].clone()]
                };

                b.iter(|| {
                    let x = array![Float::with_val(DWP, 1.0), Float::with_val(DWP, 0.0)];
                    let t = Float::with_val(DWP, 0.0);

                    let (_, x) = $solver(x, t, Float::with_val(DWP, 2.0 * PI), f);
                    approx_eq(x[0].to_f64(), 1.0, 1.0, 0.0);
                    approx_eq(x[1].to_f64(), 0.0, 1.0, 10.0f64.powf(-1.0));
                });
            }
        };
    }

    bench_sine!(bench_sine_bs32, solve_ode_bs32);
    bench_sine!(bench_sine_ck54, solve_ode_ck54);
    bench_sine!(bench_sine_dp54, solve_ode_dp54);
    bench_sine!(bench_sine_dp87, solve_ode_dp87);
    bench_sine!(bench_sine_rkf54, solve_ode_rkf54);

    macro_rules! bench_lotka_volterra {
        ( $name:ident, $solver:ident ) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let f = |_t: Float, x: &Array1<Float>| -> Array1<Float> {
                    array![&x[0] * (2.0 - x[1].clone()), &x[1] * (x[0].clone() - 1.0)]
                };

                b.iter(|| {
                    let x = array![Float::with_val(DWP, 1.0), Float::with_val(DWP, 2.7)];
                    let t = Float::with_val(DWP, 0.0);

                    let (_, x) = $solver(x, t, Float::with_val(DWP, 10.0), f);
                    approx_eq(x[0].to_f64(), 0.622_374_063_518_922_9, 1.0, 0.0);
                    approx_eq(x[1].to_f64(), 2.115_331_268_162_712_8, 1.0, 0.0);
                });
            }
        };
    }

    bench_lotka_volterra!(bench_lotka_volterra_bs32, solve_ode_bs32);
    bench_lotka_volterra!(bench_lotka_volterra_ck54, solve_ode_ck54);
    bench_lotka_volterra!(bench_lotka_volterra_dp54, solve_ode_dp54);
    bench_lotka_volterra!(bench_lotka_volterra_dp87, solve_ode_dp87);
    bench_lotka_volterra!(bench_lotka_volterra_rkf54, solve_ode_rkf54);
}
