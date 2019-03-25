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
pub(crate) mod midpoint;
pub(crate) mod rk4;
pub(crate) mod rk4_38;
pub(crate) mod rkf45;

#[cfg(test)]
mod tests {
    use ndarray::{array, Array1};
    use std::f64::consts::PI;

    const TWO_PI: f64 = 2.0 * PI;

    macro_rules! solve_ode {
        ( $name:ident, $method:ident ) => {
            pub(crate) fn $name<F>(
                mut x: Array1<f64>,
                mut t: f64,
                tf: f64,
                h: f64,
                f: F,
            ) -> (f64, Array1<f64>)
            where
                F: Fn(f64, &Array1<f64>) -> Array1<f64>,
            {
                use super::$method::*;

                let mut dx = [Array1::zeros(x.dim()), Array1::zeros(x.dim())];
                let mut k: [Array1<f64>; RK_ORDER + 1];
                unsafe {
                    k = std::mem::uninitialized();
                    for ki in &mut k[..] {
                        std::ptr::write(ki, Array1::zeros(2));
                    }
                };

                while t < tf {
                    k[0] = f(t, &x) * h;
                    for i in 0..(RK_ORDER - 1) {
                        let t_tmp = t + RK_C[i] * h;
                        let x_tmp = (0..=i).fold(x.clone(), |total, j| total + &k[j] * RK_A[i][j]);
                        k[i + 1] = f(t_tmp, &x_tmp) * h;
                    }

                    dx[0] = (0..RK_ORDER)
                        .fold(Array1::zeros(2), |total, i| total + &(&k[i] * RK_B[0][i]));
                    dx[1] = (0..RK_ORDER)
                        .fold(Array1::zeros(2), |total, i| total + &(&k[i] * RK_B[1][i]));

                    x += &dx[1];
                    t += h;
                }

                (t, x)
            }
        };
    }

    solve_ode!(solve_ode_midpoint, midpoint);
    solve_ode!(solve_ode_rk4, rk4);
    solve_ode!(solve_ode_rk4_38, rk4_38);
    solve_ode!(solve_ode_rkf45, rkf45);

    macro_rules! test_sine {
        ( $name:ident, $solver:ident ) => {
            #[test]
            fn $name() {
                let f = |_t: f64, x: &Array1<f64>| -> Array1<f64> { array![-x[1], x[0]] };

                let h = TWO_PI / 1e5;
                let x = array![1.0, 0.0];
                let t = 0.0;

                let (_, x) = $solver(x, t, TWO_PI, h, f);
                assert!((dbg!(x[0]) - 1.0).abs() < h);
                assert!((dbg!(x[1]) - h).abs() < h);
            }
        };
    }

    test_sine!(test_sine_midpoint, solve_ode_midpoint);
    test_sine!(test_sine_rk4, solve_ode_rk4);
    test_sine!(test_sine_rk4_38, solve_ode_rk4_38);
    test_sine!(test_sine_rkf45, solve_ode_rkf45);

    macro_rules! test_lotka_volterra {
        ( $name:ident, $solver:ident ) => {
            #[test]
            fn $name() {
                let f = |_t: f64, x: &Array1<f64>| -> Array1<f64> {
                    array![x[0] * (2.0 - x[1]), x[1] * (x[0] - 1.0)]
                };

                let h = 10.0 / 1e5;
                let x = array![1.0, 2.7];
                let t = 0.0;

                let (_, x) = $solver(x, t, 10.0, h, f);
                assert!((dbg!(x[0]) - 0.62237406351892283361214420345).abs() < h);
                assert!((dbg!(x[1]) - 2.1153312681627127969862394480).abs() < h);
            }
        };
    }

    test_lotka_volterra!(test_lotka_volterra_midpoint, solve_ode_midpoint);
    test_lotka_volterra!(test_lotka_volterra_rk4, solve_ode_rk4);
    test_lotka_volterra!(test_lotka_volterra_rk4_38, solve_ode_rk4_38);
    test_lotka_volterra!(test_lotka_volterra_rkf45, solve_ode_rkf45);
}

#[cfg(feature = "nightly")]
#[cfg(test)]
mod bench {
    use super::tests::{solve_ode_midpoint, solve_ode_rk4, solve_ode_rk4_38, solve_ode_rkf45};
    use ndarray::{array, Array1};
    use std::f64::consts::PI;
    use test::{black_box, Bencher};

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
                    assert!((dbg!(x[0]) - 1.0).abs() < h);
                    assert!((dbg!(x[1]) - h).abs() < h);
                });
            }
        };
    }

    bench_sine!(bench_sine_midpoint, solve_ode_midpoint);
    bench_sine!(bench_sine_rk4, solve_ode_rk4);
    bench_sine!(bench_sine_rk4_38, solve_ode_rk4_38);
    bench_sine!(bench_sine_rkf45, solve_ode_rkf45);

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
                    assert!((dbg!(x[0]) - 0.62237406351892283361214420345).abs() < h);
                    assert!((dbg!(x[1]) - 2.1153312681627127969862394480).abs() < h);
                });
            }
        };
    }

    bench_lotka_volterra!(bench_lotka_volterra_midpoint, solve_ode_midpoint);
    bench_lotka_volterra!(bench_lotka_volterra_rk4, solve_ode_rk4);
    bench_lotka_volterra!(bench_lotka_volterra_rk4_38, solve_ode_rk4_38);
    bench_lotka_volterra!(bench_lotka_volterra_rkf45, solve_ode_rkf45);
}
