//! Runge-Kutta method of order 4 in 5 stages.
//!
//! Coefficients are exported from Mathematica and can be obtained from
//! references therein.

#![allow(dead_code)]

pub const RK_ORDER: i32 = 4;
pub const RK_S: usize = 5;
pub const RK_A: [[f64; RK_S - 1]; RK_S] = [[0., 0., 0., 0.], [0.4, 0., 0., 0.], [-0.15, 0.75, 0., 0.], [0.4318181818181818, -0.3409090909090909, 0.9090909090909091, 0.], [0.1527777777777778, 0.3472222222222222, 0.3472222222222222, 0.1527777777777778]];
pub const RK_B: [f64; RK_S] = [0.1527777777777778, 0.3472222222222222, 0.3472222222222222, 0.1527777777777778, 0.];
pub const RK_C: [f64; RK_S] = [0., 0.4, 0.6, 1., 1.];
pub const RK_E: [f64; RK_S] = [0.013269665336144196, -0.06634832668072098, 0.06634832668072098, 0.14596631869758617, -0.15923598403373035];