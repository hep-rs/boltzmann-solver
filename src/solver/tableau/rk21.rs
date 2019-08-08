//! Runge-Kutta method of order 2 in 3 stages.
//!
//! Coefficients are exported from Mathematica and can be obtained from
//! references therein.

#![allow(dead_code)]

pub const RK_ORDER: i32 = 2;
pub const RK_S: usize = 3;
pub const RK_A: [[f64; RK_S - 1]; RK_S] = [[0., 0.], [1., 0.], [0.5, 0.5]];
pub const RK_B: [f64; RK_S] = [0.5, 0.5, 0.];
pub const RK_C: [f64; RK_S] = [0., 1., 1.];
pub const RK_E: [f64; RK_S] = [-0.5, 0.6666666666666666, -0.16666666666666666];