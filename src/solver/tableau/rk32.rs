//! Runge-Kutta method of order 3 in 4 stages.
//!
//! Coefficients are exported from Mathematica and can be obtained from
//! references therein.

#![allow(dead_code)]

pub const RK_ORDER: i32 = 3;
pub const RK_S: usize = 4;
pub const RK_A: [[f64; RK_S - 1]; RK_S] = [[0., 0., 0.], [0.5, 0., 0.], [-1., 2., 0.], [0.16666666666666666, 0.6666666666666666, 0.16666666666666666]];
pub const RK_B: [f64; RK_S] = [0.16666666666666666, 0.6666666666666666, 0.16666666666666666, 0.];
pub const RK_C: [f64; RK_S] = [0., 0.5, 1., 1.];
pub const RK_E: [f64; RK_S] = [-0.013119650859202537, 0.026239301718405075, 0.13155982542960126, -0.1446794762888038];