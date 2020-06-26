//! Runge-Kutta method of order 3 in 4 stages.
//!
//! Coefficients are exported from Mathematica and can be obtained from
//! references therein.

#![allow(clippy::all)]
#![allow(dead_code)]

pub const RK_ORDER: i32 = 3;
pub const RK_S: usize = 4;
pub const RK_A: [[f64; RK_S - 1]; RK_S] = [
    [0., 0., 0.],
    [0.5, 0., 0.],
    [-1., 2., 0.],
    [
        0.166_666_666_666_666_66,
        0.666_666_666_666_666_6,
        0.166_666_666_666_666_66,
    ],
];
pub const RK_B: [f64; RK_S] = [
    0.166_666_666_666_666_66,
    0.666_666_666_666_666_6,
    0.166_666_666_666_666_66,
    0.,
];
pub const RK_C: [f64; RK_S] = [0., 0.5, 1., 1.];
pub const RK_E: [f64; RK_S] = [
    -0.013_119_650_859_202_537,
    0.026_239_301_718_405_075,
    0.131_559_825_429_601_26,
    -0.144_679_476_288_803_8,
];
