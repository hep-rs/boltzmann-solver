//! Runge-Kutta method of order 7 in 10 stages.
//!
//! Coefficients are exported from Mathematica and can be obtained from
//! references therein.

#![allow(clippy::all)]
#![allow(dead_code)]

pub const RK_ORDER: i32 = 7;
pub const RK_S: usize = 10;
pub const RK_A: [[f64; RK_S - 1]; RK_S] = [
    [0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0.005, 0., 0., 0., 0., 0., 0., 0., 0.],
    [
        -1.07679012345679,
        1.185679012345679,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
    ],
    [0.04083333333333333, 0., 0.1225, 0., 0., 0., 0., 0., 0.],
    [
        0.6360714285714286,
        0.,
        -2.4444642857142855,
        2.263392857142857,
        0.,
        0.,
        0.,
        0.,
        0.,
    ],
    [
        -2.5351211079349243,
        0.,
        10.299374654449268,
        -7.951303288599059,
        0.793011489231006,
        0.,
        0.,
        0.,
        0.,
    ],
    [
        1.0018765812524633,
        0.,
        -4.16657128244238,
        3.8343432929128642,
        -0.5023333356071085,
        0.6676847438841608,
        0.,
        0.,
        0.,
    ],
    [
        27.255018354630767,
        0.,
        -42.00461727841064,
        -10.53571312661949,
        80.49553671141194,
        -67.34388227179052,
        13.048657610777939,
        0.,
        0.,
    ],
    [
        -3.0397378057114963,
        0.,
        10.1381614103298,
        -6.429305674864722,
        -1.5864371483408275,
        1.8921781841968426,
        0.01969933540760887,
        0.005441698982793324,
        0.,
    ],
    [
        -1.4449518916777735,
        0.,
        8.031891385995593,
        -7.583174166340134,
        3.5816169353190075,
        -2.436972263219953,
        0.8515899999232618,
        0.,
        0.,
    ],
];
pub const RK_B: [f64; RK_S] = [
    0.047425837833706755,
    0.,
    0.,
    0.2562236165937056,
    0.2695137683307421,
    0.12686622409092785,
    0.2488722594206007,
    0.003074483740820063,
    0.04802380998949694,
    0.,
];
pub const RK_C: [f64; RK_S] = [
    0.,
    0.005,
    0.10888888888888888,
    0.16333333333333333,
    0.455,
    0.6059617471462914,
    0.835,
    0.915,
    1.,
    1.,
];
pub const RK_E: [f64; RK_S] = [
    -0.00005940986559287495,
    0.,
    0.,
    0.00022949070679929363,
    -0.0010710124799348211,
    0.0018100372466678992,
    -0.003172427816837889,
    0.003074483740820063,
    0.04802380998949694,
    -0.048834971521418614,
];
