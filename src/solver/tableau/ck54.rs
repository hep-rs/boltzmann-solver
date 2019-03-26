//! Cash–Karp Method
pub const RK_ORDER: usize = 5;
pub const RK_DIM: usize = 6;
pub const RK_A: [[f64; RK_DIM - 1]; RK_DIM - 1] = [
    [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0],
    [3.0 / 10.0, -9.0 / 10.0, 6.0 / 5.0, 0.0, 0.0],
    [-11.0 / 54.0, 5.0 / 2.0, -70.0 / 27.0, 35.0 / 27.0, 0.0],
    [
        1631.0 / 55_296.0,
        175.0 / 512.0,
        575.0 / 13_824.0,
        44_275.0 / 110_592.0,
        253.0 / 4096.0,
    ],
];
pub const RK_B: [[f64; RK_DIM]; 2] = [
    [
        37.0 / 378.0,
        0.0,
        250.0 / 621.0,
        125.0 / 594.0,
        0.0,
        512.0 / 1771.0,
    ],
    [
        2825.0 / 27_648.0,
        0.0,
        18_575.0 / 48_384.0,
        13_525.0 / 55_296.0,
        277.0 / 14_336.0,
        1.0 / 4.0,
    ],
];
pub const RK_C: [f64; RK_DIM - 1] = [1.0 / 5.0, 3.0 / 10.0, 3.0 / 5.0, 1.0, 7.0 / 8.0];
