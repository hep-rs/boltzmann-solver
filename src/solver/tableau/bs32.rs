//! Bogacki–Shampine Method
pub const RK_ORDER: usize = 3;
pub const RK_DIM: usize = 4;
pub const RK_A: [[f64; RK_DIM - 1]; RK_DIM - 1] = [
    [1.0 / 2.0, 0.0, 0.0],
    [0.0, 3.0 / 4.0, 0.0],
    [2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0],
];
pub const RK_B: [[f64; RK_DIM]; 2] = [
    [2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0],
    [7.0 / 24.0, 1.0 / 4.0, 1.0 / 3.0, 1.0 / 8.0],
];
pub const RK_C: [f64; RK_DIM - 1] = [1.0 / 2.0, 3.0 / 4.0, 1.0];
