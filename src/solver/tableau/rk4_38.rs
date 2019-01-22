#[allow(dead_code)]
pub const RK_ORDER: usize = 4;
#[allow(dead_code)]
pub const RK_A: [[f64; RK_ORDER - 1]; RK_ORDER - 1] = [
    [1.0 / 3.0, 0.0, 0.0],
    [-1.0 / 3.0, 1.0, 0.0],
    [1.0, -1.0, 1.0],
];
#[allow(dead_code)]
pub const RK_B: [[f64; RK_ORDER]; 2] = [
    [1.0, 0.0, 0.0, 0.0],
    [1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0],
];
#[allow(dead_code)]
pub const RK_C: [f64; RK_ORDER - 1] = [1.0 / 3.0, 2.0 / 3.0, 1.];
