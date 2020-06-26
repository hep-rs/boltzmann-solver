//! Runge-Kutta method of order 6 in 9 stages.
//!
//! Coefficients are exported from Mathematica and can be obtained from
//! references therein.

#![allow(clippy::all)]
#![allow(dead_code)]

pub const RK_ORDER: i32 = 6;
pub const RK_S: usize = 9;
pub const RK_A: [[f64; RK_S - 1]; RK_S] = [
    [0., 0., 0., 0., 0., 0., 0., 0.],
    [0.18, 0., 0., 0., 0., 0., 0., 0.],
    [
        0.089_506_172_839_506_17,
        0.077_160_493_827_160_49,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
    ],
    [0.0625, 0., 0.1875, 0., 0., 0., 0., 0.],
    [0.316_516, 0., -1.044_948, 1.258_4322, 0., 0., 0., 0.],
    [
        0.272_326_127_364_856_3,
        0.,
        -0.825_133_603_238_866_4,
        1.048_091_767_881_241_6,
        0.104_715_707_992_768_57,
        0.,
        0.,
        0.,
    ],
    [
        -0.166_994_185_997_165_15,
        0.,
        0.631_708_502_024_291_5,
        0.174_610_445_527_738_77,
        -1.066_535_645_908_606_6,
        1.227_210_884_353_741_5,
        0.,
        0.,
    ],
    [
        0.364_237_516_869_095_83,
        0.,
        -0.204_048_582_995_951_4,
        -0.348_837_378_160_686_44,
        3.261_932_303_285_686_6,
        -2.755_102_040_816_326_7,
        0.681_818_181_818_181_8,
        0.,
    ],
    [
        0.076_388_888_888_888_9,
        0.,
        0.,
        0.369_408_369_408_369_4,
        0.,
        0.248_015_873_015_873_02,
        0.236_742_424_242_424_25,
        0.069_444_444_444_444_45,
    ],
];
pub const RK_B: [f64; RK_S] = [
    0.076_388_888_888_888_9,
    0.,
    0.,
    0.369_408_369_408_369_4,
    0.,
    0.248_015_873_015_873_02,
    0.236_742_424_242_424_25,
    0.069_444_444_444_444_45,
    0.,
];
pub const RK_C: [f64; RK_S] = [
    0.,
    0.18,
    0.166_666_666_666_666_66,
    0.25,
    0.53,
    0.6,
    0.8,
    1.,
    1.,
];
pub const RK_E: [f64; RK_S] = [
    0.017_688_679_245_283_02,
    0.,
    0.,
    -0.111_317_254_174_397_03,
    0.853_412_420_769_190_9,
    -0.956_632_653_061_224_5,
    0.236_742_424_242_424_25,
    0.128_686_817_516_604_74,
    -0.168_580_434_537_881_35,
];
