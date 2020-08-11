//! Runge-Kutta method of order 9 in 16 stages.
//!
//! Coefficients are exported from Mathematica and can be obtained from
//! references therein.

#![allow(clippy::all)]
#![allow(dead_code)]

pub const RK_ORDER: i32 = 9;
pub const RK_S: usize = 16;
pub const RK_A: [[f64; RK_S - 1]; RK_S] = [
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0.04, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [
        -0.019_885_273_191_822_924,
        0.116_372_633_329_696_54,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
    ],
    [
        0.036_182_760_051_702_61,
        0.,
        0.108_548_280_155_107_81,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
    ],
    [
        2.272_114_264_290_177_5,
        0.,
        -8.526_886_447_976_398,
        6.830_772_183_686_220_5,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
    ],
    [
        0.050_943_855_353_893_744,
        0.,
        0.,
        0.175_586_504_980_907_14,
        0.000_702_296_127_075_751_5,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
    ],
    [
        0.142_478_366_868_328_5,
        0.,
        0.,
        -0.354_179_943_466_868_4,
        0.075_953_154_502_951_01,
        0.676_515_765_633_712_1,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
    ],
    [
        0.071_111_111_111_111_11,
        0.,
        0.,
        0.,
        0.,
        0.327_990_928_760_589_83,
        0.240_897_960_128_299_03,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
    ],
    [
        0.07125,
        0.,
        0.,
        0.,
        0.,
        0.326_884_245_157_524_6,
        0.115_615_754_842_475_45,
        -0.03375,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
    ],
    [
        0.048_226_773_224_658_1,
        0.,
        0.,
        0.,
        0.,
        0.039_485_599_804_954,
        0.105_885_116_193_465_81,
        -0.021_520_063_204_743_093,
        -0.104_537_426_018_334_82,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
    ],
    [
        -0.026_091_134_357_549_232,
        0.,
        0.,
        0.,
        0.,
        0.033_333_333_333_333_33,
        -0.165_250_400_663_810_5,
        0.034_346_641_183_686_17,
        0.159_575_828_321_520_9,
        0.214_085_732_182_819_32,
        0.,
        0.,
        0.,
        0.,
        0.,
    ],
    [
        -0.036_284_233_962_556_61,
        0.,
        0.,
        0.,
        0.,
        -1.096_167_597_427_208_7,
        0.182_603_550_432_133_1,
        0.070_822_544_441_706_85,
        -0.023_136_470_184_824_23,
        0.271_120_472_632_093_3,
        1.308_133_749_422_980_6,
        0.,
        0.,
        0.,
        0.,
    ],
    [
        -0.507_463_505_641_697_5,
        0.,
        0.,
        0.,
        0.,
        -6.631_342_198_657_237,
        -0.252_748_010_090_880_5,
        -0.495_261_238_003_609_44,
        0.293_252_554_525_389_3,
        1.440_108_693_768_281_4,
        6.237_934_498_647_055,
        0.727_019_205_452_698_8,
        0.,
        0.,
        0.,
    ],
    [
        0.613_011_825_695_593_3,
        0.,
        0.,
        0.,
        0.,
        9.088_803_891_640_463,
        -0.407_378_815_629_345_36,
        1.790_733_389_490_374_7,
        0.714_927_166_761_755,
        -1.438_580_857_841_722_7,
        -8.263_329_312_064_74,
        -1.537_570_570_808_865,
        0.345_383_282_756_487_1,
        0.,
        0.,
    ],
    [
        -1.211_697_910_343_874,
        0.,
        0.,
        0.,
        0.,
        -19.055_818_715_595_95,
        1.263_060_675_389_874,
        -6.913_916_969_178_459,
        -0.676_462_266_509_498_8,
        3.367_860_445_026_608,
        18.006_751_643_125_906,
        6.838_828_926_794_28,
        -1.031_516_451_921_950_6,
        0.412_910_623_213_062_27,
        0.,
    ],
    [
        2.157_389_007_494_053_6,
        0.,
        0.,
        0.,
        0.,
        23.807_122_198_095_808,
        0.886_277_924_921_658_9,
        13.139_130_397_598_764,
        -2.604_415_709_287_714,
        -5.193_859_949_783_872,
        -20.412_340_711_541_51,
        -12.300_856_252_505_723,
        1.521_553_095_008_539_4,
        0.,
        0.,
    ],
];
pub const RK_B: [f64; RK_S] = [
    0.014_588_852_784_055_398,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.002_024_197_887_889_332_5,
    0.217_804_708_456_971_67,
    0.127_489_534_085_438_95,
    0.224_461_774_546_313_2,
    0.178_725_449_125_990_3,
    0.075_943_447_580_965_58,
    0.129_484_587_919_756_14,
    0.029_477_447_612_619_417,
    0.,
];
pub const RK_C: [f64; RK_S] = [
    0.,
    0.04,
    0.096_487_360_137_873_62,
    0.144_731_040_206_810_44,
    0.576,
    0.227_232_656_461_876_63,
    0.540_767_343_538_123_4,
    0.64,
    0.48,
    0.06754,
    0.25,
    0.677_092_015_354_324_3,
    0.8115,
    0.906,
    1.,
    1.,
];
pub const RK_E: [f64; RK_S] = [
    -0.005_757_813_768_188_949,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    -1.067_593_453_094_810_8,
    0.140_996_361_343_939_78,
    0.014_411_715_396_914_923,
    -0.030_796_961_251_883_03,
    1.161_315_257_817_906_7,
    -0.322_211_134_861_185_8,
    0.129_484_587_919_756_14,
    0.029_477_447_612_619_417,
    -0.049_326_007_115_068_4,
];