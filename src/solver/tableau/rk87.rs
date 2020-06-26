//! Runge-Kutta method of order 8 in 13 stages.
//!
//! Coefficients are exported from Mathematica and can be obtained from
//! references therein.

#![allow(clippy::all)]
#![allow(dead_code)]

pub const RK_ORDER: i32 = 8;
pub const RK_S: usize = 13;
pub const RK_A: [[f64; RK_S - 1]; RK_S] = [
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [
        0.087_400_846_504_915_24,
        0.025_487_604_938_654_32,
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
        0.042_333_169_291_338_58,
        0.,
        0.126_999_507_874_015_76,
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
        0.426_095_058_887_422_6,
        0.,
        -1.598_795_284_659_152_2,
        1.596_700_225_771_729_8,
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
        0.050_719_337_296_713_93,
        0.,
        0.,
        0.254_333_772_646_004_1,
        0.203_946_890_057_282,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
    ],
    [
        -0.290_003_747_175_231_1,
        0.,
        0.,
        1.344_187_391_026_078_9,
        -2.864_777_943_361_443,
        2.677_594_299_510_595,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
    ],
    [
        0.098_535_011_337_993_54,
        0.,
        0.,
        0.,
        0.221_926_806_307_513_85,
        -0.181_406_229_118_069_94,
        0.010_944_411_472_562_547,
        0.,
        0.,
        0.,
        0.,
        0.,
    ],
    [
        0.387_110_525_457_311_4,
        0.,
        0.,
        -1.442_445_497_485_527_6,
        2.905_398_189_069_950_7,
        -1.853_771_069_630_106,
        0.140_036_480_987_281_53,
        0.572_739_408_114_958_2,
        0.,
        0.,
        0.,
        0.,
    ],
    [
        -0.161_244_034_444_393_06,
        0.,
        0.,
        -0.173_396_029_573_589_85,
        -1.301_289_281_406_514_7,
        1.137_950_375_173_861_8,
        -0.031_747_649_663_966_88,
        0.933_512_938_249_336_8,
        -0.083_786_318_334_733_85,
        0.,
        0.,
        0.,
    ],
    [
        -0.019_199_444_881_589_534,
        0.,
        0.,
        0.273_308_572_652_642_86,
        -0.675_349_732_069_443_7,
        0.341_518_498_138_460_14,
        -0.067_950_064_803_375_77,
        0.096_591_752_247_623_87,
        0.132_530_825_111_821_02,
        0.368_549_593_603_861_1,
        0.,
        0.,
    ],
    [
        0.609_187_740_364_528_9,
        0.,
        0.,
        -2.272_569_085_898_002,
        4.757_898_342_694_03,
        -5.516_106_706_692_758,
        0.290_059_636_968_011_96,
        0.569_142_396_335_903_7,
        0.792_679_576_033_216_7,
        0.154_737_204_532_888_22,
        1.614_970_895_662_181_5,
        0.,
    ],
    [
        0.887_357_622_085_347_2,
        0.,
        0.,
        -2.975_459_782_108_536_5,
        5.600_717_009_488_163_5,
        -5.915_607_450_536_674,
        0.220_296_891_561_349_27,
        0.101_550_978_244_622_17,
        1.151_434_564_738_605_5,
        1.929_710_166_527_124_1,
        0.,
        0.,
    ],
];
pub const RK_B: [f64; RK_S] = [
    0.044_729_564_666_695_705,
    0.,
    0.,
    0.,
    0.,
    0.156_910_335_277_082,
    0.184_609_734_081_516_37,
    0.225_163_806_020_869_9,
    0.147_946_156_519_702_33,
    0.076_055_542_444_955_83,
    0.122_772_902_350_186_19,
    0.041_811_958_638_991_63,
    0.,
];
pub const RK_C: [f64; RK_S] = [
    0.,
    0.25,
    0.112_888_451_443_569_56,
    0.169_332_677_165_354_33,
    0.424,
    0.509,
    0.867,
    0.15,
    0.709_068_036_513_868_4,
    0.32,
    0.45,
    1.,
    1.,
];
pub const RK_E: [f64; RK_S] = [
    -0.001_117_546_733_800_211_4,
    0.,
    0.,
    0.,
    0.,
    -0.105_408_578_764_441_87,
    -0.007_083_989_297_009_742,
    0.008_072_082_741_843_727,
    0.020_564_260_271_365_28,
    -0.039_049_761_408_697_44,
    0.122_772_902_350_186_19,
    0.041_811_958_638_991_63,
    -0.040_561_327_798_437_56,
];
