//! Runge-Kutta method of order 8 in 13 stages.
//!
//! Coefficients are exported from Mathematica and can be obtained from
//! references therein.

#![allow(dead_code)]

pub const RK_ORDER: i32 = 8;
pub const RK_S: usize = 13;
pub const RK_A: [[f64; RK_S - 1]; RK_S] = [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0.08740084650491524, 0.02548760493865432, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0.04233316929133858, 0., 0.12699950787401576, 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0.4260950588874226, 0., -1.5987952846591522, 1.5967002257717298, 0., 0., 0., 0., 0., 0., 0., 0.], [0.05071933729671393, 0., 0., 0.2543337726460041, 0.203946890057282, 0., 0., 0., 0., 0., 0., 0.], [-0.2900037471752311, 0., 0., 1.3441873910260789, -2.864777943361443, 2.677594299510595, 0., 0., 0., 0., 0., 0.], [0.09853501133799354, 0., 0., 0., 0.22192680630751385, -0.18140622911806994, 0.010944411472562547, 0., 0., 0., 0., 0.], [0.3871105254573114, 0., 0., -1.4424454974855276, 2.9053981890699507, -1.853771069630106, 0.14003648098728153, 0.5727394081149582, 0., 0., 0., 0.], [-0.16124403444439306, 0., 0., -0.17339602957358985, -1.3012892814065147, 1.1379503751738618, -0.03174764966396688, 0.9335129382493368, -0.08378631833473385, 0., 0., 0.], [-0.019199444881589534, 0., 0., 0.27330857265264286, -0.6753497320694437, 0.34151849813846014, -0.06795006480337577, 0.09659175224762387, 0.13253082511182102, 0.3685495936038611, 0., 0.], [0.6091877403645289, 0., 0., -2.272569085898002, 4.75789834269403, -5.516106706692758, 0.29005963696801196, 0.5691423963359037, 0.7926795760332167, 0.15473720453288822, 1.6149708956621815, 0.], [0.8873576220853472, 0., 0., -2.9754597821085365, 5.6007170094881635, -5.915607450536674, 0.22029689156134927, 0.10155097824462217, 1.1514345647386055, 1.9297101665271241, 0., 0.]];
pub const RK_B: [f64; RK_S] = [0.044729564666695705, 0., 0., 0., 0., 0.156910335277082, 0.18460973408151637, 0.2251638060208699, 0.14794615651970233, 0.07605554244495583, 0.12277290235018619, 0.04181195863899163, 0.];
pub const RK_C: [f64; RK_S] = [0., 0.25, 0.11288845144356956, 0.16933267716535433, 0.424, 0.509, 0.867, 0.15, 0.7090680365138684, 0.32, 0.45, 1., 1.];
pub const RK_E: [f64; RK_S] = [-0.0011175467338002114, 0., 0., 0., 0., -0.10540857876444187, -0.007083989297009742, 0.008072082741843727, 0.02056426027136528, -0.03904976140869744, 0.12277290235018619, 0.04181195863899163, -0.04056132779843756];