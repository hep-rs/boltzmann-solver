use boltzmann_solver::solver::Model;
use ndarray::{array, Array2};
use num::{zero, Complex};
use std::f64;

////////////////////////////////////////////////////////////////////////////////
// Particle Names
////////////////////////////////////////////////////////////////////////////////

#[rustfmt::skip]
pub const PARTICLE_NAMES: [&str; 4] = [
    "B-L",
    "N₁", "N₂", "N₃",
];

////////////////////////////////////////////////////////////////////////////////
// Couplings
////////////////////////////////////////////////////////////////////////////////

/// Struct containing all the model parameters.
pub struct Couplings {
    /// Yukawa coupling: [H, Q, u]
    pub y_u: Array2<Complex<f64>>,
    /// Yukawa coupling: [H, Q, d]
    pub y_d: Array2<Complex<f64>>,
    /// Yukawa coupling: [H, L, e]
    pub y_e: Array2<Complex<f64>>,
    /// Yukawa coupling: [H, ν, L]
    pub y_v: Array2<Complex<f64>>,
}

impl Couplings {
    fn new(_beta: f64) -> Self {
        Couplings {
            y_u: array![
                [Complex::new(172.200, 0.0), zero(), zero()],
                [zero(), Complex::new(95e-3, 0.0), zero()],
                [zero(), zero(), Complex::new(2.2e-3, 0.0)],
            ] * f64::sqrt(2.0)
                / 246.0,
            y_d: array![
                [Complex::new(4.2, 0.0), zero(), zero()],
                [zero(), Complex::new(1.25, 0.0), zero()],
                [zero(), zero(), Complex::new(5e-3, 0.0)],
            ] * f64::sqrt(2.0)
                / 246.0,
            y_e: array![
                [Complex::new(5.109_989e-4, 0.0), zero(), zero()],
                [zero(), Complex::new(1.056_583_7e-1, 0.0), zero()],
                [zero(), zero(), Complex::new(1.77699, 0.0)],
            ] * f64::sqrt(2.0)
                / 246.0,
            y_v: array![
                [
                    Complex::new(1.0, 0.0),
                    Complex::new(0.1, 0.1),
                    Complex::new(0.1, 0.1)
                ],
                [
                    Complex::new(0.1, -0.1),
                    Complex::new(1.0, 0.0),
                    Complex::new(0.1, 0.1)
                ],
                [
                    Complex::new(0.1, -0.1),
                    Complex::new(0.1, -0.1),
                    Complex::new(1.0, 0.0)
                ],
            ],
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Masses and Widths
////////////////////////////////////////////////////////////////////////////////

#[allow(dead_code)]
pub struct Masses {
    pub n: [f64; 3],
    pub h: f64,
}

impl Masses {
    fn new(beta: f64) -> Self {
        Masses {
            n: [1e10, 1e11, 1e12],
            h: 0.4 / beta,
        }
    }
}

#[allow(dead_code)]
pub struct SquaredMasses {
    pub n: [f64; 3],
    pub h: f64,
}

impl SquaredMasses {
    fn new(beta: f64) -> Self {
        let m = Masses::new(beta);
        SquaredMasses {
            n: [m.n[0].powi(2), m.n[1].powi(2), m.n[2].powi(2)],
            h: m.h.powi(2),
        }
    }
}

#[allow(dead_code)]
pub struct Widths {
    pub n: [f64; 3],
    pub h: f64,
}

impl Widths {
    fn new(beta: f64) -> Self {
        let m = Masses::new(beta);
        Widths {
            n: [0., 0., 0.],
            h: 0.1 * m.h,
        }
    }
}

#[allow(dead_code)]
pub struct SquaredWidths {
    pub n: [f64; 3],
    pub h: f64,
}

impl SquaredWidths {
    fn new(beta: f64) -> Self {
        let w = Widths::new(beta);
        SquaredWidths {
            n: [w.n[0].powi(2), w.n[1].powi(2), w.n[2].powi(2)],
            h: w.h.powi(2),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Model
////////////////////////////////////////////////////////////////////////////////

#[allow(dead_code)]
pub struct VanillaLeptogenesisModel {
    pub coupling: Couplings,
    pub mass: Masses,
    pub mass2: SquaredMasses,
    pub width: Widths,
    pub width2: SquaredWidths,
    pub epsilon: f64,
}

impl Model for VanillaLeptogenesisModel {
    fn new(beta: f64) -> Self {
        VanillaLeptogenesisModel {
            coupling: Couplings::new(beta),
            mass: Masses::new(beta),
            mass2: SquaredMasses::new(beta),
            width: Widths::new(beta),
            width2: SquaredWidths::new(beta),
            epsilon: 1e-6,
        }
    }
}
