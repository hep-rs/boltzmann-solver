//! Construct the [`Model`] instance

// We allow `dead_code` in this module as we prefer to define all parameters in
// case we need them.  If certain parameters are quite computationally costly to
// compute and ultimately not used, it would be preferable to comment these out.
#![allow(dead_code)]

use boltzmann_solver::solver::Model;
use ndarray::{array, Array2};
use num::{zero, Complex};
use std::f64;

////////////////////////////////////////////////////////////////////////////////
// Particle Names
////////////////////////////////////////////////////////////////////////////////

/// All the particle names in the same order as they are added to the solver.
#[rustfmt::skip]
pub const NAMES: [&str; 4] = [
    "B-L",
    "N₁", "N₂", "N₃",
];

////////////////////////////////////////////////////////////////////////////////
// Model Parameter Sub-cateogories
////////////////////////////////////////////////////////////////////////////////

/// Struct containing all the Lagrangian coupling parameters.
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

/// Particle masses (in GeV)
pub struct Masses {
    /// Right-handed neutrinos
    pub n: [f64; 3],
    /// Higgs
    pub h: f64,
}

/// Particle squared masses (in GeV\\(^2\\))
pub struct SquaredMasses {
    /// Right-handed neutrinos
    pub n: [f64; 3],
    /// Higgs
    pub h: f64,
}

/// Particle widths (in GeV)
pub struct Widths {
    /// Right-handed neutrinos
    pub n: [f64; 3],
    /// Higgs
    pub h: f64,
}

/// Particle squared widths (in GeV\\(^2\\))
pub struct SquaredWidths {
    /// Right-handed neutrinos
    pub n: [f64; 3],
    /// Higgs
    pub h: f64,
}

////////////////////////////////////////////////////////////////////////////////
// Model
////////////////////////////////////////////////////////////////////////////////

/// Leptogenesis model parameters
pub struct LeptogenesisModel {
    pub coupling: Couplings,
    pub mass: Masses,
    pub mass2: SquaredMasses,
    pub width: Widths,
    pub width2: SquaredWidths,
    pub epsilon: f64,
}

impl Model for LeptogenesisModel {
    fn new(beta: f64) -> Self {
        let coupling = Couplings {
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
            ] * 1e-4,
        };

        let mass = Masses {
            n: [1e10, 1e11, 1e12],
            h: 0.4 / beta,
        };
        let mass2 = SquaredMasses {
            n: [mass.n[0].powi(2), mass.n[1].powi(2), mass.n[2].powi(2)],
            h: mass.h.powi(2),
        };
        let width = Widths {
            n: [0.0, 0.0, 0.0],
            h: 0.1 * mass.h,
        };
        let width2 = SquaredWidths {
            n: [width.n[0].powi(2), width.n[1].powi(2), width.n[2].powi(2)],
            h: width.h.powi(2),
        };

        LeptogenesisModel {
            coupling,
            mass,
            mass2,
            width,
            width2,
            epsilon: 1e-6,
        }
    }
}
