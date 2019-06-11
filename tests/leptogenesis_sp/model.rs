//! Construct the [`Model`] instance

// We allow `dead_code` in this module as we prefer to define all parameters in
// case we need them.  If certain parameters are quite computationally costly to
// compute and ultimately not used, it would be preferable to comment these out.
#![allow(dead_code)]

use boltzmann_solver::{solver::Model, statistic::Statistic};
use ndarray::{array, prelude::*};
use num::{zero, Complex};
use std::f64;

////////////////////////////////////////////////////////////////////////////////
// Particle Names
////////////////////////////////////////////////////////////////////////////////

/// All the particle names in the same order as they are added to the solver.
#[rustfmt::skip]
pub const NAMES: [&str; 5] = [
    "B-L",
    "N₁", "N₂", "N₃",
    "H",
];

/// Function to map a more memorable name to the array index for the number
/// density, mass, etc.
///
/// We are using the programming convention of 0-indexing all particles; thus
/// the index of "N₁" is obtained with `p_i("N", 0)`.
pub fn p_i(p: &str, n: usize) -> usize {
    match (p, n) {
        ("BL", _) | ("B-L", _) => 0,
        ("N", n) => n + 1,
        ("H", _) => 4,
        _ => unreachable!(),
    }
}

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

////////////////////////////////////////////////////////////////////////////////
// Model
////////////////////////////////////////////////////////////////////////////////

/// Leptogenesis model parameters
pub struct LeptogenesisModel {
    pub coupling: Couplings,
    pub statistic: Array1<(Statistic, f64)>,
    pub mass: Array1<f64>,
    pub mass2: Array1<f64>,
    pub width: Array1<f64>,
    pub width2: Array1<f64>,
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

        let statistic = array![
            (Statistic::BoseEinstein, 0.0), // B-L
            (Statistic::FermiDirac, 1.0),   // N1
            (Statistic::FermiDirac, 1.0),   // N2
            (Statistic::FermiDirac, 1.0),   // N3
            (Statistic::BoseEinstein, 1.0), // Higgs
        ];

        let mass = array![
            0.0,        // B-L
            1e10,       // N1
            1e11,       // N2
            1e12,       // N3
            0.4 / beta, // Higgs
        ];
        let mass2 = mass.map(|v| v.powi(2));
        let width = array![
            0.0,           // B-L
            0.0,           // N1
            0.0,           // N2
            0.0,           // N3
            0.1 * mass[4], // Higgs
        ];
        let width2 = width.map(|v| v.powi(2));

        LeptogenesisModel {
            coupling,
            statistic,
            mass,
            mass2,
            width,
            width2,
            epsilon: 1e-6,
        }
    }

    fn statistic(&self) -> &Array1<(Statistic, f64)> {
        &self.statistic
    }

    fn mass(&self) -> &Array1<f64> {
        &self.mass
    }

    fn mass2(&self) -> &Array1<f64> {
        &self.mass2
    }

    fn width(&self) -> &Array1<f64> {
        &self.width
    }

    fn width2(&self) -> &Array1<f64> {
        &self.width2
    }
}
