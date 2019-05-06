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
pub const PARTICLE_NAMES: [&str; 4] = [
    "B-L",
    "N₁", "N₂", "N₃",
];

////////////////////////////////////////////////////////////////////////////////
// Model
////////////////////////////////////////////////////////////////////////////////

/// Leptogenesis model parameters.
pub struct VanillaLeptogenesisModel {
    /// Yukawa coupling: [H, Q, u]
    pub y_u: Array2<Complex<f64>>,
    /// Yukawa coupling: [H, Q, d]
    pub y_d: Array2<Complex<f64>>,
    /// Yukawa coupling: [H, L, e]
    pub y_e: Array2<Complex<f64>>,
    /// Yukawa coupling: [H, ν, L]
    pub y_v: Array2<Complex<f64>>,

    /// Right handed neutrinos mass (in GeV)
    pub m_n: [f64; 3],
    /// Higgs mass (in GeV)
    pub m_h: f64,

    /// Right handed neutrinos width (in GeV)
    pub w_n: [f64; 3],
    /// Higgs width (in GeV)
    pub w_h: f64,

    /// Epsilon parameter
    pub epsilon: f64,
}

impl Model for VanillaLeptogenesisModel {
    fn new(beta: f64) -> Self {
        VanillaLeptogenesisModel {
            // Yukawa couplings
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

            // Masses
            m_n: [1e10, 1e11, 1e12],
            m_h: 0.4 / beta,

            // Widths
            w_n: [0., 0., 0.],
            w_h: 0.1 * 0.4 / beta,

            // Epsilon
            epsilon: 1e-6,
        }
    }
}
