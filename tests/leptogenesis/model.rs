//! Construct the [`Model`] instance

// We allow `dead_code` in this module as we prefer to define all parameters in
// case we need them.  If certain parameters are quite computationally costly to
// compute and ultimately not used, it would be preferable to comment these out.
#![allow(dead_code)]

use boltzmann_solver::{particle::Particle, solver::Model};
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
    "H",
    "N₁", "N₂", "N₃",
];

/// Function to map a more memorable name to the array index for the number
/// density, mass, etc.
///
/// We are using the programming convention of 0-indexing all particles; thus
/// the index of "N₁" is obtained with `p_i("N", 0)`.
#[inline]
pub fn p_i(p: &str, n: usize) -> usize {
    match (p, n) {
        ("BL", _) | ("B-L", _) => 0,
        ("H", _) => 1,
        ("N", n) if n < 3 => n + 2,
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

/// Particle masses (in GeV)
pub struct Masses {
    /// Right-handed neutrinos
    pub n: [f64; 3],
    /// Higgs
    pub h: f64,
}

impl Masses {
    fn new(particles: &Array1<Particle>) -> Self {
        Masses {
            n: [particles[2].mass, particles[3].mass, particles[4].mass],
            h: particles[1].mass,
        }
    }
}

/// Particle squared masses (in GeV\\(^2\\))
pub struct SquaredMasses {
    /// Right-handed neutrinos
    pub n: [f64; 3],
    /// Higgs
    pub h: f64,
}

impl SquaredMasses {
    fn new(particles: &Array1<Particle>) -> Self {
        SquaredMasses {
            n: [particles[2].mass2, particles[3].mass2, particles[4].mass2],
            h: particles[1].mass2,
        }
    }
}

/// Particle widths (in GeV)
pub struct Widths {
    /// Right-handed neutrinos
    pub n: [f64; 3],
    /// Higgs
    pub h: f64,
}

impl Widths {
    fn new(particles: &Array1<Particle>) -> Self {
        Widths {
            n: [particles[2].width, particles[3].width, particles[4].width],
            h: particles[1].width,
        }
    }
}

/// Particle squared widths (in GeV\\(^2\\))
pub struct SquaredWidths {
    /// Right-handed neutrinos
    pub n: [f64; 3],
    /// Higgs
    pub h: f64,
}

impl SquaredWidths {
    fn new(particles: &Array1<Particle>) -> Self {
        SquaredWidths {
            n: [
                particles[2].width2,
                particles[3].width2,
                particles[4].width2,
            ],
            h: particles[1].width2,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Model
////////////////////////////////////////////////////////////////////////////////

/// Leptogenesis model parameters
pub struct LeptogenesisModel {
    pub coupling: Couplings,
    pub particles: Array1<Particle>,
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
                [Complex::new(2.2e-3, 0.0), zero(), zero()],
                [zero(), Complex::new(95e-3, 0.0), zero()],
                [zero(), zero(), Complex::new(172.200, 0.0)],
            ] * f64::sqrt(2.0)
                / 246.0,
            y_d: array![
                [Complex::new(5e-3, 0.0), zero(), zero()],
                [zero(), Complex::new(1.25, 0.0), zero()],
                [zero(), zero(), Complex::new(4.2, 0.0)],
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
            ] * 1e-8,
        };

        let particles = array![
            Particle::new(0, 0.0, 0.0).dof(0.0), // B-L
            Particle::new(0, 0.4 / beta, 0.1 * 0.4 / beta)
                .complex()
                .dof(2.0), // Higgs
            Particle::new(1, 1e10, 0.0),         // N1
            Particle::new(1, 1e15, 0.0),         // N2
            Particle::new(1, 1e16, 0.0),         // N3
        ];

        let mass = Masses::new(&particles);
        let mass2 = SquaredMasses::new(&particles);
        let width = Widths::new(&particles);
        let width2 = SquaredWidths::new(&particles);

        LeptogenesisModel {
            coupling,
            particles,
            mass,
            mass2,
            width,
            width2,
            epsilon: 1e-6,
        }
    }

    fn particles(&self) -> &Array1<Particle> {
        &self.particles
    }
}
