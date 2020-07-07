pub mod data;

use crate::model::{Model, Particle};
use ndarray::{array, prelude::*};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::f64;

/// The Standard Model of particle physics.
// #[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct StandardModel {
    /// Inverse temperature in \\(GeV^{-1}\\)
    pub beta: f64,

    /// Particles
    pub particles: Vec<Particle>,

    // Gauge couplings
    /// Hypercharge gauge coupling
    pub g1: f64,
    /// Weak gauge coupling
    pub g2: f64,
    /// Strong gauge coupling
    pub g3: f64,

    // Yukawa couplings
    /// Up-quark Yukawa
    pub yu: Array2<f64>,
    /// Down-quark Yukawa
    pub yd: Array2<f64>,
    /// Electron Yukawa
    pub ye: Array2<f64>,

    // Scalar potential
    /// 0-temperature mass of the Higgs
    pub mh: f64,
    /// Vacuum expectation value of the Higgs
    pub vev: f64,
    /// Quadratic coupling of the Higgs
    pub mu2: f64,
    /// Quartic term in scalar potential
    pub lambda: f64,
}

impl Model for StandardModel {
    fn zero() -> Self {
        let particles = vec![
            Particle::new(0, 0.0, 0.0)
                .name("none")
                .dof(0.0)
                .own_antiparticle(), // dummy particle so particles start at index `1`
            Particle::new(2, 0.0, 0.0).name("A").own_antiparticle(), // hypercharge gauge boson
            Particle::new(2, 0.0, 0.0)
                .name("W")
                .dof(3.0)
                .own_antiparticle(), // SU(2) gauge boson
            Particle::new(2, 0.0, 0.0)
                .name("G")
                .dof(8.0)
                .own_antiparticle(), // SU(3) gauge boson
            Particle::new(0, 0.0, 0.0).name("H").dof(2.0).complex(), // Higgs
            Particle::new(1, 0.0, 0.0).name("L1").dof(2.0),
            Particle::new(1, 0.0, 0.0).name("L2").dof(2.0),
            Particle::new(1, 0.0, 0.0).name("L3").dof(2.0),
            Particle::new(1, 0.0, 0.0).name("e1").dof(1.0),
            Particle::new(1, 0.0, 0.0).name("e2").dof(1.0),
            Particle::new(1, 0.0, 0.0).name("e3").dof(1.0),
            Particle::new(1, 0.0, 0.0).name("Q1").dof(2.0 * 3.0),
            Particle::new(1, 0.0, 0.0).name("Q2").dof(2.0 * 3.0),
            Particle::new(1, 0.0, 0.0).name("Q3").dof(2.0 * 3.0),
            Particle::new(1, 0.0, 0.0).name("u1").dof(3.0),
            Particle::new(1, 0.0, 0.0).name("u2").dof(3.0),
            Particle::new(1, 0.0, 0.0).name("u3").dof(3.0),
            Particle::new(1, 0.0, 0.0).name("d1").dof(3.0),
            Particle::new(1, 0.0, 0.0).name("d2").dof(3.0),
            Particle::new(1, 0.0, 0.0).name("d3").dof(3.0),
        ];

        let mh: f64 = 125.10;
        let vev: f64 = 246.0;
        let mu2 = -2.0 * mh.powi(2);
        let lambda = (mh / vev).powi(2);

        StandardModel {
            beta: f64::INFINITY,
            g1: 3.585e-01,
            g2: 6.476e-01,
            g3: 1.164e+00,
            yu: array![
                [7.497e-06, 0.0, 0.0],
                [0.0, 3.413e-03, 0.0],
                [0.0, 0.0, 9.346e-01],
            ],
            yd: array![
                [1.491e-05, 0.0, 0.0],
                [0.0, 3.265e-04, 0.0],
                [0.0, 0.0, 1.556e-02],
            ],
            ye: array![
                [2.880e-06, 0.0, 0.0],
                [0.0, 5.956e-04, 0.0],
                [0.0, 0.0, 1.001e-02],
            ],
            mh,
            vev,
            mu2,
            lambda,
            particles,
        }
    }

    /// Update beta for the model.
    ///
    /// This does **not** compute the widths as it is assumed that most
    /// implementation will add more particles, thus making any Standard Model
    /// specific computation redundant.
    ///
    /// This does update the thermal masses of all Standard Model particles
    /// given the Standard Model interactions.  If other interactions are
    /// included, the contributions of these interactions should also be
    /// included.
    fn set_beta(&mut self, beta: f64) {
        self.beta = beta;

        // Update the values of the gauge coupling for the running
        self.g1 = data::G1_RUNNING.sample(beta);
        self.g2 = data::G2_RUNNING.sample(beta);
        self.g3 = data::G3_RUNNING.sample(beta);

        // Precompute the squared couplings to be used in the thermal masses
        let g1 = self.g1.powi(2) / 8.0;
        let g2 = self.g2.powi(2) / 8.0;
        let g3 = self.g3.powi(2) / 8.0;
        let yu = self.yu.diag().mapv(|y| y.powi(2) / 16.0);
        let yd = self.yd.diag().mapv(|y| y.powi(2) / 16.0);
        let ye = self.ye.diag().mapv(|y| y.powi(2) / 16.0);
        // let mh = self.mh;
        let mu2 = self.mu2;
        let lambda = self.lambda / 4.0;

        // Update the thermal masses
        self.particle_mut("H", 0).set_mass(
            std::f64::consts::SQRT_2
                * f64::sqrt(
                    -mu2 + lambda
                        + g1 / 4.0
                        + (3.0 / 4.0) * g2
                        + 2.0 * yu.sum()
                        + 2.0 * yd.sum()
                        + 2.0 * ye.sum(),
                )
                / beta,
        );
        self.particle_mut("A", 0)
            .set_mass(f64::sqrt((22.0 / 3.0) * g1) / beta);
        self.particle_mut("W", 0)
            .set_mass(f64::sqrt((22.0 / 3.0) * g2) / beta);

        for i in 0..3 {
            self.particle_mut("L", i)
                .set_mass(f64::sqrt(g1 / 4.0 + (3.0 / 4.0) * g2 + ye[i]) / beta);
            self.particle_mut("e", i)
                .set_mass(f64::sqrt(g1 + ye[i]) / beta);
            self.particle_mut("Q", i)
                .set_mass(f64::sqrt(g1 / 36.0 + (3.0 / 4.0) * g2 + g3 + yd[i] + yu[i]) / beta);
            self.particle_mut("u", i)
                .set_mass(f64::sqrt(g1 * 4.0 / 9.0 + g3 + yu[i]) / beta);
            self.particle_mut("d", i)
                .set_mass(f64::sqrt(g1 / 9.0 + g3 + yd[i]) / beta);
        }
    }

    fn get_beta(&self) -> f64 {
        self.beta
    }

    fn entropy_dof(&self, beta: f64) -> f64 {
        debug_assert!(beta > 0.0, "beta must be positive.");
        data::STANDARD_MODEL_GSTAR.sample(beta.ln())
    }

    fn particles(&self) -> &[Particle] {
        &self.particles
    }

    fn particles_mut(&mut self) -> &mut [Particle] {
        &mut self.particles
    }

    fn particle_idx<S: AsRef<str>>(name: S, i: usize) -> Result<usize, (S, usize)> {
        match (name.as_ref(), i) {
            ("A", _) => Ok(1),
            ("W", _) => Ok(2),
            ("G", _) => Ok(3),
            ("H", _) => Ok(4),
            ("L", i) if i < 3 => Ok(5 + i),
            ("e", i) if i < 3 => Ok(8 + i),
            ("Q", i) if i < 3 => Ok(11 + i),
            ("u", i) if i < 3 => Ok(14 + i),
            ("d", i) if i < 3 => Ok(17 + i),
            (_, i) => Err((name, i)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::StandardModel;
    use crate::model::Model;

    #[test]
    fn particle_indices() {
        let model = StandardModel::zero();

        for (i, p) in model.particles().iter().enumerate() {
            let name = &p.name;
            if name.len() == 1 {
                assert_eq!(Ok(i), StandardModel::particle_idx(name, 0));
            } else if name.len() == 2 {
                let mut chars = name.chars();
                let head = chars.next().unwrap();
                let idx = chars.next().unwrap() as usize - 49;
                assert_eq!(Ok(i), StandardModel::particle_idx(&head.to_string(), idx));
            }
        }
    }
}
