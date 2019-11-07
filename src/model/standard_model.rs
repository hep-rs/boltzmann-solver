use crate::model::{data, Interaction, Model, Particle};
use ndarray::{array, prelude::*};
use special_functions::approximations::interpolation;

/// The Standard Model of particle physics.
pub struct StandardModel {
    pub g1: f64,
    pub g2: f64,
    pub g3: f64,
    pub yu: Array2<f64>,
    pub yd: Array2<f64>,
    pub ye: Array2<f64>,
    pub mu2: f64,
    pub lambda: f64,
    pub particles: Vec<Particle>,
    pub interactions: Vec<Interaction<Self>>,
}

impl Model for StandardModel {
    fn new() -> Self {
        let particles = vec![
            Particle::new(0, 0.0, 0.0).name("none").dof(0.0), // dummy particle so particles start at index `1`
            Particle::new(0, 0.0, 0.0).name("H").dof(2.0).complex(), // Higgs
            Particle::new(1, 0.0, 0.0).name("L1").dof(2.0),   // L1
            Particle::new(1, 0.0, 0.0).name("L2").dof(2.0),   // L2
            Particle::new(1, 0.0, 0.0).name("L3").dof(2.0),   // L3
            Particle::new(1, 0.0, 0.0).name("e1").dof(1.0),   // e1
            Particle::new(1, 0.0, 0.0).name("e2").dof(1.0),   // e2
            Particle::new(1, 0.0, 0.0).name("e3").dof(1.0),   // e3
            Particle::new(1, 0.0, 0.0).name("Q1").dof(2.0 * 3.0), // Q1
            Particle::new(1, 0.0, 0.0).name("Q2").dof(2.0 * 3.0), // Q2
            Particle::new(1, 0.0, 0.0).name("Q3").dof(2.0 * 3.0), // Q3
            Particle::new(1, 0.0, 0.0).name("u1").dof(3.0),   // u1
            Particle::new(1, 0.0, 0.0).name("u2").dof(3.0),   // u2
            Particle::new(1, 0.0, 0.0).name("u3").dof(3.0),   // u3
            Particle::new(1, 0.0, 0.0).name("d1").dof(3.0),   // d1
            Particle::new(1, 0.0, 0.0).name("d2").dof(3.0),   // d2
            Particle::new(1, 0.0, 0.0).name("d3").dof(3.0),   // d3
        ];
        let interactions = Vec::new();

        StandardModel {
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
            mu2: -1.139e+04,
            lambda: 3.475e-01,
            particles,
            interactions,
        }
    }

    fn beta(&mut self, beta: f64) {
        let g1 = self.g1.powi(2) / 8.0;
        let g2 = 0.75 * self.g2.powi(2) / 8.0;
        let g3 = 0.75 * self.g3.powi(2) / 8.0;
        let yu = self.yu.diag().mapv(|y| y.powi(2) / 16.0);
        let yd = self.yd.diag().mapv(|y| y.powi(2) / 16.0);
        let ye = self.ye.diag().mapv(|y| y.powi(2) / 16.0);

        self.particle_mut("H", 0).set_mass(0.559 / beta); // To confirm
        self.particle_mut("H", 0).set_width(0.1 * 0.559 / beta); // To confirm

        for i in 0..3 {
            self.particle_mut("L", i)
                .set_mass(f64::sqrt(g1 / 4.0 + g2 + ye[i]) / beta);
            // .set_width(f64::sqrt(g1 / 4.0 + g2 + ye[i]) / beta);
            self.particle_mut("e", i)
                .set_mass(f64::sqrt(g1 + ye[i]) / beta);
            // .set_width(f64::sqrt(g1 + ye[i]) / beta);
            self.particle_mut("Q", i)
                .set_mass(f64::sqrt(g1 / 36.0 + g2 + g3 + yd[i] + yu[i]) / beta);
            // .set_width(f64::sqrt(g1 / 36.0 + g2 + g3 + yd[i] + yu[i]) / beta);
            self.particle_mut("u", i)
                .set_mass(f64::sqrt(g1 * 4.0 / 9.0 + g3 + yu[i]) / beta);
            // .set_width(f64::sqrt(g1 * 4.0 / 9.0 + g3 + yu[i]) / beta);
            self.particle_mut("d", i)
                .set_mass(f64::sqrt(g1 / 9.0 + g3 + yd[i]) / beta);
            // .set_width(f64::sqrt(g1 / 9.0 + g3 + yd[i]) / beta);
        }
    }

    fn entropy_dof(&self, beta: f64) -> f64 {
        debug_assert!(beta > 0.0, "beta must be positive.");
        interpolation::linear(&data::STANDARD_MODEL_GSTAR, beta.ln())
    }

    fn particles(&self) -> &Vec<Particle> {
        &self.particles
    }

    fn particles_mut(&mut self) -> &mut Vec<Particle> {
        &mut self.particles
    }

    fn particle_idx(name: &str, i: usize) -> Result<usize, (&str, usize)> {
        match (name, i) {
            ("H", _) => Ok(1),
            ("L", i) if i < 3 => Ok(2 + i),
            ("e", i) if i < 3 => Ok(5 + i),
            ("Q", i) if i < 3 => Ok(8 + i),
            ("u", i) if i < 3 => Ok(11 + i),
            ("d", i) if i < 3 => Ok(14 + i),
            (name, i) => Err((name, i)),
        }
    }

    fn interactions(&self) -> &Vec<Interaction<Self>> {
        &self.interactions
    }
}
