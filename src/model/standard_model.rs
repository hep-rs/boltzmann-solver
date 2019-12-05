use crate::model::{data, Interaction, Model, Particle};
use ndarray::{array, prelude::*};

/// The Standard Model of particle physics.
pub struct StandardModel {
    pub beta: f64,
    // Particle and Interations
    pub particles: Vec<Particle>,
    pub interactions: Vec<Interaction<Self>>,
    // Gauge couplings
    pub g1: f64,
    pub g2: f64,
    pub g3: f64,
    // Yukawa couplings
    pub yu: Array2<f64>,
    pub yd: Array2<f64>,
    pub ye: Array2<f64>,
    // Scalar potential
    pub mu2: f64,
    pub lambda: f64,
}

impl Model for StandardModel {
    fn zero() -> Self {
        let particles = vec![
            Particle::new(0, 0.0, 0.0).name("none").dof(0.0), // dummy particle so particles start at index `1`
            Particle::new(2, 0.0, 0.0).name("A"),             // hypercharge gauge boson
            Particle::new(2, 0.0, 0.0).name("W").dof(3.0),    // SU(2) gauge boson
            Particle::new(2, 0.0, 0.0).name("G").dof(8.0),    // SU(3) gauge boson
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
        let interactions = Vec::new();

        StandardModel {
            beta: std::f64::INFINITY,
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
            lambda: 3.475e-01, // TODO: Verify value
            particles,
            interactions,
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
        let lambda = 0.0 * self.lambda / 4.0; // TODO: Verify this as it seems wrong

        // Update the thermal masses
        self.particle_mut("H", 0).set_mass(
            std::f64::consts::SQRT_2
                * f64::sqrt(
                    g1 / 4.0
                        + (3.0 / 4.0) * g2
                        + 2.0 * yu.sum()
                        + 2.0 * yd.sum()
                        + 2.0 * ye.sum()
                        + lambda,
                )
                / beta,
        );

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

    fn particles(&self) -> &Vec<Particle> {
        &self.particles
    }

    fn particles_mut(&mut self) -> &mut Vec<Particle> {
        &mut self.particles
    }

    fn particle_idx(name: &str, i: usize) -> Result<usize, (&str, usize)> {
        match (name, i) {
            ("A", _) => Ok(1),
            ("W", _) => Ok(2),
            ("G", _) => Ok(3),
            ("H", _) => Ok(4),
            ("L", i) if i < 3 => Ok(5 + i),
            ("e", i) if i < 3 => Ok(8 + i),
            ("Q", i) if i < 3 => Ok(11 + i),
            ("u", i) if i < 3 => Ok(14 + i),
            ("d", i) if i < 3 => Ok(17 + i),
            (name, i) => Err((name, i)),
        }
    }

    fn interactions(&self) -> &Vec<Interaction<Self>> {
        &self.interactions
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
            let name = p.name;
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
