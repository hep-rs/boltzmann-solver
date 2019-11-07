//! Construct the [`Model`] instance

// We allow `dead_code` in this module as we prefer to define all parameters in
// case we need them.  If certain parameters are quite computationally costly to
// compute and ultimately not used, it would be preferable to comment these out.
#![allow(dead_code)]

mod interaction;

use boltzmann_solver::{constants::PI, prelude::*};
use ndarray::{array, prelude::*};
use num::Complex;
use std::f64;

////////////////////////////////////////////////////////////////////////////////
// Model
////////////////////////////////////////////////////////////////////////////////

/// Leptogenesis model parameters
pub struct LeptogenesisModel {
    pub sm: StandardModel,
    pub yv: Array2<Complex<f64>>,
    pub mn: Array1<f64>,
    pub particles: Vec<Particle>,
    pub interactions: Vec<Interaction<Self>>,
    pub epsilon: Array2<f64>,
}

impl Model for LeptogenesisModel {
    fn new() -> Self {
        let sm = StandardModel::new();

        let yv = array![
            [
                Complex::new(1.0, 0.0),
                Complex::new(0.1, 0.1),
                Complex::new(0.2, 0.2)
            ],
            [
                Complex::new(0.1, -0.1),
                Complex::new(2.0, 0.0),
                Complex::new(0.3, 0.3)
            ],
            [
                Complex::new(0.2, -0.2),
                Complex::new(0.3, -0.3),
                Complex::new(3.0, 0.0)
            ],
        ] * 1e-4;

        let mn = array![1e10, 1e15, 5e15];

        let mut particles = sm.particles.clone();
        particles.push(Particle::new(1, mn[0], 0.0).name("N1"));
        particles.push(Particle::new(1, mn[1], 0.0).name("N2"));
        particles.push(Particle::new(1, mn[2], 0.0).name("N3"));

        // epsilon[[i, j]] is the asymmetry in Ni -> H Lj.
        let yvd_yv: Array2<Complex<f64>> = Array2::from_shape_fn((3, 3), |(i, j)| {
            (0..3).map(|k| yv[[k, i]].conj() * yv[[k, j]]).sum()
        });
        let epsilon = Array2::from_shape_fn((3, 3), |(i, a)| {
            (0..3)
                .map(|j| {
                    if j == i {
                        0.0
                    } else {
                        let x = (mn[j] / mn[i]).powi(2);
                        let g =
                            x.sqrt() * (1.0 / (1.0 - x) + 1.0 - (1.0 + x) * f64::ln((1.0 + x) / x));

                        (yv[[a, i]].conj() * yvd_yv[[i, j]] * yv[[a, j]]).im * (g + 1.0 / (1.0 - x))
                    }
                })
                .sum::<f64>()
                / (8.0 * PI * yvd_yv[[i, i]].re)
        });

        let mut interactions = Vec::new();
        // interactions.append(&mut interaction::nn());
        interactions.append(&mut interaction::hln());
        interactions.append(&mut interaction::hhll1());
        // interactions.append(&mut interaction::hhll2());
        // interactions.append(&mut interaction::nlqd());
        // interactions.append(&mut interaction::nlqu());

        log::info!("Interactions in model: {}", interactions.len());

        LeptogenesisModel {
            sm,
            yv,
            mn,
            particles,
            interactions,
            epsilon,
        }
    }

    fn beta(&mut self, beta: f64) {
        self.sm.beta(beta);

        self.particles = self.sm.particles.clone();
        let yv = self.yv.diag().mapv(|v| v.norm_sqr() / 16.0);
        for i in 0..3 {
            self.particles
                .push(Particle::new(1, self.mn[i] + f64::sqrt(yv[i]) / beta, 0.0))
        }
    }

    fn entropy_dof(&self, beta: f64) -> f64 {
        self.sm.entropy_dof(beta)
    }

    fn particles(&self) -> &Vec<Particle> {
        &self.particles
    }

    fn particles_mut(&mut self) -> &mut Vec<Particle> {
        &mut self.particles
    }

    fn particle_idx(name: &str, i: usize) -> Result<usize, (&str, usize)> {
        StandardModel::particle_idx(name, i).or_else(|(name, i)| match (name, i) {
            ("N", i) if i < 3 => Ok(17 + i),
            (name, i) => Err((name, i)),
        })
    }

    fn interactions(&self) -> &Vec<Interaction<Self>> {
        &self.interactions
    }
}
