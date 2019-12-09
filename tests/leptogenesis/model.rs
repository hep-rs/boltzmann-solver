//! Construct the [`Model`] instance

// We allow `dead_code` in this module as we prefer to define all parameters in
// case we need them.  If certain parameters are quite computationally costly to
// compute and ultimately not used, it would be preferable to comment these out.
#![allow(dead_code)]

pub mod interaction;

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
    pub interactions: Vec<Interaction<Self>>,
    pub epsilon: Array2<f64>,
}

impl Model for LeptogenesisModel {
    fn zero() -> Self {
        let mut sm = StandardModel::zero();

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

        sm.particles.push(Particle::new(1, mn[0], 0.0).name("N1"));
        sm.particles.push(Particle::new(1, mn[1], 0.0).name("N2"));
        sm.particles.push(Particle::new(1, mn[2], 0.0).name("N3"));

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

        let interactions = Vec::new();

        // Add them on a per-test basis in order to avoid every test using all
        // of them, possibly risking slowing down many tests.

        // interactions.append(&mut interaction::hh());
        // interactions.append(&mut interaction::nn());
        // interactions.append(&mut interaction::hle());
        // interactions.append(&mut interaction::hln());
        // interactions.append(&mut interaction::hqu());
        // interactions.append(&mut interaction::hqd());
        // interactions.append(&mut interaction::hhll1());
        // interactions.append(&mut interaction::hhll2());
        // interactions.append(&mut interaction::nlqd());
        // interactions.append(&mut interaction::nlqu());

        LeptogenesisModel {
            sm,
            yv,
            mn,
            interactions,
            epsilon,
        }
    }

    fn set_beta(&mut self, beta: f64) {
        self.sm.set_beta(beta);

        // Precompute the squared couplings to be used in the thermal masses
        let g1 = self.sm.g1.powi(2) / 8.0;
        let g2 = self.sm.g2.powi(2) / 8.0;
        let yu = self.sm.yu.diag().mapv(|y| y.powi(2) / 16.0);
        let yd = self.sm.yd.diag().mapv(|y| y.powi(2) / 16.0);
        let ye = self.sm.ye.diag().mapv(|y| y.powi(2) / 16.0);
        let yv = Array1::from_shape_fn(3, |i| (0..3).map(|k| self.yv[[k, i]].norm_sqr()).sum());
        let lambda = 0.0 * self.sm.lambda / 4.0;

        self.particle_mut("H", 0).set_mass(
            std::f64::consts::SQRT_2
                * f64::sqrt(
                    g1 / 4.0
                        + (3.0 / 4.0) * g2
                        + 2.0 * yu.sum()
                        + 2.0 * yd.sum()
                        + 2.0 * ye.sum()
                        + 2.0 * yv.sum()
                        + lambda,
                )
                / beta,
        );

        for i in 0..3 {
            let mi = self.mn[i] + f64::sqrt(yv[i]) / beta;

            self.particle_mut("N", i).set_mass(mi);
            self.particle_mut("L", i)
                .set_mass(f64::sqrt(g1 / 4.0 + (3.0 / 4.0) * g2 + ye[i] + yv[i]) / beta);
        }

        self.update_widths();

        // let mh = self.particle("H", 0).mass;
        // self.particle_mut("H", 0).set_width(1e-2 * mh);
    }

    fn get_beta(&self) -> f64 {
        self.sm.get_beta()
    }

    fn entropy_dof(&self, beta: f64) -> f64 {
        self.sm.entropy_dof(beta)
    }

    fn particles(&self) -> &Vec<Particle> {
        &self.sm.particles
    }

    fn particles_mut(&mut self) -> &mut Vec<Particle> {
        &mut self.sm.particles
    }

    fn particle_idx(name: &str, i: usize) -> Result<usize, (&str, usize)> {
        StandardModel::particle_idx(name, i).or_else(|(name, i)| match (name, i) {
            ("N", i) if i < 3 => Ok(20 + i),
            (name, i) => Err((name, i)),
        })
    }

    fn interactions(&self) -> &Vec<Interaction<Self>> {
        &self.interactions
    }
}
