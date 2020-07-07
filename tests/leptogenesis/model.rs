//! Construct the [`Model`] instance

// We allow `dead_code` in this module as we prefer to define all parameters in
// case we need them.  If certain parameters are quite computationally costly to
// compute and ultimately not used, it would be preferable to comment these out.
#![allow(dead_code)]

pub mod interaction;

use boltzmann_solver::prelude::*;
use ndarray::{array, prelude::*};
use num::Complex;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::f64;

// //////////////////////////////////////////////////////////////////////////////
// Model
// //////////////////////////////////////////////////////////////////////////////

/// Leptogenesis model parameters
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct LeptogenesisModel {
    pub sm: StandardModel,
    /// Coupling between H L(i1) and N(i2)
    pub yv: Array2<Complex<f64>>,
    pub mn: Array1<f64>,
    #[cfg(feature = "parallel")]
    #[serde(skip)]
    pub interactions: Vec<Box<dyn Interaction<Self> + Sync>>,
    #[cfg(not(feature = "parallel"))]
    #[serde(skip)]
    pub interactions: Vec<Box<dyn Interaction<Self>>>,
}

impl Model for LeptogenesisModel {
    fn zero() -> Self {
        let mut sm = StandardModel::zero();

        // These values correspond to m̃ = 0.06 eV
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
        ] * 1e-4
            * 30.0;

        let mn = array![1e10, 1e15, 5e15];
        sm.particles
            .push(Particle::new(1, mn[0], 0.0).name("N1").own_antiparticle());
        sm.particles
            .push(Particle::new(1, mn[1], 0.0).name("N2").own_antiparticle());
        sm.particles
            .push(Particle::new(1, mn[2], 0.0).name("N3").own_antiparticle());

        LeptogenesisModel {
            sm,
            yv,
            mn,
            interactions: Vec::new(),
        }
    }

    fn set_beta(&mut self, beta: f64) {
        self.sm.set_beta(beta);

        // Precompute the squared couplings to be used in the thermal masses
        let g1 = self.sm.g1.powi(2) / 8.0;
        let g2 = self.sm.g2.powi(2) / 8.0;
        let yu = self.sm.yu.dot(&self.sm.yu).into_diag() / 16.0;
        let yd = self.sm.yd.dot(&self.sm.yd).into_diag() / 16.0;
        let ye = self.sm.ye.dot(&self.sm.ye).into_diag() / 16.0;
        let yv = self
            .yv
            .dot(&self.yv.t().map(|y| y.conj()))
            .into_diag()
            .map(|y| y.re)
            / 16.0;
        let lambda = 0.0 * self.sm.lambda / 4.0;

        self.particle_mut("H", 0).set_mass(
            f64::consts::SQRT_2
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
            + self.particles()[20..]
                .iter()
                .map(|p| p.entropy_dof(self.sm.beta))
                .sum::<f64>()
    }

    fn particles(&self) -> &[Particle] {
        &self.sm.particles
    }

    fn particles_mut(&mut self) -> &mut [Particle] {
        &mut self.sm.particles
    }

    fn particle_idx<S: AsRef<str>>(name: S, i: usize) -> Result<usize, (S, usize)> {
        let idx = match (name.as_ref(), i) {
            ("N", i) if i < 3 => Ok(20 + i),
            (_, i) => Err((name, i)),
        };

        idx.or_else(|(name, i)| StandardModel::particle_idx(name, i))
    }
}

#[cfg(not(feature = "parallel"))]
impl ModelInteractions for LeptogenesisModel {
    // type Iter = &Vec<Self::Item>;
    type Item = Box<dyn Interaction<Self>>;

    fn interactions(&self) -> &[Self::Item] {
        &self.interactions
    }
}

#[cfg(feature = "parallel")]
impl ModelInteractions for LeptogenesisModel {
    // type Iter = &'data Vec<Self::Item>;
    type Item = Box<dyn Interaction<Self> + Sync>;

    fn interactions(&self) -> &[Self::Item] {
        &self.interactions
    }
}
