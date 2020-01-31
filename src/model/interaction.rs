mod four_particle;
mod partial_width;
mod rate_density;
mod three_particle;

pub use four_particle::FourParticle;
pub use partial_width::PartialWidth;
pub use rate_density::RateDensity;
pub use three_particle::ThreeParticle;

use crate::{model::Model, solver::Context};
use ndarray::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// List of particles involved in the interaction.
///
/// The particles are signed such that particles are > 0 and antiparticles are <
/// 0.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct InteractionParticles {
    ingoing: Vec<isize>,
    outgoing: Vec<isize>,
}

impl InteractionParticles {
    /// Convert the signed particle numbers to indices which can be used to
    /// index the model's particle.
    pub fn as_idx(&self) -> InteractionParticleIndices {
        InteractionParticleIndices {
            ingoing: self.ingoing.iter().map(|p| p.abs() as usize).collect(),
            outgoing: self.outgoing.iter().map(|p| p.abs() as usize).collect(),
        }
    }

    /// Convert the signed particle to its signum (as a floating point).
    ///
    /// This is used in calculations of changes in asymmetry to determine which
    /// sign the change really ought to be.
    pub fn as_sign(&self) -> InteractionParticleSigns {
        InteractionParticleSigns {
            ingoing: self.ingoing.iter().map(|p| p.signum() as f64).collect(),
            outgoing: self.outgoing.iter().map(|p| p.signum() as f64).collect(),
        }
    }
}

/// List of particle indices involved in the interaction.
///
/// This can be used to obtain the particle from the model's
/// [`Model::particles`] function.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct InteractionParticleIndices {
    ingoing: Vec<usize>,
    outgoing: Vec<usize>,
}

/// The signed particle to its signum (as a floating point).
///
/// This is used in calculations of changes in asymmetry to determine which sign
/// the change really ought to be.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct InteractionParticleSigns {
    ingoing: Vec<f64>,
    outgoing: Vec<f64>,
}

/// Generic interaction between particles.
pub trait Interaction<M: Model> {
    /// Return the particles involved in this interaction
    fn particles(&self) -> &InteractionParticles;

    /// Return the particles involved in this interaction, not distinguishing
    /// between particles an antiparticles.
    fn particles_idx(&self) -> &InteractionParticleIndices;

    /// Return the sign of the particles, with +1 for a particle and -1 for an
    /// antiparticle.
    fn particles_sign(&self) -> &InteractionParticleSigns;

    /// Calculate the decay width associated with a particular interaction.
    ///
    /// There may not be a result if the decay is not kinematically allowed, or
    /// not relevant.
    ///
    /// The default implementation simply returns `None` and must be implemented
    /// manually.
    fn width(&self, _: &Context<M>) -> Option<PartialWidth> {
        None
    }

    /// Calculate the reaction rate density.
    ///
    /// This is returned as a vector for each possible configuration.
    fn gamma(&self, c: &Context<M>) -> f64;

    /// Calculate the interaction rates density from the interaction rate
    /// density by taking into account the number densities prefactors.
    fn calculate_rate(&self, gamma: f64, c: &Context<M>) -> RateDensity {
        // If the interaction rate is 0 to begin with, there's no need to adjust
        // it to the particles' number densities.
        if gamma == 0.0 {
            return RateDensity {
                forward: 0.0,
                backward: 0.0,
                asymmetric_forward: 0.0,
                asymmetric_backward: 0.0,
            };
        }

        let particles_idx = self.particles_idx();
        let particles_sign = self.particles_sign();

        // A NaN should only occur from `0.0 / 0.0`, in which case the correct
        // value ought to be 0.
        let forward = nan_to_zero(
            particles_idx
                .ingoing
                .iter()
                .map(|&p| c.n[p] / c.eq[p])
                .product(),
        );
        let backward = nan_to_zero(
            particles_idx
                .outgoing
                .iter()
                .map(|&p| c.n[p] / c.eq[p])
                .product(),
        );
        let asymmetric_forward = forward
            - nan_to_zero(
                particles_idx
                    .ingoing
                    .iter()
                    .zip(&particles_sign.ingoing)
                    .map(|(&p, &a)| (c.n[p] - a * c.na[p]) / c.eq[p])
                    .product(),
            );
        let asymmetric_backward = forward
            - nan_to_zero(
                particles_idx
                    .outgoing
                    .iter()
                    .zip(&particles_sign.outgoing)
                    .map(|(&p, &a)| (c.n[p] - a * c.na[p]) / c.eq[p])
                    .product(),
            );

        let mut rate = RateDensity {
            forward,
            backward,
            asymmetric_forward,
            asymmetric_backward,
        };
        rate *= gamma;
        rate
    }

    /// Adjust the backward and forward rates such that they do not
    /// overshoot the equilibrium number densities.
    fn adjust_rate_overshoot(&self, mut rate: RateDensity, c: &Context<M>) -> RateDensity {
        // If an overshoot of the interaction rate is detected, the rate is
        // adjusted such that `dn` satisfies:
        //
        // ```text
        // n + dn = (eq + (ALPHA - 1) * n) / ALPHA
        // ```
        //
        // Large values means that the correction towards equilibrium are
        // weaker, while smaller values make the move towards equilibrium
        // stronger.  Values of ALPHA less than 1 will overshoot the
        // equilibrium.
        const ALPHA_N: f64 = 1.1;
        const ALPHA_NA: f64 = 1.1;

        let particles_idx = self.particles_idx();
        let particles_sign = self.particles_sign();

        let mut net_rate = rate.net_rate();
        let mut net_asymmetric_rate = rate.net_asymmetric_rate();

        if net_rate != 0.0 || net_asymmetric_rate != 0.0 {
            for (&p, a) in particles_idx.ingoing.iter().zip(&particles_sign.ingoing) {
                if overshoots(c, p, -net_rate) {
                    rate.forward = (c.n[p] - c.eq[p]) / ALPHA_N + rate.backward;
                    net_rate = rate.net_rate();
                }
                if asymmetry_overshoots(c, p, -a * net_asymmetric_rate) {
                    rate.asymmetric_forward = c.na[p] / ALPHA_NA + rate.asymmetric_backward;
                    net_asymmetric_rate = rate.net_asymmetric_rate();
                }
            }
            for (&p, a) in particles_idx.outgoing.iter().zip(&particles_sign.outgoing) {
                if overshoots(c, p, net_rate) {
                    rate.backward = (c.n[p] - c.eq[p]) / ALPHA_N + rate.forward;
                    net_rate = rate.net_rate();
                }
                if asymmetry_overshoots(c, p, a * net_asymmetric_rate) {
                    rate.asymmetric_backward = c.na[p] / ALPHA_NA + rate.asymmetric_forward;
                    net_asymmetric_rate = rate.net_asymmetric_rate();
                }
            }
        }
        rate
    }

    /// Add this interaction to the `dn` array.
    ///
    /// The changes in `dn` should contain only the effect from the integrated
    /// collision operator and not take into account the normalization to
    /// inverse-temperature evolution nor the dilation factor from expansion of
    /// the Universe.  These factors are handled separately and automatically.
    ///
    /// This function automatically adjusts the rate calculated by
    /// [`Interaction::calculate_rate`] so that overshooting is avoided.
    fn change(
        &self,
        mut dn: Array1<f64>,
        mut dna: Array1<f64>,
        c: &Context<M>,
    ) -> (Array1<f64>, Array1<f64>) {
        let rate = self.adjust_rate_overshoot(self.calculate_rate(self.gamma(c), c), c);
        let particles = self.particles();
        let particles_idx = self.particles_idx();
        let particles_sign = self.particles_sign();

        let net_rate = rate.net_rate();
        let net_asymmetric_rate = rate.net_asymmetric_rate();
        log::trace!(
            "γ({:?} ↔ {:?}) = {:<10.3e}",
            particles.ingoing,
            particles.outgoing,
            net_rate
        );
        log::trace!(
            "γ'({:?} ↔ {:?}) = {:<10.3e}",
            particles.ingoing,
            particles.outgoing,
            net_asymmetric_rate
        );

        for (&p, a) in particles_idx.ingoing.iter().zip(&particles_sign.ingoing) {
            dn[p] -= net_rate;
            dna[p] -= a * net_asymmetric_rate;
        }
        for (&p, a) in particles_idx.outgoing.iter().zip(&particles_sign.outgoing) {
            dn[p] += net_rate;
            dna[p] += a * net_asymmetric_rate;
        }

        (dn, dna)
    }
}

/// Check whether particle `i` from the model with the given rate change will
/// overshoot equilibrium.
pub fn overshoots<M: Model>(c: &Context<M>, i: usize, rate: f64) -> bool {
    (c.n[i] > c.eq[i] && c.n[i] + rate < c.eq[i]) || (c.n[i] < c.eq[i] && c.n[i] + rate > c.eq[i])
}

/// Check whether particle asymmetry `i` from the model with the given rate
/// change will overshoot 0.
pub fn asymmetry_overshoots<M: Model>(c: &Context<M>, i: usize, rate: f64) -> bool {
    (c.na[i] > 0.0 && c.na[i] + rate < 0.0) || (c.na[i] < 0.0 && c.na[i] + rate > 0.0)
}

/// Converts NaN floating points to 0
fn nan_to_zero(v: f64) -> f64 {
    if v.is_nan() {
        0.0
    } else {
        v
    }
}
