//! Common trait and implementations for interactions.

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
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct InteractionParticles {
    /// Initial state particles
    pub incoming: Vec<isize>,
    /// Final state particles
    pub outgoing: Vec<isize>,
}

impl InteractionParticles {
    /// Convert the signed particle numbers to indices which can be used to
    /// index the model's particle.
    #[must_use]
    pub fn as_idx(&self) -> InteractionParticleIndices {
        InteractionParticleIndices {
            incoming: self.incoming.iter().map(|p| p.abs() as usize).collect(),
            outgoing: self.outgoing.iter().map(|p| p.abs() as usize).collect(),
        }
    }

    /// Convert the signed particle to its signum (as a floating point).
    ///
    /// This is used in calculations of changes in asymmetry to determine which
    /// sign the change really ought to be.
    #[must_use]
    pub fn as_sign(&self) -> InteractionParticleSigns {
        InteractionParticleSigns {
            incoming: self.incoming.iter().map(|p| p.signum() as f64).collect(),
            outgoing: self.outgoing.iter().map(|p| p.signum() as f64).collect(),
        }
    }

    /// Output a 'pretty' version of the interaction particles using the
    /// particle names from the model.
    ///
    /// # Errors
    ///
    /// If any particles can't be found in the model, this will produce an
    /// error.
    pub fn display<M>(&self, model: &M) -> Result<String, ()>
    where
        M: Model,
    {
        let mut s = String::new();

        if let Some(&p) = self.incoming.first() {
            s.push_str(&model.particle_name(p).map_err(|_| ())?);
            s.push(' ');
        }
        for &p in self.incoming.iter().skip(1) {
            s.push_str(&model.particle_name(p).map_err(|_| ())?);
            s.push(' ');
        }

        s.push_str("↔");

        if let Some(&p) = self.outgoing.first() {
            s.push(' ');
            s.push_str(&model.particle_name(p).map_err(|_| ())?);
        }
        for &p in self.outgoing.iter().skip(1) {
            s.push(' ');
            s.push_str(&model.particle_name(p).map_err(|_| ())?);
        }

        Ok(s)
    }
}

/// List of particle indices involved in the interaction.
///
/// This can be used to obtain the particle from the model's
/// [`Model::particles`] function.
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct InteractionParticleIndices {
    /// Initial state particles
    pub incoming: Vec<usize>,
    /// Final state particles
    pub outgoing: Vec<usize>,
}

/// The signed particle to its signum (as a floating point).
///
/// This is used in calculations of changes in asymmetry to determine which sign
/// the change really ought to be.
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct InteractionParticleSigns {
    /// Initial state particles
    pub incoming: Vec<f64>,
    /// Final state particles
    pub outgoing: Vec<f64>,
}

/// Generic interaction between particles.
pub trait Interaction<M>
where
    M: Model,
{
    /// Return the particles involved in this interaction
    fn particles(&self) -> &InteractionParticles;

    /// Return the particles involved in this interaction, not distinguishing
    /// between particles an antiparticles.
    fn particles_idx(&self) -> &InteractionParticleIndices;

    /// Return the sign of the particles, with +1 for a particle and -1 for an
    /// antiparticle.
    fn particles_sign(&self) -> &InteractionParticleSigns;

    /// Whether this interaction is to be used to determine decays.
    ///
    /// If width calcuilations are disable, then [`Interaction::width`] is
    /// expected to return `None` all the time.  If it is enabled, then
    /// [`Interaction::width`] is expected to return `None` only when the decay
    /// is not kinematically allowed.
    fn width_enabled(&self) -> bool;

    /// Calculate the decay width associated with a particular interaction.
    ///
    /// There may not be a result if the decay is not kinematically allowed, or
    /// not relevant.
    ///
    /// The default implementation simply returns `None` and must be implemented
    /// manually.  Care must be taken to obey [`Interaction::width_enabled`] in
    /// order to avoid unnecessary computation (though the incorrect
    /// implementation will not be detrimental other than in performance).
    fn width(&self, _: &Context<M>) -> Option<PartialWidth> {
        None
    }

    /// Whether this interaction is to be used within the Boltzmann equations.
    ///
    /// If this returns true, then [`Interaction::gamma`] is expected to return
    /// `None`.
    fn gamma_enabled(&self) -> bool;

    /// Calculate the reaction rate density of this interaction.
    ///
    /// This must return the result *before* it is normalized by the Hubble rate
    /// or other factors related to the number densities of particles involved.
    /// It also must *not* be normalized to the integration step size.
    ///
    /// Care must be taken to obey [`Interaction::gamma_enabled`] in order to
    /// avoid computation.  Specifically, this should always return `None` when
    /// `self.gamma_enabled() == true`.
    fn gamma(&self, c: &Context<M>) -> Option<f64>;

    /// Asymmetry
    ///
    /// TODO: Document better
    fn asymmetry(&self, _c: &Context<M>) -> Option<f64> {
        None
    }

    /// Calculate the actual interaction rates density taking into account
    /// factors related to the number densities of particles involved.
    ///
    /// The result must not be normalized by the Hubble rate, nor include the
    /// factors relating to the integration step size.
    fn rate(&self, gamma: Option<f64>, c: &Context<M>) -> Option<RateDensity> {
        // If there's no interaction rate or it is 0 to begin with, there's no
        // need to adjust it to the particles' number densities.
        if gamma.map_or(true, |gamma| gamma == 0.0) {
            return None;
        }
        let gamma = gamma.unwrap();

        let particles_idx = self.particles_idx();
        let particles_sign = self.particles_sign();

        // A NaN should only occur from `0.0 / 0.0`, in which case the correct
        // value ought to be 0.
        let forward = particles_idx
            .incoming
            .iter()
            .map(|&p| checked_div(c.n[p], c.eq[p]))
            .product::<f64>();
        let backward = particles_idx
            .outgoing
            .iter()
            .map(|&p| checked_div(c.n[p], c.eq[p]))
            .product::<f64>();
        let asymmetric_forward = forward
            - particles_idx
                .incoming
                .iter()
                .zip(&particles_sign.incoming)
                .map(|(&p, &a)| checked_div(c.n[p] - a * c.na[p], c.eq[p]))
                .product::<f64>();
        let asymmetric_backward = forward
            - particles_idx
                .outgoing
                .iter()
                .zip(&particles_sign.outgoing)
                .map(|(&p, &a)| checked_div(c.n[p] - a * c.na[p], c.eq[p]))
                .product::<f64>();

        let mut rate = RateDensity {
            forward,
            backward,
            asymmetric_forward,
            asymmetric_backward,
        };
        rate *= gamma;
        Some(rate)
    }

    /// Adjust the backward and forward rates such that they do not overshoot
    /// the equilibrium number densities.
    ///
    /// The rate density going into this function is expected to be the output
    /// of [`Interaction::rate`] and must not be previously normalized to the
    /// Hubble rate nor include factors relating to the integration step size.
    ///
    /// The output of this function however *will* be normalized by the HUbble
    /// rate and *will* factors relating to the numerical integration.
    fn adjust_rate_overshoot(
        &self,
        rate: Option<RateDensity>,
        c: &Context<M>,
    ) -> Option<RateDensity> {
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

        let mut rate = rate? * c.step_size * c.normalization;

        let particles_idx = self.particles_idx();
        let particles_sign = self.particles_sign();

        // DEBUG
        // let particles = self.particles();
        // log::trace!(
        //     "{:<10}: ↔ {:<12.3e} | → {:<12.3e} | ← {:<12.3e} | a↔ {:<12.3e} | a→ {:<12.3e} | a← {:<12.3e}",
        //     particles.display(c.model).unwrap(),
        //     rate.net_rate(),
        //     rate.forward,
        //     rate.backward,
        //     rate.net_asymmetric_rate(),
        //     rate.asymmetric_forward,
        //     rate.asymmetric_backward
        // );

        let mut net_rate = rate.net_rate();
        let mut net_asymmetric_rate = rate.net_asymmetric_rate();

        let mut changed = net_rate != 0.0 || net_asymmetric_rate != 0.0;
        while changed {
            changed = false;

            for (&p, a) in particles_idx.incoming.iter().zip(&particles_sign.incoming) {
                if overshoots(c, p, -net_rate) {
                    rate.forward = (c.n[p] - c.eq[p]) / ALPHA_N + rate.backward;
                    changed = ((net_rate - rate.net_rate()) / net_rate).abs() > 1e-10;
                    net_rate = rate.net_rate();
                }
                if asymmetry_overshoots(c, p, -a * net_asymmetric_rate) {
                    rate.asymmetric_forward = c.na[p] / ALPHA_NA + rate.asymmetric_backward;
                    changed = ((net_asymmetric_rate - rate.net_asymmetric_rate())
                        / net_asymmetric_rate)
                        .abs()
                        > 1e-10;
                    net_asymmetric_rate = rate.net_asymmetric_rate();
                }
            }

            for (&p, a) in particles_idx.outgoing.iter().zip(&particles_sign.outgoing) {
                if overshoots(c, p, net_rate) {
                    rate.backward = (c.n[p] - c.eq[p]) / ALPHA_N + rate.forward;
                    changed = ((net_rate - rate.net_rate()) / net_rate).abs() > 1e-10;
                    net_rate = rate.net_rate();
                }
                if asymmetry_overshoots(c, p, a * net_asymmetric_rate) {
                    rate.asymmetric_backward = c.na[p] / ALPHA_NA + rate.asymmetric_forward;
                    changed = ((net_asymmetric_rate - rate.net_asymmetric_rate())
                        / net_asymmetric_rate)
                        .abs()
                        > 1e-10;
                    net_asymmetric_rate = rate.net_asymmetric_rate();
                }
            }

            // if changed {
            //     log::trace!(
            //         "  updated : ↔ {:<12.3e} | → {:<12.3e} | ← {:<12.3e} | a↔ {:<12.3e} | a→ {:<12.3e} | a← {:<12.3e}",
            //         particles.display(c.model).unwrap(),
            //         rate.net_rate(),
            //         rate.forward,
            //         rate.backward,
            //         rate.net_asymmetric_rate(),
            //         rate.asymmetric_forward,
            //         rate.asymmetric_backward
            //     );
            // }
        }

        Some(rate)
    }

    /// Add this interaction to the `dn` and `dna` array.
    ///
    /// The changes in `dn` should contain only the effect from the integrated
    /// collision operator and not take into account the normalization to
    /// inverse-temperature evolution nor the dilation factor from expansion of
    /// the Universe.  These factors are handled separately and automatically.
    ///
    /// This function automatically adjusts the rate calculated by
    /// [`Interaction::rate`] so that overshooting is avoided.
    fn change(&self, dn: &mut Array1<f64>, dna: &mut Array1<f64>, c: &Context<M>) {
        let rate = self.adjust_rate_overshoot(self.rate(self.gamma(c), c), c);

        if rate.is_none() {
            return;
        }
        let rate = rate.unwrap();

        let particles_idx = self.particles_idx();
        let particles_sign = self.particles_sign();

        let net_rate = rate.net_rate();
        let net_asymmetric_rate = rate.net_asymmetric_rate();

        // DEBUG
        // let particles = self.particles();
        // if let Ok(interaction) = particles.display(c.model) {
        //     log::trace!(" γ({}) = {:<10.3e}", interaction, net_rate);
        //     log::trace!("γ'({}) = {:<10.3e}", interaction, net_asymmetric_rate);
        // } else {
        //     log::trace!(
        //         " γ({:?} ↔ {:?}) = {:<10.3e}",
        //         particles.incoming,
        //         particles.outgoing,
        //         net_rate
        //     );
        //     log::trace!(
        //         "γ'({:?} ↔ {:?}) = {:<10.3e}",
        //         particles.incoming,
        //         particles.outgoing,
        //         net_asymmetric_rate
        //     );
        // }

        for (&p, a) in particles_idx.incoming.iter().zip(&particles_sign.incoming) {
            dn[p] -= net_rate;
            dna[p] -= a * net_asymmetric_rate;
        }
        for (&p, a) in particles_idx.outgoing.iter().zip(&particles_sign.outgoing) {
            dn[p] += net_rate;
            dna[p] += a * net_asymmetric_rate;
        }

        if let Some(asymmetry) = self.asymmetry(c) {
            let source = net_rate * asymmetry;
            for (&p, a) in particles_idx.outgoing.iter().zip(&particles_sign.outgoing) {
                dna[p] += a * source;
            }
        }
    }
}

/// Check whether particle `i` from the model with the given rate change will
/// overshoot equilibrium.
#[must_use]
pub fn overshoots<M>(c: &Context<M>, i: usize, rate: f64) -> bool {
    (c.n[i] > c.eq[i] && c.n[i] + rate < c.eq[i]) || (c.n[i] < c.eq[i] && c.n[i] + rate > c.eq[i])
}

/// Check whether particle asymmetry `i` from the model with the given rate
/// change will overshoot 0.
#[must_use]
pub fn asymmetry_overshoots<M>(c: &Context<M>, i: usize, rate: f64) -> bool {
    (c.na[i] > 0.0 && c.na[i] + rate < 0.0) || (c.na[i] < 0.0 && c.na[i] + rate > 0.0)
}

/// Converts NaN floating points to 0
#[must_use]
#[inline]
pub(crate) fn checked_div(a: f64, b: f64) -> f64 {
    let v = a / b;
    if v.is_nan() {
        0.0
    } else {
        v
    }
}

impl<I: ?Sized, M> Interaction<M> for &I
where
    I: Interaction<M>,
    M: Model,
{
    fn particles(&self) -> &InteractionParticles {
        (*self).particles()
    }

    fn particles_idx(&self) -> &InteractionParticleIndices {
        (*self).particles_idx()
    }

    fn particles_sign(&self) -> &InteractionParticleSigns {
        (*self).particles_sign()
    }

    fn width_enabled(&self) -> bool {
        (*self).width_enabled()
    }

    fn width(&self, c: &Context<M>) -> Option<PartialWidth> {
        (*self).width(c)
    }

    fn gamma_enabled(&self) -> bool {
        (*self).gamma_enabled()
    }

    fn gamma(&self, c: &Context<M>) -> Option<f64> {
        (*self).gamma(c)
    }

    fn rate(&self, gamma: Option<f64>, c: &Context<M>) -> Option<RateDensity> {
        (*self).rate(gamma, c)
    }

    fn adjust_rate_overshoot(
        &self,
        rate: Option<RateDensity>,
        c: &Context<M>,
    ) -> Option<RateDensity> {
        (*self).adjust_rate_overshoot(rate, c)
    }

    fn change(&self, dn: &mut Array1<f64>, dna: &mut Array1<f64>, c: &Context<M>) {
        (*self).change(dn, dna, c)
    }
}

impl<I: ?Sized, M> Interaction<M> for Box<I>
where
    I: Interaction<M>,
    M: Model,
{
    fn particles(&self) -> &InteractionParticles {
        self.as_ref().particles()
    }

    fn particles_idx(&self) -> &InteractionParticleIndices {
        self.as_ref().particles_idx()
    }

    fn particles_sign(&self) -> &InteractionParticleSigns {
        self.as_ref().particles_sign()
    }

    fn width_enabled(&self) -> bool {
        self.as_ref().width_enabled()
    }

    fn width(&self, c: &Context<M>) -> Option<PartialWidth> {
        self.as_ref().width(c)
    }

    fn gamma_enabled(&self) -> bool {
        self.as_ref().gamma_enabled()
    }

    fn gamma(&self, c: &Context<M>) -> Option<f64> {
        self.as_ref().gamma(c)
    }

    fn rate(&self, gamma: Option<f64>, c: &Context<M>) -> Option<RateDensity> {
        self.as_ref().rate(gamma, c)
    }

    fn adjust_rate_overshoot(
        &self,
        rate: Option<RateDensity>,
        c: &Context<M>,
    ) -> Option<RateDensity> {
        self.as_ref().adjust_rate_overshoot(rate, c)
    }

    fn change(&self, dn: &mut Array1<f64>, dna: &mut Array1<f64>, c: &Context<M>) {
        self.as_ref().change(dn, dna, c)
    }
}
