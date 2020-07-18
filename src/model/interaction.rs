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
use std::{collections::HashMap, fmt, hash};

struct IdentityHasher {
    state: u64,
}

impl hash::Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.state
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut s: [u8; 8] = [0, 0, 0, 0, 0, 0, 0, 0];
        for i in (0..bytes.len()).take(8) {
            s[i] = bytes[i]
        }
        self.state = u64::from_be_bytes(s);
    }
}

struct BuildIdentityHasher;

impl hash::BuildHasher for BuildIdentityHasher {
    type Hasher = IdentityHasher;

    fn build_hasher(&self) -> Self::Hasher {
        IdentityHasher { state: 0 }
    }
}

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

        for &p in &self.incoming {
            s.push_str(&model.particle_name(p).map_err(|_| ())?);
            s.push(' ');
        }

        s.push_str("↔");

        for &p in &self.outgoing {
            s.push(' ');
            s.push_str(&model.particle_name(p).map_err(|_| ())?);
        }

        Ok(s)
    }
}

impl fmt::Display for InteractionParticles {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();

        for &p in &self.incoming {
            s.push_str(&format!("{} ", p));
        }

        s.push_str("↔");

        for &p in &self.outgoing {
            s.push_str(&format!(" {}", p));
        }

        write!(f, "{}", s)
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
    fn width(&self, _c: &Context<M>) -> Option<PartialWidth> {
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
    /// avoid unnecessary computations.  Specifically, this should always return
    /// `None` when `self.gamma_enabled() == true`.
    ///
    /// As there can be some nice mathematical cancellations between the
    /// interaction rate and the number density normalizations, the result may
    /// not be the 'real' interaction rate and may be normalized by another
    /// factor.  For example, decays will be normalized by the equilibrium
    /// number density of the decaying particle in order to avoid possible `0 /
    /// 0` errors.  In order to get the real interaction rate, `real` should be
    /// set to true.
    fn gamma(&self, c: &Context<M>, real: bool) -> Option<f64>;

    /// Asymmetry between the interaction and its `$\CP$` conjugate:
    ///
    /// ```math
    /// \delta\gamma
    ///   \defeq \gamma(\vt a \to \vt b) - \gamma(\overline{\vt a} \to \overline{\vt b})
    ///   = \gamma(\vt a \to \vt b) - \gamma(\vt b \to \vt a)
    /// ```
    ///
    /// If there is no (relevant) asymmetry, then this should return `None`.
    ///
    /// Note that his is not the same as the asymmetry specified in creating the
    /// interaction, with the latter being defined as the asymmetry in the
    /// squared amplitudes and the former being subsequently computed.
    ///
    /// As there can be some nice mathematical cancellations between the
    /// interaction rate and the number density normalizations, the result may
    /// not be the 'real' interaction rate and may be normalized by another
    /// factor.  For example, decays will be normalized by the equilibrium
    /// number density of the decaying particle in order to avoid possible `0 /
    /// 0` errors.  In order to get the real interaction rate, `real` should be
    /// set to true.
    fn asymmetry(&self, _c: &Context<M>, _real: bool) -> Option<f64> {
        None
    }

    /// Calculate the actual interaction rates density taking into account
    /// factors related to the number densities of particles involved.
    ///
    /// The result must not be normalized by the Hubble rate, nor include the
    /// factors relating to the integration step size.
    ///
    /// The rate density is defined such that the change initial state particles
    /// is proportional to the negative of the rates contained, while the change
    /// for final state particles is proportional to the rates themselves.
    fn rate(&self, c: &Context<M>) -> Option<RateDensity> {
        let gamma = self.gamma(c, false).unwrap_or(0.0);
        let asymmetry = self.asymmetry(c, false).unwrap_or(0.0);

        // If both rates are 0, there's no need to adjust it to the particles'
        // number densities.
        if gamma == 0.0 && asymmetry == 0.0 {
            return None;
        }

        let particles_idx = self.particles_idx();
        let particles_sign = self.particles_sign();

        // A NaN should only occur from `0.0 / 0.0`, in which case the correct
        // value ought to be 0 as there are no actual particles to decay (in the
        // numerator).
        let forward_prefactor = particles_idx
            .incoming
            .iter()
            .map(|&p| checked_div(c.n[p], c.eq[p]))
            .product::<f64>();
        let backward_prefactor = particles_idx
            .outgoing
            .iter()
            .map(|&p| checked_div(c.n[p], c.eq[p]))
            .product::<f64>();

        let mut rate = RateDensity::zero();
        rate.symmetric = gamma * (forward_prefactor - backward_prefactor);
        rate.asymmetric = asymmetry * (forward_prefactor + backward_prefactor)
            + gamma
                * (checked_div(
                    particles_idx
                        .incoming
                        .iter()
                        .zip(&particles_sign.incoming)
                        .enumerate()
                        .map(|(i, (&p, &a))| {
                            a * c.na[p]
                                * particles_idx
                                    .incoming
                                    .iter()
                                    .enumerate()
                                    .filter_map(|(j, &p)| if i == j { None } else { Some(c.n[p]) })
                                    .product::<f64>()
                        })
                        .sum::<f64>(),
                    particles_idx
                        .incoming
                        .iter()
                        .map(|&p| c.n[p])
                        .product::<f64>(),
                ) - checked_div(
                    particles_idx
                        .outgoing
                        .iter()
                        .zip(&particles_sign.outgoing)
                        .enumerate()
                        .map(|(i, (&p, &a))| {
                            a * c.na[p]
                                * particles_idx
                                    .outgoing
                                    .iter()
                                    .enumerate()
                                    .filter_map(|(j, &p)| if i == j { None } else { Some(c.n[p]) })
                                    .product::<f64>()
                        })
                        .sum::<f64>(),
                    particles_idx
                        .outgoing
                        .iter()
                        .map(|&p| c.n[p])
                        .product::<f64>(),
                ));

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
    fn adjusted_rate(&self, c: &Context<M>) -> Option<RateDensity> {
        let mut rate = self.rate(c)? * c.step_size * c.normalization;

        // No need to do anything of both rates are 0.
        if rate.symmetric == 0.0 && rate.asymmetric == 0.0 {
            return None;
        }

        let particles_idx = self.particles_idx();
        let particles_sign = self.particles_sign();

        // Aggregate the particles to take into account multiplicities.
        let mut counts = HashMap::with_capacity_and_hasher(
            particles_idx.incoming.len() + particles_idx.outgoing.len(),
            BuildIdentityHasher,
        );
        for (&p, &a) in particles_idx.incoming.iter().zip(&particles_sign.incoming) {
            let c = counts.entry(p).or_insert((0.0, 0.0));
            (*c).0 -= 1.0;
            (*c).1 -= a;
        }
        for (&p, &a) in particles_idx.outgoing.iter().zip(&particles_sign.outgoing) {
            let c = counts.entry(p).or_insert((0.0, 0.0));
            (*c).0 += 1.0;
            (*c).1 += a;
        }

        // DEBUG
        // if log::log_enabled!(log::Level::Info) {
        //     log::info!(
        //         "γ({}) = {:>10.3e} | {:>10.3e}",
        //         self.particles().display(c.model).unwrap(),
        //         rate.symmetric,
        //         rate.asymmetric,
        //     );
        //     log::info!("counts: {:?}", counts);
        // }

        let mut changed = true;
        while changed {
            changed = false;

            for (&p, &(symmetric_count, asymmetric_count)) in &counts {
                // In the rate adjustment, the division by the count should
                // never be division by 0 as there should never be an overshoot
                // if the multiplicity factor is 0.
                if overshoots(c, p, symmetric_count * rate.symmetric) {
                    let new_rate =
                        ((c.n[p] - c.eq[p]) / symmetric_count).abs() * rate.symmetric.signum();
                    changed = changed
                        || (rate.symmetric - new_rate).abs()
                            / (rate.symmetric.abs() + new_rate.abs())
                            > 1e-10;
                    rate.symmetric = new_rate;
                }
                if asymmetry_overshoots(c, p, asymmetric_count * rate.asymmetric) {
                    let new_rate = (c.na[p] / asymmetric_count).abs() * rate.asymmetric.signum();
                    changed = changed
                        || (rate.asymmetric - new_rate).abs()
                            / (rate.asymmetric.abs() + new_rate.abs())
                            > 1e-10;
                    rate.asymmetric = new_rate;
                }
            }

            // if changed && log::log_enabled!(log::Level::Info) {
            //     log::info!(
            //         "--> γ({}) = {:>10.3e} | {:>10.3e}",
            //         self.particles().display(c.model).unwrap(),
            //         rate.symmetric,
            //         rate.asymmetric,
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
        if let Some(rate) = self.adjusted_rate(c) {
            let particles_idx = self.particles_idx();
            let particles_sign = self.particles_sign();

            for (&p, a) in particles_idx.incoming.iter().zip(&particles_sign.incoming) {
                dn[p] -= rate.symmetric;
                dna[p] -= a * rate.asymmetric;
            }
            for (&p, a) in particles_idx.outgoing.iter().zip(&particles_sign.outgoing) {
                dn[p] += rate.symmetric;
                dna[p] += a * rate.asymmetric;
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

    fn gamma(&self, c: &Context<M>, real: bool) -> Option<f64> {
        (*self).gamma(c, real)
    }

    fn asymmetry(&self, c: &Context<M>, real: bool) -> Option<f64> {
        (*self).asymmetry(c, real)
    }
    fn rate(&self, c: &Context<M>) -> Option<RateDensity> {
        (*self).rate(c)
    }

    fn adjusted_rate(&self, c: &Context<M>) -> Option<RateDensity> {
        (*self).adjusted_rate(c)
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

    fn gamma(&self, c: &Context<M>, real: bool) -> Option<f64> {
        self.as_ref().gamma(c, real)
    }

    fn asymmetry(&self, c: &Context<M>, real: bool) -> Option<f64> {
        self.as_ref().asymmetry(c, real)
    }

    fn rate(&self, c: &Context<M>) -> Option<RateDensity> {
        self.as_ref().rate(c)
    }

    fn adjusted_rate(&self, c: &Context<M>) -> Option<RateDensity> {
        self.as_ref().adjusted_rate(c)
    }

    fn change(&self, dn: &mut Array1<f64>, dna: &mut Array1<f64>, c: &Context<M>) {
        self.as_ref().change(dn, dna, c)
    }
}
