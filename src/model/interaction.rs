//! Common trait and implementations for interactions.

mod fast_interaction_result;
mod four_particle;
mod interaction_particles;
mod partial_width;
mod rate_density;
mod three_particle;

pub use fast_interaction_result::FastInteractionResult;
pub use interaction_particles::{DisplayError, InteractionParticles};
pub use partial_width::PartialWidth;
pub use rate_density::RateDensity;

pub use four_particle::FourParticle;
pub use three_particle::ThreeParticle;

use crate::{model::Model, solver::Context};
use ndarray::prelude::*;
#[cfg(feature = "serde")]
use std::cmp::Ordering;

/// Generic interaction between particles.
pub trait Interaction<M: Model> {
    /// Return the particles involved in this interaction
    fn particles(&self) -> &InteractionParticles;

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
    #[allow(unused_variables)]
    fn width(&self, c: &Context<M>) -> Option<PartialWidth> {
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
    /// Specifically, this corresponds to `$\gamma$` in the following expression:
    ///
    /// ```math
    /// H \beta n_1 \pfrac{n_a}{\beta} =
    ///   - \left( \frac{n_a}{n_a^{(0)}} \prod_{i \in X} \frac{n_i}{n_i^{(0)}} \right) \gamma(a X \to Y)
    ///   + \left( \prod_{i \in Y} \frac{n_i}{n_i^{(0)}} \right) \gamma(Y \to a X)
    /// ```
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
    #[allow(unused_variables)]
    fn delta_gamma(&self, c: &Context<M>, real: bool) -> Option<f64> {
        None
    }

    /// Net prefactor scaling the interaction rate.
    ///
    /// This is defined for a particle `$X \to Y$` as
    ///
    /// ```math
    /// \left( \prod_{i \in X} \frac{n_i}{n_i^{(0)}} \right)
    /// - \left( \prod_{i \in Y} \frac{n_i}{n_i^{(0)}} \right)
    /// ```
    ///
    /// If there are some `$n_i^{(0)}$` which are zero, then the prefactor will
    /// be infinite as the interaction will try and get rid of the particle in
    /// question.
    ///
    /// If there are multiple particles for which `$n_i^{(0)}$` is zero located
    /// on both sides, then the direction of the interaction is first determined
    /// by the side which contains the most zero equilibrium number densities.
    /// If these are also equal, then the result is 0.
    ///
    /// ```math
    /// \left( \prod_{i \in X} n_i \right) - \left( \prod_{i \in Y} n_i \right)
    /// ```
    ///
    /// This can produce a result which is infinite if some of the `$n_i^{(0)}`$
    /// are identically 0 and thus care must be taken to handle these.
    fn symmetric_prefactor(&self, c: &Context<M>) -> f64 {
        let ((forward_numerator, forward_denominator), (backward_numerator, backward_denominator)) =
            self.particles()
                .symmetric_prefactor(&c.n, &c.eq, &c.in_equilibrium);

        let forward_prefactor = checked_div(forward_numerator, forward_denominator);
        let backward_prefactor = checked_div(backward_numerator, backward_denominator);

        if forward_prefactor.is_finite() || backward_prefactor.is_finite() {
            // If either one or both are finite, we can compute the rate.
            forward_prefactor - backward_prefactor
        } else {
            // If they are both infinite, then there are equilibrium number
            // densities on both sides which are 0.
            //
            // In order to determine the direction, the number of zero
            // equilibrium number densities on each side is used first.  If they
            // are both equal, then the product of number densities is used.
            let particles = self.particles();

            let forward_zeros = particles
                .incoming_idx
                .iter()
                .filter(|&&p| c.eq[p] == 0.0)
                .count();
            let backward_zeros = particles
                .outgoing_idx
                .iter()
                .filter(|&&p| c.eq[p] == 0.0)
                .count();

            match forward_zeros.cmp(&backward_zeros) {
                Ordering::Less => f64::NEG_INFINITY,
                Ordering::Greater => f64::INFINITY,
                Ordering::Equal => 0.0,
            }
        }
    }

    /// Compute the prefactor to the symmetric interaction rate which alters the
    /// number density asymmetries.
    ///
    /// In the forward direction, this is calculated by
    ///
    /// ```math
    /// \frac{
    ///   \sum_{i \in X} \Delta_i \prod_{j \neq i} n_j
    /// }{
    ///   \prod_{i \in X} n_i^{(0)}
    /// }
    /// ```
    ///
    /// and similarly for the backward rate.  An example of a `$1 \to 2, 3$`
    /// rate is:
    ///
    /// ```math
    /// \frac{\Delta_1}{n_1^{(0)}} - \frac{\Delta_2 n_3 + \Delta_3 n_2}{n_2^{(0)} n_3^{(0)}}
    /// ```
    ///
    /// This can produce a result which is infinite if some of the `$n_i^{(0)}`$
    /// are identically 0 and thus care must be taken to handle these.
    fn asymmetric_prefactor(&self, c: &Context<M>) -> f64 {
        let ((forward_numerator, forward_denominator), (backward_numerator, backward_denominator)) =
            self.particles().asymmetric_prefactor(
                &c.n,
                &c.na,
                &c.eq,
                &c.in_equilibrium,
                &c.no_asymmetry,
            );
        let forward = checked_div(forward_numerator, forward_denominator);
        let backward = checked_div(backward_numerator, backward_denominator);

        forward - backward
    }

    /// Calculate the actual interaction rates density taking into account
    /// factors related to the number densities of particles involved.
    ///
    /// The result is normalized by the Hubble rate, but does not include
    /// factors relating to the integration step size.  Specifically, it
    /// corresponds to the right hand side of the equation:
    ///
    /// ```math
    /// \pfrac{n_a}{\beta} = \frac{1}{H \beta n_1} \left[
    ///   - \left( \frac{n_a}{n_a^{(0)}} \prod_{i \in X} \frac{n_i}{n_i^{(0)}} \right) \gamma(a X \to Y)
    ///   + \left( \prod_{i \in Y} \frac{n_i}{n_i^{(0)}} \right) \gamma(Y \to a X)
    /// \right].
    /// ```
    ///
    /// The normalization factor `$(H \beta n_1)^{-1}$` is accessible within the
    /// current context through [`Context::normalization`].
    ///
    /// The rate density is defined such that the change initial state particles
    /// is proportional to the negative of the rates contained, while the change
    /// for final state particles is proportional to the rates themselves.
    ///
    /// The default implementation uses the output of [`Interaction::gamma`] and
    /// [`Interaction::delta_gamma`] in order to computer the actual rate.
    fn rate(&self, c: &Context<M>) -> Option<RateDensity> {
        let gamma = self.gamma(c, false).unwrap_or_default();
        let delta_gamma = self.delta_gamma(c, false);

        // If both rates are 0, there's no need to adjust it to the particles'
        // number densities.
        if gamma == 0.0 && (delta_gamma.is_none() || delta_gamma.unwrap() == 0.0) {
            return None;
        }

        let mut rate = RateDensity::zero();
        let symmetric_prefactor = self.symmetric_prefactor(c);
        rate.gamma = gamma;
        rate.symmetric = gamma * symmetric_prefactor;
        rate.delta_gamma = delta_gamma;
        let delta_gamma = delta_gamma.unwrap_or_default();
        rate.asymmetric = delta_gamma * symmetric_prefactor + gamma * self.asymmetric_prefactor(c);

        Some(rate * c.normalization)
    }

    /// Checks whether the interaction has already been deemed fast.
    ///
    /// On substep 0, this will always return `false` even is [`is_fast`] will
    /// return `true`.  This will also only work is [`is_fast`] was checked at
    /// substep 0.
    ///
    /// If fast interactions are disabled, this always returns `false`.
    fn is_fast_check(&self, c: &Context<M>) -> bool {
        c.fast_interactions
            .as_ref()
            .map_or(false, |fast_interactions| {
                let fast_interactions = fast_interactions.read().unwrap();
                fast_interactions.contains(self.particles())
            })
    }

    /// Check if the interaction is determined as 'fast'.
    ///
    /// Physically, this occurs when `$\ddfrac{N}{t} \propto \gamma / H \gg 1$`,
    /// or  `$\ddfrac{n}{\beta} \propto \gamma / H N \gg 1$`.
    ///
    /// Note that the comparison is only done on the first substep and the
    /// result is stored for subsequent substeps in order to avoid unecessarily
    /// compute the interaction rate.  This means that if the interaction is
    /// initially slow and becomes fast at a subsequent substep, this will still
    /// return `false`.  For subsequent substeps, this returns the same result
    /// as [`is_fast_check`].
    ///
    /// A `true` result must mean that we are using fast interactions.  If fast
    /// interactions are disabled, this always returns `false`.
    fn is_fast(&self, rate: &RateDensity, c: &Context<M>) -> bool {
        const FAST_THRESHOLD: f64 = 1e1;

        c.fast_interactions
            .as_ref()
            .map_or(false, |fast_interactions| {
                if c.substep == 0 {
                    // We have to multiply by beta as gamma is normalized by `$1
                    // / H N \beta$`, but we must check whether `$\gamma / H N$`
                    // is larger.
                    if rate.gamma.abs() * c.beta > FAST_THRESHOLD {
                        log::debug!(
                            "[{}.{:02}|{:>9.3e}] Fast interaction γ = {:.3e}",
                            c.step,
                            c.substep,
                            c.beta,
                            rate.gamma,
                        );

                        let mut fast_interactions = fast_interactions.write().unwrap();
                        let mut interaction = self.particles().clone();
                        interaction.gamma_ratio = rate.gamma_ratio();
                        fast_interactions.insert(interaction);

                        true
                    } else {
                        false
                    }
                } else {
                    let fast_interactions = fast_interactions.read().unwrap();
                    fast_interactions.contains(self.particles())
                }
            })
    }

    /// Adjust the overshoot as calculated by the interaction.
    ///
    /// Returns `true` if any adjustments were made, and `false` otherwise.
    fn adjust_overshoot(&self, rate: &mut RateDensity, c: &Context<M>) -> bool {
        // Although an overshoot factor results in the exact solution in a
        // single step, when this is incorporated into the Runge-Kutta method it
        // tends to undershoot the result.  As a result, an overshoot factor is
        // introduced to balance this.  This *must* be in the range [0, 2], with
        // values greater than 2 causing the results to diverge.
        // const OVERSHOOT_FACTOR: f64 = 1.693_147;
        const OVERSHOOT_FACTOR: f64 = 1.5;
        let particles = self.particles();
        let mut result = false;

        if overshoots(c, particles, rate) {
            let delta = particles.symmetric_delta(&c.n, &c.eqn, &c.in_equilibrium);
            rate.symmetric = if rate.symmetric.abs() < delta.abs() {
                rate.symmetric
            } else {
                OVERSHOOT_FACTOR * delta
            };
            result = true;
        }

        if asymmetry_overshoots(c, particles, rate) {
            let delta =
                particles.asymmetric_delta(&c.n, &c.na, &c.eqn, &c.in_equilibrium, &c.no_asymmetry);
            rate.asymmetric = if rate.asymmetric.abs() < delta.abs() {
                rate.asymmetric
            } else {
                OVERSHOOT_FACTOR * delta
            };
            result = true;
        }

        result
    }

    /// Compute the adjusted rate to handle possible overshoots of the
    /// equilibrium number density.
    ///
    /// The output of this function should be in principle the same as
    /// [`Interaction::rate`], but may be smaller if the interaction rate would
    /// overshoot an equilibrium number density.  Furthermore, it is scale by
    /// the current integration step size `$h$`:
    ///
    /// ```math
    /// \Delta n = \pfrac{n}{\beta} h.
    /// ```
    ///
    /// The default implementation uses the output of [`Interaction::rate`] in
    /// order to computer the actual rate.
    ///
    /// This method should generally not be implemented.  If it is implemented
    /// separately, one must take care to take into account the integration step
    /// size which is available in [`Context::step_size`].
    fn adjusted_rate(&self, c: &Context<M>) -> Option<RateDensity> {
        if self.is_fast_check(c) {
            return None;
        }

        let mut rate = self.rate(c)?;
        log::trace!(
            "[{}.{:02}|{:>9.3e}]          Rate: {:<12.5e}|{:<12.5e}",
            c.step,
            c.substep,
            c.beta,
            rate.gamma,
            rate.symmetric
        );

        if self.is_fast(&rate, c) {
            return None;
        }
        rate *= c.step_size;

        if self.adjust_overshoot(&mut rate, c) {
            log::trace!(
                "[{}.{:02}|{:>9.3e}] Adjusted Rate: {:<12.5e}|{:<12.5e}",
                c.step,
                c.substep,
                c.beta,
                rate.gamma / c.step_size,
                rate.symmetric / c.step_size
            );
        }

        debug_assert!(
            rate.symmetric.is_finite(),
            "[{}.{:02}|{:>9.3e}] Non-finite adjusted interaction rate for interaction {}: {}",
            c.step,
            c.substep,
            c.beta,
            self.particles()
                .display(c.model)
                .unwrap_or_else(|_| self.particles().short_display()),
            rate.symmetric
        );
        debug_assert!(
            rate.asymmetric.is_finite(),
            "[{}.{:02}|{:>9.3e}] Non-finite asymmetric adjusted interaction rate for interaction {}: {}",
            c.step,
            c.substep,
            c.beta,
            self.particles()
                .display(c.model)
                .unwrap_or_else(|_| self.particles().short_display()),
            rate.asymmetric
        );

        Some(rate)
    }

    /// Add this interaction to the `dn` and `dna` array.
    ///
    /// This must use the final `$\Delta n$` taking into account all
    /// normalization (`$(H \beta n_1)^{-1}$`) and numerical integration factors
    /// (the step size `$h$`).
    ///
    /// The default implementation uses the output of
    /// [`Interaction::adjusted_rate`] in order to computer the actual rate.
    ///
    /// This method should generally not be implemented.
    fn change(&self, dn: &mut Array1<f64>, dna: &mut Array1<f64>, c: &Context<M>) {
        if let Some(rate) = self.adjusted_rate(c) {
            let particles = self.particles();

            for (&p, &(c, ca)) in &particles.particle_counts {
                dn[p] += c * rate.symmetric;
                dna[p] += ca * rate.asymmetric;
            }
        }
    }

    /// Output a 'pretty' version of the interaction particles using the
    /// particle names from the model.
    ///
    /// See [`InteractionParticles::display`] for more information.
    #[allow(clippy::missing_errors_doc)]
    fn display(&self, model: &M) -> Result<String, DisplayError> {
        self.particles().display(model)
    }
}

/// Check whether the computed change in particle number density will cause an
/// overshoot of equilibrium.
#[must_use]
pub fn overshoots<M>(c: &Context<M>, particles: &InteractionParticles, rate: &RateDensity) -> bool {
    let f = particles.symmetric_prefactor_fn(&c.n, &c.eqn, &c.in_equilibrium);
    let a = f(0.0);
    if !a.is_finite() {
        true
    } else if a == 0.0 {
        false
    } else {
        let b = f(rate.symmetric);
        if !b.is_finite() {
            true
        } else if b == 0.0 {
            false
        } else {
            a.signum() != b.signum()
        }
    }
}

/// Check whether the computed change in particle number density asymmetry will
/// caan overshoot of equilibrium.
#[must_use]
pub fn asymmetry_overshoots<M>(
    c: &Context<M>,
    particles: &InteractionParticles,
    rate: &RateDensity,
) -> bool {
    let f =
        particles.asymmetric_prefactor_fn(&c.n, &c.na, &c.eqn, &c.in_equilibrium, &c.no_asymmetry);
    let a = f(0.0);
    if !a.is_finite() {
        true
    } else if a == 0.0 {
        false
    } else {
        let b = f(rate.asymmetric);
        if !b.is_finite() {
            true
        } else if b == 0.0 {
            false
        } else {
            a.signum() != b.signum()
        }
    }
}

/// Computes the ratio `$n / eq$` in a manner that never returns NaN.
///
/// This is to be used in the context of calculating the scaling of the
/// interaction density by the number density ratios `$n / eq$`.
///
/// Provided the both inputs are numbers, then a NaN can only arise from `$n =
/// eq = 0$`, in which case the actual answer should really be 0 as there are no
/// particles to interaction.
///
/// Note that if `$eq = 0$` and `$n \neq 0$`, then the result will be infinite.
#[must_use]
#[inline]
pub(crate) fn checked_div(n: f64, eq: f64) -> f64 {
    let v = n / eq;
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

    fn delta_gamma(&self, c: &Context<M>, real: bool) -> Option<f64> {
        (*self).delta_gamma(c, real)
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

    fn delta_gamma(&self, c: &Context<M>, real: bool) -> Option<f64> {
        self.as_ref().delta_gamma(c, real)
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

#[cfg(test)]
mod tests {
    #[test]
    #[allow(clippy::float_cmp)]
    fn checked_div() {
        let vals = [-10.0, -2.0, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 2.0, 10.0];
        for &a in &vals {
            for &b in &vals {
                assert_eq!(a / b, super::checked_div(a, b));
            }
        }

        for &a in &vals {
            if a > 0.0 {
                assert_eq!(f64::INFINITY, super::checked_div(a, 0.0));
            } else {
                assert_eq!(f64::NEG_INFINITY, super::checked_div(a, 0.0));
            }
        }

        for &a in &vals {
            assert_eq!(0.0, super::checked_div(0.0, a));
        }

        assert_eq!(0.0, super::checked_div(0.0, 0.0));
    }
}
