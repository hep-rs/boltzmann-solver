//! Common trait and implementations for interactions.

mod fast_interaction_result;
mod four_particle;
mod interaction_particles;
mod partial_width;
mod rate_density;
mod three_particle;

pub use fast_interaction_result::FastInteractionResult;
pub use interaction_particles::{DisplayError, Particles};
pub use partial_width::PartialWidth;
pub use rate_density::RateDensity;

pub use four_particle::FourParticle;
pub use three_particle::ThreeParticle;

use crate::{model::Model, solver::Context};
use ndarray::prelude::*;
use std::cmp::Ordering;

/// Threshold value of `$\gamma / H N$` which is used to determine when an
/// interaction is fast.
const FAST_THRESHOLD: f64 = 1e0;

/// Threshold value of `$m \beta$` above which particle's equilibrium number
/// density is deemed to be too small which might lead to inaccurate results if
/// `$\gamma$` is also very small.
const M_BETA_THRESHOLD: f64 = 1e1;

/// Generic interaction between particles.
pub trait Interaction<M: Model> {
    /// Return the particles involved in this interaction
    fn particles(&self) -> &Particles;

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
    fn width(&self, context: &Context<M>) -> Option<PartialWidth> {
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
    fn gamma(&self, context: &Context<M>, real: bool) -> Option<f64>;

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
    fn delta_gamma(&self, context: &Context<M>, real: bool) -> Option<f64> {
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
    fn symmetric_prefactor(&self, context: &Context<M>) -> f64 {
        let ((forward_numerator, forward_denominator), (backward_numerator, backward_denominator)) =
            self.particles()
                .symmetric_prefactor(&context.n, &context.eq, context.in_equilibrium);

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
                .filter(|&&p| context.eq[p] == 0.0)
                .count();
            let backward_zeros = particles
                .outgoing_idx
                .iter()
                .filter(|&&p| context.eq[p] == 0.0)
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
    fn asymmetric_prefactor(&self, context: &Context<M>) -> f64 {
        let ((forward_numerator, forward_denominator), (backward_numerator, backward_denominator)) =
            self.particles().asymmetric_prefactor(
                &context.n,
                &context.na,
                &context.eq,
                context.in_equilibrium,
                context.no_asymmetry,
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
    fn rate(&self, context: &Context<M>) -> Option<RateDensity> {
        let gamma = self.gamma(context, true)?;
        let delta_gamma = self.delta_gamma(context, true);

        // If both rates are 0, there's no need to adjust it to the particles'
        // number densities.
        if gamma == 0.0 && delta_gamma.unwrap_or_default() == 0.0 {
            return None;
        }

        let mut rate = RateDensity::zero();
        let symmetric_prefactor = self.symmetric_prefactor(context);
        rate.gamma = gamma;
        rate.symmetric = checked_mul(gamma, symmetric_prefactor);
        rate.delta_gamma = delta_gamma;
        rate.asymmetric = checked_mul(delta_gamma.unwrap_or_default(), symmetric_prefactor)
            + checked_mul(gamma, self.asymmetric_prefactor(context));

        Some(rate * context.normalization)
    }

    /// Checks whether the interaction has already been deemed fast.
    ///
    /// On substep 0, this will always return `false` even is [`is_fast`] will
    /// return `true`.  This will also only work is [`is_fast`] was checked at
    /// substep 0.
    ///
    /// If fast interactions are disabled, this always returns `false`.
    fn is_fast_check(&self, context: &Context<M>) -> bool {
        context
            .fast_interactions
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
    fn is_fast(&self, rate: &RateDensity, context: &Context<M>) -> bool {
        context
            .fast_interactions
            .as_ref()
            .map_or(false, |fast_interactions| {
                if context.substep == 0 {
                    // We have to multiply by beta as gamma is normalized by `$1
                    // / H N \beta$`, but we must check whether `$\gamma / H N$`
                    // is larger.
                    if rate.gamma_tilde.abs() * context.beta > FAST_THRESHOLD {
                        log::trace!(
                            "[{}.{:02}|{:>10.4e}] Detected fast interaction {} with γ̃ = {:.3e}",
                            context.step,
                            context.substep,
                            context.beta,
                            self.particles()
                                .display(context.model)
                                .unwrap_or_else(|_| self.particles().short_display()),
                            rate.gamma_tilde,
                        );

                        let mut fast_interactions = fast_interactions.write().unwrap();
                        let mut particles = self.particles().clone();
                        particles.gamma_tilde = Some(rate.gamma_tilde);
                        particles.delta_gamma_tilde = rate.delta_gamma_tilde;
                        particles.heavy = context
                            .model
                            .particles()
                            .iter()
                            .enumerate()
                            .filter_map(|(i, p)| {
                                if p.mass * context.beta > M_BETA_THRESHOLD {
                                    Some(i)
                                } else {
                                    None
                                }
                            })
                            .collect();
                        fast_interactions.insert(particles);

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
    ///
    /// If an interaction is deeemed to be too fast for the main runge-Kutta
    /// routine to provide an accurate result, `None` is returned and the
    /// particle's [`interaction::Particles`] should be added to the context's
    /// list of fast interactions (if being used).  This allows for the fast
    /// interaction to be handled separated.
    fn adjusted_rate(&self, context: &Context<M>) -> Option<RateDensity> {
        // If the interaction was already deemed to be fast (from a previous
        // substep), bypass any computation and return None.
        if self.is_fast_check(context) {
            return None;
        }

        let mut rate = self.rate(context)?;

        if self.is_fast(&rate, context) {
            return None;
        }

        rate *= context.step_size;

        if self.particles().adjust_overshoot(
            &mut rate.symmetric,
            &mut rate.asymmetric,
            &context.n,
            &context.na,
            &context.eq,
            &context.eqn,
            context.in_equilibrium,
            context.no_asymmetry,
        ) {
            // log::trace!(
            //     "[{}.{:02}|{:>10.4e}] Adjusted Rate: {:<12.5e}|{:<12.5e}",
            //     c.step,
            //     c.substep,
            //     c.beta,
            //     rate.gamma / c.step_size,
            //     rate.symmetric / c.step_size
            // );
        }

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
    fn apply_change(
        &self,
        dn: &mut ArrayViewMut1<f64>,
        dna: &mut ArrayViewMut1<f64>,
        context: &Context<M>,
    ) {
        if let Some(rate) = self.adjusted_rate(context) {
            let particles = self.particles();

            for (&p, &(c, ca)) in &particles.particle_counts {
                if context.in_equilibrium.binary_search(&p).is_err() {
                    dn[p] += c * rate.symmetric;
                }
                if context.no_asymmetry.binary_search(&p).is_err() {
                    dna[p] += ca * rate.asymmetric;
                }
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

/// Computes the ratio `$a / b$` such that if `$a = 0$` the result is 0.
///
/// This is to be used in the context of calculating the scaling of the
/// interaction density by the number density ratios `$n / n^{(0)}$`.  Provided
/// the both inputs are numbers, then a NaN can only arise from `$n = eq = 0$`,
/// in which case the actual answer should really be 0 as there are no particles
/// to interaction.
///
/// Note that if `$b = 0$` and `$a \neq 0$`, then the result will be infinite.
#[must_use]
#[inline]
pub(crate) fn checked_div(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        0.0
    } else {
        a / b
    }
}

/// Computes the product `$a \cdot b$` such that if either is 0, the result is
/// 0.
///
/// This is used in the context of scaling `$\gamma$` by the prefactor of `$n /
/// n^{(0)}$`.  If either is exactly 0, the result is assumed to be 0 even if
/// the other is infinite.
///
/// If either inputs are NaN, then the result will be NaN irrespective of the
/// other.
#[must_use]
#[inline]
pub(crate) fn checked_mul(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == 0.0 || b == 0.0 {
        0.0
    } else {
        a * b
    }
}

/// Adjust `dn` and/or `dna` for to account for:
///
/// - Particle which are held in equilibrium, irrespective if any
///   interaction going on.
/// - Particles which have no asymmetry, irrespective if any interaction
///   going on.
pub fn fix_equilibrium<M>(context: &Context<M>, dn: &mut Array1<f64>, dna: &mut Array1<f64>) {
    for &p in context.in_equilibrium {
        dn[p] = context.eqn[p] - context.n[p];
    }
    for &p in context.no_asymmetry {
        dna[p] = -context.na[p];
    }
}

impl<I: ?Sized, M> Interaction<M> for &I
where
    I: Interaction<M>,
    M: Model,
{
    fn particles(&self) -> &Particles {
        (*self).particles()
    }

    fn width_enabled(&self) -> bool {
        (*self).width_enabled()
    }

    fn width(&self, context: &Context<M>) -> Option<PartialWidth> {
        (*self).width(context)
    }

    fn gamma_enabled(&self) -> bool {
        (*self).gamma_enabled()
    }

    fn gamma(&self, context: &Context<M>, real: bool) -> Option<f64> {
        (*self).gamma(context, real)
    }

    fn delta_gamma(&self, context: &Context<M>, real: bool) -> Option<f64> {
        (*self).delta_gamma(context, real)
    }
    fn rate(&self, context: &Context<M>) -> Option<RateDensity> {
        (*self).rate(context)
    }

    fn adjusted_rate(&self, context: &Context<M>) -> Option<RateDensity> {
        (*self).adjusted_rate(context)
    }

    fn apply_change(
        &self,
        dn: &mut ArrayViewMut1<f64>,
        dna: &mut ArrayViewMut1<f64>,
        context: &Context<M>,
    ) {
        (*self).apply_change(dn, dna, context);
    }
}

impl<I: ?Sized, M> Interaction<M> for Box<I>
where
    I: Interaction<M>,
    M: Model,
{
    fn particles(&self) -> &Particles {
        self.as_ref().particles()
    }

    fn width_enabled(&self) -> bool {
        self.as_ref().width_enabled()
    }

    fn width(&self, context: &Context<M>) -> Option<PartialWidth> {
        self.as_ref().width(context)
    }

    fn gamma_enabled(&self) -> bool {
        self.as_ref().gamma_enabled()
    }

    fn gamma(&self, context: &Context<M>, real: bool) -> Option<f64> {
        self.as_ref().gamma(context, real)
    }

    fn delta_gamma(&self, context: &Context<M>, real: bool) -> Option<f64> {
        self.as_ref().delta_gamma(context, real)
    }

    fn rate(&self, context: &Context<M>) -> Option<RateDensity> {
        self.as_ref().rate(context)
    }

    fn adjusted_rate(&self, context: &Context<M>) -> Option<RateDensity> {
        self.as_ref().adjusted_rate(context)
    }

    fn apply_change(
        &self,
        dn: &mut ArrayViewMut1<f64>,
        dna: &mut ArrayViewMut1<f64>,
        context: &Context<M>,
    ) {
        self.as_ref().apply_change(dn, dna, context);
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
