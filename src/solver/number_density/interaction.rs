use crate::solver::{number_density::Context, Model};
use ndarray::prelude::*;
use special_functions::bessel;
use std::{cmp, ops};

/// If an overshoot of the interaction rate is detected, the rate is adjusted such that `dn` satisfies:
///
/// ```
/// n + dn = (eq + (ALPHA - 1) * n) / ALPHA
/// ```
///
/// Large values means that the correction towards equilibrium are weaker, while
/// smaller values make the move towards equilibrium stronger.  Values of ALPHA
/// less than 1 will overshoot the equilibrium.
const ALPHA: f64 = 0.5;

/// Indicates how a particle participates in an interaction.
#[derive(Copy, Clone)]
pub enum Interacting {
    /// Initial state particles.  If the rate is positive, the number densities of these particles will be decreased.
    Initial(usize),
    /// Final state particles.  If the rate is positive, the number densities of these particles will be increased.
    Final(usize),
    /// Other particles.  This are `by-products` of the interaction and do not
    /// have a feedback effect on the overall interaction rate. If the
    /// interaction rate is \\(\gamma\\), this particle's number density will
    /// change according to \\(\Delta n = f \gamma\\).
    ///
    /// As an explicit example, asymmetry in \\(B-L\\) in generated as a
    /// by-product of the decay of heavy right-handed neutrinos and thus the
    /// abundance of heavy right-handed neutrinos affects the rate at which
    /// \\(B-L\\) asymmetry is changing, but the amount of \\(B-L\\) asymmetry
    /// does not affect the rate at which right-handed neutrinos decay.
    ///
    /// The three fields contain the multiplicative factors relating to:
    /// - Net rate:
    ///   \\begin{equation}
    ///     \left(\frac{n_{\vt i}}{n_{\vt i}^{(0)}} - \frac{n_{\vt f}}{n_{\vt f}^{(0)}}) \gamma
    ///   \\end{equation}
    /// - Forward rate:
    ///   \\begin{equation}
    ///     \frac{n_{\vt i}}{n_{\vt i}^{(0)}} \gamma
    ///   \\end{equation}
    /// - Backward rate (note there is *no* minus sign here):
    ///   \\begin{equation}
    ///     \frac{n_{\vt f}}{n_{\vt f}^{(0)}} \gamma
    ///   \\end{equation}
    /// - Original rate, \\(\gamma\\).
    Other(usize, f64, f64, f64, f64),
}

impl Interacting {
    /// Return the index of the particle.
    fn index(&self) -> usize {
        match *self {
            Interacting::Initial(i) => i,
            Interacting::Final(i) => i,
            Interacting::Other(i, ..) => i,
        }
    }

    /// Determine whether the particle is an initial particle or not.
    fn is_initial(&self) -> bool {
        match self {
            Interacting::Initial(..) => true,
            _ => false,
        }
    }

    /// Determine whether the particle is an initial particle or not.
    fn is_final(&self) -> bool {
        match self {
            Interacting::Final(..) => true,
            _ => false,
        }
    }

    /// Determine whether the particle is an initial particle or not.
    fn is_other(&self) -> bool {
        match self {
            Interacting::Other(..) => true,
            _ => false,
        }
    }
}

/// Equality is based *only* on the particle index.
impl cmp::PartialEq for Interacting {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Interacting::Initial(a), Interacting::Initial(b)) => a == b,
            (Interacting::Final(a), Interacting::Final(b)) => a == b,
            (Interacting::Other(a, ..), Interacting::Other(b, ..)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Interacting {}

/// Sorting is such that initial particles are first, Final particles are next,
/// and Other are all last.  Within each category, particles are sorted by their
/// index.
impl cmp::Ord for Interacting {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        match (self, other) {
            (Interacting::Initial(a), Interacting::Initial(b)) => a.cmp(b),
            (Interacting::Initial(_), _) => cmp::Ordering::Less,
            (Interacting::Final(_), Interacting::Initial(_)) => cmp::Ordering::Greater,
            (Interacting::Final(a), Interacting::Final(b)) => a.cmp(b),
            (Interacting::Final(_), Interacting::Other(..)) => cmp::Ordering::Less,
            (Interacting::Other(a, ..), Interacting::Other(b, ..)) => a.cmp(b),
            (Interacting::Other(..), _) => cmp::Ordering::Greater,
        }
    }
}

impl cmp::PartialOrd for Interacting {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Store an interaction.
///
/// The interaction consists of two elements: a list of particles involved and
/// the rate at which this reaction is going.
///
/// All interactions are always assumed to be of the form the general form
///
/// \\begin{equation}
///   \left( \frac{n_{\vt i}}{n_{\vt i}^{(0)}} - \frac{n_{\vt f}}{n_{\vt f}^{(0)}} \right) \gamma,
/// \\end{equation}
///
/// Where each \\(n_{\vt i}\\) and \\(n_{\vt f}\\) are initial and final state
/// particles.  As an explicit example, the interaction
///
/// ```ignore
/// Interaction::new([Initial(1), Final(2), Final(2)], gamma)
/// ```
///
/// is equivalent to the following interactions:
///
/// \\begin{equation}
/// \\begin{aligned}
///   \ddfrac{n_1}{t} &= -\left( \frac{n_1}{n_1^{(0)}} - \frac{n_2}{n_2^{(0)}} \frac{n_3}{n_3^{(0)}} \right) \gamma, \\\\
///   \ddfrac{n_2}{t} &= \left( \frac{n_1}{n_1^{(0)}} - \frac{n_2}{n_2^{(0)}} \frac{n_3}{n_3^{(0)}} \right) \gamma, \\\\
///   \ddfrac{n_3}{t} &= \left( \frac{n_1}{n_1^{(0)}} - \frac{n_2}{n_2^{(0)}} \frac{n_3}{n_3^{(0)}} \right) \gamma.
/// \\end{aligned}
/// \\end{equation}
pub struct Interaction {
    /// The list particles involved in the interaction, and how they are
    /// involved.
    ///
    /// This must always remain sorted.
    particles: Vec<Interacting>,
    /// The standard interaction rate for this interaction.
    rate: f64,
    /// The squared amplitude to be used in a two-body decay.
    decay: Option<f64>,
}

impl Interaction {
    /// Create a new interaction with the specified particles involved.
    pub fn new<I>(particles: I, rate: f64) -> Self
    where
        I: IntoIterator<Item = Interacting>,
    {
        let mut particles: Vec<_> = particles.into_iter().collect::<Vec<_>>();
        debug_assert!(
            !particles.is_empty(),
            "List of particles in interaction must be non-zero."
        );
        particles.sort_unstable();

        Interaction {
            particles,
            rate,
            decay: None,
        }
    }

    /// Specify the amplitude squared for a two-body decay.
    ///
    /// This option assumes that there is only *one* initial particle and that
    /// is decaying in a two-body decay.  By specifying this, we avoid computing
    /// \\(\gamma / n^{(0)}\\) as both numerator and denominator go to 0 very
    /// fast and fail to compute.
    ///
    /// The decay rate is computed using:
    ///
    /// \\begin{equation}
    /// \\begin{aligned}
    ///   \frac{n_a}{n_a^{(0)}} \gamma(a \to bc) &= n_a \frac{\abs{\mathcal M}^2}{16 \pi m_a} \frac{K_1(m_a \beta)}{K_2(m_b \beta)}, \\\\
    ///   \frac{Y_a}{Y_a^{(0)}} \gamma(a \to bc) &= Y_a \frac{\abs{\mathcal M}^2}{16 \pi m_a} \frac{K_1(m_a \beta)}{K_2(m_b \beta)}
    /// \\end{aligned}
    /// \\end{equation}
    ///
    /// and the inverse decay rate is adapted from that.
    pub fn decay(mut self, m2: f64) -> Self {
        self.decay = Some(m2);
        self
    }

    /// Calculate the common prefactor for initial and final state particles:
    ///
    /// \\begin{equation}
    ///   \prod_i \frac{n_i}{n_i^{(0)}} - \prod_f \frac{n_f}{n_f^{(0)}}.
    /// \\end{equation}
    ///
    /// If there are either no initial or final states, it is assumed that these
    /// number densities are in equilibrium, and thus the product is 1.
    ///
    /// The reaction rate density is adjusted internal by absorbing both
    /// numerators, and the functions returns the common prefactor such that the
    /// effective rate is `prefactor * self.rate`.
    fn calculate_rate<M: Model + Sync>(&mut self, n: &Array1<f64>, c: &Context<M>) -> (f64, f64) {
        let (mut rf, mut rb);

        if let Some(m2) = self.decay {
            // If we have a decay, use that to calculate the rate instead of the
            // more conventional way.

            let i = self.particles[0].index();
            let m = c.model.particles()[i].mass;

            // 0.0024230112251823004 = ζ(3) / 16 π³
            let gamma_tilde =
                0.0024230112251823004 * m2 * bessel::k1_on_k2(m * c.beta) / c.beta.powi(3) / m;
            log::trace!("̃γ: {:.3e}", gamma_tilde);

            rf = gamma_tilde;
            rb = gamma_tilde;
            for &p in &self.particles {
                match p {
                    Interacting::Initial(i) => {
                        rf *= n[i];
                        rb *= c.eq[i];
                    }
                    Interacting::Final(i) => {
                        rb *= n[i] / c.eq[i];
                    }
                    Interacting::Other(..) => break,
                }
            }
        } else {
            // Calculate the forward and backward rates the conventional way.
            rf = self.rate;
            rb = self.rate;

            for &p in &self.particles {
                match p {
                    Interacting::Initial(i) => {
                        rf *= n[i] / c.eq[i];
                    }
                    Interacting::Final(i) => {
                        rb *= n[i] / c.eq[i];
                    }
                    Interacting::Other(..) => break,
                }
            }
        }

        let mut net_rate = rf - rb;
        log::trace!(
            "Forward rate: {:.3e}, Backward rate: {:.3e}, Net: {:.3e}",
            rf,
            rb,
            net_rate
        );

        // If the rates are both zero, or either is nan, just return 0 .
        if (rf == 0.0 && rb == 0.0) || net_rate.is_nan() {
            return (0.0, 0.0);
        }

        // Adjust the rate so that it doesn't overshoot equilibrium for any of
        // the initial or final state particles.
        let check_overshoot = |i: usize, rate| {
            (n[i] > c.eq[i] && n[i] + rate < c.eq[i]) || (n[i] < c.eq[i] && n[i] + rate > c.eq[i])
        };
        for p in &mut self.particles {
            match p {
                &mut Interacting::Initial(i) => {
                    if check_overshoot(i, -net_rate) {
                        log::trace!(
                            "Scaling forward rate from {:.3e} to {:.3e}",
                            rf,
                            (n[i] - c.eq[i]) / ALPHA + rb
                        );
                        rf = (n[i] - c.eq[i]) / ALPHA + rb;
                        net_rate = rf - rb;
                    }
                }
                &mut Interacting::Final(i) => {
                    if check_overshoot(i, net_rate) {
                        log::trace!(
                            "Scaling backward rate from {:.3e} to {:.3e}",
                            rb,
                            (n[i] - c.eq[i]) / ALPHA + rf
                        );
                        rb = (n[i] - c.eq[i]) / ALPHA + rf;
                        net_rate = rf - rb;
                    }
                }
                Interacting::Other(i, net, forward, backward, custom) => {
                    let base = (*forward + *net) * rf + (*backward - *net) * rb;
                    if check_overshoot(*i, base + *custom * self.rate) {
                        log::trace!(
                            "Scaling custom rate from {:.3e} to {:.3e}",
                            custom,
                            ((c.eq[*i] - n[*i]) / ALPHA - base) / self.rate
                        );
                        *custom = ((c.eq[*i] - n[*i]) - base) / self.rate;
                    }
                }
            }
        }

        (rf, rb)
    }

    /// Add this interaction to the `dn` array.
    ///
    /// The changes in `dn` should contain only the effect from the integrated
    /// collision operator and not take into account the normalization to
    /// inverse-temperature evolution nor the dilation factor from expansion of
    /// the Universe.  These factors are handled separately and automatically.
    ///
    /// This function automatically adjusts the rate so that overshooting is
    /// avoided.
    pub fn dn<M: Model + Sync>(
        &mut self,
        mut dn: Array1<f64>,
        n: &Array1<f64>,
        c: &Context<M>,
    ) -> Array1<f64> {
        let (rf, rb) = self.calculate_rate(n, c);
        let net_rate = rf - rb;

        // Construct the array of `dn` for each particle.
        for &p in &self.particles {
            match p {
                Interacting::Initial(i) => dn[i] -= net_rate,
                Interacting::Final(i) => dn[i] += net_rate,
                Interacting::Other(i, net, forward, backward, custom) => {
                    dn[i] += (forward + net) * rf + (backward - net) * rb + custom * self.rate
                }
            }
        }

        dn
    }
}

impl<T> ops::Mul<T> for Interaction
where
    T: ops::Mul<f64, Output = f64> + Copy,
{
    type Output = Self;

    fn mul(mut self, other: T) -> Self {
        self.rate = other * self.rate;
        self.decay = self.decay.map(|factor| other * factor);
        self
    }
}

impl<T> ops::MulAssign<T> for Interaction
where
    T: ops::Mul<f64, Output = f64> + Copy,
{
    fn mul_assign(&mut self, other: T) {
        self.rate = other * self.rate;
        self.decay = self.decay.map(|factor| other * factor);
    }
}
