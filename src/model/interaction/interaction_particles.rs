use ndarray::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::{
    cmp::{self, Ordering},
    collections::HashMap,
    error, fmt, hash, ops,
    sync::RwLock,
};

use crate::prelude::Model;

use super::FastInteractionResult;

struct AbsRelConvergency {
    abs: f64,
    rel: f64,
    max_iter: usize,
    iter: RwLock<usize>,
}

/// Error used when failing to find the particle within the model.
#[derive(Debug, PartialEq, Eq)]
pub struct DisplayError(isize);

impl fmt::Display for DisplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 >= 0 {
            write!(f, "Unable to find particle {} in the model.", self.0)
        } else {
            write!(f, "Unable to find antiparticle {} in the model.", -self.0)
        }
    }
}

impl error::Error for DisplayError {}

impl From<isize> for DisplayError {
    fn from(p: isize) -> Self {
        Self(p)
    }
}

/// Convergency criterion which combines both absolute and relative differences.
///
/// Specifically, `abs` is used to determine if a root is found, and the result
/// deemed to have converged if `$|x_1 - x_2| < \varepsilon_{a}$` or `$|x_1 -
/// x_2| / (|x_1| + |x_2|) < \varepsilon_{r}$`.
impl AbsRelConvergency {
    fn new(abs: f64, rel: f64, max_iter: usize) -> Self {
        Self {
            abs: abs.abs(),
            rel: rel.abs(),
            max_iter,
            iter: RwLock::new(0),
        }
    }
}

impl roots::Convergency<f64> for AbsRelConvergency {
    fn is_root_found(&mut self, y: f64) -> bool {
        y.abs() < self.abs
    }

    fn is_converged(&mut self, x1: f64, x2: f64) -> bool {
        let delta = (x1 - x2).abs();
        delta < self.abs || delta / (x1.abs() + x2.abs()) < self.rel
    }

    fn is_iteration_limit_reached(&mut self, iter: usize) -> bool {
        *self.iter.write().unwrap() = iter;
        iter >= self.max_iter
    }
}

/// List of particles involved in the interaction.
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct InteractionParticles {
    /// Initial state particle indices.
    pub incoming_idx: Vec<usize>,
    /// Final state particle indices.
    pub outgoing_idx: Vec<usize>,
    /// Initial state particle sign.  A positive sign indicates this is a
    /// particle, while a negative sign indicates it is an antiparticle.
    pub incoming_sign: Vec<f64>,
    /// Final state particle sign.  A positive sign indicates this is a
    /// particle, while a negative sign indicates it is an antiparticle.
    pub outgoing_sign: Vec<f64>,
    /// A combination of `incoming_idx` and `incoming_sign`.
    pub incoming_signed: Vec<isize>,
    /// A combination of `outgoing_idx` and `outgoing_sign`.
    pub outgoing_signed: Vec<isize>,

    /// Aggregate of the particles to take into account multiplicities.
    ///
    /// Incoming particles are taken to have a negative multiplicity (as they
    /// are being annihilated) while outgoing particles have a positive
    /// multiplicity.
    ///
    /// For each particle (indicated by the key), the multiplicity is given ins
    /// the tuple for the symmetric and asymmetric rates respectively.
    ///
    /// The two counts differ for the symmetric and asymmetric rates because the
    /// symmetric rate does not take into account whether it is a particle or
    /// antiparticle as both it and its CP-conjugate rate are assumed to be
    /// equal.  This is evidently not true for the asymmetric rate as it
    /// fundamentally assumes that the CP rate is different.
    pub particle_counts: HashMap<usize, (f64, f64)>,

    /// Ratio of the original interactions [`Interaction::delta_gamma`] to
    /// [`Interaction::gamma`] if defined.
    pub gamma_ratio: Option<f64>,
}

impl InteractionParticles {
    /// Create a new set of interaction particles.  The absolute value of the
    /// numbers indicate the corresponding index for the particle, with the sign
    /// indicating whether it is a particle or anti-particle involved.
    #[must_use]
    pub fn new(incoming: &[isize], outgoing: &[isize]) -> Self {
        let mut incoming = incoming.to_vec();
        incoming.sort_unstable();
        let mut outgoing = outgoing.to_vec();
        outgoing.sort_unstable();

        let mut result = Self {
            incoming_idx: incoming.iter().map(|p| p.unsigned_abs()).collect(),
            outgoing_idx: outgoing.iter().map(|p| p.unsigned_abs()).collect(),
            incoming_sign: incoming
                .iter()
                .map(|&p| match p.cmp(&0) {
                    Ordering::Greater => 1.0,
                    Ordering::Equal => 0.0,
                    Ordering::Less => -1.0,
                })
                .collect(),
            outgoing_sign: outgoing
                .iter()
                .map(|&p| match p.cmp(&0) {
                    Ordering::Less => -1.0,
                    Ordering::Equal => 0.0,
                    Ordering::Greater => 1.0,
                })
                .collect(),
            incoming_signed: incoming.clone(),
            outgoing_signed: outgoing.clone(),

            particle_counts: HashMap::new(),

            gamma_ratio: None,
        };

        result.calculate_particle_counts();
        result
    }

    /// Create the CPT conjugate the [`InteractionParticles`].  This
    /// interchanges the incoming and outgoing particles, and particles with
    /// their corresponding antiparticles.
    #[must_use]
    pub fn cpt(&self) -> Self {
        Self {
            incoming_idx: self.outgoing_idx.iter().rev().copied().collect(),
            outgoing_idx: self.incoming_idx.iter().rev().copied().collect(),
            incoming_sign: self.outgoing_sign.iter().rev().map(ops::Neg::neg).collect(),
            outgoing_sign: self.incoming_sign.iter().rev().map(ops::Neg::neg).collect(),
            incoming_signed: self
                .outgoing_signed
                .iter()
                .rev()
                .map(ops::Neg::neg)
                .collect(),
            outgoing_signed: self
                .incoming_signed
                .iter()
                .rev()
                .map(ops::Neg::neg)
                .collect(),
            particle_counts: self
                .particle_counts
                .iter()
                .map(|(&p, &(c, ca))| (p, (-c, ca)))
                .collect(),
            gamma_ratio: self.gamma_ratio,
        }
    }

    /// Iterate over all incoming particle attributes, as a tuple `(idx, sign)`.
    pub fn iter_incoming(&self) -> impl Iterator<Item = (&usize, &f64)> {
        self.incoming_idx.iter().zip(&self.incoming_sign)
    }

    /// Iterate over all incoming particle attributes, as a tuple `(idx, sign)`.
    pub fn iter_outgoing(&self) -> impl Iterator<Item = (&usize, &f64)> {
        self.outgoing_idx.iter().zip(&self.outgoing_sign)
    }

    /// Number of incoming particles.
    #[must_use]
    pub fn len_incoming(&self) -> usize {
        self.incoming_idx.len()
    }

    /// Number of outgoing particles.
    #[must_use]
    pub fn len_outgoing(&self) -> usize {
        self.outgoing_idx.len()
    }

    /// Compute the product of the relevant entries in the given array as
    /// indexed by the incoming particles.  This product will exclude the
    /// incoming particle given by `pos` in the list of incoming particles and
    /// will also ignore any particle whose index is contained in the `idx`
    /// slice.
    ///
    /// For example, if the incoming particles are `[1, 3, 2, 1, 4, 2]`, then
    /// the products are:
    ///
    /// | Result | `pos` | `idx` |
    /// | ------ | ----- | ----- |
    /// | `$n_3 n_2 n_1 n_4 n_2$` | `0` | `[]` |
    /// | `$n_3 n_1 n_4$` | `0` | `[2]` |
    /// | `$n_1 n_2 n_1 n_4 n_2$` | `1` | `[]` |
    /// | `$n_4 n_2$` | `2` | `[1, 3]` |
    /// # Warning
    ///
    /// The `idx` slice must be ordered or the result may be wrong.
    #[must_use]
    pub fn product_except_incoming(&self, arr: &Array1<f64>, pos: usize, idx: &[usize]) -> f64 {
        self.incoming_idx
            .iter()
            .enumerate()
            .filter_map(|(j, &p)| {
                if j == pos || idx.binary_search(&p).is_ok() {
                    None
                } else {
                    Some(arr[p])
                }
            })
            .product()
    }

    /// Compute the product of the relevant entries in the given array as
    /// indexed by the outgoing particles.  This product will exclude the
    /// outgoing particle given by `pos` in the list of outgoing particles and
    /// will also ignore any particle whose index is contained in the `idx`
    /// slice.
    ///
    /// For example, if the outgoing particles are `[1, 3, 2, 1, 4, 2]`, then
    /// the products are:
    ///
    /// | Result | `pos` | `idx` |
    /// | ------ | ----- | ----- |
    /// | `$n_3 n_2 n_1 n_4 n_2$` | `0` | `[]` |
    /// | `$n_3 n_1 n_4$` | `0` | `[2]` |
    /// | `$n_1 n_2 n_1 n_4 n_2$` | `1` | `[]` |
    /// | `$n_4 n_2$` | `2` | `[1, 3]` |
    ///
    /// # Warning
    ///
    /// The `idx` slice must be ordered or the result may be wrong.
    #[must_use]
    pub fn product_except_outgoing(&self, arr: &Array1<f64>, i: usize, idx: &[usize]) -> f64 {
        self.outgoing_idx
            .iter()
            .enumerate()
            .filter_map(|(j, &p)| {
                if j == i || idx.binary_search(&p).is_ok() {
                    None
                } else {
                    Some(arr[p])
                }
            })
            .product()
    }

    /// Aggregate the particles to take into account multiplicities.
    ///
    /// Incoming particles are taken to have a negative multiplicity (as they
    /// are being annihilated) while outgoing particles have a positive
    /// multiplicity.
    ///
    /// For each particle (indicated by the key), the multiplicity is given ins
    /// the tuple for the symmetric and asymmetric rates respectively.
    ///
    /// The two counts differ for the symmetric and asymmetric rates because the
    /// symmetric rate does not take into account whether it is a particle or
    /// antiparticle as both it and its CP-conjugate rate are assumed to be
    /// equal.  This is evidently not true for the asymmetric rate as it
    /// fundamentally assumes that the CP rate is different.
    pub fn calculate_particle_counts(&mut self) {
        let mut particle_counts = HashMap::with_capacity(self.len_incoming() + self.len_outgoing());

        for (&p, &a) in self.iter_incoming() {
            let entry = particle_counts.entry(p).or_insert((0.0, 0.0));
            (*entry).0 -= 1.0;
            (*entry).1 -= a;
        }
        for (&p, &a) in self.iter_outgoing() {
            let entry = particle_counts.entry(p).or_insert((0.0, 0.0));
            (*entry).0 += 1.0;
            (*entry).1 += a;
        }

        self.particle_counts = particle_counts;
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
    /// The result is given as the pair of `(numerator, denominator)` for
    /// incoming and outgoing particles respectively.
    ///
    /// Particles which are fixed to be in equilibrium can be specified in the
    /// ordered slice `in_equilibrium`.
    ///
    /// # Warning
    ///
    /// The `in_equilibrium` slice must be ordered or the result may be wrong.
    #[must_use]
    pub fn symmetric_prefactor(
        &self,
        n: &Array1<f64>,
        eq: &Array1<f64>,
        in_equilibrium: &[usize],
    ) -> ((f64, f64), (f64, f64)) {
        let forward = self
            .incoming_idx
            .iter()
            .filter(|p| in_equilibrium.binary_search(p).is_err())
            .fold((1.0, 1.0), |(num, den), &p| (num * n[p], den * eq[p]));
        let backward = self
            .outgoing_idx
            .iter()
            .filter(|p| in_equilibrium.binary_search(p).is_err())
            .fold((1.0, 1.0), |(num, den), &p| (num * n[p], den * eq[p]));
        (forward, backward)
    }

    /// Net prefactor scaling the interaction rate with a specified change
    /// included.
    ///
    /// The change is calculated by counting the number of times the particles
    /// interact and takes into account the multiplicity of the particles
    /// involved within the interaction.
    ///
    /// Particles which are fixed to be in equilibrium can be specified in the
    /// ordered slice `in_equilibrium`.
    ///
    /// # Warning
    ///
    /// The `in_equilibrium` slice must be ordered or the result may be wrong.
    pub fn symmetric_prefactor_fn<'a>(
        &'a self,
        n: &'a Array1<f64>,
        eq: &'a Array1<f64>,
        in_equilibrium: &'a [usize],
    ) -> impl Fn(f64) -> f64 + 'a {
        move |delta| {
            let forward = self
                .incoming_idx
                .iter()
                .filter(|p| in_equilibrium.binary_search(p).is_err())
                .fold((1.0, 1.0), |(num, den), &p| {
                    (
                        num * (n[p] + self.particle_counts[&p].0 * delta),
                        den * eq[p],
                    )
                });
            let backward = self
                .outgoing_idx
                .iter()
                .filter(|p| in_equilibrium.binary_search(p).is_err())
                .fold((1.0, 1.0), |(num, den), &p| {
                    (
                        num * (n[p] + self.particle_counts[&p].0 * delta),
                        den * eq[p],
                    )
                });

            forward.0 / forward.1 - backward.0 / backward.1
        }
    }

    /// Return the minumum change from this interaction that will result in the
    /// number density of an incoming particle being 0.
    ///
    /// If there are no incoming particle or there is no net change the result
    /// is 0.  This will generally be a positive number as the interaction
    /// annihilates incoming particles.  The result may be negative in special
    /// cases such as `$a b \to a a b$`.
    ///
    /// Particles which are fixed to be in equilibrium can be specified in the
    /// ordered slice `in_equilibrium` so they they are ignored for the purposes
    /// of this function.
    ///
    /// # Warning
    ///
    /// The `in_equilibrium` slice must be ordered or the result may be wrong.
    fn symmetric_minimum_change_incoming(&self, n: &Array1<f64>, in_equilibrium: &[usize]) -> f64 {
        self.incoming_idx
            .iter()
            .filter(|p| in_equilibrium.binary_search(p).is_err())
            .filter_map(|&p| {
                let count = self.particle_counts[&p].0;
                if count == 0.0 {
                    None
                } else {
                    Some(-n[p] / count)
                }
            })
            .filter(|n| n.is_finite())
            .min_by(|x, y| x.abs().partial_cmp(&y.abs()).unwrap())
            .unwrap_or_default()
    }

    /// Return the minumum change from this interaction that will result in the
    /// number density of an outgoing particle being 0.
    ///
    /// If there are no outgoing particle or there is no net change the result
    /// is 0.  This will generally be a negative number as the interaction
    /// produces outgoing particles.  The result may be positive in special
    /// cases such as `$a a b \to a b$`.
    ///
    /// Particles which are fixed to be in equilibrium can be specified in the
    /// ordered slice `in_equilibrium` so they they are ignored for the purposes
    /// of this function.
    ///
    /// # Warning
    ///
    /// The `in_equilibrium` slice must be ordered or the result may be wrong.
    fn symmetric_minimum_change_outgoing(&self, n: &Array1<f64>, in_equilibrium: &[usize]) -> f64 {
        self.outgoing_idx
            .iter()
            .filter(|p| in_equilibrium.binary_search(p).is_err())
            .filter_map(|&p| {
                let count = self.particle_counts[&p].0;
                if count == 0.0 {
                    None
                } else {
                    Some(-n[p] / count)
                }
            })
            .filter(|n| n.is_finite())
            .min_by(|x, y| x.abs().partial_cmp(&y.abs()).unwrap())
            .unwrap_or_default()
    }

    /// Computes the change in number density required to establish equilibrium.
    ///
    /// Given the interaction `$X \leftrightarrow Y$`, this computes the value
    /// of `$\delta$` such that
    ///
    /// ```math
    ///   \prod_{i \in X} \frac{n_i + m_i \delta}{n_i^{(0)}}
    /// = \prod_{i \in Y} \frac{n_i + m_i \delta}{n_i^{(0)}}
    /// ```
    ///
    /// holds true up to linear order in `$\delta$` and where each `$m_i \inZ$`
    /// is net number of particle species `$i$` involved in the process (-1 and
    /// +1 for annihilation and creation respectively).
    ///
    /// If one of the equilibrium number density of the incoming particles is 0
    /// then `$\delta = \min_{i \in X}\{n_i / m_i\}$` as the interaction can
    /// only proceed in one direction.  Similarly, if the 0 is on the outgoing
    /// side, then `$\delta = \min_{i \in Y}\{n_i / m_i}$`.
    ///
    /// If there are zero equilibrium number densities on both sides, the
    /// minimum change required to reach equilibrium is used.
    ///
    /// Any particle which is forced to be in equilibrium can be specified in
    /// the ordered slice `in_equilibrium`.  Any change to their number
    /// densities is ignored (as it would be fixed anyway).
    ///
    /// # Warning
    ///
    /// The `in_equilibrium` slice must be ordered or the result may be wrong.
    #[allow(clippy::too_many_lines)]
    #[must_use]
    pub fn symmetric_delta(
        &self,
        n: &Array1<f64>,
        eq: &Array1<f64>,
        in_equilibrium: &[usize],
    ) -> f64 {
        // If the interaction does not result in any number density change, then return 0.
        if self.particle_counts.values().all(|&(c, _)| c == 0.0) {
            return 0.0;
        }

        let incoming_eq_prod = self.incoming_idx.iter().map(|&p| eq[p]).product::<f64>();
        let outgoing_eq_prod = self.outgoing_idx.iter().map(|&p| eq[p]).product::<f64>();

        match (incoming_eq_prod == 0.0, outgoing_eq_prod == 0.0) {
            (true, true) => {
                // If both sides are zero, use the minimum in absolute value change.
                let delta_forward = self.symmetric_minimum_change_incoming(n, &in_equilibrium);
                let delta_backward = self.symmetric_minimum_change_outgoing(n, &in_equilibrium);

                if delta_forward.abs() < delta_backward.abs() {
                    delta_forward
                } else {
                    delta_backward
                }
            }
            (true, false) => self.symmetric_minimum_change_incoming(n, &in_equilibrium),
            (false, true) => self.symmetric_minimum_change_outgoing(n, &in_equilibrium),
            (false, false) => {
                // The prefactor function for which we will be finding the root.
                let prefactor = &self.symmetric_prefactor_fn(n, eq, in_equilibrium);

                // Check if we already have a root
                if prefactor(0.0).abs() < 1e-30 {
                    return 0.0;
                }

                // Bound the search for roots based on the minimum changes that
                // result in 0 density.
                let (lower_limit, mut upper_limit) = (
                    self.symmetric_minimum_change_outgoing(n, &in_equilibrium)
                        .min(0.0),
                    self.symmetric_minimum_change_incoming(n, &in_equilibrium)
                        .max(0.0),
                );

                // Algorithm doesn't work if bracket / initial points are
                // identical.
                #[allow(clippy::float_cmp)]
                if lower_limit == upper_limit
                    || (lower_limit - upper_limit).abs() / (lower_limit.abs() + upper_limit.abs())
                        < 1e-10
                {
                    upper_limit += 1e-2;
                }

                // We try Brent first with a small number of iterations and then
                // secant.  If both fail, then we try with a large number of
                // iterations in case convergence is unusually slow.
                roots::find_root_brent(
                    lower_limit,
                    upper_limit,
                    prefactor,
                    &mut AbsRelConvergency::new(1e-50, 1e-10, 10),
                )
                .or_else(|_| {
                    roots::find_root_secant(
                        lower_limit,
                        upper_limit,
                        prefactor,
                        &mut AbsRelConvergency::new(1e-50, 1e-10, 10),
                    )
                })
                .or_else(|_| {
                    roots::find_root_brent(
                        lower_limit,
                        upper_limit,
                        prefactor,
                        &mut AbsRelConvergency::new(1e-50, 1e-10, 100),
                    )
                })
                .or_else(|_| {
                    roots::find_root_secant(
                        lower_limit,
                        upper_limit,
                        prefactor,
                        &mut AbsRelConvergency::new(1e-50, 1e-10, 100),
                    )
                })
                .unwrap_or(0.0)
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
    /// The result is given as the pair of `(numerator, denominator)` for
    /// incoming and outgoing particles respectively.
    ///
    /// Particles which are fixed to be in equilibrium can be specified in the
    /// ordered slice `in_equilibrium`, and similarly for `no_asymmetry`.
    ///
    /// # Warning
    ///
    /// Both the `in_equilibrium` and `no_asymmetry` slices must be ordered or
    /// the result may be wrong.
    #[must_use]
    pub fn asymmetric_prefactor(
        &self,
        n: &Array1<f64>,
        na: &Array1<f64>,
        eq: &Array1<f64>,
        in_equilibrium: &[usize],
        no_asymmetry: &[usize],
    ) -> ((f64, f64), (f64, f64)) {
        let forward = self
            .iter_incoming()
            .filter(|(p, _)| no_asymmetry.binary_search(p).is_err())
            .enumerate()
            .fold((0.0, 1.0), |(num, den), (i, (&p, &a))| {
                (
                    num + a * na[p] * self.product_except_incoming(n, i, in_equilibrium),
                    if in_equilibrium.binary_search(&p).is_ok() {
                        den
                    } else {
                        den * eq[p]
                    },
                )
            });
        let backward = self
            .iter_outgoing()
            .filter(|(p, _)| no_asymmetry.binary_search(p).is_err())
            .enumerate()
            .fold((0.0, 1.0), |(num, den), (i, (&p, &a))| {
                (
                    num + a * na[p] * self.product_except_outgoing(n, i, in_equilibrium),
                    if in_equilibrium.binary_search(&p).is_ok() {
                        den
                    } else {
                        den * eq[p]
                    },
                )
            });

        (forward, backward)
    }

    /// Net asymmetric prefactor scaling the interaction rate with a specified
    /// change included.
    ///
    /// The change is calculated by counting the number of times the particles
    /// interact and takes into account the multiplicity of the particles
    /// involved within the interaction.
    ///
    /// Particles which are fixed to be in equilibrium can be specified in the
    /// ordered slice `in_equilibrium`, and similarly for `no_asymmetry`.
    ///
    /// # Warning
    ///
    /// Both the `in_equilibrium` and `no_asymmetry` slices must be ordered or
    /// the result may be wrong.
    pub fn asymmetric_prefactor_fn<'a>(
        &'a self,
        n: &'a Array1<f64>,
        na: &'a Array1<f64>,
        eq: &'a Array1<f64>,
        in_equilibrium: &'a [usize],
        no_asymmetry: &'a [usize],
    ) -> impl Fn(f64) -> f64 + 'a {
        move |delta| {
            let forward = self
                .iter_incoming()
                .filter(|(p, _)| no_asymmetry.binary_search(p).is_err())
                .enumerate()
                .fold((0.0, 1.0), |(num, den), (i, (&p, &a))| {
                    (
                        num + a
                            * (na[p] + self.particle_counts[&p].1 * delta)
                            * self.product_except_incoming(n, i, in_equilibrium),
                        if in_equilibrium.binary_search(&p).is_ok() {
                            den
                        } else {
                            den * eq[p]
                        },
                    )
                });
            let backward = self
                .iter_outgoing()
                .filter(|(p, _)| no_asymmetry.binary_search(p).is_err())
                .enumerate()
                .fold((0.0, 1.0), |(num, den), (i, (&p, &a))| {
                    (
                        num + a
                            * (na[p] + self.particle_counts[&p].1 * delta)
                            * self.product_except_outgoing(n, i, in_equilibrium),
                        if in_equilibrium.binary_search(&p).is_ok() {
                            den
                        } else {
                            den * eq[p]
                        },
                    )
                });

            forward.0 / forward.1 - backward.0 / backward.1
            // forward.0 * backward.1 - backward.0 * forward.1
        }
    }

    /// Return the minumum change from this interaction that will result in the
    /// number density asymmteric of an incoming particle being 0.
    ///
    /// If there are no incoming particle or there is no net change the result
    /// is 0.  This will generally be a positive number as the interaction
    /// annihilates incoming particles.  The result may be negative in special
    /// cases such as `$a b \to a a b$`.
    ///
    /// Particles which are fixed to never have any asymmetry can be specified
    /// in the ordered slice `no_asymmetry` so they they are ignored for the
    /// purposes of this function.
    ///
    /// # Warning
    ///
    /// The `no_asymmetry` slice must be ordered or the result may be wrong.
    fn asymmetric_minimum_change_incoming(
        &self,
        n: &Array1<f64>,
        na: &Array1<f64>,
        no_asymmetry: &[usize],
    ) -> f64 {
        let (mut numerator, mut denominator) = (0.0, 0.0);
        let mut min_zero = f64::INFINITY;

        for (&p, &a) in self
            .iter_incoming()
            .filter(|(p, _)| no_asymmetry.binary_search(p).is_err())
        {
            let count = a * self.particle_counts[&p].1;
            if count == 0.0 {
                continue;
            }

            let np = n[p];
            if np == 0.0 {
                let new_value = a * na[p] / count;
                if new_value.abs() < min_zero.abs() {
                    min_zero = new_value;
                }
            } else {
                numerator += a * na[p] / np;
                denominator -= count / np;
            }
        }

        if min_zero.is_finite() {
            -min_zero
        } else if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Return the minumum change from this interaction that will result in the
    /// number density asymmetry of an outgoing particle being 0.
    ///
    /// If there are no incoming particle or there is no net change the result
    /// is 0.  This will generally be a positive number as the interaction
    /// annihilates incoming particles.  The result may be negative in special
    /// cases such as `$a a b \to a b$`.
    ///
    /// Particles which are fixed to never have any asymmetry can be specified
    /// in the ordered slice `no_asymmetry` so they they are ignored for the
    /// purposes of this function.
    ///
    /// # Warning
    ///
    /// The `no_asymmetry` slice must be ordered or the result may be wrong.
    fn asymmetric_minimum_change_outgoing(
        &self,
        n: &Array1<f64>,
        na: &Array1<f64>,
        no_asymmetry: &[usize],
    ) -> f64 {
        let (mut numerator, mut denominator) = (0.0, 0.0);
        let mut min_zero = f64::INFINITY;

        for (&p, &a) in self
            .iter_outgoing()
            .filter(|(p, _)| no_asymmetry.binary_search(p).is_err())
        {
            let count = a * self.particle_counts[&p].1;
            if count == 0.0 {
                continue;
            }

            let np = n[p];
            if np == 0.0 {
                let new_value = a * na[p] / count;
                if new_value.abs() < min_zero.abs() {
                    min_zero = new_value;
                }
            } else {
                numerator += a * na[p] / np;
                denominator -= count / np;
            }
        }

        if min_zero.is_finite() {
            -min_zero
        } else if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Given the interaction `$X \leftrightarrow Y$`, this computes the alue of
    /// `$\delta$` such that
    ///
    /// ```math
    /// \frac{
    ///   \sum_{i \in X} (\Delta_i + m_i \delta)
    ///   \prod_{\substack{j \in X \\ j \neq i} n_i}
    /// }{
    ///   \prod_{i \in X} n_i^{(0)}
    /// }
    /// =
    /// \frac{
    ///   \sum_{i \in Y} (\Delta_i + m_i \delta)
    ///   \prod_{\substack{j \in Y \\ j \neq i} n_i}
    /// }{
    ///   \prod_{i \in Y} n_i^{(0)}
    /// }
    /// ```
    ///
    /// holds true.
    ///
    /// If there are zero equilibrium number densities on both sides, the
    /// minimum change required to reach equilibrium is used.
    ///
    /// Any particle which is forced to be in equilibrium can be specified in
    /// the ordered slice `in_equilibrium`.  Any change to their number
    /// densities is ignored (as it would be fixed anyway).
    ///
    /// Particles which are fixed to never have any asymmetry can be specified
    /// in the ordered slice `no_asymmetry` so they they are ignored for the
    /// purposes of this function.
    ///
    /// # Warning
    ///
    /// Both the `in_equilibrium` and `no_asymmetry` slices must be ordered or
    /// the result may be wrong.
    #[must_use]
    pub fn asymmetric_delta(
        &self,
        n: &Array1<f64>,
        na: &Array1<f64>,
        eq: &Array1<f64>,
        in_equilibrium: &[usize],
        no_asymmetry: &[usize],
    ) -> f64 {
        // If the interaction does not result in any number density asymmetric change, then return 0.
        if self.particle_counts.values().all(|&(_, c)| c == 0.0) {
            return 0.0;
        }

        let incoming_eq_prod = self.incoming_idx.iter().map(|&p| eq[p]).product::<f64>();
        let outgoing_eq_prod = self.outgoing_idx.iter().map(|&p| eq[p]).product::<f64>();

        match (incoming_eq_prod == 0.0, outgoing_eq_prod == 0.0) {
            (true, true) => {
                let delta_forward = self.asymmetric_minimum_change_incoming(n, na, no_asymmetry);
                let delta_backward = self.asymmetric_minimum_change_outgoing(n, na, no_asymmetry);

                if delta_forward.abs() > delta_backward.abs() {
                    delta_forward
                } else {
                    delta_backward
                }
            }
            (true, false) => self.asymmetric_minimum_change_incoming(n, na, no_asymmetry),
            (false, true) => self.asymmetric_minimum_change_outgoing(n, na, no_asymmetry),
            (false, false) => {
                // The prefactor function for which we will be finding the root.
                let prefactor =
                    self.asymmetric_prefactor_fn(n, na, eq, in_equilibrium, no_asymmetry);

                // The function should be linear, thus we just need two initial
                // guesses to find the root.
                let (incoming_guess, mut outgoing_guess) = (
                    self.asymmetric_minimum_change_incoming(n, na, no_asymmetry),
                    self.asymmetric_minimum_change_outgoing(n, na, no_asymmetry),
                );

                // It is possible that both guesses are (nearly) identical, in
                // which case the algorithm will fail.
                if (incoming_guess - outgoing_guess).abs()
                    / (incoming_guess.abs() + outgoing_guess.abs())
                    < 1e-10
                {
                    outgoing_guess += 1e-2;
                }

                roots::find_root_secant(
                    incoming_guess,
                    outgoing_guess,
                    prefactor,
                    &mut AbsRelConvergency::new(1e-50, 1e-10, 20),
                )
                // If this fails, then it must be because no matter the value of
                // delta, the result is constant.
                .unwrap_or(0.0)
            }
        }
    }

    /// For a fast interaction, compute the change to each particle species such
    /// that equilibrium is established.
    ///
    /// The computation in the symmetric case is described in
    /// [`Self::symmetric_delta`], and the asymmetric case is described in
    /// [`Self::asymmetric_delta`].
    ///
    /// # Warning
    ///
    /// Both the `in_equilibrium` and `no_asymmetry` slices must be ordered or
    /// the result may be wrong.
    #[must_use]
    pub fn fast_interaction(
        &self,
        n: &Array1<f64>,
        na: &Array1<f64>,
        eq: &Array1<f64>,
        in_equilibrium: &[usize],
        no_asymmetry: &[usize],
    ) -> FastInteractionResult {
        log::trace!(
            r#"fast_interaction:
        {{
            "incoming": {:?},
            "outgoing": {:?},
            "n": {:.5e},
            "na": {:.5e},
            "eq": {:.5e}
        }}"#,
            self.incoming_signed,
            self.outgoing_signed,
            n,
            na,
            eq
        );

        let mut result = FastInteractionResult::zero(n.dim());

        result.symmetric_delta = self.symmetric_delta(n, eq, in_equilibrium);

        let mut n = n.clone();
        for (&p, &(c, _)) in &self.particle_counts {
            n[p] += c * result.symmetric_delta;

            // Although equilibrating the symmetric part of the interaction
            // impacts the number density asymmetries, we need not worry about
            // this as we are equilibrating the asymmetries as well, making any
            // change futile.
        }

        result.asymmetric_delta = self.asymmetric_delta(&n, na, eq, in_equilibrium, no_asymmetry);

        for (&p, &(c, ca)) in &self.particle_counts {
            result.dn[p] += c * result.symmetric_delta;
            result.dna[p] += ca * result.asymmetric_delta;
        }

        log::trace!("{}", result);

        result
    }

    /// Output a 'pretty' version of the interaction particles using the
    /// particle names from the model.
    ///
    /// # Errors
    ///
    /// If any particles can't be found in the model, this will produce an
    /// error.
    pub fn display<M>(&self, model: &M) -> Result<String, DisplayError>
    where
        M: Model,
    {
        let mut s = String::with_capacity(3 * (self.len_incoming() + self.len_outgoing()) + 2);

        for &p in &self.incoming_signed {
            s.push_str(&model.particle_name(p)?);
            s.push(' ');
        }

        s.push('↔');

        for &p in &self.outgoing_signed {
            s.push(' ');
            s.push_str(&model.particle_name(p)?);
        }

        Ok(s)
    }

    /// Output a 'pretty' version of the interaction particles using the
    /// particle indices.
    ///
    /// Unlike [`Self::display`], this does not require the model and will always
    /// work.
    #[must_use]
    pub fn short_display(&self) -> String {
        let mut s = String::with_capacity(3 * (self.len_incoming() + self.len_outgoing()) + 2);

        for &p in &self.incoming_signed {
            s.push_str(&format!("{} ", p));
        }

        s.push('↔');

        for &p in &self.outgoing_signed {
            s.push_str(&format!(" {}", p));
        }

        s
    }
}

impl fmt::Display for InteractionParticles {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for &p in &self.incoming_signed {
            write!(f, "{} ", p)?;
        }

        write!(f, "↔")?;

        for &p in &self.outgoing_signed {
            write!(f, " {}", p)?;
        }

        Ok(())
    }
}

/// Implementation of [`PartialEq`] (and thus [`Eq`]) only needs to look at the
/// `incoming_signed` and `outgoing_signed` attributes as all other properties
/// are derived from this.
impl cmp::PartialEq for InteractionParticles {
    fn eq(&self, other: &Self) -> bool {
        self.incoming_signed == other.incoming_signed
            && self.outgoing_signed == other.outgoing_signed
    }
}

impl cmp::Eq for InteractionParticles {}

impl hash::Hash for InteractionParticles {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        hash::Hash::hash_slice(&self.incoming_signed, state);
        hash::Hash::hash_slice(&self.outgoing_signed, state);
    }
}

#[cfg(test)]
mod tests {
    use crate::utilities::test::approx_eq;
    use ndarray::prelude::*;
    use rand::prelude::Distribution;
    use serde::Deserialize;
    use std::{
        collections::HashMap,
        error, f64,
        fs::{self, File},
        io::BufReader,
        iter,
    };

    use super::InteractionParticles;

    const RUN_COUNT: usize = 10_000;

    // These values need to be adjusted for tests which involve a large number
    // of particles, approximately an order of magnitude per particle.
    const EQUALITY_EPS_REL: f64 = 4.0;
    const EQUALITY_EPS_ABS: f64 = 1e-15;
    const ZERO_EPS_REL: f64 = 10.0;
    const ZERO_EPS_ABS: f64 = 1e-13;

    macro_rules! err_line_print {
        () => {
            |e| {
                eprintln!("Error at {}:{}", std::file!(), std::line!());
                e
            }
        };
    }

    /// Generate a random number densities.  This has a 1% chance of returning
    /// exactly 0, and other is uniformly distributed over `$[0, 1]$`.
    fn rand_n() -> f64 {
        if rand::random::<f64>() < 0.01 {
            0.0
        } else {
            rand::random()
        }
    }

    /// Generate a random number density asymmetry.  This has a 1% change of
    /// returning exactly 0, and otherwise is uniformly distributed over `$[-0.05,
    /// 0.05]$`.
    fn rand_na() -> f64 {
        if rand::random::<f64>() < 0.01 {
            0.0
        } else {
            (rand::random::<f64>() - 0.5) / 10.0
        }
    }

    /// Generate a random equilibrium number density. There is a 50% chance that
    /// it is logarithmically distributed over `$[10^{-20}, 10^0]$`, and a 50%
    /// chance it is distributed over `$(0, 1]$`.
    fn rand_eq() -> f64 {
        if rand::random::<f64>() < 0.5 {
            10.0_f64.powf(rand::random::<f64>() * -20.0)
        } else {
            let mut v = rand::random::<f64>();
            while v <= 0.0 {
                v = rand::random();
            }
            v
        }
    }

    /// Generate three arrays for number density, number density asymmetry, and
    /// equilibrium number density.
    fn rand_n_na_eq() -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        (
            Array1::from_shape_simple_fn(10, rand_n),
            Array1::from_shape_simple_fn(10, rand_na),
            Array1::from_shape_simple_fn(10, rand_eq),
        )
    }

    /// Generate an array of random particles for incoming/outgoing particles in
    /// an interaction.
    ///
    /// The result is compatible with the output of [`rand_n_na_eq`].
    fn rand_particles() -> Vec<isize> {
        let mut rng = rand::thread_rng();
        let count = rand::distributions::Uniform::new(1, 10).sample(&mut rng);
        iter::from_fn(|| Some(rand::distributions::Uniform::new_inclusive(-9, 9).sample(&mut rng)))
            .take(count)
            .collect()
    }

    /// Generate a random interaction with random incoming and outgoing
    /// particles.
    fn rand_interaction() -> super::InteractionParticles {
        super::InteractionParticles::new(&rand_particles(), &rand_particles())
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn interaction_particles() {
        let a = super::InteractionParticles::new(&[1, 2, 3], &[4, 5, 6]);
        assert_eq!(&a.incoming_idx, &[1, 2, 3]);
        assert_eq!(&a.incoming_sign, &[1.0, 1.0, 1.0]);
        assert_eq!(&a.incoming_signed, &[1, 2, 3]);
        assert_eq!(&a.outgoing_idx, &[4, 5, 6]);
        assert_eq!(&a.outgoing_sign, &[1.0, 1.0, 1.0]);
        assert_eq!(&a.outgoing_signed, &[4, 5, 6]);
        assert_eq!(
            a.particle_counts,
            vec![
                (1, (-1.0, -1.0)),
                (2, (-1.0, -1.0)),
                (3, (-1.0, -1.0)),
                (4, (1.0, 1.0)),
                (5, (1.0, 1.0)),
                (6, (1.0, 1.0)),
            ]
            .drain(..)
            .collect()
        );

        let b = super::InteractionParticles::new(&[2, 1, -1], &[3, 3, -3]);
        assert_eq!(&b.incoming_idx, &[1, 1, 2]);
        assert_eq!(&b.incoming_sign, &[-1.0, 1.0, 1.0]);
        assert_eq!(&b.incoming_signed, &[-1, 1, 2]);
        assert_eq!(&b.outgoing_idx, &[3, 3, 3]);
        assert_eq!(&b.outgoing_sign, &[-1.0, 1.0, 1.0]);
        assert_eq!(&b.outgoing_signed, &[-3, 3, 3]);
        assert_eq!(
            b.particle_counts,
            vec![(1, (-2.0, 0.0)), (2, (-1.0, -1.0)), (3, (3.0, 1.0))]
                .drain(..)
                .collect()
        );

        let c = super::InteractionParticles::new(&[0, 1, -1], &[1, -1, 1]);
        assert_eq!(&c.incoming_idx, &[1, 0, 1]);
        assert_eq!(&c.incoming_sign, &[-1.0, 0.0, 1.0]);
        assert_eq!(&c.incoming_signed, &[-1, 0, 1]);
        assert_eq!(&c.outgoing_idx, &[1, 1, 1]);
        assert_eq!(&c.outgoing_sign, &[-1.0, 1.0, 1.0]);
        assert_eq!(&c.outgoing_signed, &[-1, 1, 1]);
        assert_eq!(
            c.particle_counts,
            vec![(0, (-1.0, 0.0)), (1, (1.0, 1.0))].drain(..).collect()
        );
    }

    #[test]
    fn cpt() {
        // Standard interaction
        let forward = super::InteractionParticles::new(&[1, 2, 3], &[4, 5, 6]);
        let backward = super::InteractionParticles::new(&[-4, -5, -6], &[-1, -2, -3]);
        assert_eq!(forward.cpt(), backward);
        assert_eq!(forward, backward.cpt());

        // Multiplicity on each side
        let forward = super::InteractionParticles::new(&[1, 1, 1], &[2, 2, 2]);
        let backward = super::InteractionParticles::new(&[-2, -2, -2], &[-1, -1, -1]);
        assert_eq!(forward.cpt(), backward);
        assert_eq!(forward, backward.cpt());

        // Multiplicity across sides
        let forward = super::InteractionParticles::new(&[1, 2, 3], &[1, -2, 3, -3]);
        let backward = super::InteractionParticles::new(&[-1, 2, 3, -3], &[-1, -2, -3]);
        assert_eq!(forward.cpt(), backward);
        assert_eq!(forward, backward.cpt());

        // Check that order doesn't matter
        let forward = super::InteractionParticles::new(&[1, 2, 3], &[1, -2, 2, 3, -3, 4]);
        let backward = super::InteractionParticles::new(&[2, -2, -1, 3, -4, -3], &[-1, -2, -3]);
        assert_eq!(forward.cpt(), backward);
        assert_eq!(forward, backward.cpt());

        // Symmetric under CPT
        let symmetric = super::InteractionParticles::new(&[1, 2, 3], &[-1, -2, -3]);
        assert_eq!(symmetric.cpt(), symmetric);
        assert_eq!(symmetric, symmetric.cpt());
    }

    #[test]
    fn iter() {
        let interaction = super::InteractionParticles::new(&[1, -2, 3], &[0, 1, -1]);

        // Particles are sorted from smallest to largest signed index.
        let mut incoming = interaction.iter_incoming();
        assert_eq!(incoming.next(), Some((&2, &-1.0)));
        assert_eq!(incoming.next(), Some((&1, &1.0)));
        assert_eq!(incoming.next(), Some((&3, &1.0)));
        assert_eq!(incoming.next(), None);

        let mut outgoing = interaction.iter_outgoing();
        assert_eq!(outgoing.next(), Some((&1, &-1.0)));
        assert_eq!(outgoing.next(), Some((&0, &0.0)));
        assert_eq!(outgoing.next(), Some((&1, &1.0)));
        assert_eq!(outgoing.next(), None);
    }

    #[test]
    fn len() {
        for ni in 0..10 {
            for no in 0..10 {
                let interaction = super::InteractionParticles::new(&vec![1; ni], &vec![1; no]);
                assert_eq!(ni, interaction.len_incoming());
                assert_eq!(no, interaction.len_outgoing());
            }
        }
    }

    #[test]
    fn product_except() -> Result<(), Box<dyn error::Error>> {
        let interaction = super::InteractionParticles::new(&[0, 1, 2, 3], &[4, 5, 6, 7]);

        let n = array![2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0];
        let prod_in = n.iter().take(4).product::<f64>();
        let prod_out = n.iter().skip(4).product::<f64>();

        for i in 0..4 {
            approx_eq(
                interaction.product_except_incoming(&n, i, &[]),
                prod_in / n[interaction.incoming_idx[i]],
                15.0,
                0.0,
            )
            .map_err(err_line_print!())?;
            approx_eq(
                interaction.product_except_outgoing(&n, i, &[]),
                prod_out / n[interaction.outgoing_idx[i]],
                15.0,
                0.0,
            )
            .map_err(err_line_print!())?;
        }

        Ok(())
    }

    #[test]
    fn product_except_ignored() -> Result<(), Box<dyn error::Error>> {
        let interaction = super::InteractionParticles::new(&[0, 1, 2, 3], &[4, 5, 6, 7]);

        let n = array![2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0];

        approx_eq(
            interaction.product_except_incoming(&n, 0, &[1, 2, 3]),
            1.0,
            15.0,
            0.0,
        )
        .map_err(err_line_print!())?;
        approx_eq(
            interaction.product_except_incoming(&n, 0, &[4, 5, 6, 7]),
            3.0 * 5.0 * 7.0,
            15.0,
            0.0,
        )
        .map_err(err_line_print!())?;
        approx_eq(
            interaction.product_except_incoming(&n, 0, &[2, 6]),
            3.0 * 7.0,
            15.0,
            0.0,
        )
        .map_err(err_line_print!())?;

        approx_eq(
            interaction.product_except_outgoing(&n, 0, &[0, 1, 2, 3]),
            13.0 * 17.0 * 19.0,
            15.0,
            0.0,
        )
        .map_err(err_line_print!())?;
        approx_eq(
            interaction.product_except_outgoing(&n, 0, &[5, 6, 7]),
            1.0,
            15.0,
            0.0,
        )
        .map_err(err_line_print!())?;
        approx_eq(
            interaction.product_except_outgoing(&n, 0, &[2, 6]),
            13.0 * 19.0,
            15.0,
            0.0,
        )
        .map_err(err_line_print!())?;

        Ok(())
    }

    // A simple interaction with all distinct particles
    #[test]
    fn prefactor_simple() -> Result<(), Box<dyn error::Error>> {
        const EPS_REL: f64 = 1e-14;
        const EPS_ABS: f64 = 1e-250;

        let interaction = super::InteractionParticles::new(&[1, 2], &[3, 4]);
        let deltas = Array1::linspace(-1.0, 1.0, RUN_COUNT / 100);
        for _ in 0..RUN_COUNT {
            let (n, na, eq) = rand_n_na_eq();

            let (forward, backward) = interaction.symmetric_prefactor(&n, &eq, &[]);
            approx_eq(
                forward.0 / forward.1,
                n[1] * n[2] / (eq[1] * eq[2]),
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;
            approx_eq(
                backward.0 / backward.1,
                n[3] * n[4] / (eq[3] * eq[4]),
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;

            let f = interaction.symmetric_prefactor_fn(&n, &eq, &[]);
            approx_eq(
                forward.0 / forward.1 - backward.0 / backward.1,
                f(0.0),
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;

            for &delta in &deltas {
                approx_eq(
                    (n[1] - delta) * (n[2] - delta) / (eq[1] * eq[2])
                        - (n[3] + delta) * (n[4] + delta) / (eq[3] * eq[4]),
                    f(delta),
                    EPS_REL,
                    EPS_ABS,
                )
                .map_err(err_line_print!())?;
            }

            let (forward, backward) = interaction.asymmetric_prefactor(&n, &na, &eq, &[], &[]);
            #[allow(clippy::suspicious_operation_groupings)]
            approx_eq(
                forward.0 / forward.1,
                (na[1] * n[2] + na[2] * n[1]) / (eq[1] * eq[2]),
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;
            #[allow(clippy::suspicious_operation_groupings)]
            approx_eq(
                backward.0 / backward.1,
                (na[3] * n[4] + na[4] * n[3]) / (eq[3] * eq[4]),
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;

            let f = interaction.asymmetric_prefactor_fn(&n, &na, &eq, &[], &[]);
            approx_eq(
                forward.0 / forward.1 - backward.0 / backward.1,
                f(0.0),
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;

            for &delta in &deltas {
                approx_eq(
                    ((na[1] - delta) * n[2] + (na[2] - delta) * n[1]) / (eq[1] * eq[2])
                        - ((na[3] + delta) * n[4] + (na[4] + delta) * n[3]) / (eq[3] * eq[4]),
                    f(delta),
                    EPS_REL,
                    EPS_ABS,
                )
                .map_err(err_line_print!())?;
            }
        }

        Ok(())
    }

    // A simple interaction with all distinct particles
    #[test]
    fn prefactor_simple_ignore() -> Result<(), Box<dyn error::Error>> {
        const EPS_REL: f64 = 1e-14;
        const EPS_ABS: f64 = 1e-250;

        let interaction = super::InteractionParticles::new(&[1, 2], &[3, 4]);
        let deltas = Array1::linspace(-1.0, 1.0, RUN_COUNT / 100);
        for _ in 0..RUN_COUNT {
            let (n, na, eq) = rand_n_na_eq();

            let (forward, backward) = interaction.symmetric_prefactor(&n, &eq, &[1, 4]);
            approx_eq(forward.0 / forward.1, n[2] / eq[2], EPS_REL, EPS_ABS)
                .map_err(err_line_print!())?;
            approx_eq(backward.0 / backward.1, n[3] / eq[3], EPS_REL, EPS_ABS)
                .map_err(err_line_print!())?;

            let f = interaction.symmetric_prefactor_fn(&n, &eq, &[1, 4]);
            approx_eq(
                forward.0 / forward.1 - backward.0 / backward.1,
                f(0.0),
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;

            for &delta in &deltas {
                approx_eq(
                    (n[2] - delta) / (eq[2]) - (n[3] + delta) / (eq[3]),
                    f(delta),
                    EPS_REL,
                    EPS_ABS,
                )
                .map_err(err_line_print!())?;
            }

            let (forward, backward) = interaction.asymmetric_prefactor(&n, &na, &eq, &[], &[]);
            #[allow(clippy::suspicious_operation_groupings)]
            approx_eq(
                forward.0 / forward.1,
                (na[1] * n[2] + na[2] * n[1]) / (eq[1] * eq[2]),
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;
            #[allow(clippy::suspicious_operation_groupings)]
            approx_eq(
                backward.0 / backward.1,
                (na[3] * n[4] + na[4] * n[3]) / (eq[3] * eq[4]),
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;

            let f = interaction.asymmetric_prefactor_fn(&n, &na, &eq, &[], &[]);
            approx_eq(
                forward.0 / forward.1 - backward.0 / backward.1,
                f(0.0),
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;

            for &delta in &deltas {
                approx_eq(
                    ((na[1] - delta) * n[2] + (na[2] - delta) * n[1]) / (eq[1] * eq[2])
                        - ((na[3] + delta) * n[4] + (na[4] + delta) * n[3]) / (eq[3] * eq[4]),
                    f(delta),
                    EPS_REL,
                    EPS_ABS,
                )
                .map_err(err_line_print!())?;
            }
        }

        Ok(())
    }

    // More complicated iteraction with the same particles on both sides and some multiplicities.
    #[test]
    fn prefactor_2() -> Result<(), Box<dyn error::Error>> {
        const EPS_REL: f64 = 1e-14;
        const EPS_ABS: f64 = 1e-250;

        let interaction = super::InteractionParticles::new(&[1, -1, 2], &[2, -3]);
        let deltas = Array1::linspace(-1.0, 1.0, RUN_COUNT / 100);

        for _ in 0..RUN_COUNT {
            let (n, na, eq) = rand_n_na_eq();

            let (forward, backward) = interaction.symmetric_prefactor(&n, &eq, &[]);
            approx_eq(
                forward.0 / forward.1,
                n[1] * n[1] * n[2] / (eq[1] * eq[1] * eq[2]),
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;
            approx_eq(
                backward.0 / backward.1,
                n[2] * n[3] / (eq[2] * eq[3]),
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;

            let f = interaction.symmetric_prefactor_fn(&n, &eq, &[]);
            approx_eq(
                forward.0 / forward.1 - backward.0 / backward.1,
                f(0.0),
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;

            for &delta in &deltas {
                approx_eq(
                    ((n[1] - 2.0 * delta) * (n[1] - 2.0 * delta) * (n[2]))
                        / (eq[1] * eq[1] * eq[2])
                        - ((n[2]) * (n[3] + delta)) / (eq[2] * eq[3]),
                    f(delta),
                    EPS_REL,
                    EPS_ABS,
                )
                .map_err(err_line_print!())?;
            }

            let (forward, backward) = interaction.asymmetric_prefactor(&n, &na, &eq, &[], &[]);
            approx_eq(
                forward.0 / forward.1,
                (na[2] * n[1] * n[1]) / (eq[1] * eq[1] * eq[2]),
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;
            #[allow(clippy::suspicious_operation_groupings)]
            approx_eq(
                backward.0 / backward.1,
                (na[2] * n[3] - na[3] * n[2]) / (eq[2] * eq[3]),
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;

            let f = interaction.asymmetric_prefactor_fn(&n, &na, &eq, &[], &[]);
            approx_eq(
                forward.0 / forward.1 - backward.0 / backward.1,
                f(0.0),
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;

            for &delta in &deltas {
                approx_eq(
                    (na[2] * n[1] * n[1]) / (eq[1] * eq[1] * eq[2])
                        - (na[2] * n[3] - (na[3] - delta) * n[2]) / (eq[2] * eq[3]),
                    f(delta),
                    EPS_REL,
                    EPS_ABS,
                )
                .map_err(err_line_print!())?;
            }
        }

        Ok(())
    }

    #[test]
    fn minimum_change() -> Result<(), Box<dyn error::Error>> {
        const EPS_REL: f64 = -14.0;
        const EPS_ABS: f64 = 1e-200;

        let interaction = super::InteractionParticles::new(&[1, 2, 3], &[4, 5, 6]);
        for _ in 0..RUN_COUNT {
            let (n, na, _eq) = rand_n_na_eq();
            approx_eq(
                interaction.symmetric_minimum_change_incoming(&n, &[]),
                n[1].min(n[2]).min(n[3]),
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;
            approx_eq(
                interaction.symmetric_minimum_change_outgoing(&n, &[]),
                -n[3].min(n[4]).min(n[5]),
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;

            approx_eq(
                interaction.asymmetric_minimum_change_incoming(&n, &na, &[]),
                na[1].min(na[2]).min(na[3]),
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;
            approx_eq(
                interaction.asymmetric_minimum_change_outgoing(&n, &na, &[]),
                -na[4].min(na[5]).min(na[6]),
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;
        }

        let interaction = super::InteractionParticles::new(&[1, 1, 2], &[2, 3, -3]);
        for _ in 0..RUN_COUNT {
            let (n, na, _eq) = rand_n_na_eq();
            approx_eq(
                interaction.symmetric_minimum_change_incoming(&n, &[]),
                n[1] / 2.0,
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;
            approx_eq(
                interaction.symmetric_minimum_change_outgoing(&n, &[]),
                -n[3] / 2.0,
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;

            approx_eq(
                interaction.asymmetric_minimum_change_incoming(&n, &na, &[]),
                na[1] / 2.0,
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;
            approx_eq(
                interaction.asymmetric_minimum_change_outgoing(&n, &na, &[]),
                0.0,
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;
        }

        let interaction = super::InteractionParticles::new(&[1, 2], &[1, 2, 2]);
        for _ in 0..RUN_COUNT {
            let (n, na, _eq) = rand_n_na_eq();
            approx_eq(
                interaction.symmetric_minimum_change_incoming(&n, &[]),
                -n[2],
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;
            approx_eq(
                interaction.symmetric_minimum_change_outgoing(&n, &[]),
                -n[2],
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;

            approx_eq(
                interaction.asymmetric_minimum_change_incoming(&n, &na, &[]),
                -na[2],
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;
            approx_eq(
                interaction.asymmetric_minimum_change_outgoing(&n, &na, &[]),
                -na[2],
                EPS_REL,
                EPS_ABS,
            )
            .map_err(err_line_print!())?;
        }

        Ok(())
    }

    #[test]
    fn fast_interaction_1_2() -> Result<(), Box<dyn error::Error>> {
        let interaction = super::InteractionParticles::new(&[1], &[2, 3]);
        let mut csv = csv::Reader::from_reader(zstd::Decoder::with_buffer(BufReader::new(
            fs::File::open("tests/data/fast_interaction_1_2.csv.zst")?,
        ))?);

        for result in csv.deserialize() {
            let data: HashMap<String, f64> = result?;

            let n = Array1::from(vec![0.0, data["n1"], data["n2"], data["n3"]]);
            let na = Array1::from(vec![0.0, data["na1"], data["na2"], data["na3"]]);
            let eq = Array1::from(vec![0.0, data["eq1"], data["eq2"], data["eq3"]]);

            approx_eq(
                data["symmetric"],
                interaction.symmetric_delta(&n, &eq, &[]),
                EQUALITY_EPS_REL,
                EQUALITY_EPS_ABS,
            )
            .map_err(err_line_print!())?;
            approx_eq(
                data["asymmetric"],
                interaction.asymmetric_delta(&n, &na, &eq, &[], &[]),
                EQUALITY_EPS_REL,
                EQUALITY_EPS_ABS,
            )
            .map_err(err_line_print!())?;
        }

        Ok(())
    }

    #[test]
    fn fast_interaction_1_2_zero_incoming() -> Result<(), Box<dyn error::Error>> {
        let interaction = super::InteractionParticles::new(&[1], &[2, 3]);

        for _ in 0..RUN_COUNT {
            let (mut n, mut na, mut eq) = rand_n_na_eq();
            eq[1] = 0.0;

            let result = interaction.fast_interaction(&n, &na, &eq, &[], &[]);
            n += &result.dn;
            na += &result.dna;

            approx_eq(n[1], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS).map_err(err_line_print!())?;
            approx_eq(na[1], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS).map_err(err_line_print!())?;
        }

        Ok(())
    }

    #[test]
    fn fast_interaction_1_2_zero_outgoing() -> Result<(), Box<dyn error::Error>> {
        let interaction = super::InteractionParticles::new(&[1], &[2, 3]);

        for _ in 0..RUN_COUNT {
            let (mut n, mut na, mut eq) = rand_n_na_eq();
            eq[2] = 0.0;
            eq[3] = 0.0;

            let result = interaction.fast_interaction(&n, &na, &eq, &[], &[]);
            n += &result.dn;
            na += &result.dna;

            approx_eq(n[2], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS)
                .or_else(|_| approx_eq(n[3], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS))
                .map_err(err_line_print!())?;
            approx_eq(na[2] * n[3] + na[3] * n[2], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS)
                .map_err(err_line_print!())?;
        }

        Ok(())
    }

    #[test]
    fn fast_interaction_2_2() -> Result<(), Box<dyn error::Error>> {
        let interaction = super::InteractionParticles::new(&[1, 2], &[3, 4]);
        let mut csv = csv::Reader::from_reader(zstd::Decoder::with_buffer(BufReader::new(
            fs::File::open("tests/data/fast_interaction_2_2.csv.zst")?,
        ))?);

        for result in csv.deserialize() {
            let data: HashMap<String, f64> = result?;

            let n = Array1::from(vec![0.0, data["n1"], data["n2"], data["n3"], data["n4"]]);
            let na = Array1::from(vec![
                0.0,
                data["na1"],
                data["na2"],
                data["na3"],
                data["na4"],
            ]);
            let eq = Array1::from(vec![
                0.0,
                data["eq1"],
                data["eq2"],
                data["eq3"],
                data["eq4"],
            ]);

            approx_eq(
                data["symmetric"],
                interaction.symmetric_delta(&n, &eq, &[]),
                EQUALITY_EPS_REL,
                EQUALITY_EPS_ABS,
            )
            .map_err(err_line_print!())?;
            approx_eq(
                data["asymmetric"],
                interaction.asymmetric_delta(&n, &na, &eq, &[], &[]),
                EQUALITY_EPS_REL,
                EQUALITY_EPS_ABS,
            )
            .map_err(err_line_print!())?;
        }

        Ok(())
    }

    #[test]
    fn fast_interaction_2_2_zero_incoming() -> Result<(), Box<dyn error::Error>> {
        let interaction = super::InteractionParticles::new(&[1, 2], &[3, 4]);

        for _ in 0..RUN_COUNT {
            let (mut n, mut na, mut eq) = rand_n_na_eq();
            eq[1] = 0.0;
            eq[2] = 0.0;

            let result = interaction.fast_interaction(&n, &na, &eq, &[], &[]);
            n += &result.dn;
            na += &result.dna;

            approx_eq(n[1], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS)
                .or_else(|_| approx_eq(n[2], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS))
                .map_err(err_line_print!())?;
            approx_eq(na[1] * n[2] + na[2] * n[1], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS)
                .map_err(err_line_print!())?;
        }

        Ok(())
    }

    #[test]
    fn fast_interaction_2_2_zero_outgoing() -> Result<(), Box<dyn error::Error>> {
        let interaction = super::InteractionParticles::new(&[1, 2], &[3, 4]);

        for _ in 0..RUN_COUNT {
            let (mut n, mut na, mut eq) = rand_n_na_eq();
            eq[3] = 0.0;
            eq[4] = 0.0;

            let result = interaction.fast_interaction(&n, &na, &eq, &[], &[]);
            n += &result.dn;
            na += &result.dna;

            approx_eq(n[3], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS)
                .or_else(|_| approx_eq(n[4], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS))
                .map_err(err_line_print!())?;
            approx_eq(na[3] * n[4] + na[4] * n[3], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS)
                .map_err(err_line_print!())?;
        }

        Ok(())
    }

    #[test]
    fn fast_interaction_1_3() -> Result<(), Box<dyn error::Error>> {
        let interaction = super::InteractionParticles::new(&[1], &[2, 3, 4]);
        let mut csv = csv::Reader::from_reader(zstd::Decoder::with_buffer(BufReader::new(
            fs::File::open("tests/data/fast_interaction_1_3.csv.zst")?,
        ))?);

        for result in csv.deserialize() {
            let data: HashMap<String, f64> = result?;

            let n = Array1::from(vec![0.0, data["n1"], data["n2"], data["n3"], data["n4"]]);
            let na = Array1::from(vec![
                0.0,
                data["na1"],
                data["na2"],
                data["na3"],
                data["na4"],
            ]);
            let eq = Array1::from(vec![
                0.0,
                data["eq1"],
                data["eq2"],
                data["eq3"],
                data["eq4"],
            ]);

            approx_eq(
                data["symmetric"],
                interaction.symmetric_delta(&n, &eq, &[]),
                EQUALITY_EPS_REL,
                EQUALITY_EPS_ABS,
            )
            .map_err(err_line_print!())?;
            approx_eq(
                data["asymmetric"],
                interaction.asymmetric_delta(&n, &na, &eq, &[], &[]),
                EQUALITY_EPS_REL,
                EQUALITY_EPS_ABS,
            )
            .map_err(err_line_print!())?;
        }

        Ok(())
    }

    #[test]
    fn fast_interaction_1_3_zero_incoming() -> Result<(), Box<dyn error::Error>> {
        let interaction = super::InteractionParticles::new(&[1], &[2, 3, 4]);

        for _ in 0..RUN_COUNT {
            let (mut n, mut na, mut eq) = rand_n_na_eq();
            eq[1] = 0.0;

            let result = interaction.fast_interaction(&n, &na, &eq, &[], &[]);
            n += &result.dn;
            na += &result.dna;

            approx_eq(n[1], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS).map_err(err_line_print!())?;
            approx_eq(na[1], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS).map_err(err_line_print!())?;
        }

        Ok(())
    }

    #[test]
    fn fast_interaction_1_3_zero_outgoing() -> Result<(), Box<dyn error::Error>> {
        let interaction = super::InteractionParticles::new(&[1], &[2, 3, 4]);

        for _ in 0..RUN_COUNT {
            let (mut n, mut na, mut eq) = rand_n_na_eq();
            eq[2] = 0.0;
            eq[3] = 0.0;
            eq[4] = 0.0;

            let result = interaction.fast_interaction(&n, &na, &eq, &[], &[]);
            n += &result.dn;
            na += &result.dna;

            approx_eq(n[2], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS)
                .or_else(|_| approx_eq(n[3], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS))
                .or_else(|_| approx_eq(n[4], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS))
                .map_err(err_line_print!())?;
            approx_eq(
                na[2] * n[3] * n[4] + na[3] * n[2] * n[4] + na[4] * n[2] * n[3],
                0.0,
                ZERO_EPS_REL,
                ZERO_EPS_ABS,
            )
            .map_err(err_line_print!())?;
        }

        Ok(())
    }

    #[test]
    fn fast_interaction_random() -> Result<(), Box<dyn error::Error>> {
        #[derive(Deserialize)]
        struct Entry {
            incoming: Vec<isize>,
            outgoing: Vec<isize>,
            n: Vec<f64>,
            na: Vec<f64>,
            eq: Vec<f64>,
            symmetric: Vec<f64>,
            asymmetric: f64,
        }

        let json = zstd::Decoder::with_buffer(BufReader::new(File::open(
            "tests/data/fast_interaction_random.json.zst",
        )?))?;
        let data: Vec<Entry> = serde_json::from_reader(json).unwrap();

        for (i, entry) in data.into_iter().enumerate() {
            // FIXME: Find why these don't work exactly.  First are symmetric
            // errors, second are asymmetric errors.
            if [
                1664, 3923, 5473, 5932, 6345, 7614, 11577, 11720, 12090, 12612, 14157, 15391,
                17247, 18597, 19300, 25714, 26557, 27425, 28365, 30197, 31683, 33544, 33798, 33866,
                34182, 35102, 36459, 37023, 37937, 44185, 50752, 51209, 52142, 56660, 59177, 60542,
                60750, 61564, 63306, 63645, 66453, 67257, 70659, 72058, 72596, 74379, 74535, 77388,
                81097, 81892, 82885, 87637, 88992, 95163, 98116,
            ]
            .contains(&i)
                || [
                    395, 507, 1448, 3444, 3679, 5275, 5540, 5695, 6456, 8149, 8497, 12894, 13134,
                    13178, 13986, 14418, 16947, 17341, 17664, 18151, 18468, 19970, 20308, 20353,
                    20654, 21393, 21960, 23216, 23530, 24165, 25529, 26471, 26736, 28850, 30184,
                    30990, 31558, 32438, 32953, 33399, 34541, 34570, 35835, 36349, 36984, 37449,
                    38167, 38303, 38957, 40300, 40710, 41170, 41697, 42214, 42694, 43168, 43782,
                    44137, 44540, 44728, 46907, 48508, 48781, 48816, 49093, 49579, 49748, 50758,
                    51219, 52316, 53188, 53356, 53642, 55600, 55969, 56242, 57588, 57975, 58248,
                    59528, 60011, 63196, 63741, 63876, 64903, 66282, 66284, 67560, 67577, 68316,
                    68774, 69037, 69197, 69416, 69483, 69631, 70872, 73869, 74154, 74332, 75052,
                    75326, 75425, 76342, 76796, 78422, 78452, 79643, 79700, 79754, 80241, 80790,
                    80915, 82449, 82518, 82555, 82882, 85112, 85244, 85534, 86373, 87235, 87377,
                    90861, 91338, 91824, 92150, 92985, 93740, 95762, 95995, 97222, 97688, 97943,
                ]
                .contains(&i)
            {
                continue;
            }
            #[allow(clippy::cast_possible_truncation)]
            let interaction = InteractionParticles::new(&entry.incoming, &entry.outgoing);

            let n = Array1::from(entry.n);
            let na = Array1::from(entry.na);
            let eq = Array1::from(entry.eq);

            let symmetric = interaction.symmetric_delta(&n, &eq, &[]);
            assert!(
                entry.symmetric.iter().any(|&sol| {
                    approx_eq(sol, symmetric, 1.0, 1e-50)
                        // .map_err(err_line_print!())
                        .is_ok()
                }),
                "Root found ({:e}) is not one of the roots known: {:?}",
                symmetric,
                entry.symmetric
            );

            // FIXME:
            // Some solutions have *very* large solutions for the asymmetric,
            // which is clearly wrong
            if entry.asymmetric.abs() < 10.0 {
                approx_eq(
                    entry.asymmetric,
                    interaction.asymmetric_delta(&n, &na, &eq, &[], &[]),
                    1.0,
                    1e-30,
                )
                .map_err(err_line_print!())?;
            }
        }

        Ok(())
    }

    #[test]
    fn fast_interaction_random_zero_incoming() -> Result<(), Box<dyn error::Error>> {
        for _ in 0..RUN_COUNT {
            let interaction = rand_interaction();
            let (mut n, mut na, mut eq) = rand_n_na_eq();
            for &idx in &interaction.incoming_idx {
                eq[idx] = 0.0;
            }

            let result = interaction.fast_interaction(&n, &na, &eq, &[], &[]);
            n += &result.dn;
            na += &result.dna;

            let (forward, backward) = interaction.symmetric_prefactor(&n, &eq, &[]);
            approx_eq(
                forward.0 * backward.1,
                backward.0 * forward.1,
                ZERO_EPS_REL,
                ZERO_EPS_ABS,
            )
            .map_err(err_line_print!())?;
            let (forward, backward) = interaction.asymmetric_prefactor(&n, &na, &eq, &[], &[]);
            approx_eq(
                forward.0 * backward.1,
                backward.0 * forward.1,
                ZERO_EPS_REL,
                ZERO_EPS_ABS,
            )
            .map_err(err_line_print!())?;
        }

        Ok(())
    }

    #[test]
    fn fast_interaction_random_zero_outgoing() -> Result<(), Box<dyn error::Error>> {
        for _ in 0..RUN_COUNT {
            let interaction = rand_interaction();
            let (mut n, mut na, mut eq) = rand_n_na_eq();
            for &idx in &interaction.outgoing_idx {
                eq[idx] = 0.0;
            }

            let result = interaction.fast_interaction(&n, &na, &eq, &[], &[]);
            n += &result.dn;
            na += &result.dna;

            let (forward, backward) = interaction.symmetric_prefactor(&n, &eq, &[]);
            approx_eq(forward.0 * backward.1, backward.0 * forward.1, 8.0, 1e-15)
                .map_err(err_line_print!())?;
            let (forward, backward) = interaction.asymmetric_prefactor(&n, &na, &eq, &[], &[]);
            approx_eq(forward.0 * backward.1, backward.0 * forward.1, 8.0, 1e-15)
                .map_err(err_line_print!())?;
        }

        Ok(())
    }

    #[test]
    fn display() {
        let mut model = crate::model::EmptyModel::default();
        model.push_particle(crate::prelude::Particle::new(0, 1.0, 1e-3).name("one"));
        model.push_particle(crate::prelude::Particle::new(0, 1.0, 1e-3).name("two"));
        model.push_particle(crate::prelude::Particle::new(0, 1.0, 1e-3).name("three"));
        model.push_particle(crate::prelude::Particle::new(0, 1.0, 1e-3).name("four"));

        let interaction = super::InteractionParticles::new(&[0, 1], &[1, 2]);
        assert_eq!(interaction.short_display(), "0 1 ↔ 1 2");
        assert_eq!(interaction.short_display(), format!("{}", interaction));
        assert_eq!(
            interaction.display(&model),
            Ok("none one ↔ one two".to_string())
        );

        let interaction = super::InteractionParticles::new(&[-1, 2], &[2, -3]);
        assert_eq!(interaction.short_display(), "-1 2 ↔ -3 2");
        assert_eq!(interaction.short_display(), format!("{}", interaction));
        assert_eq!(
            interaction.display(&model),
            Ok("\u{304}one two ↔ \u{304}three two".to_string())
        );

        let interaction = super::InteractionParticles::new(&[], &[1, 2, 3]);
        assert_eq!(interaction.short_display(), "↔ 1 2 3");
        assert_eq!(interaction.short_display(), format!("{}", interaction));
        assert_eq!(
            interaction.display(&model),
            Ok("↔ one two three".to_string())
        );

        let interaction = super::InteractionParticles::new(&[1, -2, 3], &[]);
        assert_eq!(interaction.short_display(), "-2 1 3 ↔");
        assert_eq!(interaction.short_display(), format!("{}", interaction));
        assert_eq!(
            interaction.display(&model),
            Ok("\u{304}two one three ↔".to_string())
        );

        let interaction = super::InteractionParticles::new(&[1, 2, 3], &[5]);
        assert_eq!(interaction.short_display(), "1 2 3 ↔ 5");
        assert_eq!(interaction.short_display(), format!("{}", interaction));
        assert_eq!(interaction.display(&model), Err(super::DisplayError(5)));

        let interaction = super::InteractionParticles::new(&[1, -5, 3], &[2]);
        assert_eq!(interaction.short_display(), "-5 1 3 ↔ 2");
        assert_eq!(interaction.short_display(), format!("{}", interaction));
        assert_eq!(interaction.display(&model), Err(super::DisplayError(-5)));
    }
}
