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
use rand::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, collections::HashMap, convert::TryFrom, fmt, ops};

/// Result from a fast interaction.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct FastInteractionResult {
    /// Array of changes to be added to the number densities.
    pub dn: Array1<f64>,
    /// Value of the change in number density.  This is equivalent to
    /// `self.dn.abs().max()` provided that no particle is repeated in the
    /// interaction.
    pub symmetric_delta: f64,
    /// Array of changes to be added to the number density asymmetries.
    pub dna: Array1<f64>,
    /// Value of teh change in number density asymmetry.  This is equivalent to
    /// `self.dna.abs().max()` provided that no particle is repeated.
    pub asymmetric_delta: f64,
}

impl FastInteractionResult {
    /// Create a new interaction result filled with 0 values and the specified
    /// size for the [`dn`](FastInteractionResult::dn) and
    /// [`dna`](FastInteractionResult::dna) arrays.
    #[must_use]
    pub fn zero(n: usize) -> Self {
        Self {
            dn: Array1::zeros(n),
            symmetric_delta: 0.0,
            dna: Array1::zeros(n),
            asymmetric_delta: 0.0,
        }
    }
}

impl ops::AddAssign<&Self> for FastInteractionResult {
    fn add_assign(&mut self, rhs: &Self) {
        self.dn += &rhs.dn;
        self.symmetric_delta += rhs.symmetric_delta;
        self.dna += &rhs.dna;
        self.asymmetric_delta += rhs.asymmetric_delta;
    }
}

impl fmt::Display for FastInteractionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Fast Interaction Result: δ = {}, δ' = {}",
            self.symmetric_delta, self.asymmetric_delta
        )
    }
}

/// List of particles involved in the interaction.
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct InteractionParticles {
    /// Initial sate particle indices.
    pub incoming_idx: Vec<usize>,
    /// Final sate particle indices.
    pub outgoing_idx: Vec<usize>,
    /// Initial sate particle sign.  A positive sign indicates this is a
    /// particle, while a negative sign indicates it is an antiparticle.
    pub incoming_sign: Vec<f64>,
    /// Final sate particle sign.  A positive sign indicates this is a particle,
    /// while a negative sign indicates it is an antiparticle.
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
}

impl InteractionParticles {
    /// Create a new set of interaction particles.  The absolute value of the
    /// numbers indicate the corresponding index for the particle, with the sign
    /// indicating whether it is a particle or anti-particle involved.
    #[must_use]
    pub fn new(incoming: &[isize], outgoing: &[isize]) -> Self {
        let mut result = Self {
            incoming_idx: incoming
                .iter()
                .map(|p| usize::try_from(p.abs()).unwrap())
                .collect(),
            outgoing_idx: outgoing
                .iter()
                .map(|p| usize::try_from(p.abs()).unwrap())
                .collect(),
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
            incoming_signed: incoming.to_vec(),
            outgoing_signed: outgoing.to_vec(),

            particle_counts: HashMap::new(),
        };

        result.calculate_particle_counts();
        result
    }

    /// Create a reversed version of the InteractionParticles with the incoming
    /// and outgoing particles reversed.  This assumes CPT symmetry, thus the
    /// status if incoming/outgoing particles are swapped alongside with the
    /// status as particle or antiparticle.
    #[must_use]
    pub fn reversed(&self) -> Self {
        Self {
            incoming_idx: self.outgoing_idx.clone(),
            outgoing_idx: self.incoming_idx.clone(),
            incoming_sign: self.outgoing_sign.iter().map(ops::Neg::neg).collect(),
            outgoing_sign: self.incoming_sign.iter().map(ops::Neg::neg).collect(),
            incoming_signed: self.outgoing_signed.iter().map(ops::Neg::neg).collect(),
            outgoing_signed: self.incoming_signed.iter().map(ops::Neg::neg).collect(),
            particle_counts: self
                .particle_counts
                .iter()
                .map(|(&p, &(c, ca))| (p, (-c, -ca)))
                .collect(),
        }
    }

    /// Iterate over all incoming particle attributes, as a tuple `(idx, sign)`.
    fn iter_incoming(&self) -> std::iter::Zip<std::slice::Iter<usize>, std::slice::Iter<f64>> {
        self.incoming_idx.iter().zip(&self.incoming_sign)
    }

    /// Iterate over all incoming particle attributes, as a tuple `(idx, sign)`.
    fn iter_outgoing(&self) -> std::iter::Zip<std::slice::Iter<usize>, std::slice::Iter<f64>> {
        self.outgoing_idx.iter().zip(&self.outgoing_sign)
    }

    /// Number of incoming particles.
    #[must_use]
    fn incoming_len(&self) -> usize {
        self.incoming_idx.len()
    }

    /// Number of outgoing particles.
    #[must_use]
    fn outgoing_len(&self) -> usize {
        self.outgoing_idx.len()
    }

    /// Compute the product of the relevant entries in the given array as
    /// indexed by the incoming particles, excluding the `$i$`th incoming
    /// particle.
    ///
    /// The array specified should be an array of number densities or
    /// equilibrium number densities depending on context.  The index refers to
    /// the position of the particle in the incoming array, not the index of the
    /// particle itself.
    #[must_use]
    pub fn incoming_product_except(&self, arr: &Array1<f64>, i: usize) -> f64 {
        self.incoming_idx
            .iter()
            .enumerate()
            .filter_map(|(j, &p)| if j == i { None } else { Some(arr[p]) })
            .product()
    }

    /// Compute the product of the relevant entries in the given array as
    /// indexed by the outgoing particles, excluding the `$i$`th outgoing
    /// particle.
    ///
    /// ```math
    /// \prod_{j \neq i} arr[j]
    /// ```
    ///
    /// The array specified should be an array of number densities or
    /// equilibrium number densities depending on context.  The index refers to
    /// the position of the particle in the outgoing array, not the index of the
    /// particle itself.
    #[must_use]
    pub fn outgoing_product_except(&self, arr: &Array1<f64>, i: usize) -> f64 {
        self.outgoing_idx
            .iter()
            .enumerate()
            .filter_map(|(j, &p)| if j == i { None } else { Some(arr[p]) })
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
        let mut particle_counts = HashMap::with_capacity(self.incoming_len() + self.outgoing_len());

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
    #[must_use]
    pub fn symmetric_prefactor(
        &self,
        n: &Array1<f64>,
        eq: &Array1<f64>,
    ) -> ((f64, f64), (f64, f64)) {
        let forward = self
            .incoming_idx
            .iter()
            .fold((1.0, 1.0), |(num, den), &p| (num * n[p], den * eq[p]));
        let backward = self
            .outgoing_idx
            .iter()
            .fold((1.0, 1.0), |(num, den), &p| (num * n[p], den * eq[p]));
        (forward, backward)
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
    fn asymmetric_prefactor(
        &self,
        n: &Array1<f64>,
        na: &Array1<f64>,
        eq: &Array1<f64>,
    ) -> ((f64, f64), (f64, f64)) {
        let forward =
            self.iter_incoming()
                .enumerate()
                .fold((0.0, 1.0), |(num, den), (i, (&p, &a))| {
                    (
                        num + a * na[p] * self.incoming_product_except(n, i),
                        den * eq[p],
                    )
                });
        let backward =
            self.iter_outgoing()
                .enumerate()
                .fold((0.0, 1.0), |(num, den), (i, (&p, &a))| {
                    (
                        num + a * na[p] * self.outgoing_product_except(n, i),
                        den * eq[p],
                    )
                });

        (forward, backward)
    }

    /// Return the minimum change in number density that will result in an
    /// incoming particle number density to be zero.
    ///
    /// The smallest change in magnitude is used and if there are no incoming
    /// particle or there is no net change the result is 0.
    fn minimum_change_symmetric_incoming(&self, n: &Array1<f64>) -> f64 {
        self.incoming_idx
            .iter()
            .filter_map(|&p| {
                let count = self.particle_counts[&p].0;
                if count == 0.0 {
                    None
                } else {
                    Some(-n[p] / count)
                }
            })
            .min_by(|x, y| x.abs().partial_cmp(&y.abs()).unwrap())
            .unwrap_or_default()
    }

    /// Return the minimum change in number density that will result in an
    /// outgoing particle number density to be zero.
    ///
    /// The smallest change in magnitude is used and if there are no outging
    /// particle or there is no net change the result is 0.
    fn minimum_change_symmetric_outgoing(&self, n: &Array1<f64>) -> f64 {
        self.outgoing_idx
            .iter()
            .filter_map(|&p| {
                let count = self.particle_counts[&p].0;
                if count == 0.0 {
                    None
                } else {
                    Some(-n[p] / count)
                }
            })
            .min_by(|x, y| x.abs().partial_cmp(&y.abs()).unwrap())
            .unwrap_or_default()
    }

    /// Computes the change in number density required to establish equilibrium,
    /// up to linear order.
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
    /// +1 for annihilation and creation respectively).  The exact solution can
    /// be obtained by recursively applying the change until convergence is
    /// achieved.
    ///
    /// If one of the equilibrium number density of the incoming particles is 0
    /// then `$\delta = \min_{i \in X}\{n_i\}$` as the interaction can only
    /// proceed in one direction.  Similarly, if the 0 is on the outgoing side,
    /// then `$\delta = -\min_{i \in Y}\{n_i}$`.
    ///
    /// If there are zero equilibrium number densities on both sides, the number
    /// of 0s is used to determine which direction the interaction proceeds, and
    /// if equal then is taken to be the one with the smallest absolute value of
    /// the two above.
    ///
    /// As this algorithm only solves the linearized equation, it must be
    /// recursively applied.  The cumulative sum of changes is to be specified
    /// in `cumulative_delta`.
    #[must_use]
    pub fn symmetric_delta(&self, n: &Array1<f64>, eq: &Array1<f64>) -> f64 {
        // If the interaction does not result in any number density change, then return 0.
        if self.particle_counts.values().all(|&(c, _)| c == 0.0) {
            return 0.0;
        }

        let (incoming_n_prod, incoming_eq_prod) = self
            .incoming_idx
            .iter()
            .fold((1.0, 1.0), |(n_prod, eq_prod), &p| {
                (n_prod * n[p], eq_prod * eq[p])
            });
        let (outgoing_n_prod, outgoing_eq_prod) = self
            .outgoing_idx
            .iter()
            .fold((1.0, 1.0), |(n_prod, eq_prod), &p| {
                (n_prod * n[p], eq_prod * eq[p])
            });

        match (incoming_eq_prod == 0.0, outgoing_eq_prod == 0.0) {
            (true, true) => {
                // If both sides are zero, use the number of zeros to determine
                // which side is favored, and if they are equal, is the minimum
                // in absolute value.
                let incoming_zeros = self.incoming_idx.iter().filter(|&&p| eq[p] == 0.0).count();
                let outgoing_zeros = self.outgoing_idx.iter().filter(|&&p| eq[p] == 0.0).count();

                match incoming_zeros.cmp(&outgoing_zeros) {
                    Ordering::Less => self.minimum_change_symmetric_incoming(n),
                    Ordering::Equal => {
                        let delta_forward = self.minimum_change_symmetric_incoming(n);
                        let delta_backward = self.minimum_change_symmetric_outgoing(n);

                        if delta_forward.abs() > delta_backward.abs() {
                            delta_forward
                        } else {
                            delta_backward
                        }
                    }
                    Ordering::Greater => self.minimum_change_symmetric_outgoing(n),
                }
            }
            (true, false) => self.minimum_change_symmetric_incoming(n),
            (false, true) => self.minimum_change_symmetric_outgoing(n),
            (false, false) => {
                let numerator =
                    incoming_n_prod * outgoing_eq_prod - outgoing_n_prod * incoming_eq_prod;

                if numerator == 0.0 {
                    0.0
                } else {
                    let denominator = incoming_eq_prod
                        * self
                            .outgoing_idx
                            .iter()
                            .enumerate()
                            .map(|(i, &p)| {
                                self.particle_counts[&p].0 * self.outgoing_product_except(n, i)
                            })
                            .sum::<f64>()
                        - outgoing_eq_prod
                            * self
                                .incoming_idx
                                .iter()
                                .enumerate()
                                .map(|(i, &p)| {
                                    self.particle_counts[&p].0 * self.incoming_product_except(n, i)
                                })
                                .sum::<f64>();

                    if denominator == 0.0 {
                        0.0
                    } else {
                        numerator / denominator
                    }
                }
            }
        }
    }

    /// Return the minimum change in number density asymmetric that will result
    /// in an incoming particle number density to be zero.
    ///
    /// The smallest change in magnitude is used and if there are no incoming
    /// particle or there is no net change the result is 0.
    fn minimum_change_asymmetric_incoming(&self, n: &Array1<f64>, na: &Array1<f64>) -> f64 {
        let (mut numerator, mut denominator) = (0.0, 0.0);
        let mut min_zero = f64::INFINITY;

        for (&p, &a) in self.iter_incoming() {
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

    /// Return the minimum change in number density asymmetric that will result
    /// in an outgoing particle number density to be zero.
    ///
    /// The smallest change in magnitude is used and if there are no outgoing
    /// particle or there is no net change1 the result is 0.
    fn minimum_change_asymmetric_outgoing(&self, n: &Array1<f64>, na: &Array1<f64>) -> f64 {
        let (mut numerator, mut denominator) = (0.0, 0.0);
        let mut min_zero = f64::INFINITY;

        for (&p, &a) in self.iter_outgoing() {
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
    /// holds true.  Unlike the symmetric case, this is always linear in
    /// `$\delta$` and thus the result is exact.
    #[must_use]
    pub fn asymmetric_delta(&self, n: &Array1<f64>, na: &Array1<f64>, eq: &Array1<f64>) -> f64 {
        let incoming_eq_prod = self.incoming_idx.iter().map(|&p| eq[p]).product::<f64>();
        let outgoing_eq_prod = self.outgoing_idx.iter().map(|&p| eq[p]).product::<f64>();

        match (incoming_eq_prod == 0.0, outgoing_eq_prod == 0.0) {
            (true, true) => {
                // If both sides are zero, use the number of zeros to determine
                // which side is favored, and if they are equal, have delta = 0.
                // If equal
                let incoming_zeros = self.incoming_idx.iter().filter(|&&p| n[p] == 0.0).count();
                let outgoing_zeros = self.outgoing_idx.iter().filter(|&&p| n[p] == 0.0).count();

                match incoming_zeros.cmp(&outgoing_zeros) {
                    Ordering::Less => self.minimum_change_asymmetric_outgoing(n, na),
                    Ordering::Equal => {
                        let delta_forward = self.minimum_change_asymmetric_incoming(n, na);
                        let delta_backward = self.minimum_change_asymmetric_outgoing(n, na);

                        if delta_forward.abs() > delta_backward.abs() {
                            delta_forward
                        } else {
                            delta_backward
                        }
                    }
                    Ordering::Greater => self.minimum_change_asymmetric_incoming(n, na),
                }
            }
            (true, false) => self.minimum_change_asymmetric_incoming(n, na),
            (false, true) => self.minimum_change_asymmetric_outgoing(n, na),
            (false, false) => {
                let mut numerator = 0.0;
                let mut denominator = 0.0;

                for (i, (&p, &a)) in self.iter_incoming().enumerate() {
                    let product = self.incoming_product_except(n, i);
                    numerator += a * na[p] * product * outgoing_eq_prod;
                    denominator -= a * self.particle_counts[&p].1 * product * outgoing_eq_prod;
                }
                for (i, (&p, &a)) in self.iter_outgoing().enumerate() {
                    let product = self.outgoing_product_except(n, i);
                    numerator -= a * na[p] * product * incoming_eq_prod;
                    denominator += a * self.particle_counts[&p].1 * product * incoming_eq_prod;
                }

                numerator / denominator
            }
        }
    }

    /// For a fast interaction, compute the change to each particle species such
    /// that equilibrium is established.
    ///
    /// The computation in the symmetric case is described in
    /// [`symmetric_delta`], and the asymmetric case is described in [`asymmetric_delta`].
    #[must_use]
    pub fn fast_interaction(
        &self,
        n: &Array1<f64>,
        na: &Array1<f64>,
        eq: &Array1<f64>,
    ) -> FastInteractionResult {
        debug_assert!(
            n.dim() == na.dim() && n.dim() == eq.dim(),
            "The number density, number density asymmetry and equilibrium number density array shapes must be all the same"
        );
        let mut result = FastInteractionResult::zero(n.dim());

        result.symmetric_delta = self.symmetric_delta(n, eq);
        result.asymmetric_delta = self.asymmetric_delta(n, na, eq);

        for (&p, &(c, ca)) in &self.particle_counts {
            result.dn[p] += c * result.symmetric_delta;
            result.dna[p] += ca * result.asymmetric_delta;
        }

        result
    }

    /// Apply [`fast_interaction`] recursively until the exact solution is
    /// found.
    ///
    /// As the original implementation only works to linear order in `$\Delta$`,
    /// the solution it provides is only approximate.  Recursive applications of
    /// the function should converge to the exact solution.
    #[must_use]
    pub fn fast_interaction_exact(
        &self,
        n: &Array1<f64>,
        na: &Array1<f64>,
        eq: &Array1<f64>,
    ) -> FastInteractionResult {
        debug_assert!(
            n.dim() == na.dim() && n.dim() == eq.dim(),
            "The number density, number density asymmetry and equilibrium number density array shapes must be all the same."
        );

        let mut rng = rand::thread_rng();

        // For the initial guess for delta, we use the absolute value of the
        // smallest number density change that would result in a 0 density.  We
        // scale this down a little just to avoid some accidental 0s or being at
        // places where the linear approximation is highly inaccurate.
        let mut delta = 0.5
            * self
                .particle_counts
                .iter()
                .map(|(&p, &(c, _))| {
                    if c == 0.0 {
                        f64::INFINITY
                    } else {
                        (n[p] / c).abs()
                    }
                })
                .min_by(|x, y| x.partial_cmp(y).unwrap())
                .unwrap();

        // If delta is infinite at this stage, then all symmetric particle
        // counts were 0 and the final delta ought to be 0.
        if delta.is_infinite() {
            let mut result = FastInteractionResult::zero(n.dim());
            result.asymmetric_delta = self.asymmetric_delta(n, na, eq);
            for (&p, &(_, ca)) in &self.particle_counts {
                result.dna[p] += ca * result.asymmetric_delta;
            }
            // log::trace!("δ = {}", result.symmetric_delta);
            // log::trace!("δ' = {}", result.asymmetric_delta);
            return result;
        }

        // Using this initial δ, we give it a sign based on whether the
        // interaction ought to go forward or backwards.  This helps converge on
        // the right value if there are multiple.  If the net prescaling is not
        // computable or infinite, the starting point for delta should be 0.
        let (forward, backward) = self.symmetric_prefactor(n, eq);
        let net_prescaling = forward.0 / forward.1 - backward.0 / backward.1;
        delta *= if net_prescaling.is_finite() {
            net_prescaling.signum()
        } else {
            0.0
        };
        let mut cumulative_delta = delta;

        // We track of sign changes to avoid infinite loops.
        let mut sign_changes = 0_usize;
        let mut prev_sign = delta.signum();

        // For debugging, if we have too many iterations we keep track of the
        // last few entries.
        let mut cumulative_deltas = Vec::with_capacity(100);
        let mut deltas = Vec::with_capacity(100);

        // Successively calculate and apply the local change in number densities
        // until we converge.
        let mut ni;
        let mut count = 0_usize;
        loop {
            ni = n.clone();
            for (&p, &(c, _)) in &self.particle_counts {
                ni[p] += c * cumulative_delta;
            }

            delta = self.symmetric_delta(&ni, eq);
            cumulative_delta += delta;

            #[allow(clippy::float_cmp)]
            if delta.signum() != prev_sign {
                sign_changes += 1;
                prev_sign = delta.signum();

                if sign_changes > 4 {
                    delta = rng.gen();

                    sign_changes = 0;
                    prev_sign = delta.signum();
                    cumulative_delta = delta;
                }
            }

            if cumulative_delta != 0.0 && (delta / cumulative_delta).abs() < 1e-6 {
                break;
            }

            count += 1;
            if count > 1000 - cumulative_deltas.capacity() {
                cumulative_deltas.push(cumulative_delta);
                deltas.push(delta);
            }
            if count > 1000 {
                log::error!("Unable to converge fast interaction after 1000 iterations.");
                log::info!(
                    "Last cumulative and local δ:\n-> {:?}\n-> {:?}",
                    cumulative_deltas,
                    deltas
                );
                break;
            }
        }

        let mut result = FastInteractionResult::zero(ni.dim());
        result.symmetric_delta = cumulative_delta;
        result.asymmetric_delta = self.asymmetric_delta(&ni, na, eq);

        for (&p, &(c, ca)) in &self.particle_counts {
            result.dn[p] += c * result.symmetric_delta;
            result.dna[p] += ca * result.asymmetric_delta;
        }

        result
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
        let mut s = String::with_capacity(3 * (self.incoming_len() + self.outgoing_len()) + 2);

        for &p in &self.incoming_signed {
            s.push_str(&model.particle_name(p).map_err(|_| ())?);
            s.push(' ');
        }

        s.push_str("↔");

        for &p in &self.outgoing_signed {
            s.push(' ');
            s.push_str(&model.particle_name(p).map_err(|_| ())?);
        }

        Ok(s)
    }
}

impl fmt::Display for InteractionParticles {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();

        for &p in &self.incoming_signed {
            s.push_str(&format!("{} ", p));
        }

        s.push_str("↔");

        for &p in &self.outgoing_signed {
            s.push_str(&format!(" {}", p));
        }

        write!(f, "{}", s)
    }
}

/// Generic interaction between particles.
pub trait Interaction<M> {
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
    fn symmetric_prefactor(&self, c: &Context<M>) -> f64 {
        let ((forward_numerator, forward_denominator), (backward_numerator, backward_denominator)) =
            self.particles().symmetric_prefactor(&c.n, &c.eq);

        // When computing the ratio of number density to equilibrium number
        // density, a NaN should only occur from `0.0 / 0.0`, in which case the
        // correct value ought to be 0 as there are no actual particles to decay
        // in the numerator.
        let mut forward_prefactor = checked_div(forward_numerator, forward_denominator);
        let mut backward_prefactor = checked_div(backward_numerator, backward_denominator);

        // In the resulting product of ratios, a NaN value can only happen if we
        // have 0 * ∞, in which case one of the number densities was 0 and thus
        // that prefactor should really be 0.
        if forward_prefactor.is_nan() {
            forward_prefactor = 0.0;
        }
        if backward_prefactor.is_nan() {
            backward_prefactor = 0.0;
        }

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
    fn asymmetric_prefactor(&self, c: &Context<M>) -> f64 {
        let ((forward_numerator, forward_denominator), (backward_numerator, backward_denominator)) =
            self.particles().asymmetric_prefactor(&c.n, &c.na, &c.eq);
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
    /// [`Interaction::asymmetry`] in order to computer the actual rate.
    fn rate(&self, c: &Context<M>) -> Option<RateDensity> {
        let gamma = self.gamma(c, false).unwrap_or(0.0);
        let asymmetry = self.delta_gamma(c, false).unwrap_or(0.0);

        // If both rates are 0, there's no need to adjust it to the particles'
        // number densities.
        if gamma == 0.0 && asymmetry == 0.0 {
            return None;
        }

        debug_assert!(
            !gamma.is_finite(),
            "Non-finite interaction rate at step {} for interaction {:?}: {}",
            c.step,
            self.particles(),
            gamma
        );
        debug_assert!(
            !asymmetry.is_finite(),
            "Non-finite asymmetric interaction rate at step {} for interaction {:?}: {}",
            c.step,
            self.particles(),
            asymmetry
        );

        let mut rate = RateDensity::zero();
        let symmetric_prefactor = self.symmetric_prefactor(c);
        rate.symmetric = gamma * symmetric_prefactor;
        rate.asymmetric = asymmetry * symmetric_prefactor + gamma * self.asymmetric_prefactor(c);

        Some(rate * c.normalization)
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
        let rate = self.rate(c)? * c.step_size;

        // No need to do anything of both rates are 0.
        if rate.symmetric == 0.0 && rate.asymmetric == 0.0 {
            return None;
        }

        // If the rate is large, we add the interacting particles to the list of
        // fast interactions.
        if rate.symmetric.abs() > 0.1 {
            let mut fast_interactions = c.fast_interactions.write().unwrap();
            fast_interactions.push(self.particles().clone());
            return None;
        }

        debug_assert!(
            !rate.symmetric.is_finite(),
            "Non-finite interaction adjusted rate at step {} for interaction {:?}: {}",
            c.step,
            self.particles(),
            rate.symmetric
        );
        debug_assert!(
            !rate.asymmetric.is_finite(),
            "Non-finite asymmetric adjusted interaction rate at step {} for interaction {:?}: {}",
            c.step,
            self.particles(),
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

            for (&p, a) in particles.iter_incoming() {
                dn[p] -= rate.symmetric;
                dna[p] -= a * rate.asymmetric;
            }
            for (&p, a) in particles.iter_outgoing() {
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

/// Computes the ratio `a / b` in a manner that never returns NaN
///
/// This is to be used in the context of calculating the scaling of the
/// interaction density by the number density ratios `n / eq`.  Note that a NaN
/// value arises only from have `n == eq == 0.0`, and the result is
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
    use crate::utilities::test::approx_eq;
    use ndarray::prelude::*;
    use rand::prelude::Distribution;
    use std::{error, f64, iter};

    const RUN_COUNT: usize = 1_000;

    // These values need to be adjusted for tests which involve a large number
    // of particles, approximately an order of magnitude per particle.
    const EQUALITY_EPS_REL: f64 = 4.5;
    const EQUALITY_EPS_ABS: f64 = 1e-13;
    const ZERO_EPS_REL: f64 = 10.0;
    const ZERO_EPS_ABS: f64 = 1e-13;

    fn rand_na() -> f64 {
        let x: f64 = rand::random();
        (x - 0.5) / 10.0
    }

    fn rand_n() -> f64 {
        rand::random()
    }

    fn rand_eq() -> f64 {
        rand::random()
    }

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

    #[test]
    fn product_except() -> Result<(), Box<dyn error::Error>> {
        let interaction = super::InteractionParticles::new(&[0, 1, 2, 3], &[4, 5, 6, 7]);

        let n = array![2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0];
        let prod_in = n.iter().take(4).product::<f64>();
        let prod_out = n.iter().skip(4).product::<f64>();

        for i in 0..4 {
            approx_eq(
                interaction.incoming_product_except(&n, i),
                prod_in / n[interaction.incoming_idx[i]],
                15.0,
                0.0,
            )?;
            approx_eq(
                interaction.outgoing_product_except(&n, i),
                prod_out / n[interaction.outgoing_idx[i]],
                15.0,
                0.0,
            )?;
        }

        Ok(())
    }

    #[test]
    fn fast_interaction_1_2() -> Result<(), Box<dyn error::Error>> {
        let interaction = super::InteractionParticles::new(&[1], &[2, 3]);
        let mut n = Array1::zeros(5);
        let mut na = Array1::zeros(5);
        let eq = Array1::from_shape_simple_fn(5, rand_eq);

        for _ in 0..RUN_COUNT {
            n.mapv_inplace(|_| rand_n());
            na.mapv_inplace(|_| rand_na());

            let result = interaction.fast_interaction_exact(&n, &na, &eq);
            n += &result.dn;
            na += &result.dna;

            approx_eq(
                super::checked_div(n[1], eq[1]),
                super::checked_div(n[2], eq[2]) * super::checked_div(n[3], eq[3]),
                EQUALITY_EPS_REL,
                EQUALITY_EPS_ABS,
            )
            .or_else(|e| {
                if [1, 2, 3].iter().any(|&p| n[p] == 0.0) {
                    Ok(())
                } else {
                    Err(format!(
                        "Symmetric equality error: {}\n\
                        None of the number densities were 0.",
                        e
                    ))
                }
            })?;

            approx_eq(
                super::checked_div(na[1], eq[1]),
                super::checked_div(na[2] * n[3] + na[3] * n[2], eq[2] * eq[3]),
                EQUALITY_EPS_REL,
                EQUALITY_EPS_ABS * 1_000.0,
            )
            .map_err(|e| format!("Asymmetric equality error: {}", e))?;
        }

        Ok(())
    }

    #[test]
    fn fast_interaction_1_2_zero_incoming() -> Result<(), Box<dyn error::Error>> {
        let interaction = super::InteractionParticles::new(&[1], &[2, 3]);
        let mut n = Array1::zeros(5);
        let mut na = Array1::zeros(5);
        let mut eq = Array1::from_shape_simple_fn(5, rand_eq);

        eq[1] = 0.0;
        for _ in 0..RUN_COUNT {
            n.mapv_inplace(|_| rand_n());
            na.mapv_inplace(|_| rand_na());

            let result = interaction.fast_interaction_exact(&n, &na, &eq);
            n += &result.dn;
            na += &result.dna;

            approx_eq(n[1], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS)
                .map_err(|_| format!("Expected number density of 0, but got {}", n[1]))?;
            approx_eq(na[1], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS)
                .map_err(|_| format!("Expected number density asymmetry of 0, gut go {}", na[1]))?;
        }

        Ok(())
    }

    #[test]
    fn fast_interaction_1_2_zero_outgoing() -> Result<(), Box<dyn error::Error>> {
        let interaction = super::InteractionParticles::new(&[1], &[2, 3]);
        let mut n = Array1::zeros(5);
        let mut na = Array1::zeros(5);
        let mut eq = Array1::from_shape_simple_fn(5, rand_eq);

        eq.mapv_inplace(|_| rand_eq());
        eq[2] = 0.0;
        eq[3] = 0.0;
        for _ in 0..RUN_COUNT {
            n.mapv_inplace(|_| rand_n());
            na.mapv_inplace(|_| rand_na());

            let result = interaction.fast_interaction_exact(&n, &na, &eq);
            n += &result.dn;
            na += &result.dna;

            approx_eq(n[2], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS)
                .or_else(|_| approx_eq(n[3], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS))
                .map_err(|_| {
                    format!(
                        "Expected a number density of 0 but got {} and {}",
                        n[2], n[3]
                    )
                })?;
            approx_eq(na[2] * n[3] + na[3] * n[2], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS)
                .map_err(|e| format!("Number density asymmetry unequal: {}", e))?;
        }

        Ok(())
    }

    #[test]
    fn fast_interaction_2_2() -> Result<(), Box<dyn error::Error>> {
        let interaction = super::InteractionParticles::new(&[1, 2], &[3, 4]);
        let mut n = Array1::zeros(5);
        let mut na = Array1::zeros(5);
        let eq = Array1::from_shape_simple_fn(5, rand_eq);

        for _ in 0..RUN_COUNT {
            n.mapv_inplace(|_| rand_n());
            na.mapv_inplace(|_| rand_na());

            let result = interaction.fast_interaction_exact(&n, &na, &eq);
            n += &result.dn;
            na += &result.dna;

            approx_eq(
                super::checked_div(n[1], eq[1]) * super::checked_div(n[2], eq[2]),
                super::checked_div(n[3], eq[3]) * super::checked_div(n[4], eq[4]),
                EQUALITY_EPS_REL,
                EQUALITY_EPS_ABS * 100.0,
            )
            .or_else(|e| {
                if [1, 2, 3, 4].iter().any(|&p| n[p] == 0.0) {
                    Ok(())
                } else {
                    Err(format!(
                        "Symmetric equality error: {}\n\
                            None of the number densities were 0.",
                        e
                    ))
                }
            })?;

            approx_eq(
                super::checked_div(na[1] * n[2] + na[2] * n[1], eq[1] * eq[2]),
                super::checked_div(na[3] * n[4] + na[4] * n[3], eq[3] * eq[4]),
                EQUALITY_EPS_REL,
                EQUALITY_EPS_ABS * 10_000.0,
            )
            .map_err(|e| format!("Asymmetric equality error: {}", e))?;
        }

        Ok(())
    }

    #[test]
    fn fast_interaction_2_2_zero_incoming() -> Result<(), Box<dyn error::Error>> {
        let interaction = super::InteractionParticles::new(&[1, 2], &[3, 4]);
        let mut n = Array1::zeros(5);
        let mut na = Array1::zeros(5);
        let mut eq = Array1::from_shape_simple_fn(5, rand_eq);

        eq[1] = 0.0;
        eq[2] = 0.0;
        for _ in 0..RUN_COUNT {
            n.mapv_inplace(|_| rand_n());
            na.mapv_inplace(|_| rand_na());

            let result = interaction.fast_interaction_exact(&n, &na, &eq);
            n += &result.dn;
            na += &result.dna;

            approx_eq(n[1], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS)
                .or_else(|_| approx_eq(n[2], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS))
                .map_err(|_| {
                    format!("Expected a 0 number density, but got {} and {}", n[1], n[2])
                })?;
            approx_eq(na[1] * n[2] + na[2] * n[1], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS)
                .map_err(|e| format!("Asymmetric equality error: {}", e))?;
        }

        Ok(())
    }

    #[test]
    fn fast_interaction_2_2_zero_outgoing() -> Result<(), Box<dyn error::Error>> {
        let interaction = super::InteractionParticles::new(&[1, 2], &[3, 4]);
        let mut n = Array1::zeros(5);
        let mut na = Array1::zeros(5);
        let mut eq = Array1::from_shape_simple_fn(5, rand_eq);

        eq.mapv_inplace(|_| rand_eq());
        eq[3] = 0.0;
        eq[4] = 0.0;
        for _ in 0..RUN_COUNT {
            n.mapv_inplace(|_| rand_n());
            na.mapv_inplace(|_| rand_na());

            let result = interaction.fast_interaction_exact(&n, &na, &eq);
            n += &result.dn;
            na += &result.dna;

            approx_eq(n[3], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS)
                .or_else(|_| approx_eq(n[4], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS))
                .map_err(|_| {
                    format!("Expected a 0 number density, but got {} and {}", n[3], n[4])
                })?;
            approx_eq(na[3] * n[4] + na[4] * n[3], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS)
                .map_err(|e| format!("Asymmetric equality error: {}", e))?;
        }

        Ok(())
    }

    #[test]
    fn fast_interaction_1_3() -> Result<(), Box<dyn error::Error>> {
        let interaction = super::InteractionParticles::new(&[1], &[2, 3, 4]);
        let mut n = Array1::zeros(5);
        let mut na = Array1::zeros(5);
        let eq = Array1::from_shape_simple_fn(5, rand_eq);

        for _ in 0..RUN_COUNT {
            n.mapv_inplace(|_| rand_n());
            na.mapv_inplace(|_| rand_na());

            let result = interaction.fast_interaction_exact(&n, &na, &eq);
            n += &result.dn;
            na += &result.dna;

            approx_eq(
                super::checked_div(n[1], eq[1]),
                super::checked_div(n[2], eq[2])
                    * super::checked_div(n[3], eq[3])
                    * super::checked_div(n[4], eq[4]),
                EQUALITY_EPS_REL,
                EQUALITY_EPS_ABS * 100.0,
            )
            .or_else(|e| {
                if [1, 2, 3, 4].iter().any(|&p| n[p] == 0.0) {
                    Ok(())
                } else {
                    Err(format!(
                        "Symmetric equality error: {}\n\
                        None of the number densities were 0.",
                        e
                    ))
                }
            })?;

            approx_eq(
                super::checked_div(na[1], eq[1]),
                super::checked_div(
                    na[2] * n[3] * n[4] + na[3] * n[2] * n[4] + na[4] * n[2] * n[3],
                    eq[2] * eq[3] * eq[4],
                ),
                EQUALITY_EPS_REL,
                EQUALITY_EPS_ABS * 10_000.0,
            )
            .map_err(|e| format!("Asymmetric equality error: {}", e))?;
        }

        Ok(())
    }

    #[test]
    fn fast_interaction_1_3_zero_incoming() -> Result<(), Box<dyn error::Error>> {
        let interaction = super::InteractionParticles::new(&[1], &[2, 3, 4]);
        let mut n = Array1::zeros(5);
        let mut na = Array1::zeros(5);
        let mut eq = Array1::from_shape_simple_fn(5, rand_eq);

        eq[1] = 0.0;
        for _ in 0..RUN_COUNT {
            n.mapv_inplace(|_| rand_n());
            na.mapv_inplace(|_| rand_na());

            let result = interaction.fast_interaction_exact(&n, &na, &eq);
            n += &result.dn;
            na += &result.dna;

            approx_eq(n[1], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS)
                .map_err(|_| format!("Expected 0 number density but got {}", n[1]))?;
            approx_eq(na[1], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS)
                .map_err(|_| format!("Expected 0 number density asymmetry, but got {}", na[1]))?;
        }

        Ok(())
    }

    #[test]
    fn fast_interaction_1_3_zero_outgoing() -> Result<(), Box<dyn error::Error>> {
        let interaction = super::InteractionParticles::new(&[1], &[2, 3, 4]);
        let mut n = Array1::zeros(5);
        let mut na = Array1::zeros(5);
        let mut eq = Array1::from_shape_simple_fn(5, rand_eq);

        eq.mapv_inplace(|_| rand_eq());
        eq[2] = 0.0;
        eq[3] = 0.0;
        eq[4] = 0.0;
        for _ in 0..RUN_COUNT {
            n.mapv_inplace(|_| rand_n());
            na.mapv_inplace(|_| rand_na());

            let result = interaction.fast_interaction_exact(&n, &na, &eq);
            n += &result.dn;
            na += &result.dna;

            approx_eq(n[2], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS)
                .or_else(|_| approx_eq(n[3], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS))
                .or_else(|_| approx_eq(n[4], 0.0, ZERO_EPS_REL, ZERO_EPS_ABS))
                .map_err(|_| {
                    format!(
                        "Expected a 0 number density, but got {}, {} and {}",
                        n[2], n[3], n[4]
                    )
                })?;
            approx_eq(
                na[2] * n[3] * n[4] + na[3] * n[2] * n[4] + na[4] * n[2] * n[3],
                0.0,
                ZERO_EPS_REL,
                ZERO_EPS_ABS,
            )
            .map_err(|e| format!("Asymmetric equality error: {}", e))?;
        }

        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    #[test]
    fn fast_interaction_random() -> Result<(), Box<dyn error::Error>> {
        let mut rng = rand::thread_rng();
        let particles_dist = rand::distributions::Uniform::new_inclusive(-5, 5);
        let size_dist = rand::distributions::Uniform::new(1, 10);

        let mut particles = || {
            let count = size_dist.sample(&mut rng);
            iter::from_fn(|| Some(particles_dist.sample(&mut rng)))
                .take(count)
                .collect()
        };

        let mut n = Array1::zeros(6);
        let mut na = Array1::zeros(6);
        let mut eq = Array1::zeros(6);

        for _ in 0..RUN_COUNT {
            let incoming: Vec<_> = particles();
            let outgoing: Vec<_> = particles();
            let interaction = super::InteractionParticles::new(&incoming, &outgoing);

            n.mapv_inplace(|_| rand_n());
            na.mapv_inplace(|_| rand_na());
            eq.mapv_inplace(|_| rand_eq());

            let result = interaction.fast_interaction_exact(&n, &na, &eq);
            n += &result.dn;
            na += &result.dna;

            let (forward, backward) = interaction.symmetric_prefactor(&n, &eq);
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            approx_eq(
                forward.0 * backward.1,
                backward.0 * forward.1,
                EQUALITY_EPS_REL,
                EQUALITY_EPS_ABS
                    * 10.0_f64
                        .powi((interaction.incoming_len() + interaction.outgoing_len()) as i32 / 2),
            )
            .or_else(|e| {
                if interaction
                    .incoming_idx
                    .iter()
                    .all(|p| interaction.particle_counts[p].0 == 0.0)
                    || interaction
                        .outgoing_idx
                        .iter()
                        .all(|p| interaction.particle_counts[p].0 == 0.0)
                {
                    Ok(())
                } else {
                    Err(format!(
                        "Symmetric equality error: {}\n\
                        None of the particle multiplicities were 0\n\
                        None of the particle densities were 0.",
                        e
                    ))
                }
            })?;
            let (forward, backward) = interaction.asymmetric_prefactor(&n, &na, &eq);
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            approx_eq(
                forward.0 * backward.1,
                backward.0 * forward.1,
                EQUALITY_EPS_REL,
                EQUALITY_EPS_ABS
                    * 10.0_f64
                        .powi((interaction.incoming_len() + interaction.outgoing_len()) as i32),
            )
            .or_else(|e| {
                #[allow(clippy::blocks_in_if_conditions)]
                let incoming_eq_product = interaction
                    .incoming_idx
                    .iter()
                    .map(|&p| eq[p])
                    .product::<f64>();
                let outgoing_eq_product = interaction
                    .outgoing_idx
                    .iter()
                    .map(|&p| eq[p])
                    .product::<f64>();

                // It is possible that no value of δ, no solution can be found.
                // This happens if change in asymmetry does not affect the
                // result which can eiher be because there net count from the
                // interaction is 0, or because the number multiplying it are 0.
                #[allow(clippy::blocks_in_if_conditions)]
                if interaction.incoming_idx.iter().enumerate().all(|(i, p)| {
                    (interaction.particle_counts[p].1) == 0.0
                        || (interaction.incoming_product_except(&n, i) * outgoing_eq_product).abs()
                            < 1e-20
                }) || interaction.outgoing_idx.iter().enumerate().all(|(i, p)| {
                    (interaction.particle_counts[p].1) == 0.0
                        || (interaction.outgoing_product_except(&n, i) * incoming_eq_product).abs()
                            < 1e-20
                }) {
                    Ok(())
                } else {
                    Err(format!(
                        "Asymmetric equality error: {}\\n\
                        None of the particle multiplicities were 0.",
                        e
                    ))
                }
            })?;
        }

        Ok(())
    }

    #[test]
    fn fast_interaction_random_zero_incoming() -> Result<(), Box<dyn error::Error>> {
        let mut rng = rand::thread_rng();
        let particles_dist = rand::distributions::Uniform::new_inclusive(-5, 5);
        let size_dist = rand::distributions::Uniform::new(1, 10);

        let mut particles = || {
            let count = size_dist.sample(&mut rng);
            iter::from_fn(|| Some(particles_dist.sample(&mut rng)))
                .take(count)
                .collect()
        };

        let mut n = Array1::zeros(6);
        let mut na = Array1::zeros(6);
        let mut eq = Array1::zeros(6);

        for _ in 0..RUN_COUNT {
            let incoming: Vec<_> = particles();
            let outgoing: Vec<_> = particles();
            let interaction = super::InteractionParticles::new(&incoming, &outgoing);

            n.mapv_inplace(|_| rand_n());
            na.mapv_inplace(|_| rand_na());
            eq.mapv_inplace(|_| rand_eq());
            for &idx in &interaction.incoming_idx {
                eq[idx] = 0.0;
            }

            let result = interaction.fast_interaction_exact(&n, &na, &eq);
            n += &result.dn;
            na += &result.dna;

            let (forward, backward) = interaction.symmetric_prefactor(&n, &eq);
            approx_eq(
                forward.0 * backward.1,
                backward.0 * forward.1,
                ZERO_EPS_REL,
                ZERO_EPS_ABS,
            )
            .map_err(|e| format!("Symmetric equality error: {}", e))?;
            let (forward, backward) = interaction.asymmetric_prefactor(&n, &na, &eq);
            approx_eq(
                forward.0 * backward.1,
                backward.0 * forward.1,
                ZERO_EPS_REL,
                ZERO_EPS_ABS,
            )
            .map_err(|e| format!("Asymmetric equality error: {}", e))?;
        }

        Ok(())
    }

    #[test]
    fn fast_interaction_random_zero_outgoing() -> Result<(), Box<dyn error::Error>> {
        let mut rng = rand::thread_rng();
        let particles_dist = rand::distributions::Uniform::new_inclusive(-5, 5);
        let size_dist = rand::distributions::Uniform::new(1, 10);

        let mut particles = || {
            let count = size_dist.sample(&mut rng);
            iter::from_fn(|| Some(particles_dist.sample(&mut rng)))
                .take(count)
                .collect()
        };

        let mut n = Array1::zeros(6);
        let mut na = Array1::zeros(6);
        let mut eq = Array1::zeros(6);

        eq.mapv_inplace(|_| rand::random());
        for _ in 0..RUN_COUNT {
            let incoming: Vec<_> = particles();
            let outgoing: Vec<_> = particles();
            let interaction = super::InteractionParticles::new(&incoming, &outgoing);

            n.mapv_inplace(|_| rand_n());
            na.mapv_inplace(|_| rand_na());
            eq.mapv_inplace(|_| rand_eq());
            for &idx in &interaction.outgoing_idx {
                eq[idx] = 0.0;
            }

            let result = interaction.fast_interaction_exact(&n, &na, &eq);
            n += &result.dn;
            na += &result.dna;

            let (forward, backward) = interaction.symmetric_prefactor(&n, &eq);
            approx_eq(forward.0 * backward.1, backward.0 * forward.1, 8.0, 1e-15)
                .map_err(|e| format!("Symmetric eqality error: {}", e))?;
            let (forward, backward) = interaction.asymmetric_prefactor(&n, &na, &eq);
            approx_eq(forward.0 * backward.1, backward.0 * forward.1, 8.0, 1e-15)
                .map_err(|e| format!("Asymmetric eqality error: {}", e))?;
        }

        Ok(())
    }
}
