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
}

impl InteractionParticles {
    /// Create a new set of interaction particles.  The absolute value of the
    /// numbers indicate the corresponding index for the particle, with the sign
    /// indicating whether it is a particle or anti-particle involved.
    #[must_use]
    pub fn new(incoming: &[isize], outgoing: &[isize]) -> Self {
        Self {
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
        }
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
    #[must_use]
    pub fn particle_counts(&self) -> HashMap<usize, (f64, f64)> {
        let mut counts = HashMap::with_capacity(self.incoming_len() + self.outgoing_len());

        for (&p, &a) in self.iter_incoming() {
            let entry = counts.entry(p).or_insert((0.0, 0.0));
            (*entry).0 -= 1.0;
            (*entry).1 -= a;
        }
        for (&p, &a) in self.iter_outgoing() {
            let entry = counts.entry(p).or_insert((0.0, 0.0));
            (*entry).0 += 1.0;
            (*entry).1 += a;
        }

        counts
    }

    /// Aggregate the incoming particles to take into account multiplicities.
    ///
    /// The two counts differ for the symmetric and asymmetric rates because the
    /// symmetric rate does not take into account whether it is a particle or
    /// antiparticle as both it and its CP-conjugate rate are assumed to be
    /// equal.  This is evidently not true for the asymmetric rate as it
    /// fundamentally assumes that the CP rate is different.
    #[must_use]
    pub fn incoming_particle_counts(&self) -> HashMap<usize, (f64, f64)> {
        let mut counts = HashMap::with_capacity(self.incoming_len());

        for (&p, &a) in self.iter_incoming() {
            let entry = counts.entry(p).or_insert((0.0, 0.0));
            (*entry).0 += 1.0;
            (*entry).1 += a;
        }

        counts
    }

    /// Aggregate the outgoing particles to take into account multiplicities.
    ///
    /// The two counts differ for the symmetric and asymmetric rates because the
    /// symmetric rate does not take into account whether it is a particle or
    /// antiparticle as both it and its CP-conjugate rate are assumed to be
    /// equal.  This is evidently not true for the asymmetric rate as it
    /// fundamentally assumes that the CP rate is different.
    #[must_use]
    pub fn outgoing_particle_counts(&self) -> HashMap<usize, (f64, f64)> {
        let mut counts = HashMap::with_capacity(self.outgoing_len());

        for (&p, &a) in self.iter_outgoing() {
            let entry = counts.entry(p).or_insert((0.0, 0.0));
            (*entry).0 += 1.0;
            (*entry).1 += a;
        }

        counts
    }

    /// Computes the change in number density required to establish equilibrium,
    /// up to linear order.
    ///
    /// Given the interaction `$X \leftrightarrow Y$`, this computes the value
    /// of `$\delta$` such that
    ///
    /// ```math
    ///   \prod_{i \in X} \frac{n_i - \delta}{n_i^{(0)}}
    /// = \prod_{i \in Y} \frac{n_i + \delta}{n_i^{(0)}}
    /// ```
    ///
    /// holds true, up to linear order in `$\delta$`.  The exact solution can be
    /// obtained by recursively applying the change until convergence is
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
    #[must_use]
    pub fn symmetric_delta(&self, n: &Array1<f64>, eq: &Array1<f64>) -> f64 {
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
                let incoming_zeros = self.incoming_idx.iter().filter(|&&p| n[p] == 0.0).count();
                let outgoing_zeros = self.outgoing_idx.iter().filter(|&&p| n[p] == 0.0).count();
                let counts = self.particle_counts();

                match incoming_zeros.cmp(&outgoing_zeros) {
                    Ordering::Less => self
                        .incoming_idx
                        .iter()
                        .map(|&p| n[p] / counts[&p].0)
                        .min_by(|x, y| x.partial_cmp(y).unwrap())
                        .unwrap(),
                    Ordering::Equal => {
                        let delta_forward = self
                            .incoming_idx
                            .iter()
                            .map(|&i| n[i])
                            .min_by(|x, y| x.partial_cmp(y).unwrap())
                            .unwrap();
                        let delta_backward = self
                            .outgoing_idx
                            .iter()
                            .map(|&i| n[i])
                            .min_by(|x, y| x.partial_cmp(y).unwrap())
                            .unwrap();

                        if delta_forward > delta_backward {
                            delta_forward
                        } else {
                            -delta_backward
                        }
                    }
                    Ordering::Greater => -self
                        .outgoing_idx
                        .iter()
                        .map(|&p| n[p] / -counts[&p].0)
                        .min_by(|x, y| x.partial_cmp(y).unwrap())
                        .unwrap(),
                }
            }
            (true, false) => {
                let counts = self.incoming_particle_counts();

                self.incoming_idx
                    .iter()
                    .map(|&p| n[p] / counts[&p].0)
                    .min_by(|x, y| x.partial_cmp(y).unwrap())
                    .unwrap()
            }
            (false, true) => {
                let counts = self.outgoing_particle_counts();

                -self
                    .outgoing_idx
                    .iter()
                    .map(|&p| n[p] / counts[&p].0)
                    .min_by(|x, y| x.partial_cmp(y).unwrap())
                    .unwrap()
            }
            (false, false) => {
                let numerator =
                    incoming_n_prod * outgoing_eq_prod - outgoing_n_prod * incoming_eq_prod;

                if numerator == 0.0 {
                    0.0
                } else {
                    let denominator = outgoing_eq_prod
                        * (0..self.incoming_idx.len())
                            .map(|i| self.incoming_product_except(n, i))
                            .sum::<f64>()
                        + incoming_eq_prod
                            * (0..self.outgoing_idx.len())
                                .map(|i| self.outgoing_product_except(n, i))
                                .sum::<f64>();

                    numerator / denominator
                }
            }
        }
    }

    /// Given the interaction `$X \leftrightarrow Y$`, this computes the alue of
    /// `$\delta$` such that
    ///
    /// ```math
    /// \frac{
    ///   \sum_{i \in X} (\Delta_i - \delta)
    ///   \prod_{\substack{j \in X \\ j \neq i} n_i}
    /// }{
    ///   \prod_{i \in X} n_i^{(0)}
    /// }
    /// =
    /// \frac{
    ///   \sum_{i \in Y} (\Delta_i - \delta)
    ///   \prod_{\substack{j \in Y \\ j \neq i} n_i}
    /// }{
    ///   \prod_{i \in Y} n_i^{(0)}
    /// }
    /// ```
    ///
    /// holds true.  Unlike the symmetric case, this is always linear in
    /// `$\delta$` and thus the result is exact.
    #[must_use]
    #[allow(clippy::too_many_lines)]
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
                    Ordering::Less => {
                        let (numerator, denominator) = self.incoming_idx.iter().enumerate().fold(
                            (0.0, 0.0),
                            |(numerator, denominator), (i, &p)| {
                                let product = self.incoming_product_except(n, i);
                                (numerator + na[p] * product, denominator + product)
                            },
                        );

                        numerator / denominator
                    }
                    Ordering::Equal => 0.0,
                    Ordering::Greater => {
                        let (numerator, denominator) = self.outgoing_idx.iter().enumerate().fold(
                            (0.0, 0.0),
                            |(numerator, denominator), (i, &p)| {
                                let product = self.outgoing_product_except(n, i);
                                (numerator + na[p] * product, denominator + product)
                            },
                        );

                        -numerator / denominator
                    }
                }
            }
            (true, false) => {
                let (numerator, denominator) = self.incoming_idx.iter().enumerate().fold(
                    (0.0, 0.0),
                    |(numerator, denominator), (i, &p)| {
                        let product = self.incoming_product_except(n, i);
                        (numerator + na[p] * product, denominator + product)
                    },
                );

                numerator / denominator
            }
            (false, true) => {
                let (numerator, denominator) = self.outgoing_idx.iter().enumerate().fold(
                    (0.0, 0.0),
                    |(numerator, denominator), (i, &p)| {
                        let product = self.outgoing_product_except(n, i);
                        (numerator + na[p] * product, denominator + product)
                    },
                );

                -numerator / denominator
            }
            (false, false) => {
                let mut numerator = 0.0;
                let mut denominator = 0.0;

                for (i, &p) in self.incoming_idx.iter().enumerate() {
                    let product = self.incoming_product_except(n, i);
                    numerator += na[p] * product * outgoing_eq_prod;
                    denominator += product * outgoing_eq_prod;
                }
                for (i, &p) in self.outgoing_idx.iter().enumerate() {
                    let product = self.outgoing_product_except(n, i);
                    numerator -= na[p] * product * incoming_eq_prod;
                    denominator += product * incoming_eq_prod;
                }

                numerator / denominator
            }
        }
    }

    /// For a fast interaction, compute the change to each particle species such
    /// that equilibrium is established.
    ///
    /// This functions returns the tuple of arrays `(dn, dna)` where the first
    /// contains the change to the number densities and the second contains the
    /// change to the number density asymmetries.
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

        for &p in &self.incoming_idx {
            result.dn[p] -= result.symmetric_delta;
            result.dna[p] -= result.asymmetric_delta;
        }
        for &p in &self.outgoing_idx {
            result.dn[p] += result.symmetric_delta;
            result.dna[p] += result.asymmetric_delta;
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
            "The number density, number density asymmetry and equilibrium number density array shapes must be all the same"
        );

        // Successively calculate the local change and apply the change to the
        // number densities.
        let mut delta = 0.0;
        let mut n = n.clone();
        loop {
            let di = self.symmetric_delta(&n, eq);
            delta += di;

            for &p in &self.incoming_idx {
                n[p] -= di;
            }
            for &p in &self.outgoing_idx {
                n[p] += di;
            }

            if di.abs() < 1e-10 {
                break;
            }
        }

        let mut result = FastInteractionResult::zero(n.dim());
        result.symmetric_delta = delta;
        result.asymmetric_delta = self.asymmetric_delta(&n, na, eq);

        for &p in &self.incoming_idx {
            result.dn[p] -= result.symmetric_delta;
            result.dna[p] -= result.asymmetric_delta;
        }
        for &p in &self.outgoing_idx {
            result.dn[p] += result.symmetric_delta;
            result.dna[p] += result.asymmetric_delta;
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
    fn asymmetry(&self, c: &Context<M>, real: bool) -> Option<f64> {
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
        let particles = self.particles();

        // When computing the ratio of number density to equilibrium number
        // density, a NaN should only occur from `0.0 / 0.0`, in which case the
        // correct value ought to be 0 as there are no actual particles to decay
        // in the numerator.
        let mut forward_prefactor = particles
            .incoming_idx
            .iter()
            .map(|&p| checked_div(c.n[p], c.eq[p]))
            .product::<f64>();
        let mut backward_prefactor = particles
            .outgoing_idx
            .iter()
            .map(|&p| checked_div(c.n[p], c.eq[p]))
            .product::<f64>();

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
        let particles = self.particles();

        let (forward_numerator, forward_denominator) =
            particles
                .iter_incoming()
                .enumerate()
                .fold((0.0, 1.0), |(n, d), (i, (&p, &a))| {
                    (
                        n + a * c.na[p] * particles.incoming_product_except(&c.n, i),
                        d * c.eq[p],
                    )
                });
        let forward = checked_div(forward_numerator, forward_denominator);

        let (backward_numerator, backward_denominator) = particles
            .iter_outgoing()
            .enumerate()
            .fold((0.0, 1.0), |(n, d), (i, (&p, &a))| {
                (
                    n + a * c.na[p] * particles.outgoing_product_except(&c.n, i),
                    d * c.eq[p],
                )
            });
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
        let asymmetry = self.asymmetry(c, false).unwrap_or(0.0);

        // If both rates are 0, there's no need to adjust it to the particles'
        // number densities.
        if gamma == 0.0 && asymmetry == 0.0 {
            return None;
        }

        debug_assert!(!gamma.is_nan(), "Interaction rate is NaN");
        debug_assert!(!asymmetry.is_nan(), "Asymmetric interaction rate is NaN");

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

#[cfg(test)]
mod tests {
    use crate::utilities::test::approx_eq;
    use ndarray::prelude::*;
    use std::{error, f64};

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
        let mut n = Array1::from_shape_simple_fn(5, rand::random);
        let mut na = Array1::from_shape_simple_fn(5, rand::random);
        let mut eq = Array1::from_shape_simple_fn(5, rand::random);

        for _ in 0..10_000 {
            n.mapv_inplace(|_| rand::random());
            na.mapv_inplace(|_| rand::random());

            let result = interaction.fast_interaction_exact(&n, &na, &eq);
            n += &result.dn;
            na += &result.dna;

            approx_eq(
                super::checked_div(n[1], eq[1]),
                super::checked_div(n[2], eq[2]) * super::checked_div(n[3], eq[3]),
                8.0,
                0.0,
            )
            .or_else(|e| {
                if n.iter().any(|&ni| ni == 0.0) {
                    Ok(())
                } else {
                    Err(e)
                }
            })?;

            approx_eq(
                super::checked_div(na[1], eq[1]),
                super::checked_div(na[2] * n[3] + na[3] * n[2], eq[2] * eq[3]),
                8.0,
                1e-14,
            )?;
        }

        // 0 equilibrium on LHS
        eq[1] = 0.0;
        for _ in 0..10_000 {
            n.mapv_inplace(|_| rand::random());
            na.mapv_inplace(|_| rand::random());

            let result = interaction.fast_interaction_exact(&n, &na, &eq);
            n += &result.dn;
            na += &result.dna;

            approx_eq(n[1], 0.0, 8.0, 0.0)?;
            approx_eq(na[1], 0.0, 8.0, 0.0)?;
        }

        // 0 equilibrium on RHS
        eq.mapv_inplace(|_| rand::random());
        eq[2] = 0.0;
        eq[3] = 0.0;
        for _ in 0..10_000 {
            n.mapv_inplace(|_| rand::random());
            na.mapv_inplace(|_| rand::random());

            let result = interaction.fast_interaction_exact(&n, &na, &eq);
            n += &result.dn;
            na += &result.dna;

            approx_eq(n[2], 0.0, 8.0, 0.0).or_else(|_| approx_eq(n[3], 0.0, 8.0, 0.0))?;
            approx_eq(na[2] * n[3] + na[3] * n[2], 0.0, 8.0, 1e-14)?;
        }

        Ok(())
    }

    #[test]
    fn fast_interaction_2_2() -> Result<(), Box<dyn error::Error>> {
        let interaction = super::InteractionParticles::new(&[1, 2], &[3, 4]);
        let mut n = Array1::from_shape_simple_fn(5, rand::random);
        let mut na = Array1::from_shape_simple_fn(5, rand::random);
        let mut eq = Array1::from_shape_simple_fn(5, rand::random);

        for _ in 0..10_000 {
            n.mapv_inplace(|_| rand::random());
            na.mapv_inplace(|_| rand::random());

            let result = interaction.fast_interaction_exact(&n, &na, &eq);
            n += &result.dn;
            na += &result.dna;

            approx_eq(
                super::checked_div(n[1], eq[1]) * super::checked_div(n[2], eq[2]),
                super::checked_div(n[3], eq[3]) * super::checked_div(n[4], eq[4]),
                8.0,
                0.0,
            )
            .or_else(|e| {
                if n.iter().any(|&ni| ni == 0.0) {
                    Ok(())
                } else {
                    Err(e)
                }
            })?;

            approx_eq(
                super::checked_div(na[1] * n[2] + na[2] * n[1], eq[1] * eq[2]),
                super::checked_div(na[3] * n[4] + na[4] * n[3], eq[3] * eq[4]),
                8.0,
                0.0,
            )?;
        }

        // 0 equilibrium on LHS
        eq[1] = 0.0;
        eq[2] = 0.0;
        for _ in 0..10_000 {
            n.mapv_inplace(|_| rand::random());
            na.mapv_inplace(|_| rand::random());

            let result = interaction.fast_interaction_exact(&n, &na, &eq);
            n += &result.dn;
            na += &result.dna;

            approx_eq(n[1], 0.0, 8.0, 0.0).or_else(|_| approx_eq(n[2], 0.0, 8.0, 0.0))?;
            approx_eq(na[1] * n[2] + na[2] * n[1], 0.0, 8.0, 1e-14)?;
        }

        // 0 equilibrium on RHS
        eq.mapv_inplace(|_| rand::random());
        eq[3] = 0.0;
        eq[4] = 0.0;
        for _ in 0..10_000 {
            n.mapv_inplace(|_| rand::random());
            na.mapv_inplace(|_| rand::random());

            let result = interaction.fast_interaction_exact(&n, &na, &eq);
            n += &result.dn;
            na += &result.dna;

            approx_eq(n[3], 0.0, 8.0, 0.0).or_else(|_| approx_eq(n[4], 0.0, 8.0, 0.0))?;
            approx_eq(na[3] * n[4] + na[4] * n[3], 0.0, 8.0, 1e-14)?;
        }

        Ok(())
    }

    #[test]
    fn fast_interaction_1_3() -> Result<(), Box<dyn error::Error>> {
        let interaction = super::InteractionParticles::new(&[1], &[2, 3, 4]);
        let mut n = Array1::from_shape_simple_fn(5, rand::random);
        let mut na = Array1::from_shape_simple_fn(5, rand::random);
        let mut eq = Array1::from_shape_simple_fn(5, rand::random);

        for _ in 0..10_000 {
            n.mapv_inplace(|_| rand::random());
            na.mapv_inplace(|_| rand::random());

            let result = interaction.fast_interaction_exact(&n, &na, &eq);
            n += &result.dn;
            na += &result.dna;

            approx_eq(
                super::checked_div(n[1], eq[1]),
                super::checked_div(n[2], eq[2])
                    * super::checked_div(n[3], eq[3])
                    * super::checked_div(n[4], eq[4]),
                8.0,
                0.0,
            )
            .or_else(|e| {
                if n.iter().any(|&ni| ni == 0.0) {
                    Ok(())
                } else {
                    Err(e)
                }
            })?;

            approx_eq(
                super::checked_div(na[1], eq[1]),
                super::checked_div(
                    na[2] * n[3] * n[4] + na[3] * n[2] * n[4] + na[4] * n[2] * n[3],
                    eq[2] * eq[3] * eq[4],
                ),
                8.0,
                0.0,
            )
            .or_else(|e| {
                if n.iter().any(|&ni| ni == 0.0) {
                    Ok(())
                } else {
                    Err(e)
                }
            })?;
        }

        // 0 equilibrium on LHS
        eq[1] = 0.0;
        for _ in 0..10_000 {
            n.mapv_inplace(|_| rand::random());
            na.mapv_inplace(|_| rand::random());

            let result = interaction.fast_interaction_exact(&n, &na, &eq);
            n += &result.dn;
            na += &result.dna;

            approx_eq(n[1], 0.0, 8.0, 0.0)?;
            approx_eq(na[1], 0.0, 8.0, 0.0)?;
        }

        // 0 equilibrium on RHS
        eq.mapv_inplace(|_| rand::random());
        eq[2] = 0.0;
        eq[3] = 0.0;
        eq[4] = 0.0;
        for _ in 0..10_000 {
            n.mapv_inplace(|_| rand::random());
            na.mapv_inplace(|_| rand::random());

            let result = interaction.fast_interaction_exact(&n, &na, &eq);
            n += &result.dn;
            na += &result.dna;

            approx_eq(n[2], 0.0, 8.0, 0.0)
                .or_else(|_| approx_eq(n[3], 0.0, 8.0, 0.0))
                .or_else(|_| approx_eq(n[4], 0.0, 8.0, 0.0))?;
            approx_eq(
                na[2] * n[3] * n[4] + na[3] * n[2] * n[4] + na[4] * n[2] * n[3],
                0.0,
                8.0,
                1e-14,
            )?;
        }

        Ok(())
    }
}
