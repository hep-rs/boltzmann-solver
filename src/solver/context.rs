use crate::model::interaction::InteractionParticles;
use ndarray::prelude::*;
use std::{fmt, sync::RwLock};

/// Current context at a particular step in the numerical integration.
#[derive(Debug)]
pub struct Context<'a, M> {
    /// Evaluation step
    pub step: u64,
    /// Step size
    pub step_size: f64,
    /// Inverse temperature in GeV`$^{-1}$`
    pub beta: f64,
    /// Hubble rate, in GeV
    pub hubble_rate: f64,
    /// Normalization factor, which is
    /// ```math
    /// \frac{1}{H \beta n_1}
    /// ```
    /// where `$n_1$` is the number density of a single massless bosonic degree
    /// of freedom, `$H$` is the Hubble rate and `$\beta$` is the inverse
    /// temperature.
    pub normalization: f64,
    /// Equilibrium number densities for the particles
    pub eq: Array1<f64>,
    /// Current number density
    pub n: Array1<f64>,
    /// Current number density asymmetries
    pub na: Array1<f64>,
    /// Model data
    pub model: &'a M,
    /// Keep track of which particles are in equilibrium with each other.
    /// Within each HashMap, the particles are indicated using the keys, and the
    /// sign and multiplicity is stored in the value.
    pub(crate) fast_interactions: Option<RwLock<Vec<InteractionParticles>>>,
}

impl<'a, M> fmt::Display for Context<'a, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Context {{ step: {}, beta: {:e},\n  n: {:e},\n  na: {:e},\n  eq: {:e}\n}}",
            self.step, self.beta, self.n, self.na, self.eq
        )
    }
}

impl<'a, M> Context<'a, M> {
    /// Unwrap the context leaving only the fast interactions.
    ///
    /// If fast interactions are disabled, the result is an empty vectors.
    pub(crate) fn into_fast_interactions(self) -> Vec<InteractionParticles> {
        self.fast_interactions
            .map_or_else(Vec::new, |f| f.into_inner().unwrap())
    }
}
