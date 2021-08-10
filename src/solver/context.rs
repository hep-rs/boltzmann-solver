use crate::model::interaction::InteractionParticles;
use ndarray::prelude::*;
use std::{collections::HashSet, fmt, sync::RwLock};

/// Current context at a particular step in the numerical integration.
#[derive(Debug)]
pub struct Context<'a, M> {
    /// Evaluation step.
    pub step: usize,
    /// The substep apply within Runge-Kutta integration and are numbered
    /// starting from 0.  If no substep is applicable, a negative number
    /// (typically -1) is used.
    pub substep: i8,
    /// Inverse temperature in GeV`$^{-1}$`
    ///
    /// Note that the value of beta at the next integration step will in general
    /// not be `beta + step_size` as the values of beta for substep fall in
    /// between the current and the next values of beta.
    pub beta: f64,
    /// Step size.
    ///
    /// Note that the value of beta at the next integration step will in general
    /// not be `beta + step_size` as the values of beta for substep fall in
    /// between the current and the next values of beta.
    pub step_size: f64,
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
    /// Equilibrium number densities for the particles at the current value of
    /// beta.
    pub eq: Array1<f64>,
    /// Equilibrium number densities for the particles at the next integration
    /// step.
    pub eqn: Array1<f64>,
    /// Current number density
    pub n: Array1<f64>,
    /// Current number density asymmetries
    pub na: Array1<f64>,
    /// Model data
    pub model: &'a M,
    /// Keep track of which particles are in equilibrium with each other.
    /// Within each HashMap, the particles are indicated using the keys, and the
    /// sign and multiplicity is stored in the value.
    pub(crate) fast_interactions: Option<RwLock<HashSet<InteractionParticles>>>,
    /// List of particles which are forced in equilibrium.
    ///
    /// If the context
    pub in_equilibrium: &'a [usize],
    /// List of particles which are forced to never develop any asymmetry.
    pub no_asymmetry: &'a [usize],
}

impl<'a, M> fmt::Display for Context<'a, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Context {{ step: {}, substep: {}, beta: {:e},\n  n: {:e},\n  na: {:e},\n  eq: {:e}\n  eqn:{:e}\n}}",
            self.step, self.substep, self.beta, self.n, self.na, self.eq, self.eqn,
        )
    }
}

impl<'a, M> Context<'a, M> {
    /// Unwrap the context leaving only the fast interactions.
    ///
    /// If fast interactions are disabled, the result is an empty vectors.
    pub(crate) fn into_fast_interactions(self) -> Option<RwLock<HashSet<InteractionParticles>>> {
        self.fast_interactions
    }
}
