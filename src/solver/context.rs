use crate::model::Model;
use ndarray::prelude::*;
// use std::sync::RwLock;

/// Current context at a particular step in the numerical integration.
#[derive(Debug)]
pub struct Context<'a, M: Model> {
    /// Evaluation step
    pub step: u64,
    /// Step size
    pub step_size: f64,
    /// Inverse temperature in GeV\\(^{-1}\\)
    pub beta: f64,
    /// Hubble rate, in GeV
    pub hubble_rate: f64,
    /// Normalization factor, which is
    /// \\begin{equation}
    ///   \frac{1}{H \beta n_1}
    /// \\end{equation}
    /// where \\(n_1\\) is the number density of a single massless bosonic
    /// degree of freedom, \\(H\\) is the Hubble rate and \\(\beta\\) is the
    /// inverse temperature.
    pub normalization: f64,
    /// Equilibrium number densities for the particles
    pub eq: Array1<f64>,
    /// Current number density
    pub n: Array1<f64>,
    /// Current number density asymmetries
    pub na: Array1<f64>,
    /// Model data
    pub model: &'a M,
}
