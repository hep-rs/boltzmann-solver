//! Solvers for the Boltzmann equation, or sets of Boltzmann equations

pub mod number_density;
mod options;
mod tableau;

pub use self::options::InitialCondition;
pub(crate) use self::options::StepPrecision;

use crate::particle::Particle;
use ndarray::{array, prelude::*};

/// Contains all the information relevant to a particular model, including
/// masses, widths and couplings.  All these attributes can be dependent on the
/// inverse temperature \\(\beta\\).
pub trait Model {
    /// Instantiate a new instance of the model parameters with the values
    /// calculated at the inverse temperature \\(\beta\\).
    fn new(beta: f64) -> Self;

    /// Return list of particles in the model.
    fn particles(&self) -> &Array1<Particle>;
}

/// An empty model containing no couplings, masses, etc.  This is can be used
/// for very simple implementations of the Boltzmann solver.
pub struct EmptyModel {
    particles: Array1<Particle>,
}

impl Model for EmptyModel {
    fn new(_: f64) -> Self {
        EmptyModel {
            particles: array![],
        }
    }

    fn particles(&self) -> &Array1<Particle> {
        &self.particles
    }
}
