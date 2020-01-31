//! The effects of the Universe's evolution play an important role in the
//! Boltzmann equation.  This information is provided by implementations of the
//! [`Universe`](universe::Universe) trait.

mod standard_model;

use crate::{constants, statistic::Statistic};
use special_functions::approximations::interpolation;

/// Contribution from a single particle in the Universe.
///
/// The particle can be either a fermion or boson with arbitrary (non-integral)
/// degrees of freedom.  The degrees of freedom are (typically):
///
/// - 2 for a fermion;
/// - 1 for a real scalar;
/// - 2 for a complex scalar;
/// - 2 for a massless boson;
/// - 3 for a massive boson (dependent on the gauge).
///
/// The contribution of the particle to the entropy of the Universe is based on
/// the data from [`arXiv:1609.04979`](https://arxiv.org/abs/1609.04979).
///
/// Note that this always assumes the particle to be free and does not work if
/// the particle forms bound states below a particular energy scale (such as
/// with coloured particles in the Standard Model).
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct SingleSpecies {
    statistic: Statistic,
    mass: f64,
    dof: f64,
}

impl SingleSpecies {
    /// Create a new particle with the specified statistic, mass and degrees of
    /// freedom.
    pub fn new(statistic: Statistic, mass: f64, dof: f64) -> Self {
        SingleSpecies {
            statistic,
            mass,
            dof,
        }
    }
}

impl Universe for SingleSpecies {
    fn entropy_dof(&self, beta: f64) -> f64 {
        match self.statistic {
            Statistic::FermiDirac => {
                self.dof
                    * interpolation::linear(&constants::FERMION_GSTAR, (self.mass * beta).ln())
                        .exp()
            }
            Statistic::BoseEinstein => {
                self.dof
                    * interpolation::linear(&constants::BOSON_GSTAR, (self.mass * beta).ln()).exp()
            }
            Statistic::MaxwellBoltzmann => unimplemented!(),
            Statistic::MaxwellJuttner => unimplemented!(),
        }
    }
}

impl std::fmt::Display for SingleSpecies {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SingleSpecies {{ {}, mass: {}, dof: {} }}",
            self.statistic, self.mass, self.dof
        )
    }
}
