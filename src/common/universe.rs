//! The effects of the Universe's evolution play an important role in the
//! Boltzmann equations.  The [`Universe`] trait describes the evolution of the
//! Universe.

use common::{constants, Statistic};
use utilities::interpolation;

/// Collection of properties which determine the evolution of a Universe.
///
/// Some of these values can be constant, though in general they will change
/// over time.  As these properties often have a clearer explicit dependence on
/// temperature, the inverse temperature \\(\beta\\) is used as the dependent
/// variable (specified in units of inverse gigaelectronvolts).
pub trait Universe {
    /// Return the effective degrees of freedom contributing to the entropy
    /// density of the Universe at the specified inverse temperature.
    fn entropy_dof(&self, beta: f64) -> f64;

    /// Return the Hubble rate at the specified inverse temperature.
    ///
    /// The default implementation assumes the Universe to be radiation
    /// dominated such that
    ///
    /// \\[
    ///    H(\beta) = \sqrt{\frac{\pi\^2}{90}} g_{*}^{1/2}(\beta) \frac{1}{m_{\text{Pl}}} \frac{1}{\beta\^2}.
    /// \\]
    ///
    /// Which is valid provided that the entropy density of the Universe does
    /// not change too rapidly.
    ///
    /// # Warning
    ///
    /// The dominated epoch in the Standard Model of cosmology ends at the
    /// matter--radiation equality, which occurs at an inverse temperature of
    /// \\(\beta \approx 10\^{9}\\) GeV^{-1}.
    fn hubble_rate(&self, beta: f64) -> f64 {
        debug_assert_warn!(
            beta < 1e8,
            "For β > 10⁸ GeV⁻¹, our Universe transitions into the matter
            dominated epoch where this implementation of the Hubble rate no
            longer applies."
        );

        // Prefactor: sqrt(pi^2 / 90) ≅ 0.3311529421932034
        0.3311529421932034 * self.entropy_dof(beta)
            / (constants::REDUCED_PLANCK_MASS * beta.powi(2))
    }
}

pub struct SingleSpecies {
    statistic: Statistic,
    mass: f64,
    dof: f64,
}

impl SingleSpecies {
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
        }
    }
}
