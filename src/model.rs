//! Model information

pub(crate) mod data;
pub mod interaction;
mod particle;
mod standard_model;

pub use particle::Particle;
pub use standard_model::StandardModel;

use crate::{
    model::interaction::Interaction,
    solver::Context,
    statistic::{Statistic, Statistics},
};
use ndarray::prelude::*;
use std::collections::HashMap;

/// Contains all the information relevant to a particular model, including
/// masses, widths and couplings.  All these attributes can be dependent on the
/// inverse temperature \\(\beta\\).
pub trait Model: Sized {
    /// Instantiate a new instance of the model at 0 temperature (\\(\beta =
    /// \infty\\)).
    ///
    /// This is implemented separately so that the computational peculiarities
    /// of dealing with \\(\beta = \infty\\) can be avoided.
    fn zero() -> Self;

    /// Update the model to be valid at the given inverse temperature `beta`.
    ///
    /// Beta is always strictly positive, and should never be infinite.  The
    /// zero temperature case should be obtained from [`Model::zero`].
    fn set_beta(&mut self, beta: f64);

    /// Return the current value of beta for the model
    fn get_beta(&self) -> f64;

    /// Return the effective degrees of freedom contributing to the entropy
    /// density of the Universe at the specified inverse temperature.
    fn entropy_dof(&self, beta: f64) -> f64;

    /// Return the Hubble rate at the specified inverse temperature.
    ///
    /// The default implementation assumes the Universe to be radiation
    /// dominated such that
    ///
    /// \\begin{equation}
    ///    H(\beta) = \sqrt{\frac{\pi^2}{90}} g_{*}^{1/2}(\beta) \frac{1}{m_{\text{Pl}}} \frac{1}{\beta^2}.
    /// \\end{equation}
    ///
    /// Which is valid provided that the entropy density of the Universe does
    /// not change too rapidly.
    ///
    /// # Warning
    ///
    /// The dominated epoch in the Standard Model of cosmology ends at the
    /// matter–radiation equality, which occurs at an inverse temperature of
    /// \\(\beta \approx 10^{9}\\) GeV^{-1}.
    fn hubble_rate(&self, beta: f64) -> f64 {
        debug_assert_warn!(
            beta > 1e8,
            "For β > 10⁸ GeV⁻¹, our Universe transitions into the matter
            dominated epoch where this implementation of the Hubble rate no
            longer applies."
        );

        // Prefactor: sqrt(pi^2 / 90) / REDUCED_PLANCK_MASS ≅ 1.35977e-19
        1.35977e-19 * self.entropy_dof(beta).sqrt() * beta.powi(-2)
    }

    /// Return a list of particles in the model.
    fn particles(&self) -> &Vec<Particle>;

    /// Return a mutable list of particles in the model.
    fn particles_mut(&mut self) -> &mut Vec<Particle>;

    /// Return the index corresponding to a particle's name and generation
    /// index.
    ///
    /// If the particle has no index, the `i` argument should be ignored.
    /// Indices must be 0-indexed, so that generational indices are `[0, 1, 2]`.
    ///
    /// Note that since particles are distinguished from anti-particles based on
    /// the sign of the integer, `particle_idx` should not return `0` unless
    /// that particle is its own anti-particle (so the distinction is
    /// irrelevant).
    ///
    /// If the particle is not within the model, the name and index should be
    /// returned as an error so that they can be subsequently handled.
    // fn particle_idx(name: &str, i: usize) -> Result<usize, (&str, usize)>;
    fn particle_idx<S: AsRef<str>>(name: S, i: usize) -> Result<usize, (S, usize)>;

    /// Return a reference to the matching particle by name.
    ///
    /// # Panics
    ///
    /// Panics if then name or particle is not known within the model.
    fn particle<S: AsRef<str>>(&self, name: S, i: usize) -> &Particle {
        match Self::particle_idx(name.as_ref(), i) {
            Ok(idx) => &self.particles()[idx],
            Err((name, i)) => {
                log::error!("unknown particle {}{}", name, i);
                panic!("unknown particle {}{}", name, i);
            }
        }
    }

    /// Return a mutable reference to the matching particle by name.
    ///
    /// # Panics
    ///
    /// Panics if then name or particle is not known within the model.
    fn particle_mut(&mut self, name: &str, i: usize) -> &mut Particle {
        match Self::particle_idx(name, i) {
            Ok(idx) => &mut self.particles_mut()[idx],
            Err((name, i)) => {
                log::error!("unknown particle {}{}", name, i);
                panic!("unknown particle {}{}", name, i);
            }
        }
    }

    /// Return a list of interactions in the model.
    fn interactions(&self) -> &Vec<Box<dyn Interaction<Self> + Sync>>;

    /// Return a instance of [`Context`] for the model.
    ///
    /// As this is not within the context of solving the Boltzmann equations,
    /// the following attributes have a special value:
    ///
    /// - `step = 0`,
    /// - `step_size = 1.0`,
    /// - `normalization = (hubble_rate * beta * n).recip()`, and
    /// - `eq`, `n`, `na` are empty arrays.
    ///
    /// All other attribute contexts will be as expected.
    fn as_context(&self) -> Context<'_, Self> {
        let beta = self.get_beta();
        let n = Statistic::BoseEinstein.massless_number_density(0.0, beta);
        let hubble_rate = self.hubble_rate(beta);

        Context {
            step: 0,
            step_size: 1.0,
            beta,
            hubble_rate,
            normalization: (hubble_rate * beta * n).recip(),
            eq: Array1::zeros(0),
            n: Array1::zeros(0),
            na: Array1::zeros(0),
            model: &self,
        }
    }

    /// Calculate the widths of all particles.
    ///
    /// This computes the possible decays of all particles given the
    /// interactions specified within the model in order to compute the
    /// particle's final width.
    ///
    /// The information is stored in each particle's `p.width` and the partial
    /// widths are stored in `p.decays`.
    fn update_widths(&mut self) {
        let mut widths: Vec<_> = std::iter::repeat_with(|| (0.0, HashMap::new()))
            .take(self.particles().len())
            .collect();

        let c = self.as_context();
        for interaction in self.interactions() {
            if let Some(partial_width) = interaction.width(&c) {
                let parent_idx = partial_width.parent_idx();
                widths[parent_idx].0 += partial_width.width;
                widths[parent_idx]
                    .1
                    .insert(partial_width.daughters, partial_width.width);
            }
        }

        for (i, (width, hm)) in widths.into_iter().enumerate() {
            let p = &mut self.particles_mut()[i];
            p.set_width(width);
            p.decays = hm;
        }
    }
}
