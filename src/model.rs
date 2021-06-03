//! Model information

mod empty_model;
pub mod interaction;
mod particle;
pub(crate) mod standard_model;

pub use empty_model::EmptyModel;
pub use particle::Particle;
pub use standard_model::StandardModel;

use crate::{
    model::interaction::Interaction,
    solver::Context,
    statistic::{Statistic, Statistics},
};
use ndarray::prelude::*;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "parallel")]
use std::sync::RwLock;
use std::{collections::HashMap, convert::TryFrom, iter};

/// Contains all the information relevant to a particular model, including
/// masses, widths and couplings.  All these attributes can be dependent on the
/// inverse temperature `$\beta$`.
pub trait Model
where
    Self: Sized,
{
    /// Instantiate a new instance of the model at 0 temperature (`$\beta =
    /// \infty$`).
    ///
    /// This is implemented separately so that the computational peculiarities
    /// of dealing with `$\beta = \infty$` can be avoided.
    fn zero() -> Self;

    /// Update the model to be valid at the given inverse temperature `beta`.
    ///
    /// The inverse temperature is always strictly positive, and should never be
    /// infinite.  The zero temperature case should be obtained from
    /// [`Model::zero`].
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
    /// ```math
    /// H(\beta) = \sqrt{\frac{\pi^2}{90}} g_{*}^{1/2}(\beta) \frac{1}{m_{\text{Pl}}} \frac{1}{\beta^2}.
    /// ```
    ///
    /// Which is valid provided that the entropy density of the Universe does
    /// not change too rapidly.
    ///
    /// # Warning
    ///
    /// The dominated epoch in the Standard Model of cosmology ends at the
    /// matter–radiation equality, which occurs at an inverse temperature of
    /// `$\beta \approx 10^{8}$` GeV^{-1}.
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

    /// Return the number of particles in the model.
    fn len_particles(&self) -> usize {
        self.particles().len()
    }

    /// Return a list of particles in the model.
    fn particles(&self) -> &[Particle];

    /// Return a mutable list of particles in the model.
    ///
    /// This is available to allow mutating particles within the model. This
    /// should not change the ordering of the particles as these are assumed to
    /// be fixed throughout the integration.
    fn particles_mut(&mut self) -> &mut [Particle];

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
    /// By default, this implementation defers to
    /// [`static_particle_idx`](Model::static_particle_idx) as models generally
    /// do not need access to particular instances of the model.
    ///
    /// # Errors
    ///
    /// If the particle is not within the model, the name and index should be
    /// returned as an error so that they can be subsequently handled.
    fn particle_idx<S: AsRef<str>>(&self, name: S, i: usize) -> Result<usize, (S, usize)> {
        Self::static_particle_idx(name, i)
    }

    /// A static implementation of [`particle_idx`](Model::particle_idx) which
    /// does not need a reference to an instance of the class (i.e. `&self`).
    ///
    /// # Errors
    ///
    /// If the particle is not within the model, the name and index should be
    /// returned as an error so that they can be subsequently handled.
    ///
    /// # Implementation
    ///
    /// If possible, this method should be implemented over
    /// [`particle_idx`](Model::particle_idx) as models typically will always
    /// have the same particle content (with only the value of parameters
    /// varying); however, if the particles might vary between instances of the
    /// model, this implementation should be
    /// [`unimplemented!()`](std::unimplemented) and
    /// [`particle_idx`](Model::particle_idx) should be implemented instead.
    fn static_particle_idx<S: AsRef<str>>(name: S, i: usize) -> Result<usize, (S, usize)>;

    /// The signed version of [`particle_idx`](Model::particle_idx).
    ///
    /// This is identical to [`particle_idx`](Model::particle_idx) but returns
    /// an `isize` type.
    ///
    /// # Errors
    ///
    /// If the particle is not within the model, the name and index should be
    /// returned as an error so that they can be subsequently handled.
    fn particle_num<S: AsRef<str>>(&self, name: S, i: usize) -> Result<isize, (S, usize)> {
        self.particle_idx(name, i)
            .map(|idx| isize::try_from(idx).expect("Unable to convert particle index to isize"))
    }

    /// A static implementation of [`particle_num`](Model::particle_num) which
    /// does not need a reference to an instance of the class (i.e. `&self`).
    ///
    /// # Errors
    ///
    /// If the particle is not within the model, the name and index should be
    /// returned as an error so that they can be subsequently handled.
    fn static_particle_num<S: AsRef<str>>(name: S, i: usize) -> Result<isize, (S, usize)> {
        Self::static_particle_idx(name, i)
            .map(|idx| isize::try_from(idx).expect("Unable to convert particle index to isize"))
    }

    /// Convert a signed particle number to the corresponding particle name.
    ///
    /// If the particle number is negative and the particle is not its own
    /// antiparticle, a macron (◌̄) will be place over the first character of
    /// the name.
    ///
    /// # Errors
    ///
    /// If the particle is not found within the model, the number is returned is
    /// it can be handled separately.
    fn particle_name(&self, i: isize) -> Result<String, isize> {
        let sign = i.signum();
        let idx = i.abs() as usize;

        self.particles().get(idx).map_or(Err(i), |p| {
            let mut name = p.name.clone();
            if !p.own_antiparticle && sign < 0 {
                name.insert(0, '\u{304}');
            }
            Ok(name)
        })
    }

    /// Return a reference to the matching particle by name.
    ///
    /// # Panics
    ///
    /// Panics if then name or particle is not known within the model.
    fn particle<S: AsRef<str>>(&self, name: S, i: usize) -> &Particle {
        match self.particle_idx(name.as_ref(), i) {
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
        match self.particle_idx(name, i) {
            Ok(idx) => &mut self.particles_mut()[idx],
            Err((name, i)) => {
                log::error!("unknown particle {}{}", name, i);
                panic!("unknown particle {}{}", name, i);
            }
        }
    }

    /// Return a instance of [`Context`] for the model.
    ///
    /// As this is not within the context of solving the Boltzmann equations,
    /// the following attributes have a special value:
    ///
    /// - `step = 0`,
    /// - `step_size = 1.0`,
    /// - `normalization = (hubble_rate * beta * n).recip()`,
    /// - `n` is an array of `1.0`, and
    /// - `na` is an array of `0.0`.
    ///
    /// All other attribute contexts will be as expected.
    fn as_context(&self) -> Context<'_, Self> {
        let beta = self.get_beta();
        let n = Statistic::BoseEinstein.massless_number_density(0.0, beta);
        let hubble_rate = self.hubble_rate(beta);
        let eq: Array1<f64> = self
            .particles()
            .iter()
            .map(|p| p.normalized_number_density(0.0, beta))
            .collect();

        Context {
            step: 0,
            substep: -1,
            beta,
            step_size: 1.0,
            hubble_rate,
            normalization: (hubble_rate * beta * n).recip(),
            eqn: eq.clone(),
            eq,
            n: Array1::ones(self.len_particles()),
            na: Array1::zeros(self.len_particles()),
            model: &self,
            fast_interactions: None,
        }
    }
}

/// Supertrait for [`Model`] for the handling of interactions.
#[allow(clippy::module_name_repetitions)]
#[cfg(not(feature = "parallel"))]
pub trait ModelInteractions
where
    Self: Model,
{
    /// The underlying interaction type.
    ///
    /// If only three-particle interactions are used, then
    /// [`ThreeParticle`](crate::model::interaction::ThreeParticle) would be
    /// appropriate here; however, if a combination of interactions might be
    /// used then it is necessary to us `Box<dyn Interaction<Self>>`.
    type Item: Interaction<Self>;

    /// Return an iterator over all interactions in the model.
    fn interactions(&self) -> &[Self::Item];

    /// Calculate the widths of all particles.
    ///
    /// This computes the possible decays of all particles given the
    /// interactions specified within the model in order to compute the
    /// particle's final width.
    ///
    /// The information is stored in each particle's [`Particle::width`] and the
    /// partial widths are stored in [`Particle::decays`].
    fn update_widths(&mut self) {
        let mut widths: Vec<_> = iter::repeat_with(|| (0.0, HashMap::new()))
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

/// Supertrait for [`Model`] for the handling of interactions.
#[cfg(feature = "parallel")]
#[allow(clippy::module_name_repetitions)]
pub trait ModelInteractions
where
    Self: Model + Sync,
{
    /// The underlying interaction type.
    ///
    /// If only three-particle interactions are used, then
    /// [`ThreeParticle`](crate::model::interaction::ThreeParticle) would be
    /// appropriate here; however, if a combination of interactions might be
    /// used then it is necessary to us `Box<dyn Interaction<Self>>`.
    type Item: Interaction<Self> + Sync;

    /// Return an iterator over all interactions in the model.
    fn interactions(&self) -> &[Self::Item];

    /// Calculate the widths of all particles.
    ///
    /// This computes the possible decays of all particles given the
    /// interactions specified within the model in order to compute the
    /// particle's final width.
    ///
    /// The information is stored in each particle's [`Particle::width`] and the
    /// partial widths are stored in [`Particle::decays`].
    fn update_widths(&mut self) {
        let widths: RwLock<Vec<_>> = RwLock::new(
            iter::repeat_with(|| (0.0, HashMap::new()))
                .take(self.particles().len())
                .collect(),
        );

        let c = self.as_context();

        self.interactions().par_iter().for_each(|interaction| {
            if let Some(partial_width) = interaction.width(&c) {
                let parent_idx = partial_width.parent_idx();
                let mut widths = widths
                    .write()
                    .expect("cannot get write access of widths behind RwLock");
                widths[parent_idx].0 += partial_width.width;
                widths[parent_idx]
                    .1
                    .insert(partial_width.daughters, partial_width.width);
            }
        });

        let ptcl = self.particles_mut();
        for (i, (width, hm)) in widths
            .into_inner()
            .expect("cannot unwrap RwLock protecting widths")
            .into_iter()
            .enumerate()
        {
            let p = &mut ptcl[i];
            p.set_width(width);
            p.decays = hm;
        }
    }
}
