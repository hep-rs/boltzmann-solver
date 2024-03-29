use crate::model::{particle::SCALAR, Model, ModelInteractions, ParticleData};
use crate::prelude::Interaction;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::{f64, fmt};

#[cfg(test)]
mod tests;

/// Empty model with no particles or interactions.
///
/// As the `0` index is reserved for a particle that is it's own antiparticle,
/// this is the only 'particle' included within the model.
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Empty {
    /// Inverse temperature in GeV`$^{-1}$`
    beta: f64,

    /// Particles
    particles: Vec<ParticleData>,

    /// Interactions between the particles.
    #[cfg(feature = "parallel")]
    #[cfg_attr(feature = "serde", serde(skip))]
    interactions: Vec<Box<dyn Interaction<Self> + Sync>>,
    #[cfg(not(feature = "parallel"))]
    #[cfg_attr(feature = "serde", serde(skip))]
    interactions: Vec<Box<dyn Interaction<Self>>>,
}

impl fmt::Debug for Empty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EmptyModel {{ beta: {}, particles: {:?}, interactions: <length {}> }}",
            self.beta,
            self.particles,
            self.interactions.len()
        )
    }
}

impl Default for Empty {
    fn default() -> Self {
        Self::zero()
    }
}

impl Empty {
    /// Push the specified particle to the end of the list of particles within
    /// the model.
    ///
    /// Note that the particle comes with a default `none` particle at index 0,
    /// thus the index of particles added with this start from 1.
    pub fn push_particle(&mut self, p: ParticleData) {
        self.particles.push(p);
    }

    /// Extend the particles within the model given the specified iterator.
    ///
    /// Note that the particle comes with a default `none` particle at index 0,
    /// thus the index of particles added with this start from 1.
    pub fn extend_particles<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = ParticleData>,
    {
        self.particles.extend(iter);
    }

    /// Push the specified interaction into the model.
    #[cfg(feature = "parallel")]
    pub fn push_interaction<I>(&mut self, interaction: I)
    where
        I: Interaction<Self> + Sync + 'static,
    {
        self.interactions.push(Box::new(interaction));
    }

    /// Push the specified interaction into the model.
    #[cfg(not(feature = "parallel"))]
    pub fn push_interaction<I>(&mut self, interaction: I)
    where
        I: Interaction<Self> + 'static,
    {
        self.interactions.push(Box::new(interaction));
    }

    /// Extend the particles within the model given the specified iterator.
    ///
    /// Note that the particle comes with a default `none` particle at index 0,
    /// thus the index of particles added with this start from 1.
    #[cfg(feature = "parallel")]
    pub fn extend_interactions<I>(&mut self, iter: I)
    where
        I: IntoIterator,
        <I as IntoIterator>::Item: Interaction<Self> + Sync + 'static,
    {
        self.interactions.extend(
            iter.into_iter()
                .map(|t| Box::new(t) as Box<dyn Interaction<Self> + Sync>),
        );
    }

    /// Extend the particles within the model given the specified iterator.
    ///
    /// Note that the particle comes with a default `none` particle at index 0,
    /// thus the index of particles added with this start from 1.
    #[cfg(not(feature = "parallel"))]
    pub fn extend_interactions<I>(&mut self, iter: I)
    where
        I: IntoIterator,
        <I as IntoIterator>::Item: Interaction<Self> + 'static,
    {
        self.interactions.extend(
            iter.into_iter()
                .map(|t| Box::new(t) as Box<dyn Interaction<Self>>),
        );
    }
}

impl Model for Empty {
    fn zero() -> Self {
        Self {
            beta: f64::INFINITY,
            particles: vec![ParticleData::new(SCALAR, 0.0, 0.0).name("none")],
            interactions: Vec::new(),
        }
    }

    fn set_beta(&mut self, beta: f64) {
        self.beta = beta;
    }

    fn get_beta(&self) -> f64 {
        self.beta
    }

    fn entropy_dof(&self, _beta: f64) -> f64 {
        // self.particles.iter().map(|p| p.entropy_dof(beta)).sum()
        1.0
    }

    fn particles(&self) -> &[ParticleData] {
        &self.particles
    }

    fn particles_mut(&mut self) -> &mut [ParticleData] {
        &mut self.particles
    }

    /// Unfortunately, this method cannot be implemented for the empty model as
    /// the class is not aware of the particles that will be present within the
    /// model.
    fn static_particle_idx<S: AsRef<str>>(_name: S, _i: usize) -> Result<usize, (S, usize)> {
        unimplemented!()
    }

    fn particle_idx<S: AsRef<str>>(&self, name: S, i: usize) -> Result<usize, (S, usize)> {
        let full_name = format!("{}{}", name.as_ref(), i);
        for (i, p) in self.particles.iter().enumerate() {
            if p.name == full_name || p.name == name.as_ref() {
                return Ok(i);
            }
        }
        Err((name, i))
    }

    /// As the model is empty (at least initially), this method will simply
    /// return 0 by default.
    fn self_energy_absorptive(&self, _p: &ParticleData, _momentum: f64) -> f64 {
        0.0
    }
}

#[cfg(feature = "parallel")]
impl ModelInteractions for Empty {
    type Item = Box<dyn Interaction<Self> + Sync>;

    fn interactions(&self) -> &[Self::Item] {
        &self.interactions
    }
}

#[cfg(not(feature = "parallel"))]
impl ModelInteractions for Empty {
    type Item = Box<dyn Interaction<Self>>;

    fn interactions(&self) -> &[Self::Item] {
        &self.interactions
    }
}
