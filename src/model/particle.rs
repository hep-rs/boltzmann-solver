//! Basic implementation of a particle type

mod lorentz_representation;
mod propagator;

use crate::{
    model::{standard::data, Model},
    statistic::{Statistic, Statistics},
};
pub use lorentz_representation::{
    LorentzRepresentation, DIRAC_SPINOR, LEFT_WEYL_SPINOR, RIGHT_WEYL_SPINOR, SCALAR, TENSOR,
    VECTOR,
};
use num::Complex;
pub use propagator::{Multi as MultiPropagator, Propagator, Single as SinglePropagator};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::{cmp, f64, fmt, hash};

// /// Trait for a particle belonging to a model.
// pub trait ModelParticle<M> {
//     /// Representation of the particle under the Lorentz group.
//     fn representation(&self) -> LorentzRepresentation {
//         self.particle_data().representation
//     }

//     /// Internal degrees of the particle.
//     fn dof(&self) -> f64 {
//         self.particle_data().dof
//     }

//     /// Name
//     fn name(&self) -> &str {
//         &self.particle_data().name
//     }

//     /// Get a reference to the particle data
//     fn particle_data(&self) -> &ParticleData;

//     /// Get a mutable reference to the particle data
//     fn particle_data_mut(&mut self) -> &mut ParticleData;

//     /// Mass of the particle in GeV.
//     fn mass(&self) -> f64 {
//         self.particle_data().mass
//     }

//     /// Squared mass of the particle in GeV`$^2$`.
//     fn mass2(&self) -> f64 {
//         self.particle_data().mass2
//     }

//     /// Width of the particle in GeV.
//     fn width(&self) -> f64 {
//         self.particle_data().width
//     }

//     /// Squared width of the particle in GeV`$^2$`.
//     fn width2(&self) -> f64 {
//         self.particle_data().width2
//     }

//     /// Recalculate the mass of the particle.
//     ///
//     /// As model parameters might change due to RGE running or because of
//     /// thermal contributions, the mass of the particle might change.  The
//     /// result of the calculation should be stored internally so that it can be
//     /// returned by the [`mass`] and [`mass2`] functions.
//     fn calculate_mass(&mut self, model: &M) -> f64;

//     /// Recalculate the width of the particle.
//     ///
//     /// As model parameters might change due to RGE running or because of
//     /// thermal contributions, the width of the particle might change.  The
//     /// result of the calculation should be stored internally so that it can be
//     /// returned by the [`width`] and [`width2`] functions.
//     ///
//     /// This can also be computed from [`self_energy_absorptive`] by using the
//     /// relation `$\Sigma(p^2 = m^2) = m \Gamma$`.
//     fn calculate_width(&mut self, model: &M) -> f64 {
//         let mass = self.mass();
//         let width = self.self_energy_absorptive(model, mass * mass) / mass;

//         let pdata = self.particle_data_mut();
//         pdata.width = width;
//         pdata.width2 = width * width;
//         width
//     }

//     /// Imaginary component of the self-energy at the specified momentum
//     /// transfer which is given in terms of the invariant `$p^2$`.
//     ///
//     /// Physically this corresponds to when intermediate particles in the
//     /// self-energy go on-shell.  These can be calculated using the Cutkosky
//     /// rules, or using known results from Passarino-Veltman functions.
//     fn self_energy_absorptive(&self, model: &M, momentum: f64) -> f64;
// }

// impl<M> PartialEq for dyn ModelParticle<M> {
//     fn eq(&self, other: &dyn ModelParticle<M>) -> bool {
//         self.name() == other.name()
//     }
// }

// impl<M> Eq for dyn ModelParticle<M> {}

// impl<M> std::hash::Hash for dyn ModelParticle<M> {
//     fn hash<H: hash::Hasher>(&self, state: &mut H) {
//         self.name().hash(state);
//     }
// }

/// Particle type
///
/// The particle type is only very basic, containing only the minimum
/// information required for cosmological purposes.  Specifically, it only keeps
/// track of the particle's spin, mass and whether it is complex.  It does not
/// keep track of transformation properties and interactions.
///
/// By default, the degrees of freedom is calculated based on the particle's
/// spin and whether it is complex or not.  An arbitrary number can be specified
/// which will override this calculation (which can be used for multiplets for
/// example).
///
/// For hashing purposes, only the particle's name is taken into account.
///
/// In the long run, it is intended that this type be replaced by another one.
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Debug, Clone)]
pub struct Data {
    // The spin is stored as twice the spin, so a spin-½ particle has `spin ==
    // 1` and a spin-1 particle has `spin == 2`.
    representation: LorentzRepresentation,
    /// Whether the particle is its own antiparticle or not.
    ///
    /// By default, this is assumed to be false and can be set to true with
    /// [`Particle::own_antiparticle`].
    pub own_antiparticle: bool,
    /// Mass of the particle in GeV.
    ///
    /// This should be updated using [`Particle::set_mass`] so that both the
    /// mass and squared mass are updated simultaneously.
    pub mass: f64,
    /// Squared mass of the particle in GeV².
    ///
    /// This should be updated using [`Particle::set_mass`] so that both the
    /// mass and squared mass are updated simultaneously.
    pub mass2: f64,
    /// Width of the particle in GeV.
    ///
    /// This should be updated using [`Particle::set_width`] so that both the
    /// width and squared width are updated simultaneously.
    pub width: f64,
    /// Squared width of the particle in GeV².
    ///
    /// This should be updated using [`Particle::set_width`] so that both the
    /// width and squared width are updated simultaneously.
    pub width2: f64,
    /// Internal degrees of freedom of the particle
    pub dof: f64,
    /// Name
    pub name: String,
}

impl Data {
    /// Create a new particle with the specified spin and mass.
    ///
    /// The mass is specified in units of GeV, and the spin is multiplied by a
    /// factor of two such that a spin-½ particle has `spin == 1` and a spin-1
    /// particle has `spin == 2`.
    #[must_use]
    pub fn new(representation: LorentzRepresentation, mass: f64, width: f64) -> Self {
        Self {
            representation,
            own_antiparticle: false,
            mass,
            mass2: mass.powi(2),
            width,
            width2: width.powi(2),
            dof: 1.0,
            name: "?".to_string(),
        }
    }

    /// Set the mass of the particle.
    pub fn set_mass(&mut self, mass: f64) -> &mut Self {
        self.mass = mass;
        self.mass2 = mass.powi(2);
        self
    }

    /// Set the width of the particle.
    pub fn set_width(&mut self, width: f64) -> &mut Self {
        self.width = width;
        self.width2 = width.powi(2);
        self
    }

    /// Indicate that the particle is its own antiparticle, thereby preventing
    /// any asymmetry from being generated in its density.
    #[must_use]
    pub fn own_antiparticle(mut self) -> Self {
        self.own_antiparticle = true;
        self
    }

    /// Specify how many internal degrees of freedom this particle has.  These
    /// can be because the particle is complex or any other internal gauge
    /// symmetries.
    ///
    /// This is a multiplicative factor to the degrees of freedom.  For a
    /// 'pseudo' particle such as `$B-L$`, this should be set to zero.
    ///
    /// This function returns self and should be used in constructors.
    #[must_use]
    pub fn dof(mut self, dof: f64) -> Self {
        self.dof = dof;
        self
    }

    /// Specify the particle's name.
    ///
    /// # Panics
    ///
    /// The particle name cannot be an empty string.
    #[must_use]
    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = name.into();
        assert!(!self.name.is_empty(), "Particle name cannot be empty.");
        self
    }

    /// Returns true if the particle is bosonic.
    #[must_use]
    pub fn is_bosonic(&self) -> bool {
        self.representation.is_bosonic()
    }

    /// Returns true if the particle is fermionic.
    #[must_use]
    pub fn is_fermionic(&self) -> bool {
        self.representation.is_fermionic()
    }

    /// Check whether the particle is massless.
    ///
    /// As the calculation for the mass may not be exact, this checks whether
    /// the mass square is less than the minimum positive normal floating point
    /// (equivalent to `$m \lesssim 10^{-150}$`).
    #[must_use]
    pub fn is_massless(&self) -> bool {
        self.mass2 < f64::MIN_POSITIVE
    }

    /// Return the number of degrees of freedom for the underlying particle.
    ///
    /// The general formula is:
    ///
    /// ```math
    /// N_\text{int} \times N_\text{spin}
    /// ```
    ///
    /// where `$N_\text{int}$` are the internal degrees of freedom,
    /// `$N_\text{spin}$` are the spin degrees of freedom.
    ///
    /// The spin degrees of freedom are:
    /// - spin-0: 1
    /// - spin-1/2: 2
    /// - spin-1: 2 for massless, 3 for massive
    /// - spin-3/2: 4
    /// - spin-2: 2 for massless, 5 for massive
    ///
    /// or more generally, for half-integer spins `n/2`, the degrees of freedom
    /// is `n + 1`; and for integer spin `n > 0`, the degrees of freedom are `2`
    /// for massless or `2n + 1` for massive.
    ///
    /// # Panics
    ///
    /// Panics if a particle with spin greater than 2 is given.
    #[must_use]
    pub fn degrees_of_freedom(&self) -> f64 {
        match (self.representation.spin(), self.mass) {
            (0, _) => self.dof,
            (n, _) if n % 2 == 1 => f64::from(n + 1) * self.dof,
            (n, m) if n % 2 == 0 => {
                if m == 0.0 {
                    2.0 * self.dof
                } else {
                    f64::from(n + 1) * self.dof
                }
            }
            _ => unimplemented!("Particles with spin greater than 2 are not supported."),
        }
    }

    /// Return the quantum statistic that this particle obeys.
    #[must_use]
    fn statistic(&self) -> Statistic {
        if self.is_fermionic() {
            Statistic::FermiDirac
        } else {
            Statistic::BoseEinstein
        }
    }

    /// Return the equilibrium phase space occupation of the particle.
    #[must_use]
    pub fn phase_space(&self, e: f64, mu: f64, beta: f64) -> f64 {
        self.statistic().phase_space(beta, e, self.mass, mu) * self.degrees_of_freedom()
    }

    /// Return the equilibrium number density of the particle.
    #[must_use]
    pub fn number_density(&self, beta: f64, mu: f64) -> f64 {
        self.statistic().number_density(beta, self.mass, mu) * self.degrees_of_freedom()
    }

    /// Return the equilibrium number density of the particle, normalized to the
    /// number density of a massless boson with one degree of freedom.
    #[must_use]
    pub fn normalized_number_density(&self, beta: f64, mu: f64) -> f64 {
        self.statistic()
            .normalized_number_density(beta, self.mass, mu)
            * self.degrees_of_freedom()
    }

    /// Return the entropy degrees of freedom associated with this particle.
    #[must_use]
    pub fn entropy_dof(&self, beta: f64) -> f64 {
        if self.is_bosonic() {
            data::BOSON_GSTAR.sample(f64::ln(self.mass * beta)).exp() * self.degrees_of_freedom()
        } else {
            data::FERMION_GSTAR.sample(f64::ln(self.mass * beta)).exp() * self.degrees_of_freedom()
        }
    }

    /// Return the propagator denominator for the particle.
    ///
    /// The propagator is defined as follow:
    ///
    /// ```math
    /// \mathcal{P}_i^{-1}(s) = s - m_i^2 + i \theta(s) m_i \Gamma_i,
    /// ```
    ///
    /// where `$\theta$` is the Heaviside step function.
    #[must_use]
    pub fn propagator<M: Model>(&self, model: &M, s: f64) -> Complex<f64> {
        SinglePropagator::new(self, model, s).eval()
    }

    /// Return the RIS propagator for the particle.
    ///
    /// The RIS propagator is defined such that a real intermediate state is
    /// substracted from the propagator.  This is done to avoid a
    /// double-counting issue of with `$2 \xleftrightarrow{P} 2$` interaction
    /// with some intermediate particle `$P$` and the combination of the two
    /// 3-particle interactions 2 ↔ P and P ↔ 2.
    ///
    /// The definition is follows that of a regular
    /// [propagator](`Particle::propagator`) except in the following case:
    ///
    /// ```math
    /// \mathcal{P}_i(s) \mathcal{P}_i^*(s)
    /// = \frac{(s - m_i^2)^2 - (m_i \Gamma_i)^2}{[(s - m_i^2)^2 + (m_i \Gamma_i)^2]^2}.
    /// ```
    #[must_use]
    pub fn ris_propagator<M: Model>(&self, model: &M, s: f64) -> SinglePropagator {
        SinglePropagator::new(self, model, s)
    }
}

impl cmp::PartialEq for Data {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}
impl Eq for Data {}

impl hash::Hash for Data {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

/// Display the particle by its name.
///
/// To display all of the particle's properties, use the
/// [`Debug`](std::fmt::Display).
impl fmt::Display for Data {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[cfg(test)]
mod tests {
    use super::Data;
    use crate::{
        model::particle::{DIRAC_SPINOR, SCALAR, TENSOR},
        utilities::test::approx_eq,
    };
    use std::error;

    #[test]
    fn real_scalar() -> Result<(), Box<dyn error::Error>> {
        let mut particle: Data = Data::new(SCALAR, 1.0, 0.1);

        assert!(particle.is_bosonic());
        assert!(!particle.is_fermionic());

        approx_eq(particle.entropy_dof(1e-10), 1.0, 8.0, 0.0)?;
        assert!(particle.entropy_dof(1e10) < 1e-30);
        approx_eq(
            particle.normalized_number_density(1e-10, 0.0),
            1.0,
            8.0,
            0.0,
        )?;

        particle.set_mass(0.0);
        approx_eq(
            particle.entropy_dof(1e-10),
            particle.entropy_dof(1e10),
            8.0,
            0.0,
        )?;
        approx_eq(
            particle.normalized_number_density(1e-10, 0.0),
            1.0,
            8.0,
            0.0,
        )?;

        Ok(())
    }

    #[test]
    fn complex_scalar() -> Result<(), Box<dyn error::Error>> {
        let particle: Data = Data::new(SCALAR, 1.0, 0.1).dof(2.0);

        assert!(particle.is_bosonic());
        assert!(!particle.is_fermionic());

        approx_eq(particle.entropy_dof(1e-10), 2.0, 8.0, 0.0)?;
        assert!(particle.entropy_dof(1e10) < 1e-30);

        approx_eq(
            particle.normalized_number_density(1e-10, 0.0),
            2.0,
            8.0,
            0.0,
        )?;

        Ok(())
    }

    #[test]
    fn fermion() -> Result<(), Box<dyn error::Error>> {
        let particle: Data = Data::new(DIRAC_SPINOR, 1.0, 0.1);

        assert!(!particle.is_bosonic());
        assert!(particle.is_fermionic());

        approx_eq(particle.entropy_dof(1e-10), 2.0 * 0.875, 8.0, 0.0)?;
        assert!(particle.entropy_dof(1e10) < 1e-30);

        approx_eq(
            particle.normalized_number_density(1e-10, 0.0),
            1.5,
            8.0,
            0.0,
        )?;

        Ok(())
    }

    #[test]
    fn gauge_boson() -> Result<(), Box<dyn error::Error>> {
        let particle: Data = Data::new(TENSOR, 1.0, 0.1);

        assert!(particle.is_bosonic());
        assert!(!particle.is_fermionic());

        approx_eq(particle.entropy_dof(1e-10), 3.0, 8.0, 0.0)?;
        assert!(particle.entropy_dof(1e10) < 1e-30);

        approx_eq(
            particle.normalized_number_density(1e-10, 0.0),
            3.0,
            8.0,
            0.0,
        )?;

        Ok(())
    }

    #[test]
    fn complex_scalar_dof() -> Result<(), Box<dyn error::Error>> {
        let particle: Data = Data::new(SCALAR, 1.0, 0.1).dof(2.0 * 2.5);

        assert!(particle.is_bosonic());
        assert!(!particle.is_fermionic());

        approx_eq(particle.entropy_dof(1e-10), 5.0, 8.0, 0.0)?;
        assert!(particle.entropy_dof(1e10) < 1e-30);

        approx_eq(
            particle.normalized_number_density(1e-10, 0.0),
            5.0,
            8.0,
            0.0,
        )?;

        Ok(())
    }

    #[test]
    fn fermion_dof() -> Result<(), Box<dyn error::Error>> {
        let particle: Data = Data::new(DIRAC_SPINOR, 1.0, 0.1).dof(1.2);

        assert!(!particle.is_bosonic());
        assert!(particle.is_fermionic());

        approx_eq(particle.entropy_dof(1e-10), 2.0 * 1.2 * 0.875, 8.0, 0.0)?;
        assert!(particle.entropy_dof(1e10) < 1e-30);

        approx_eq(
            particle.normalized_number_density(1e-10, 0.0),
            2.0 * 1.2 * 0.75,
            8.0,
            0.0,
        )?;

        Ok(())
    }
}
