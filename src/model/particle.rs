//! Basic implementation of a particle type

mod propagator;
pub use propagator::{Multi as MultiPropagator, Propagator, Single as SinglePropagator};

use crate::{
    model::standard::data,
    statistic::{Statistic, Statistics},
};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::{cmp, collections::HashMap, f64, fmt};

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
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Particle {
    // The spin is stored as twice the spin, so a spin-½ particle has `spin ==
    // 1` and a spin-1 particle has `spin == 2`.
    spin: u8,
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
    /// Decays
    #[cfg_attr(feature = "serde", serde(skip))]
    pub decays: HashMap<Vec<isize>, f64>,
    // Whether the particle is complex or not
    complex: bool,
    // Internal degrees of freedom of the particle
    dof: f64,
    /// Name
    pub name: String,
}

impl Particle {
    /// Create a new particle with the specified spin and mass.
    ///
    /// The mass is specified in units of GeV, and the spin is multiplied by a
    /// factor of two such that a spin-½ particle has `spin == 1` and a spin-1
    /// particle has `spin == 2`.
    #[must_use]
    pub fn new(spin: u8, mass: f64, width: f64) -> Self {
        Self {
            spin,
            own_antiparticle: false,
            mass,
            mass2: mass.powi(2),
            width,
            width2: width.powi(2),
            decays: HashMap::new(),
            complex: false,
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

    /// Indicate that the particle is real.
    ///
    /// This function returns self and should be used in constructors.
    #[must_use]
    pub fn real(mut self) -> Self {
        self.complex = false;
        self
    }

    /// Indicate that the particle is complex.
    ///
    /// This function returns self and should be used in constructors.
    #[must_use]
    pub fn complex(mut self) -> Self {
        self.complex = true;
        self
    }

    /// Indicate that the particle is its own antiparticle, thereby preventing
    /// any asymmetry from being generated in its density.
    #[must_use]
    pub fn own_antiparticle(mut self) -> Self {
        self.own_antiparticle = true;
        self
    }

    /// Specify how many internal degrees of freedom this particle has.
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
        if self.name.is_empty() {
            panic!("Particle name cannot be empty.");
        }
        self
    }

    /// Returns true if the particle is real (real scalar, Majorana fermion)
    #[must_use]
    pub fn is_real(&self) -> bool {
        !self.complex
    }

    /// Returns true if the particle is complex (complex scalar, Dirac fermion)
    #[must_use]
    pub fn is_complex(&self) -> bool {
        self.complex
    }

    /// Returns true if the particle is bosonic.
    #[must_use]
    pub fn is_bosonic(&self) -> bool {
        self.spin % 2 == 0
    }

    /// Returns true if the particle is fermionic.
    #[must_use]
    pub fn is_fermionic(&self) -> bool {
        self.spin % 2 == 1
    }

    /// Return the number of degrees of freedom for the underlying particle.
    ///
    /// The general formula is:
    ///
    /// ```math
    /// N_\text{int} \times N_\text{spin} \times N_\text{complex}
    /// ```
    ///
    /// where `$N_\text{int}$` are the internal degrees of freedom,
    /// `$N_\text{spin}$` are the spin degrees of freedom, and
    /// `$N_\text{complex}$` are the degrees of freedom for complex particle.
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
        let dof = self.dof * if self.complex { 2.0 } else { 1.0 };
        match (self.spin, self.mass) {
            (0, _) => dof,
            (n, _) if n % 2 == 1 => f64::from(n + 1) * dof,
            (n, m) if n % 2 == 0 => {
                if m == 0.0 {
                    2.0 * dof
                } else {
                    f64::from(n + 1) * dof
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
    #[must_use]
    pub fn propagator(&self, s: f64) -> SinglePropagator {
        SinglePropagator::new(self, s)
    }
}

impl cmp::PartialEq for Particle {
    fn eq(&self, other: &Self) -> bool {
        self.spin == other.spin && self.mass == other.mass
    }
}
impl Eq for Particle {}

/// Display the particle by its name.
///
/// To display all of the particle's properties, use the
/// [`Debug`](std::fmt::Display).
impl fmt::Display for Particle {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[cfg(test)]
mod tests {
    use super::Particle;
    use crate::utilities::test::approx_eq;
    use std::error;

    #[test]
    fn real_scalar() -> Result<(), Box<dyn error::Error>> {
        let mut particle = Particle::new(0, 1.0, 0.1);

        assert!(particle.is_bosonic());
        assert!(!particle.is_fermionic());
        assert!(!particle.is_complex());

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
        let particle = Particle::new(0, 1.0, 0.1).complex();

        assert!(particle.is_bosonic());
        assert!(!particle.is_fermionic());
        assert!(particle.is_complex());

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
        let particle = Particle::new(1, 1.0, 0.1);

        assert!(!particle.is_bosonic());
        assert!(particle.is_fermionic());
        assert!(!particle.is_complex());

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
        let particle = Particle::new(2, 1.0, 0.1);

        assert!(particle.is_bosonic());
        assert!(!particle.is_fermionic());
        assert!(!particle.is_complex());

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
        let particle = Particle::new(0, 1.0, 0.1).complex().dof(2.5);

        assert!(particle.is_bosonic());
        assert!(!particle.is_fermionic());
        assert!(particle.is_complex());

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
        let particle = Particle::new(1, 1.0, 0.1).dof(1.2);

        assert!(!particle.is_bosonic());
        assert!(particle.is_fermionic());
        assert!(!particle.is_complex());

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
