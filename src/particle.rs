//! Basic implementation of a particle type

use constants::{BOSON_GSTAR, FERMION_GSTAR};
use special_functions::interpolation;
use universe::Universe;

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
/// In the long run, it is intended that this type be replaced by another one.
#[derive(Debug)]
pub struct Particle {
    /// The spin is stored as twice the spin, so a spin-½ particle has `spin ==
    /// 1` and a spin-1 particle has `spin == 2`
    pub spin: u8,
    /// Mass of the particle in GeV
    pub mass: f64,
    complex: bool,
    dof: Option<f64>,
}

impl Particle {
    /// Create a new particle with the specified spin and mass.
    ///
    /// The mass is specified in units of GeV, and the spin is multiplied by a
    /// factor of two such that a spin-½ particle has `spin == 1` and a spin-1
    /// particle has `spin == 2`.
    pub fn new(spin: u8, mass: f64) -> Self {
        Self {
            spin,
            mass,
            complex: false,
            dof: None,
        }
    }

    /// Indicate that the particle is complex.
    pub fn set_complex(mut self) -> Self {
        self.complex = true;
        self
    }

    /// Specify how many degrees of freedom this particle has.  This overwrites
    /// completely the default calculation (that is, the maximum degrees of
    /// freedom will be independent of the spin of the particle and whether it
    /// is complex or not).
    pub fn set_dof(mut self, dof: f64) -> Self {
        self.dof = Some(dof);
        self
    }

    /// Returns true if the particle is complex.
    pub fn is_complex(&self) -> bool {
        self.complex
    }

    /// Returns true if the particle is bosonic.
    pub fn is_bosonic(&self) -> bool {
        self.spin % 2 == 0
    }

    /// Returns true if the particle is fermionic.
    pub fn is_fermionic(&self) -> bool {
        self.spin % 2 == 1
    }
}

impl Universe for Particle {
    fn entropy_dof(&self, beta: f64) -> f64 {
        let g1 = if self.is_bosonic() {
            interpolation::linear(&BOSON_GSTAR, (self.mass * beta).ln()).exp()
        } else {
            interpolation::linear(&FERMION_GSTAR, (self.mass * beta).ln()).exp()
        };
        if let Some(dof) = self.dof {
            g1 * dof
        } else {
            g1 * (self.spin + 1) as f64 * if self.complex { 2.0 } else { 1.0 }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn real_scalar() {
        let particle = Particle::new(0, 1.0);

        assert!(particle.is_bosonic());
        assert!(!particle.is_fermionic());
        assert!(!particle.is_complex());

        assert_eq!(particle.entropy_dof(1e-10), 1.0);
        assert!(particle.entropy_dof(1e10) < 1e-30);
    }

    #[test]
    fn complex_scalar() {
        let particle = Particle::new(0, 1.0).set_complex();

        assert!(particle.is_bosonic());
        assert!(!particle.is_fermionic());
        assert!(particle.is_complex());

        assert_eq!(particle.entropy_dof(1e-10), 2.0);
        assert!(particle.entropy_dof(1e10) < 1e-30);
    }

    #[test]
    fn fermion() {
        let particle = Particle::new(1, 1.0);

        assert!(!particle.is_bosonic());
        assert!(particle.is_fermionic());
        assert!(!particle.is_complex());

        assert_eq!(particle.entropy_dof(1e-10), 2.0 * 0.875);
        assert!(particle.entropy_dof(1e10) < 1e-30);
    }

    #[test]
    fn gauge_boson() {
        let particle = Particle::new(2, 1.0);

        assert!(particle.is_bosonic());
        assert!(!particle.is_fermionic());
        assert!(!particle.is_complex());

        assert_eq!(particle.entropy_dof(1e-10), 3.0);
        assert!(particle.entropy_dof(1e10) < 1e-30);
    }

    #[test]
    fn complex_scalar_dof() {
        let particle = Particle::new(0, 1.0).set_complex().set_dof(2.5);

        assert!(particle.is_bosonic());
        assert!(!particle.is_fermionic());
        assert!(particle.is_complex());

        assert_eq!(particle.entropy_dof(1e-10), 2.5);
        assert!(particle.entropy_dof(1e10) < 1e-30);
    }

}