//! Basic implementation of a particle type

use crate::{
    constants::{BOSON_GSTAR, FERMION_GSTAR},
    statistic::{Statistic, Statistics},
    universe::Universe,
};
use special_functions::interpolation;
use std::f64;

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
pub struct Particle {
    // The spin is stored as twice the spin, so a spin-½ particle has `spin ==
    // 1` and a spin-1 particle has `spin == 2`
    spin: u8,
    /// Mass of the particle in GeV
    pub mass: f64,
    /// Squared mass of the particle in GeV²
    pub mass2: f64,
    /// Width of the particle in GeV
    pub width: f64,
    /// Squared width of the particle in GeV²
    pub width2: f64,
    // Whether the particle is complex or not
    complex: bool,
    // Internal degrees of freedom of the particle
    dof: f64,
}

impl Particle {
    /// Create a new particle with the specified spin and mass.
    ///
    /// The mass is specified in units of GeV, and the spin is multiplied by a
    /// factor of two such that a spin-½ particle has `spin == 1` and a spin-1
    /// particle has `spin == 2`.
    pub fn new(spin: u8, mass: f64, width: f64) -> Self {
        Self {
            spin,
            mass,
            mass2: mass.powi(2),
            width,
            width2: width.powi(2),
            complex: false,
            dof: 1.0,
        }
    }

    /// Set the mass of the particle.
    pub fn set_mass(mut self, mass: f64) -> Self {
        self.mass = mass;
        self.mass2 = mass.powi(2);
        self
    }

    /// Set the width of the particle.
    pub fn set_width(mut self, width: f64) -> Self {
        self.width = width;
        self.width2 = width.powi(2);
        self
    }

    /// Indicate that the particle is complex.
    pub fn set_complex(mut self) -> Self {
        self.complex = true;
        self
    }

    /// Specify how many internal degrees of freedom this particle has.
    ///
    /// This is a multiplicative factor to the degrees of freedom.  For a
    /// 'pseudo' particle such as \\(B-L\\), this should be set to zero.
    pub fn set_dof(mut self, dof: f64) -> Self {
        self.dof = dof;
        self
    }

    /// Returns true if the particle is complex.
    #[inline]
    pub fn is_complex(&self) -> bool {
        self.complex
    }

    /// Returns true if the particle is bosonic.
    #[inline]
    pub fn is_bosonic(&self) -> bool {
        self.spin % 2 == 0
    }

    /// Returns true if the particle is fermionic.
    #[inline]
    pub fn is_fermionic(&self) -> bool {
        self.spin % 2 == 1
    }

    /// Return the number of degrees of freedom for the underlying particle.
    ///
    /// The general formula is:
    ///
    /// \\begin{equation}
    ///   N_\text{int} \times N_\text{spin} \times N_\text{complex}
    /// \\end{equation}
    ///
    /// where \\(N_\text{int}\\) are the internal degrees of freedom,
    /// \\(N_\text{spin}\\) are the spin degrees of freedom, and
    /// \\(N_\text{complex}\\) are the degrees of freedom for complex particle.
    ///
    /// The spin degrees of freedom are:
    /// - spin-0: 1
    /// - spin-1/2: 2
    /// - spin-1: 2 for massless, 3 for massive
    /// - spin-3/2: 4
    /// - spin-2: ?? for massless, 5 for massive
    // TODO: What are the massless degrees of freedom of a spin-2 particle?
    #[inline]
    pub fn degrees_of_freedom(&self) -> f64 {
        let dof = self.dof * if self.complex { 2.0 } else { 1.0 };
        match (self.spin, self.mass) {
            (0, _) => dof,
            (1, _) => 2.0 * dof,
            (2, x) if x == 0.0 => 2.0 * dof,
            (2, x) if x != 0.0 => 3.0 * dof,
            (3, _) => 4.0 * dof,
            (4, _) => 5.0 * dof,
            _ => unimplemented!("Particles with spin greater than 2 are not supported."),
        }
    }

    /// Return the quantum statistic that this particle obeys.
    #[inline]
    fn statistic(&self) -> Statistic {
        if self.is_fermionic() {
            Statistic::FermiDirac
        } else {
            Statistic::BoseEinstein
        }
    }

    /// Return the equilibrium phase space occupation of the particle.
    #[inline]
    pub fn phase_space(&self, e: f64, mu: f64, beta: f64) -> f64 {
        self.statistic().phase_space(e, self.mass, mu, beta) * self.degrees_of_freedom()
    }

    /// Return the equilibrium number density of the particle.
    #[inline]
    pub fn number_density(&self, mu: f64, beta: f64) -> f64 {
        self.statistic().number_density(self.mass, mu, beta) * self.degrees_of_freedom()
    }

    /// Return the equilibrium number density of the particle, normalized to the
    /// number density of a massless boson with one degree of freedom.
    #[inline]
    pub fn normalized_number_density(&self, mu: f64, beta: f64) -> f64 {
        self.statistic()
            .normalized_number_density(self.mass, mu, beta)
            * self.degrees_of_freedom()
    }
}

impl Universe for Particle {
    fn entropy_dof(&self, beta: f64) -> f64 {
        if self.is_bosonic() {
            interpolation::linear(&BOSON_GSTAR, (self.mass * beta).ln()).exp()
                * self.degrees_of_freedom()
        } else {
            interpolation::linear(&FERMION_GSTAR, (self.mass * beta).ln()).exp()
                * self.degrees_of_freedom()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::test::*;

    #[test]
    fn real_scalar() {
        let particle = Particle::new(0, 1.0, 0.1);

        assert!(particle.is_bosonic());
        assert!(!particle.is_fermionic());
        assert!(!particle.is_complex());

        approx_eq(particle.entropy_dof(1e-10), 1.0, 8.0, 0.0);
        assert!(particle.entropy_dof(1e10) < 1e-30);

        approx_eq(
            particle.normalized_number_density(0.0, 1e-10),
            1.0,
            8.0,
            0.0,
        );
    }

    #[test]
    fn complex_scalar() {
        let particle = Particle::new(0, 1.0, 0.1).set_complex();

        assert!(particle.is_bosonic());
        assert!(!particle.is_fermionic());
        assert!(particle.is_complex());

        approx_eq(particle.entropy_dof(1e-10), 2.0, 8.0, 0.0);
        assert!(particle.entropy_dof(1e10) < 1e-30);

        approx_eq(
            particle.normalized_number_density(0.0, 1e-10),
            2.0,
            8.0,
            0.0,
        );
    }

    #[test]
    fn fermion() {
        let particle = Particle::new(1, 1.0, 0.1);

        assert!(!particle.is_bosonic());
        assert!(particle.is_fermionic());
        assert!(!particle.is_complex());

        approx_eq(particle.entropy_dof(1e-10), 2.0 * 0.875, 8.0, 0.0);
        assert!(particle.entropy_dof(1e10) < 1e-30);

        approx_eq(
            particle.normalized_number_density(0.0, 1e-10),
            1.5,
            8.0,
            0.0,
        );
    }

    #[test]
    fn gauge_boson() {
        let particle = Particle::new(2, 1.0, 0.1);

        assert!(particle.is_bosonic());
        assert!(!particle.is_fermionic());
        assert!(!particle.is_complex());

        approx_eq(particle.entropy_dof(1e-10), 3.0, 8.0, 0.0);
        assert!(particle.entropy_dof(1e10) < 1e-30);

        approx_eq(
            particle.normalized_number_density(0.0, 1e-10),
            3.0,
            8.0,
            0.0,
        );
    }

    #[test]
    fn complex_scalar_dof() {
        let particle = Particle::new(0, 1.0, 0.1).set_complex().set_dof(2.5);

        assert!(particle.is_bosonic());
        assert!(!particle.is_fermionic());
        assert!(particle.is_complex());

        approx_eq(particle.entropy_dof(1e-10), 5.0, 8.0, 0.0);
        assert!(particle.entropy_dof(1e10) < 1e-30);

        approx_eq(
            particle.normalized_number_density(0.0, 1e-10),
            5.0,
            8.0,
            0.0,
        );
    }

    #[test]
    fn fermion_dof() {
        let particle = Particle::new(1, 1.0, 0.1).set_dof(1.2);

        assert!(!particle.is_bosonic());
        assert!(particle.is_fermionic());
        assert!(!particle.is_complex());

        approx_eq(particle.entropy_dof(1e-10), 2.0 * 1.2 * 0.875, 8.0, 0.0);
        assert!(particle.entropy_dof(1e10) < 1e-30);

        approx_eq(
            particle.normalized_number_density(0.0, 1e-10),
            2.0 * 1.2 * 0.75,
            8.0,
            0.0,
        );
    }
}
