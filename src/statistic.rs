//! If the rate of collisions between particles is sufficiently high (as is
//! usually the case), the phase space distribution of the particles will
//! quickly converge onto either the Fermi–Dirac statistic or the Bose–Einstein
//! statistic depending on whether the particle is a half-integer or integer
//! spin particle.
//!
//! Furthermore, in the limit that the occupation is really low, both statistics
//! resemble the Maxwell–Boltzmann distribution in the non-relativistic case, or
//! the Maxwell–Jüttner distribution for the relativistic case.
//!
//! The statistics are implemented, as well as calculations of the number
//! density.

use quadrature::integrate;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use special_functions::{bessel, particle_physics::statistics};
use std::{f64, fmt};

/// Equilibrium number density for massless bosons, normalized to the
/// equilibrium number density of a massless boson.  This is specified per
/// degree of freedom (that is `$g = 1$`).
pub const BOSON_EQ_DENSITY: f64 = 1.0;
/// Equilibrium number density for massless fermions, normalized to the
/// equilibrium number density of a massless boson.  This is specified per
/// degree of freedom (that is `$g = 1$`)
pub const FERMION_EQ_DENSITY: f64 = 0.75;

/// The statistics which describe the distribution of particles over energy
/// states.  Both Fermi–Dirac and Bose–Einstein quantum statistics are
/// implemented, as well as the classical Maxwell–Boltzmann statistic.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub enum Statistic {
    /// Fermi–Dirac statistic describing half-integer-spin particles:
    ///
    /// ```math
    /// f_{\text{FD}} = \frac{1}{\exp[(E - \mu) \beta] + 1}.
    /// ```
    FermiDirac,
    /// Bose–Einstein statistic describing integer-spin particles:
    ///
    /// ```math
    /// f_{\text{BE}} = \frac{1}{\exp[(E - \mu) \beta] - 1}.
    /// ```
    BoseEinstein,
    /// Maxwell–Boltzmann statistic describing classical particles:
    ///
    /// ```math
    /// f_{\text{MB}} = \exp[-(E - \mu) \beta].
    /// ```
    MaxwellBoltzmann,
}

impl fmt::Display for Statistic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FermiDirac => write!(f, "Statistic::FermiDirac"),
            Self::BoseEinstein => write!(f, "Statistic::BoseEinstein"),
            Self::MaxwellBoltzmann => write!(f, "Statistic::MaxwellBoltzmann"),
        }
    }
}

/// Equilibrium statistics.
pub trait Statistics {
    /// Evaluate the phase space distribution, for a given energy, mass,
    /// chemical potential and inverse temperature.
    fn phase_space(&self, e: f64, m: f64, mu: f64, beta: f64) -> f64;

    /// Return number density for a particle following the specified statistic.
    ///
    /// ```math
    /// n = \frac{1}{2 \pi^2} \int_{m}^{\infty} f_{i} u \sqrt{u^2 - m^2} \dd u.
    /// ```
    ///
    /// The naïve implementation will perform a numerical integration.
    ///
    /// # Note to Implementors
    ///
    /// If an analytic closed form is available for the integral, it should be
    /// preferred over the numerical integration.
    fn number_density(&self, mass: f64, mu: f64, beta: f64) -> f64 {
        let integral = integrate(
            |t| {
                let u = mass + t / (1.0 - t);
                let dudt = (t - 1.0).powi(-2);

                self.phase_space(u, mass, mu, beta) * u * f64::sqrt(u.powi(2) - mass.powi(2)) * dudt
            },
            0.0,
            1.0,
            1e-12,
        );
        log::debug!(
            "Phase space integral: {:e} ± {:e} ({} function evaluations)",
            integral.integral,
            integral.error_estimate,
            integral.num_function_evaluations
        );
        // 1/(2 π²) ≅ 0.050_660_591_821_168_89
        0.050_660_591_821_168_89 * integral.integral
    }

    /// Return number density asymmetry for a particle following the specified
    /// statistic.
    ///
    /// ```math
    /// n = \frac{1}{2 \pi^2} \int_{m}^{\infty} [f_{i}(\mu) - f_{i}(-\mu)] u \sqrt{u^2 - m^2} \dd u.
    /// ```
    ///
    /// The naïve implementation will perform a numerical integration.
    ///
    /// # Note to Implementors
    ///
    /// If an analytic closed form is available for the integral, it should be
    /// preferred over the numerical integration.
    fn number_density_asymmetry(&self, mass: f64, mu: f64, beta: f64) -> f64 {
        let integral = integrate(
            |t| {
                let u = mass + t / (1.0 - t);
                let dudt = (t - 1.0).powi(-2);

                (self.phase_space(u, mass, mu, beta) - self.phase_space(u, mass, -mu, beta))
                    * u
                    * f64::sqrt(u.powi(2) - mass.powi(2))
                    * dudt
            },
            0.0,
            1.0,
            1e-12,
        );
        log::debug!(
            "Phase space integral: {:e} ± {:e} ({} function evaluations)",
            integral.integral,
            integral.error_estimate,
            integral.num_function_evaluations
        );
        // 1/(2 π²) ≅ 0.050_660_591_821_168_89
        0.050_660_591_821_168_89 * integral.integral
    }

    /// Return number density for a particle following the specified statistic,
    /// normalized to the number density of a massless boson with a single
    /// degree of freedom.
    fn normalized_number_density(&self, mass: f64, mu: f64, beta: f64) -> f64 {
        self.number_density(mass, mu, beta) / Statistic::BoseEinstein.number_density(0.0, 0.0, beta)
    }

    /// Return number density for a particle following the specified statistic,
    /// normalized to the number density of a massless boson with a single
    /// degree of freedom.
    fn normalized_number_density_asymmetry(&self, mass: f64, mu: f64, beta: f64) -> f64 {
        self.number_density_asymmetry(mass, mu, beta)
            / Statistic::BoseEinstein.number_density(0.0, 0.0, beta)
    }
}

impl Statistics for Statistic {
    /// Evaluate the phase space distribution, `$f$` as defined above for the
    /// four statistics.
    fn phase_space(&self, e: f64, _m: f64, mu: f64, beta: f64) -> f64 {
        match self {
            Statistic::FermiDirac => (f64::exp((e - mu) * beta) + 1.0).recip(),
            Statistic::BoseEinstein => {
                let exponent = (e - mu) * beta;
                if exponent.abs() < 1.0 {
                    f64::exp_m1(exponent).recip()
                } else {
                    (f64::exp(exponent) - 1.0).recip()
                }
            }
            Statistic::MaxwellBoltzmann => f64::exp(-(e - mu) * beta),
        }
    }

    /// Return number density for a particle following the specified statistic.
    ///
    /// ```math
    /// n = \frac{1}{2 \pi^2} \int_{m}^{\infty} f_{i} u \sqrt{u^2 - m^2} \dd u
    /// ```
    ///
    /// # Implementation Details
    ///
    /// Both the Fermi–Dirac and Bose–Einstein statistic rely on numerical
    /// integration and thus are fairly slow and are prone to errors in certain
    /// regimes.  The Maxwell–Boltzmann and Maxwell–Jüttner distributions offer
    /// exact implementations.
    fn number_density(&self, mass: f64, mu: f64, beta: f64) -> f64 {
        debug_assert!(mass >= 0.0, "mass must be positive.");
        debug_assert!(beta >= 0.0, "β must be positive.");

        match (self, mu == 0.0, mass == 0.0) {
            (Statistic::FermiDirac, true, _) => statistics::fermi_dirac_massive(mass, beta),
            (Statistic::FermiDirac, _, true) => statistics::fermi_dirac_massless(mu, beta),
            (Statistic::FermiDirac, false, false) => {
                let integral = integrate(
                    |t| {
                        let u = mass + t / (1.0 - t);
                        let dudt = (t - 1.0).powi(-2);

                        self.phase_space(u, mass, mu, beta)
                            * u
                            * f64::sqrt(u.powi(2) - mass.powi(2))
                            * dudt
                    },
                    0.0,
                    1.0,
                    1e-12,
                );
                log::debug!(
                    "Fermi–Dirac integral: {:e} ± {:e} ({} function evaluations)",
                    integral.integral,
                    integral.error_estimate,
                    integral.num_function_evaluations
                );
                if cfg!(debug_assertions) && integral.error_estimate > 0.01 * integral.integral {
                    log::warn!(
                        "Fermi–Dirac integral has a relative error of {:0.2}%",
                        integral.error_estimate / integral.integral.abs()
                    );
                }
                // 1/(2 π²) ≅ 0.050_660_591_821_168_89
                0.050_660_591_821_168_89 * integral.integral
            }
            (Statistic::BoseEinstein, true, _) => statistics::bose_einstein_massive(mass, beta),
            (Statistic::BoseEinstein, _, true) => statistics::bose_einstein_massless(mu, beta),
            (Statistic::BoseEinstein, false, false) => {
                let integral = integrate(
                    |t| {
                        let u = mass + t / (1.0 - t);
                        let dudt = (t - 1.0).powi(-2);

                        self.phase_space(u, mass, mu, beta)
                            * u
                            * f64::sqrt(u.powi(2) - mass.powi(2))
                            * dudt
                    },
                    0.0,
                    1.0,
                    1e-12,
                );
                log::debug!(
                    "Bose–Einstein integral: {:e} ± {:e} ({} function evaluations)",
                    integral.integral,
                    integral.error_estimate,
                    integral.num_function_evaluations
                );
                if cfg!(debug_assertions) && integral.error_estimate > 0.01 * integral.integral {
                    log::warn!(
                        "Bose–Einstein integral has a relative error of {:0.2}%",
                        integral.error_estimate / integral.integral.abs()
                    );
                }
                // 1/(2 π²) ≅ 0.050_660_591_821_168_89
                0.050_660_591_821_168_89 * integral.integral
            }
            (Statistic::MaxwellBoltzmann, _, true) => {
                // 1 / π² ≅ 0.101_321_183_642_337_78
                0.101_321_183_642_337_78 * f64::exp(mu * beta) / beta.powi(3)
            }
            (Statistic::MaxwellBoltzmann, _, _) => {
                // 1/(2 π²) ≅ 0.050_660_591_821_168_89
                0.050_660_591_821_168_89
                    * mass.powi(2)
                    * bessel::k2(mass * beta)
                    * f64::exp(mu * beta)
                    / beta
            }
        }
    }

    /// Return number density for a particle following the specified statistic.
    ///
    /// ```math
    /// n = \frac{1}{2 \pi^2} \int_{m}^{\infty} f_{i} u \sqrt{u^2 - m^2} \dd u
    /// ```
    ///
    /// # Implementation Details
    ///
    /// Both the Fermi–Dirac and Bose–Einstein statistic rely on numerical
    /// integration and thus are fairly slow and are prone to errors in certain
    /// regimes.  The Maxwell–Boltzmann and Maxwell–Jüttner distributions offer
    /// exact implementations.
    fn number_density_asymmetry(&self, mass: f64, mu: f64, beta: f64) -> f64 {
        debug_assert!(mass >= 0.0, "mass must be positive.");
        debug_assert!(beta >= 0.0, "β must be positive.");

        match (self, mu == 0.0, mass == 0.0) {
            (_, true, _) => 0.0,
            (Statistic::FermiDirac, false, true) => {
                statistics::fermi_dirac_massless(mu, beta)
                    - statistics::fermi_dirac_massless(-mu, beta)
            }
            (Statistic::FermiDirac, false, false) => {
                let integral = integrate(
                    |t| {
                        let u = mass + t / (1.0 - t);
                        let dudt = (t - 1.0).powi(-2);

                        (self.phase_space(u, mass, mu, beta) - self.phase_space(u, mass, -mu, beta))
                            * u
                            * f64::sqrt(u.powi(2) - mass.powi(2))
                            * dudt
                    },
                    0.0,
                    1.0,
                    1e-12,
                );
                log::debug!(
                    "Fermi–Dirac integral: {:e} ± {:e} ({} function evaluations)",
                    integral.integral,
                    integral.error_estimate,
                    integral.num_function_evaluations
                );
                if cfg!(debug_assertions) && integral.error_estimate > 0.01 * integral.integral {
                    log::warn!(
                        "Fermi–Dirac integral has a relative error of {:0.2}%",
                        integral.error_estimate / integral.integral.abs()
                    );
                }
                // 1/(2 π²) ≅ 0.050_660_591_821_168_89
                0.050_660_591_821_168_89 * integral.integral
            }
            (Statistic::BoseEinstein, false, true) => {
                statistics::bose_einstein_massless(mu, beta)
                    - statistics::bose_einstein_massless(-mu, beta)
            }
            (Statistic::BoseEinstein, false, false) => {
                {
                    let integral = integrate(
                        |t| {
                            let u = mass + t / (1.0 - t);
                            let dudt = (t - 1.0).powi(-2);

                            (self.phase_space(u, mass, mu, beta)
                                - self.phase_space(u, mass, -mu, beta))
                                * u
                                * f64::sqrt(u.powi(2) - mass.powi(2))
                                * dudt
                        },
                        0.0,
                        1.0,
                        1e-12,
                    );
                    log::debug!(
                        "Bose–Einstein integral: {:e} ± {:e} ({} function evaluations)",
                        integral.integral,
                        integral.error_estimate,
                        integral.num_function_evaluations
                    );
                    if cfg!(debug_assertions) && integral.error_estimate > 0.01 * integral.integral
                    {
                        log::warn!(
                            "Bose–Einstein integral has a relative error of {:0.2}%",
                            integral.error_estimate / integral.integral.abs()
                        );
                    }
                    // 1/(2 π²) ≅ 0.050_660_591_821_168_89
                    0.050_660_591_821_168_89 * integral.integral
                }
            }
            (Statistic::MaxwellBoltzmann, false, true) => {
                // 2 / π² ≅ 0.202_642_367_284_675_55
                0.202_642_367_284_675_55 * f64::sinh(mu * beta) / beta.powi(3)
            }
            (Statistic::MaxwellBoltzmann, false, false) => {
                // 1/(π²) ≅ 0.101_321_183_642_337_78
                0.101_321_183_642_337_78
                    * mass.powi(2)
                    * bessel::k2(mass * beta)
                    * f64::sinh(mu * beta)
                    / beta
            }
        }
    }

    /// Return number density for a particle following the specified statistic,
    /// normalized to the number density of a massless boson with a single
    /// degree of freedom.
    ///
    /// # Implementation Details
    ///
    /// This implementation assumes that the chemical potential is negligible.
    fn normalized_number_density(&self, mass: f64, mu: f64, beta: f64) -> f64 {
        debug_assert!(mass >= 0.0, "mass must be positive.");
        debug_assert!(beta >= 0.0, "β must be positive.");

        match (self, mu == 0.0) {
            (Statistic::FermiDirac, true) => statistics::fermi_dirac_normalized(mass, beta),
            (Statistic::BoseEinstein, true) => statistics::bose_einstein_normalized(mass, beta),
            _ => {
                self.number_density(mass, mu, beta)
                    / Statistic::BoseEinstein.number_density(0.0, 0.0, beta)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Statistic, Statistics};
    use crate::utilities::test::approx_eq;
    use std::{error, f64, fs, io::BufReader};

    type Row6 = (f64, f64, f64, f64, f64, f64);
    type Row7 = (f64, f64, f64, f64, f64, f64, f64);
    type Row8 = (f64, f64, f64, f64, f64, f64, f64, f64);

    #[test]
    #[allow(clippy::similar_names)]
    fn phase_space() -> Result<(), Box<dyn error::Error>> {
        let fd = Statistic::FermiDirac;
        let be = Statistic::BoseEinstein;
        let mb = Statistic::MaxwellBoltzmann;

        let mut rdr = csv::Reader::from_reader(zstd::Decoder::with_buffer(BufReader::new(
            fs::File::open("tests/data/phase_space.csv.zst")?,
        ))?);

        for result in rdr.deserialize() {
            let (e, m, mu, beta, f_fd, f_be, f_mb, _f_mj): Row8 = result.unwrap();
            // println!("(e, m, μ, β) = ({:e}, {:e}, {:e}, {:e})", e, m, mu, beta);

            if !f_fd.is_nan() {
                let f = fd.phase_space(e, m, mu, beta);
                approx_eq(f_fd, f, 10.0, 0.0)?;
            }
            if !f_be.is_nan() {
                let f = be.phase_space(e, m, mu, beta);
                approx_eq(f_be, f, 10.0, 0.0)?;
            }
            if !f_mb.is_nan() {
                let f = mb.phase_space(e, m, mu, beta);
                approx_eq(f_mb, f, 10.0, 0.0)?;
            }
        }

        Ok(())
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn massive() -> Result<(), Box<dyn error::Error>> {
        let fd = Statistic::FermiDirac;
        let be = Statistic::BoseEinstein;
        let mb = Statistic::MaxwellBoltzmann;

        let mut rdr = csv::Reader::from_reader(zstd::Decoder::with_buffer(BufReader::new(
            fs::File::open("tests/data/number_density_massive.csv.zst")?,
        ))?);

        for result in rdr.deserialize() {
            let (m, mu, beta, n_fd, n_be, n_mb, _n_mj): Row7 = result.unwrap();
            // println!("(m, μ, β) = ({:e}, {:e}, {:e})", m, mu, beta);

            if !n_fd.is_nan() {
                let n = fd.number_density(m, mu, beta);
                approx_eq(n_fd, n, 10.0, 1e-100)?;
            }
            if !n_be.is_nan() {
                let n = be.number_density(m, mu, beta);
                approx_eq(n_be, n, 10.0, 1e-100)?;
            }
            if !n_mb.is_nan() {
                // TODO: Check accuracy of Maxwell–Boltzmann distribution
                let n = mb.number_density(m, mu, beta);
                if !n.is_nan() {
                    approx_eq(n_mb, n, 7.0, 1e-100)?;
                }
            }
        }

        Ok(())
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn massless() -> Result<(), Box<dyn error::Error>> {
        let fd = Statistic::FermiDirac;
        let be = Statistic::BoseEinstein;
        let mb = Statistic::MaxwellBoltzmann;

        let mut rdr = csv::Reader::from_reader(zstd::Decoder::with_buffer(BufReader::new(
            fs::File::open("tests/data/number_density_massless.csv.zst")?,
        ))?);

        for result in rdr.deserialize() {
            let (mu, beta, n_fd, n_be, n_mb, _n_mj): Row6 = result.unwrap();
            // println!("(μ, β) = ({:e}, {:e})", mu, beta);

            if !n_fd.is_nan() {
                let n = fd.number_density(0.0, mu, beta);
                approx_eq(n_fd, n, 10.0, 1e-100)?;
            }
            if !n_be.is_nan() {
                let n = be.number_density(0.0, mu, beta);
                println!("μ = {:e}, β = {:e}, n = {:e}", mu, beta, n);
                approx_eq(n_be, n, 10.0, 1e-100)?;
            }
            if !n_mb.is_nan() {
                let n = mb.number_density(0.0, mu, beta);
                approx_eq(n_mb, n, 10.0, 1e-100)?;
            }
        }

        Ok(())
    }
}
