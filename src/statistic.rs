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

use quadrature::double_exponential::integrate;
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

/// Threshold value to determine whether we can approximate the mass as 0.  It
/// is massless if `$m < k |\mu|$` where `$k$` is the threshold below.
const MASSLESS_THRESHOD: f64 = 1e-10;

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
    fn phase_space(&self, beta: f64, e: f64, m: f64, mu: f64) -> f64;

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
    fn number_density(&self, beta: f64, mass: f64, mu: f64) -> f64 {
        let integral = integrate(
            |t| {
                let u = mass + t / (1.0 - t);
                let dudt = (t - 1.0).powi(-2);

                self.phase_space(beta, u, mass, mu) * u * f64::sqrt(u.powi(2) - mass.powi(2)) * dudt
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
    fn number_density_asymmetry(&self, beta: f64, mass: f64, mu: f64) -> f64 {
        let integral = integrate(
            |t| {
                let u = mass + t / (1.0 - t);
                let dudt = (t - 1.0).powi(-2);

                (self.phase_space(beta, u, mass, mu) - self.phase_space(u, mass, -mu, beta))
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
    fn normalized_number_density(&self, beta: f64, mass: f64, mu: f64) -> f64 {
        self.number_density(beta, mass, mu) / Statistic::BoseEinstein.number_density(beta, 0.0, 0.0)
    }

    /// Return number density for a particle following the specified statistic,
    /// normalized to the number density of a massless boson with a single
    /// degree of freedom.
    fn normalized_number_density_asymmetry(&self, beta: f64, mass: f64, mu: f64) -> f64 {
        self.number_density_asymmetry(mass, mu, beta)
            / Statistic::BoseEinstein.number_density(beta, 0.0, 0.0)
    }
}

impl Statistics for Statistic {
    /// Evaluate the phase space distribution, `$f$` as defined above for the
    /// four statistics.
    fn phase_space(&self, beta: f64, e: f64, m: f64, mu: f64) -> f64 {
        debug_assert!(
            e >= m,
            "Energy cannot be less than the mass.\ne = {:e}, m = {:e}",
            e,
            m
        );
        match self {
            Statistic::FermiDirac => (f64::exp((e - mu) * beta) + 1.0).recip(),
            Statistic::BoseEinstein => {
                debug_assert!(
                    e > mu,
                    "Energy must be greater than the chemical potential.\ne = {:e}, μ = {:e}",
                    e,
                    mu
                );
                f64::exp_m1((e - mu) * beta).recip()
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
    /// regimes.  The Maxwell–Boltzmann distribution offers an exact
    /// implementations.
    fn number_density(&self, beta: f64, mass: f64, mu: f64) -> f64 {
        debug_assert!(mass >= 0.0, "mass must be positive.");
        debug_assert!(beta >= 0.0, "β must be positive.");

        match (
            self,
            mass == 0.0 || mass < mu * MASSLESS_THRESHOD,
            mu == 0.0,
        ) {
            (Statistic::FermiDirac, true, _) => statistics::fermi_dirac_massless(beta, mu),
            (Statistic::FermiDirac, false, true) => statistics::fermi_dirac_massive(beta, mass),
            (Statistic::FermiDirac, false, false) => {
                // 1/(2 π²) ≅ 0.050_660_591_821_168_89
                0.050_660_591_821_168_89
                    * fermi_dirac_integral(mass * beta, mu * beta)
                    * beta.powi(-3)
            }
            (Statistic::BoseEinstein, true, _) => statistics::bose_einstein_massless(beta, mu),
            (Statistic::BoseEinstein, false, true) => statistics::bose_einstein_massive(beta, mass),
            (Statistic::BoseEinstein, false, false) => {
                // 1/(2 π²) ≅ 0.050_660_591_821_168_89
                0.050_660_591_821_168_89
                    * bose_einstein_integral(mass * beta, mu * beta)
                    * beta.powi(-3)
            }
            (Statistic::MaxwellBoltzmann, true, _) => {
                // 1 / π² ≅ 0.101_321_183_642_337_78
                0.101_321_183_642_337_78 * f64::exp(mu * beta) / beta.powi(3)
            }
            (Statistic::MaxwellBoltzmann, false, _) => {
                let mu_exp = f64::exp(mu * beta);
                let m_beta = mass * beta;

                if mu_exp.is_infinite() && m_beta > 39.828_784_017_944_02 {
                    // In order to avoid having 0 * ∞, we use the series
                    // expansion which combines the exponential from the
                    // chemical potential and the Bessel function.
                    0.126_987_271_868_481_94
                        * f64::exp((mu - mass) * beta)
                        * f64::sqrt(m_beta)
                        * (0.5 * m_beta
                            + special_functions::approximations::polynomial(
                                m_beta.recip(),
                                &[
                                    0.9375,
                                    0.410_156_25,
                                    -0.153_808_593_75,
                                    0.158_615_112_304_687_5,
                                    -0.257_749_557_495_117_2,
                                    0.563_827_157_020_568_8,
                                    -1.540_456_339_716_911_3,
                                    5.030_552_734_388_038_5,
                                    -19.074_179_117_887_98,
                                    82.257_397_445_891_91,
                                    -397.265_839_937_546_16,
                                ],
                            ))
                        / beta.powi(3)
                } else {
                    // 1/(2 π²) ≅ 0.050_660_591_821_168_89
                    0.050_660_591_821_168_89 * mass.powi(2) * bessel::k2(m_beta) * mu_exp / beta
                }
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
    fn number_density_asymmetry(&self, beta: f64, mass: f64, mu: f64) -> f64 {
        debug_assert!(mass >= 0.0, "mass must be positive.");
        debug_assert!(beta >= 0.0, "β must be positive.");

        match (
            self,
            mass == 0.0 || mass < mu * MASSLESS_THRESHOD,
            mu == 0.0,
        ) {
            // If mu is 0, there never is any asymmetry.
            (_, _, true) => 0.0,
            (Statistic::FermiDirac, true, false) => {
                statistics::fermi_dirac_massless(mu, beta)
                    - statistics::fermi_dirac_massless(-mu, beta)
            }
            (Statistic::FermiDirac, false, false) => {
                // 1/(2 π²) ≅ 0.050_660_591_821_168_89
                0.050_660_591_821_168_89
                    * fermi_dirac_asymmetry_integral(mass * beta, mu * beta)
                    * beta.powi(-3)
            }
            (Statistic::BoseEinstein, true, false) => {
                debug_assert!(
                    mu.abs() == 0.0,
                    "|μ| must be less than the mass (m = {:e}, |μ| = {:e})",
                    mass,
                    mu.abs()
                );
                f64::NAN
            }
            (Statistic::BoseEinstein, false, false) => {
                debug_assert!(
                    mu.abs() < mass,
                    "|μ| must be less than the mass (m = {:e}, |μ| = {:e})",
                    mass,
                    mu.abs()
                );
                // 1/(2 π²) ≅ 0.050_660_591_821_168_89
                0.050_660_591_821_168_89
                    * bose_einstein_asymmetry_integral(mass * beta, mu * beta)
                    * beta.powi(-3)
            }
            (Statistic::MaxwellBoltzmann, true, false) => {
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
    fn normalized_number_density(&self, beta: f64, mass: f64, mu: f64) -> f64 {
        debug_assert!(mass >= 0.0, "mass must be positive.");
        debug_assert!(beta >= 0.0, "β must be positive.");

        match (
            self,
            mass == 0.0 || mass < mu * MASSLESS_THRESHOD,
            mu == 0.0,
        ) {
            (Statistic::FermiDirac, true, _) => {
                statistics::fermi_dirac_normalized_massless(beta, mu)
            }
            (Statistic::FermiDirac, false, true) => {
                statistics::fermi_dirac_normalized_massive(beta, mass)
            }
            (Statistic::FermiDirac, false, false) => {
                // 1/(2 ζ(3)) ≅ 0.41595368629035373
                0.415_953_686_290_353_73 * fermi_dirac_integral(mass * beta, mu * beta)
            }
            (Statistic::BoseEinstein, true, _) => {
                statistics::bose_einstein_normalized_massless(beta, mu)
            }
            (Statistic::BoseEinstein, false, true) => {
                statistics::bose_einstein_normalized_massive(beta, mass)
            }
            (Statistic::BoseEinstein, false, false) => {
                // 1/(2 ζ(3)) ≅ 0.41595368629035373
                0.415_953_686_290_353_73 * bose_einstein_integral(mass * beta, mu * beta)
            }
            (Statistic::MaxwellBoltzmann, true, _) => {
                // 1 / ζ(3) ≅ 0.8319073725807075
                0.831_907_372_580_707_5 * f64::exp(mu * beta)
            }
            (Statistic::MaxwellBoltzmann, _, _) => {
                let m_beta = mass * beta;
                let mu_exp = f64::exp(mu * beta);

                if mu_exp.is_infinite() && m_beta > 39.828_784_017_944_02 {
                    // In order to avoid having 0 * ∞, we use the series
                    // expansion which combines the exponential from the
                    // chemical potential and the Bessel function.
                    1.042_641_270_992_393_8
                        * f64::exp(beta * (mu - mass))
                        * f64::sqrt(m_beta)
                        * (0.5 * m_beta
                            + special_functions::approximations::polynomial(
                                m_beta.recip(),
                                &[
                                    0.9375,
                                    0.410_156_25,
                                    -0.153_808_593_75,
                                    0.158_615_112_304_687_5,
                                    -0.257_749_557_495_117_2,
                                    0.563_827_157_020_568_8,
                                    -1.540_456_339_716_911_3,
                                    5.030_552_734_388_038_5,
                                    -19.074_179_117_887_98,
                                    82.257_397_445_891_91,
                                    -397.265_839_937_546_16,
                                ],
                            ))
                } else {
                    // 1/(2 ζ(3)) ≅ 0.41595368629035373
                    0.415_953_686_290_353_73
                        * (m_beta).powi(2)
                        * bessel::k2(m_beta)
                        * f64::exp(mu * beta)
                }
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
    fn normalized_number_density_asymmetry(&self, beta: f64, mass: f64, mu: f64) -> f64 {
        debug_assert!(mass >= 0.0, "mass must be positive.");
        debug_assert!(beta >= 0.0, "β must be positive.");

        match (
            self,
            mass == 0.0 || mass < mu * MASSLESS_THRESHOD,
            mu == 0.0,
        ) {
            (_, _, true) => 0.0,
            (Statistic::FermiDirac, true, _) => {
                statistics::fermi_dirac_normalized_massless(beta, mu)
                    - statistics::fermi_dirac_normalized_massless(beta, -mu)
            }
            (Statistic::FermiDirac, false, false) => {
                // 1/(2 ζ(3)) ≅ 0.41595368629035373
                0.415_953_686_290_353_73 * fermi_dirac_asymmetry_integral(mass * beta, mu * beta)
            }
            (Statistic::BoseEinstein, true, _) => {
                statistics::bose_einstein_normalized_massless(beta, mu)
                    - statistics::bose_einstein_normalized_massless(beta, -mu)
            }
            (Statistic::BoseEinstein, false, false) => {
                // 1/(2 ζ(3)) ≅ 0.41595368629035373
                0.415_953_686_290_353_73 * bose_einstein_asymmetry_integral(mass * beta, mu * beta)
            }
            (Statistic::MaxwellBoltzmann, true, _) => {
                // 2 / ζ(3) ≅ 1.663814745161415
                1.663_814_745_161_415 * f64::sinh(mu * beta)
            }
            (Statistic::MaxwellBoltzmann, _, _) => {
                let m_beta = mass * beta;
                // 1 / ζ(3) ≅ 0.8319073725807075
                0.831_907_372_580_707_5
                    * (m_beta).powi(2)
                    * bessel::k2(m_beta)
                    * f64::sinh(mu * beta)
            }
        }
    }
}

fn recursive_integrate<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64, depth: u64) -> f64 {
    const EPS_REL: f64 = 1e-6;
    const EPS_ABS: f64 = 1e-200;
    const SUBDIV: f64 = 1e2;

    let result = if b.is_infinite() {
        let integrand = |y: f64| {
            let x = a + y / (1.0 - y);
            let dxdy = (y - 1.0).powi(-2);

            f(x) * dxdy
        };
        integrate(integrand, 0.0, 1.0, EPS_ABS)
    } else {
        integrate(f, a, b, EPS_ABS)
    };

    // if depth == 0 {
    //     dbg!("Max depth reached");
    // }

    if depth == 0
        || result.error_estimate < EPS_ABS
        || (result.error_estimate / result.integral).abs() < EPS_REL
    {
        result.integral
    } else if b.is_infinite() {
        recursive_integrate(f, a, SUBDIV * a, depth - 1)
            + recursive_integrate(f, SUBDIV * a, b, depth - 1)
    } else {
        recursive_integrate(f, a, (a + b) / 2.0, depth - 1)
            + recursive_integrate(f, (a + b) / 2.0, b, depth - 1)
    }
}

/// Computes the dimensionless Fermi-Dirac integral:
///
/// ```math
/// \mathcal{I}_{\textsc{FD}}(m, \mu) = \int_m^\infty \frac{u \sqrt{u^2 - m^2}}{e^{u - \mu} + 1} \dd u
/// ```
///
/// where `$m$` and `$\mu$` are dimensionless (typically from being multiplied
/// by the inverse temperature `$\beta$`).
fn fermi_dirac_integral(mass: f64, mu: f64) -> f64 {
    let integrand =
        |u: f64| f64::recip(f64::exp(u - mu) + 1.0) * u * f64::sqrt(u.powi(2) - mass.powi(2));
    recursive_integrate(&integrand, mass, f64::INFINITY, 16)

    // let integral = integrate(
    //     |t| {
    //         let u = mass + t / (1.0 - t);
    //         let dudt = (t - 1.0).powi(-2);

    //         f64::recip(f64::exp(u - mu) + 1.0) * u * f64::sqrt(u.powi(2) - mass.powi(2)) * dudt
    //     },
    //     0.0,
    //     1.0,
    //     1e-12,
    // );
    // integral.integral
}

/// Computes the dimensionless Bose-Einstein integral:
///
/// ```math
/// \mathcal{I}_{\textsc{BE}}(m, \mu) = \int_m^\infty \frac{u \sqrt{u^2 - m^2}}{e^{u - \mu} - 1} \dd u
/// ```
///
/// where `$m$` and `$\mu$` are dimensionless (typically from being multiplied
/// by the inverse temperature `$\beta$`).
fn bose_einstein_integral(mass: f64, mu: f64) -> f64 {
    let integrand =
        |u: f64| f64::recip(f64::exp(u - mu) - 1.0) * u * f64::sqrt(u.powi(2) - mass.powi(2));
    recursive_integrate(&integrand, mass, f64::INFINITY, 16)

    // let integral = integrate(
    //     |t| {
    //         let u = mass + t / (1.0 - t);
    //         let dudt = (t - 1.0).powi(-2);

    //         f64::recip(f64::exp_m1(u - mu)) * u * f64::sqrt(u.powi(2) - mass.powi(2)) * dudt
    //     },
    //     0.0,
    //     1.0,
    //     1e-12,
    // );

    // integral.integral
}

/// Computes the dimensionless Fermi-Dirac integral:
///
/// ```math
/// \mathcal{I}_{\textsc{FD}}(m, \mu) = \int_m^\infty \frac{u \sqrt{u^2 - m^2}}{e^{u - \mu} + 1} - \frac{u \sqrt{u^2 - m^2}}{e^{u + \mu} + 1}  \dd u
/// ```
///
/// where `$m$` and `$\mu$` are dimensionless (typically from being multiplied
/// by the inverse temperature `$\beta$`).
fn fermi_dirac_asymmetry_integral(mass: f64, mu: f64) -> f64 {
    let cosh_mu = f64::cosh(mu);
    let integrand =
        |u: f64| f64::recip(cosh_mu + f64::cosh(u)) * u * f64::sqrt(u.powi(2) - mass.powi(2));
    f64::sinh(mu) * recursive_integrate(&integrand, mass, f64::INFINITY, 16)

    // let integral = integrate(
    //     |t| {
    //         let u = mass + t / (1.0 - t);
    //         let dudt = (t - 1.0).powi(-2);

    //         -u * f64::sqrt(u.powi(2) - mass.powi(2)) / (cosh_mu + f64::cosh(u)) * dudt
    //     },
    //     0.0,
    //     1.0,
    //     1e-12,
    // );

    // f64::sinh(mu) * integral.integral
}

/// Computes the dimensionless Bose-Einstein integral:
///
/// ```math
/// \mathcal{I}_{\textsc{BE}}(m, \mu) = \int_m^\infty \frac{u \sqrt{u^2 - m^2}}{e^{u - \mu} - 1} - \frac{u \sqrt{u^2 - m^2}}{e^{u + \mu} - 1} \dd u
/// ```
///
/// where `$m$` and `$\mu$` are dimensionless (typically from being multiplied
/// by the inverse temperature `$\beta$`).
fn bose_einstein_asymmetry_integral(mass: f64, mu: f64) -> f64 {
    let cosh_mu = f64::cosh(mu);
    let integrand =
        |u: f64| f64::recip(cosh_mu - f64::cosh(u)) * u * f64::sqrt(u.powi(2) - mass.powi(2));
    f64::sinh(mu) * recursive_integrate(&integrand, mass, f64::INFINITY, 16)
    // let integral = integrate(
    //     |t| {
    //         let u = mass + t / (1.0 - t);
    //         let dudt = (t - 1.0).powi(-2);

    //         u * f64::sqrt(u.powi(2) - mass.powi(2)) / (cosh_mu - f64::cosh(u)) * dudt
    //     },
    //     0.0,
    //     1.0,
    //     1e-12,
    // );

    // f64::sinh(mu) * integral.integral
}

#[cfg(test)]
mod tests {
    use super::{Statistic, Statistics};
    use crate::utilities::test::approx_eq;
    use std::{error, f64, fs};

    #[test]
    #[allow(clippy::similar_names)]
    fn phase_space() -> Result<(), Box<dyn error::Error>> {
        let mut rdr = csv::Reader::from_reader(zstd::Decoder::new(fs::File::open(
            "tests/data/phase_space.csv.zst",
        )?)?);

        let f = [
            Statistic::FermiDirac,
            Statistic::BoseEinstein,
            Statistic::MaxwellBoltzmann,
        ];

        for (row, result) in rdr.deserialize().enumerate() {
            let data: [f64; 4 + 3] = result.unwrap();

            let beta = data[0];
            let e = data[1];
            let m = data[2];
            let mu = data[3];
            let y = &data[4..];

            for (i, stat) in f.iter().enumerate() {
                let yi = y[i];
                if !yi.is_nan() {
                    let nyi = stat.phase_space(beta, e, m, mu);
                    approx_eq(yi, nyi, 11.0, 10_f64.powi(-200)).map_err(|err| {
                        println!(
                            "[{}] f{}({:e}, {:e}, {:e}, {:e}) = {:e} but expected {:e}.",
                            row, i, beta, e, m, mu, nyi, yi
                        );
                        err
                    })?;
                }
            }
        }

        Ok(())
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn number_density() -> Result<(), Box<dyn error::Error>> {
        let mut rdr = csv::Reader::from_reader(zstd::Decoder::new(fs::File::open(
            "tests/data/number_density.csv.zst",
        )?)?);

        let f = [
            Statistic::BoseEinstein,
            Statistic::FermiDirac,
            Statistic::MaxwellBoltzmann,
        ];

        for (row, result) in rdr.deserialize().enumerate() {
            // if row != 16836 {
            //     continue;
            // };
            let data: [f64; 3 + 6] = result.unwrap();

            let beta = data[0];
            let m = data[1];
            let mu = data[2];
            let y = &data[3..];

            // TODO: Refine the number density calculations to work when the
            // chemical potential is large.
            if (mu / m).abs() > 1e-1 {
                continue;
            }

            for (i, &yi) in y.iter().enumerate() {
                if !yi.is_nan() {
                    let nyi = if i % 2 == 0 {
                        f[i / 2].number_density(beta, m, mu)
                    } else {
                        f[i / 2].normalized_number_density(beta, m, mu)
                    };
                    // For extremely large / infinite numbers, approximate them
                    // as equal.
                    if yi > f64::MAX / 1e2 && nyi > f64::MAX / 1e2 {
                        continue;
                    };
                    approx_eq(yi, nyi, 2.0, 10_f64.powi(-50)).map_err(|err| {
                        println!(
                            "[{}] n{}({:e}, {:e}, {:e}) = {:e} but expected {:e}.",
                            row, i, beta, m, mu, nyi, yi
                        );
                        err
                    })?;
                }
            }
        }

        Ok(())
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn number_density_asymmetry() -> Result<(), Box<dyn error::Error>> {
        let mut rdr = csv::Reader::from_reader(zstd::Decoder::new(fs::File::open(
            "tests/data/number_density_asymmetry.csv.zst",
        )?)?);

        let f = [
            Statistic::BoseEinstein,
            Statistic::FermiDirac,
            Statistic::MaxwellBoltzmann,
        ];

        for (row, result) in rdr.deserialize().enumerate() {
            let data: [f64; 3 + 6] = result.unwrap();

            let beta = data[0];
            let m = data[1];
            let mu = data[2];
            let y = &data[3..];

            // TODO: Refine the number density calculations to work when the
            // chemical potential is large.
            if (mu / m).abs() > 1e-1 {
                continue;
            }

            for (i, &yi) in y.iter().enumerate() {
                if !yi.is_nan() {
                    let nyi = if i % 2 == 0 {
                        f[i / 2].number_density_asymmetry(beta, m, mu)
                    } else {
                        f[i / 2].normalized_number_density_asymmetry(beta, m, mu)
                    };
                    approx_eq(yi, nyi, 10.0, 10_f64.powi(-200)).map_err(|err| {
                        println!(
                            "[{}] n{}({:e}, {:e}, {:e}) = {:e} but expected {:e}.",
                            row, i, beta, m, mu, nyi, yi
                        );
                        err
                    })?;
                }
            }
        }

        Ok(())
    }
}
