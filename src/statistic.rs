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

use crate::constants::PI_N2;
use quadrature::integrate;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use special_functions::{bessel, particle_statistics};
use std::{f64, fmt};

/// Equilibrium number density for massless bosons, normalized to the
/// equilibrium number density of a massless boson.  This is specified per
/// degree of freedom (that is \\(g = 1\\)).
pub const BOSON_EQ_DENSITY: f64 = 1.0;
/// Equilibrium number density for massless fermions, normalized to the
/// equilibrium number density of a massless boson.  This is specified per
/// degree of freedom (that is \\(g = 1\\))
pub const FERMION_EQ_DENSITY: f64 = 0.75;

/// The statistics which describe the distribution of particles over energy
/// states.  Both Fermi–Dirac and Bose–Einstein quantum statistics are
/// implemented, as well as the classical Maxwell–Boltzmann statistic.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub enum Statistic {
    /// Fermi–Dirac statistic describing half-integer-spin particles:
    ///
    /// \\begin{equation}
    ///   f_{\textsc{FD}} = \frac{1}{\exp[(E - \mu) \beta] + 1}.
    /// \\end{equation}
    FermiDirac,
    /// Bose–Einstein statistic describing integer-spin particles:
    ///
    /// \\begin{equation}
    ///   f_{\textsc{BE}} = \frac{1}{\exp[(E - \mu) \beta] - 1}.
    /// \\end{equation}
    BoseEinstein,
    /// Maxwell–Boltzmann statistic describing classical particles:
    ///
    /// \\begin{equation}
    ///   f_{\textsc{MB}} = \exp[-(E - \mu) \beta].
    /// \\end{equation}
    MaxwellBoltzmann,
    /// Maxwell–Jüttner statistic describing relativistic classical particles:
    ///
    /// \\begin{equation}
    ///   f_{\textsc{MJ}} = \frac{E \beta \sqrt{E^2 - m^2}}{m K_2(m \beta)} \exp[- E \beta].
    /// \\end{equation}
    MaxwellJuttner,
}

impl fmt::Display for Statistic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FermiDirac => write!(f, "Statistic::FermiDirac"),
            Self::BoseEinstein => write!(f, "Statistic::BoseEinstein"),
            Self::MaxwellBoltzmann => write!(f, "Statistic::MaxwellBoltzmann"),
            Self::MaxwellJuttner => write!(f, "Statistic::MaxwellJuttner"),
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
    /// \\begin{equation}
    ///    n = \frac{1}{2 \pi^2} \int_{m}^{\infty} f_{i} u \sqrt{u^2 - m^2} \dd u.
    /// \\end{equation}
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

    /// Return number density for a particle following the specified statistic,
    /// normalized to the number density of a massless boson with a single
    /// degree of freedom.
    fn normalized_number_density(&self, mass: f64, mu: f64, beta: f64) -> f64;

    /// Return number density for a massless particle following the specified
    /// statistic.
    ///
    /// \\begin{equation}
    ///    n = \frac{1}{2 \pi^2} \int_{0}^{\infty} f_{i} u^2 \dd u
    /// \\end{equation}
    ///
    /// The naïve implementation simply calls [`Statistics::number_density`]
    /// setting `mass = 0.0` and then uses numerical integration.
    ///
    /// # Note to Implementors
    ///
    /// If an analytic closed form is available for the integral, it should be
    /// preferred over the numerical integration.
    fn massless_number_density(&self, mu: f64, beta: f64) -> f64 {
        self.number_density(0.0, mu, beta)
    }
}

impl Statistics for Statistic {
    /// Evaluate the phase space distribution, \\(f\\) as defined above for the
    /// four statistics.
    fn phase_space(&self, e: f64, m: f64, mu: f64, beta: f64) -> f64 {
        match *self {
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
            Statistic::MaxwellJuttner => {
                // Check whether we'll likely have zero or not
                if ((m / e).powi(2) - 1.0).abs() < f64::EPSILON || e * beta > 700.0 {
                    0.0
                } else {
                    // Note that instead of `f64::sqrt(e.powi(2) - m.powi(2))`
                    // we use the more precise (but equivalent) form: `e *
                    // f64::sqrt(1.0 - (m / e).powi(2))`.
                    beta * e.powi(2) * f64::sqrt(1.0 - (m / e).powi(2))
                        / (f64::exp(e * beta) * m * bessel::k2(m * beta))
                }
            }
        }
    }

    /// Return number density for a particle following the specified statistic.
    ///
    /// \\begin{equation}
    ///    n = \frac{1}{2 \pi^2} \int_{m}^{\infty} f_{i} u \sqrt{u^2 - m^2} \dd u
    /// \\end{equation}
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

        match *self {
            Statistic::FermiDirac => {
                if mu == 0.0 {
                    particle_statistics::fermi_dirac_massive(mass, beta)
                } else {
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
                    if cfg!(debug_assertions) && integral.error_estimate > 0.01 * integral.integral
                    {
                        log::warn!(
                            "Fermi–Dirac integral has a relative error of {:0.2}%",
                            integral.error_estimate / integral.integral.abs()
                        );
                    }
                    // 1/(2 π²) ≅ 0.050_660_591_821_168_89
                    0.050_660_591_821_168_89 * integral.integral
                }
            }
            Statistic::BoseEinstein => {
                if mu == 0.0 {
                    particle_statistics::bose_einstein_massive(mass, beta)
                } else {
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
            Statistic::MaxwellBoltzmann => {
                // 1/(2 π²) ≅ 0.050_660_591_821_168_89
                0.050_660_591_821_168_89
                    * mass.powi(2)
                    * bessel::k2(mass * beta)
                    * f64::exp(mu * beta)
                    / beta
            }
            Statistic::MaxwellJuttner => {
                let m_beta = mass * beta;
                PI_N2 * (m_beta + 2.0) * (m_beta * (m_beta + 3.0) + 6.0)
                    / (beta.powi(4) * mass * f64::exp(m_beta) * bessel::k2(m_beta))
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
        if mu == 0.0 {
            match *self {
                Statistic::BoseEinstein => {
                    particle_statistics::bose_einstein_normalized(mass, beta)
                }
                Statistic::FermiDirac => particle_statistics::fermi_dirac_normalized(mass, beta),
                Statistic::MaxwellBoltzmann | Statistic::MaxwellJuttner => {
                    self.number_density(mass, mu, beta)
                        / Statistic::BoseEinstein.massless_number_density(0.0, beta)
                }
            }
        } else {
            self.number_density(mass, mu, beta)
                / Statistic::BoseEinstein.massless_number_density(0.0, beta)
        }
    }

    /// Return number density for a massless particle following the specified
    /// statistic.
    ///
    /// \\begin{equation}
    ///    n = \frac{1}{2 \pi^2} \int_{0}^{\infty} f_{i} u^2 \dd u
    /// \\end{equation}
    ///
    /// # Implementation Details
    ///
    /// All four statistics have exact implementations and do not rely on any
    /// numerical integration.
    fn massless_number_density(&self, mu: f64, beta: f64) -> f64 {
        debug_assert!(beta >= 0.0, "β must be positive.");

        match *self {
            Statistic::FermiDirac => particle_statistics::fermi_dirac_massless(mu, beta),
            Statistic::BoseEinstein => {
                debug_assert!(
                    mu <= 0.0,
                    "Bose–Einstein condensates (μ > 0) are not supported."
                );
                particle_statistics::bose_einstein_massless(mu, beta)
            }
            Statistic::MaxwellBoltzmann => PI_N2 * f64::exp(mu * beta) * beta.powi(-3),
            Statistic::MaxwellJuttner => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::test::approx_eq;
    use std::f64;

    type Row6 = (f64, f64, f64, f64, f64, f64);
    type Row7 = (f64, f64, f64, f64, f64, f64, f64);
    type Row8 = (f64, f64, f64, f64, f64, f64, f64, f64);

    #[test]
    fn phase_space() {
        let fd = Statistic::FermiDirac;
        let be = Statistic::BoseEinstein;
        let mb = Statistic::MaxwellBoltzmann;
        let mj = Statistic::MaxwellJuttner;

        let mut rdr = csv::Reader::from_path("tests/data/phase_space.csv").unwrap();

        for result in rdr.deserialize() {
            let (e, m, mu, beta, f_fd, f_be, f_mb, f_mj): Row8 = result.unwrap();
            // println!("(e, m, μ, β) = ({:e}, {:e}, {:e}, {:e})", e, m, mu, beta);

            if !f_fd.is_nan() {
                let f = fd.phase_space(e, m, mu, beta);
                approx_eq(f_fd, f, 10.0, 0.0);
            }
            if !f_be.is_nan() {
                let f = be.phase_space(e, m, mu, beta);
                approx_eq(f_be, f, 10.0, 0.0);
            }
            if !f_mb.is_nan() {
                let f = mb.phase_space(e, m, mu, beta);
                approx_eq(f_mb, f, 10.0, 0.0);
            }
            if !f_mj.is_nan() {
                let f = mj.phase_space(e, m, mu, beta);
                approx_eq(f_mj, f, 10.0, 0.0);
            }
        }
    }

    #[test]
    fn massive() {
        let fd = Statistic::FermiDirac;
        let be = Statistic::BoseEinstein;
        let mb = Statistic::MaxwellBoltzmann;
        let mj = Statistic::MaxwellJuttner;

        let mut rdr = csv::Reader::from_path("tests/data/number_density_massive.csv").unwrap();

        for result in rdr.deserialize() {
            let (m, mu, beta, n_fd, n_be, n_mb, n_mj): Row7 = result.unwrap();
            // println!("(m, μ, β) = ({:e}, {:e}, {:e})", m, mu, beta);

            if !n_fd.is_nan() {
                let n = fd.number_density(m, mu, beta);
                approx_eq(n_fd, n, 10.0, 1e-100);
            }
            if !n_be.is_nan() {
                let n = be.number_density(m, mu, beta);
                approx_eq(n_be, n, 10.0, 1e-100);
            }
            if !n_mb.is_nan() {
                // TODO: Check accuracy of Maxwell–Boltzmann distribution
                let n = mb.number_density(m, mu, beta);
                if !n.is_nan() {
                    approx_eq(n_mb, n, 7.0, 1e-100);
                }
            }
            if !n_mj.is_nan() {
                let n = mj.number_density(m, mu, beta);
                if !n.is_nan() {
                    approx_eq(n_mj, n, 10.0, 1e-100);
                }
            }
        }
    }

    #[test]
    fn massless() {
        let fd = Statistic::FermiDirac;
        let be = Statistic::BoseEinstein;
        let mb = Statistic::MaxwellBoltzmann;
        let mj = Statistic::MaxwellJuttner;

        let mut rdr = csv::Reader::from_path("tests/data/number_density_massless.csv").unwrap();

        for result in rdr.deserialize() {
            let (mu, beta, n_fd, n_be, n_mb, n_mj): Row6 = result.unwrap();
            // println!("(μ, β) = ({:e}, {:e})", mu, beta);

            if !n_fd.is_nan() {
                let n = fd.massless_number_density(mu, beta);
                approx_eq(n_fd, n, 10.0, 1e-100);
            }
            if !n_be.is_nan() {
                let n = be.massless_number_density(mu, beta);
                println!("μ = {:e}, β = {:e}, n = {:e}", mu, beta, n);
                approx_eq(n_be, n, 10.0, 1e-100);
            }
            if !n_mb.is_nan() {
                let n = mb.massless_number_density(mu, beta);
                approx_eq(n_mb, n, 10.0, 1e-100);
            }
            if !n_mj.is_nan() {
                let n = mj.massless_number_density(mu, beta);
                approx_eq(n_mj, n, 10.0, 1e-100);
            }
        }
    }
}

#[cfg(all(test, feature = "nightly"))]
mod benches {
    use super::*;
    use std::result::Result;
    use test::{black_box, Bencher};

    type Row6 = (f64, f64, f64, f64, f64, f64);
    type Row7 = (f64, f64, f64, f64, f64, f64, f64);

    const STEP_SIZE: usize = 10;

    #[bench]
    fn fermi_dirac_massive(b: &mut Bencher) {
        let fd = Statistic::FermiDirac;
        let rdr = csv::Reader::from_path("tests/data/number_density_massive.csv").unwrap();
        let data: Vec<Row7> = rdr
            .into_deserialize()
            .step_by(STEP_SIZE)
            .map(Result::unwrap)
            .collect();

        b.iter(|| {
            for &(m, mu, beta, _, _, _, _) in &data {
                let n = fd.number_density(m, mu, beta);
                black_box(n);
            }
        });
    }

    #[bench]
    fn bose_einstein_massive(b: &mut Bencher) {
        let be = Statistic::BoseEinstein;
        let rdr = csv::Reader::from_path("tests/data/number_density_massive.csv").unwrap();
        let data: Vec<Row7> = rdr
            .into_deserialize()
            .step_by(STEP_SIZE)
            .map(Result::unwrap)
            .collect();

        b.iter(|| {
            for &(m, mu, beta, _, _, _, _) in &data {
                if mu > 0.0 {
                    continue;
                }

                let n = be.number_density(m, mu, beta);
                black_box(n);
            }
        });
    }

    #[bench]
    fn maxwell_boltzmann_massive(b: &mut Bencher) {
        let mb = Statistic::MaxwellBoltzmann;
        let rdr = csv::Reader::from_path("tests/data/number_density_massive.csv").unwrap();
        let data: Vec<Row7> = rdr
            .into_deserialize()
            .step_by(STEP_SIZE)
            .map(Result::unwrap)
            .collect();

        b.iter(|| {
            for &(m, mu, beta, _, _, _, _) in &data {
                let n = mb.number_density(m, mu, beta);
                black_box(n);
            }
        });
    }

    #[bench]
    fn maxwell_juttner_massive(b: &mut Bencher) {
        let mj = Statistic::MaxwellJuttner;
        let rdr = csv::Reader::from_path("tests/data/number_density_massive.csv").unwrap();
        let data: Vec<Row7> = rdr
            .into_deserialize()
            .step_by(STEP_SIZE)
            .map(Result::unwrap)
            .collect();

        b.iter(|| {
            for &(m, mu, beta, _, _, _, _) in &data {
                let n = mj.number_density(m, mu, beta);
                black_box(n);
            }
        });
    }

    #[bench]
    fn fermi_dirac_massless(b: &mut Bencher) {
        let fd = Statistic::FermiDirac;
        let rdr = csv::Reader::from_path("tests/data/number_density_massless.csv").unwrap();
        let data: Vec<Row6> = rdr
            .into_deserialize()
            .step_by(STEP_SIZE)
            .map(Result::unwrap)
            .collect();

        b.iter(|| {
            for &(mu, beta, _, _, _, _) in &data {
                let n = fd.massless_number_density(mu, beta);
                black_box(n);
            }
        });
    }

    #[bench]
    fn bose_einstein_massless(b: &mut Bencher) {
        let be = Statistic::BoseEinstein;
        let rdr = csv::Reader::from_path("tests/data/number_density_massless.csv").unwrap();
        let data: Vec<Row6> = rdr
            .into_deserialize()
            .step_by(STEP_SIZE)
            .map(Result::unwrap)
            .collect();

        b.iter(|| {
            for &(mu, beta, _, _, _, _) in &data {
                let n = be.massless_number_density(mu, beta);
                black_box(n);
            }
        });
    }

    #[bench]
    fn maxwell_boltzmann_massless(b: &mut Bencher) {
        let mb = Statistic::MaxwellBoltzmann;
        let rdr = csv::Reader::from_path("tests/data/number_density_massless.csv").unwrap();
        let data: Vec<Row6> = rdr
            .into_deserialize()
            .step_by(STEP_SIZE)
            .map(Result::unwrap)
            .collect();

        b.iter(|| {
            for &(mu, beta, _, _, _, _) in &data {
                let n = mb.massless_number_density(mu, beta);
                black_box(n);
            }
        });
    }

    #[bench]
    fn maxwell_juttner_massless(b: &mut Bencher) {
        let mj = Statistic::MaxwellJuttner;
        let rdr = csv::Reader::from_path("tests/data/number_density_massless.csv").unwrap();
        let data: Vec<Row6> = rdr
            .into_deserialize()
            .step_by(STEP_SIZE)
            .map(Result::unwrap)
            .collect();

        b.iter(|| {
            for &(mu, beta, _, _, _, _) in &data {
                let n = mj.massless_number_density(mu, beta);
                black_box(n);
            }
        });
    }
}
