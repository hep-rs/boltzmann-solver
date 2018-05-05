use quadrature::integrate;
use special_functions::{bessel, polylog};
use std::f64;

use constants::PI_M2;

/// The statistics which describe the distribution of particles over energy
/// states.  Both Fermi--Dirac and Bose--Einstein quantum statistics are
/// implemented, as well as the classical Maxwell--Boltzmann statistic.
pub enum Statistic {
    /// Fermi--Dirac statistic describing half-integer-spin particles:
    ///
    /// \\[
    ///   f_{\textsc{FD}} = \frac{1}{\exp[(E - \mu) \beta] + 1}.
    /// \\]
    FermiDirac,
    /// Bose--Einstein statistic describing integer-spin particles:
    ///
    /// \\[
    ///   f_{\textsc{BE}} = \frac{1}{\exp[(E - \mu) \beta] - 1}.
    /// \\]
    BoseEinstein,
    /// Maxwell--Boltzmann statistic describing classical particles:
    ///
    /// \\[
    ///   f_{\textsc{MB}} = \exp[-(E - \mu) \beta].
    /// \\]
    MaxwellBoltzmann,
    /// Maxwell--Jüttner statistic describing relativistic classical particles:
    ///
    /// \\[
    ///   f_{\textsc{MJ}} = \frac{E \beta \sqrt{E\^2 - m^2}}{m K_2(m \beta)} \exp[- E \beta].
    /// \\]
    MaxwellJuttner,
}

impl Statistic {
    /// Evaluate the phase space distribution, \\(f\\) as defined above.
    pub fn phase_space(&self, e: f64, m: f64, mu: f64, beta: f64) -> f64 {
        match *self {
            Statistic::FermiDirac => 1.0 / (f64::exp((e - mu) * beta) + 1.0),
            Statistic::BoseEinstein => 1.0 / (f64::exp((e - mu) * beta) - 1.0),
            Statistic::MaxwellBoltzmann => f64::exp(-(e - mu) * beta),
            Statistic::MaxwellJuttner => {
                beta * e * f64::sqrt(e.powi(2) - m.powi(2)) * (-e * beta).exp()
                    / (m * bessel::k_2(m * beta))
            }
        }
    }

    /// Return number density for a particle following the specified statistic.
    ///
    /// \\[
    ///    n = \frac{1}{2 \pi\^2} \int_{m}\^{\infty} f_{i} u \sqrt{u\^2 - m\^2} \dd u
    /// \\]
    ///
    /// # Implementation Details
    ///
    /// If an analytic closed form is available for the integral, it will be
    /// preferred over the numerical integration.  Failing that, the
    /// semi-infinite integral is numerical evaluated.  In order to make the
    /// calculation tractable on a computer, the change of variables
    ///
    /// \\[
    ///   u \to m + \frac{t}{1 - t}, \qquad \dd u \to \frac{\dd t}{(t - 1)\^2}
    /// \\]
    ///
    /// is used such that the bounds of the integral over \\(t\\) are \\([0,
    /// 1]\\).
    pub fn number_density(&self, mass: f64, mu: f64, beta: f64) -> f64 {
        debug_assert!(mass >= 0.0, "mass must be positive.");
        debug_assert!(beta >= 0.0, "β must be positive.");

        if mass < 1e-20 {
            debug!("mass is below threshold, using massless_number_density instead.");
            return self.massless_number_density(mu, beta);
        }

        match *self {
            Statistic::FermiDirac => {
                debug_assert_warn!(
                    mu > 0.9 * mass,
                    "Evaluation of number densities for μ > 0.9 m can be inaccurate."
                );

                let integral = integrate(
                    |t| {
                        let u = mass + t / (1.0 - t);
                        let dudt = (t - 1.0).powi(-2);

                        self.phase_space(u, mass, mu, beta) * u
                            * f64::sqrt(u.powi(2) - mass.powi(2)) * dudt
                    },
                    0.0,
                    1.0,
                    1e-12,
                );
                debug!(
                    "Fermi–Dirac integral: {:e} ± {:e} ({} function evaluations)",
                    integral.integral, integral.error_estimate, integral.num_function_evaluations
                );
                // 1/(2 π²) ≅ 0.050660591821168885722
                0.050660591821168885722 * integral.integral
            }
            Statistic::BoseEinstein => {
                let integral = integrate(
                    |t| {
                        let u = mass + t / (1.0 - t);
                        let dudt = (t - 1.0).powi(-2);

                        self.phase_space(u, mass, mu, beta) * u
                            * f64::sqrt(u.powi(2) - mass.powi(2)) * dudt
                    },
                    0.0,
                    1.0,
                    1e-12,
                );
                debug!(
                    "Bose–Einstein integral: {:e} ± {:e} ({} function evaluations)",
                    integral.integral, integral.error_estimate, integral.num_function_evaluations
                );
                // 1/(2 π²) ≅ 0.050660591821168885722
                0.050660591821168885722 * integral.integral
            }
            Statistic::MaxwellBoltzmann => {
                // 1/(2 π²) ≅ 0.050660591821168885722
                0.050660591821168885722 * mass.powi(2) * bessel::k_2(mass * beta)
                    * f64::exp(mu * beta) / beta
            }
            Statistic::MaxwellJuttner => {
                let m_beta = mass * beta;
                PI_M2 * (m_beta + 2.0) * (m_beta * (m_beta + 3.0) + 6.0)
                    / (beta.powi(4) * mass * f64::exp(m_beta) * bessel::k_2(m_beta))
            }
        }
    }

    /// Return number density for a massless particle following the specified
    /// statistic.
    ///
    /// \\[
    ///    n = \frac{1}{2 \pi\^2} \int_{0}\^{\infty} f_{i} u\^2 \dd u
    /// \\]
    ///
    /// This is theoretically equivalent to setting `mass = 0` in
    /// [`Statistic::number_density`]; however, as there are analytic closed
    /// forms of the above integral for all statistics, this method will be much
    /// faster and more precise.
    pub fn massless_number_density(&self, mu: f64, beta: f64) -> f64 {
        debug_assert!(beta >= 0.0, "β must be positive.");

        match *self {
            Statistic::FermiDirac => PI_M2 * polylog::fermi_dirac(mu * beta) * beta.powi(-3),
            Statistic::BoseEinstein => {
                debug_assert!(
                    mu <= 0.0,
                    "Bose–Einstein condensates (μ > 0) are not supported."
                );
                PI_M2 * polylog::bose_einstein(mu * beta) * beta.powi(-3)
            }
            Statistic::MaxwellBoltzmann => PI_M2 * f64::exp(mu * beta) * beta.powi(-3),
            Statistic::MaxwellJuttner => 0.0,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::f64;
    use utilities::test::*;
    use csv;

    type Row6 = (f64, f64, f64, f64, f64, f64);
    type Row7 = (f64, f64, f64, f64, f64, f64, f64);
    type Row8 = (f64, f64, f64, f64, f64, f64, f64, f64);

    #[test]
    fn phase_space() {
        let fd = Statistic::FermiDirac;
        let be = Statistic::BoseEinstein;
        let mb = Statistic::MaxwellBoltzmann;
        let mj = Statistic::MaxwellJuttner;

        let mut rdr = csv::Reader::from_path("test/data/phase_space.csv").unwrap();

        for result in rdr.deserialize() {
            let (e, m, mu, beta, f_fd, f_be, f_mb, f_mj): Row8 = result.unwrap();
            println!("(e, m, μ, β) = ({:e}, {:e}, {:e}, {:e})", e, m, mu, beta);

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

        let mut rdr = csv::Reader::from_path("test/data/number_density_massive.csv").unwrap();

        for result in rdr.deserialize() {
            let (m, mu, beta, n_fd, n_be, n_mb, n_mj): Row7 = result.unwrap();
            println!("(m, μ, β) = ({:e}, {:e}, {:e})", m, mu, beta);

            if !n_fd.is_nan() {
                let n = fd.number_density(m, mu, beta);
                approx_eq(n_fd, n, 10.0, 1e-100);
            }
            if !n_be.is_nan() {
                let n = be.number_density(m, mu, beta);
                approx_eq(n_be, n, 10.0, 1e-100);
            }
            if !n_mb.is_nan() {
                let n = mb.number_density(m, mu, beta);
                if !n.is_nan() {
                    approx_eq(n_mb, n, 10.0, 1e-100);
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

        let mut rdr = csv::Reader::from_path("test/data/number_density_massless.csv").unwrap();

        for result in rdr.deserialize() {
            let (mu, beta, n_fd, n_be, n_mb, n_mj): Row6 = result.unwrap();
            println!("(μ, β) = ({:e}, {:e})", mu, beta);

            if !n_fd.is_nan() {
                let n = fd.massless_number_density(mu, beta);
                approx_eq(n_fd, n, 10.0, 1e-100);
            }
            if !n_be.is_nan() {
                let n = be.massless_number_density(mu, beta);
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

#[cfg(feature = "nightly")]
#[cfg(test)]
mod bench {
    use super::*;
    use test::{black_box, Bencher};
    use csv;

    type Row7 = (f64, f64, f64, f64, f64, f64, f64);

    const STEP_SIZE: usize = 10;

    #[bench]
    fn fermi_dirac_massive(b: &mut Bencher) {
        let fd = Statistic::FermiDirac;
        let rdr = csv::Reader::from_path("test/data/number_density_massive.csv").unwrap();
        let data: Vec<Row7> = rdr.into_deserialize()
            .step_by(STEP_SIZE)
            .map(|r| r.unwrap())
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
        let rdr = csv::Reader::from_path("test/data/number_density_massive.csv").unwrap();
        let data: Vec<Row7> = rdr.into_deserialize()
            .step_by(STEP_SIZE)
            .map(|r| r.unwrap())
            .collect();

        b.iter(|| {
            for &(m, mu, beta, _, _, _, _) in &data {
                let n = be.number_density(m, mu, beta);
                black_box(n);
            }
        });
    }

    #[bench]
    fn maxwell_boltzmann_massive(b: &mut Bencher) {
        let mb = Statistic::MaxwellBoltzmann;
        let rdr = csv::Reader::from_path("test/data/number_density_massive.csv").unwrap();
        let data: Vec<Row7> = rdr.into_deserialize()
            .step_by(STEP_SIZE)
            .map(|r| r.unwrap())
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
        let rdr = csv::Reader::from_path("test/data/number_density_massive.csv").unwrap();
        let data: Vec<Row7> = rdr.into_deserialize()
            .step_by(STEP_SIZE)
            .map(|r| r.unwrap())
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
        let rdr = csv::Reader::from_path("test/data/number_density_massive.csv").unwrap();
        let data: Vec<Row7> = rdr.into_deserialize()
            .step_by(STEP_SIZE)
            .map(|r| r.unwrap())
            .collect();

        b.iter(|| {
            for &(_, mu, beta, _, _, _, _) in &data {
                let n = fd.massless_number_density(mu, beta);
                black_box(n);
            }
        });
    }

    #[bench]
    fn bose_einstein_massless(b: &mut Bencher) {
        let be = Statistic::BoseEinstein;
        let rdr = csv::Reader::from_path("test/data/number_density_massive.csv").unwrap();
        let data: Vec<Row7> = rdr.into_deserialize()
            .step_by(STEP_SIZE)
            .map(|r| r.unwrap())
            .collect();

        b.iter(|| {
            for &(_, mu, beta, _, _, _, _) in &data {
                let n = be.massless_number_density(mu, beta);
                black_box(n);
            }
        });
    }

    #[bench]
    fn maxwell_boltzmann_massless(b: &mut Bencher) {
        let mb = Statistic::MaxwellBoltzmann;
        let rdr = csv::Reader::from_path("test/data/number_density_massive.csv").unwrap();
        let data: Vec<Row7> = rdr.into_deserialize()
            .step_by(STEP_SIZE)
            .map(|r| r.unwrap())
            .collect();

        b.iter(|| {
            for &(_, mu, beta, _, _, _, _) in &data {
                let n = mb.massless_number_density(mu, beta);
                black_box(n);
            }
        });
    }

    #[bench]
    fn maxwell_juttner_massless(b: &mut Bencher) {
        let mj = Statistic::MaxwellJuttner;
        let rdr = csv::Reader::from_path("test/data/number_density_massive.csv").unwrap();
        let data: Vec<Row7> = rdr.into_deserialize()
            .step_by(STEP_SIZE)
            .map(|r| r.unwrap())
            .collect();

        b.iter(|| {
            for &(_, mu, beta, _, _, _, _) in &data {
                let n = mj.massless_number_density(mu, beta);
                black_box(n);
            }
        });
    }
}
