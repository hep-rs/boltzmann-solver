use quadrature::integrate;
use std::f64;

use constants::{PI_2, ZETA_3};

/// The statistics which describe the distribution of particles over energy
/// states.  Both Fermi--Dirac and Bose--Einstein quantum statistics are
/// implemented, as well as the classical Maxwell--Boltzmann statistic.
pub enum Statistic {
    /// Fermi--Dirac statistic describing half-integer-spin particles:
    ///
    /// \\[
    ///   f_{\textsc{fd}} = \frac{1}{\exp[(E - \mu) \beta] + 1}.
    /// \\]
    FermiDirac,
    /// Bose--Einstein statistic describing integer-spin particles:
    ///
    /// \\[
    ///   f_{\textsc{be}} = \frac{1}{\exp[(E - \mu) \beta] - 1}.
    /// \\]
    BoseEinstein,
    /// Maxwell--Boltzmann statistic describing classical particles:
    ///
    /// \\[
    ///   f_{\textsc{mb}} = \exp[-(E - \mu) \beta]
    /// \\]
    MaxwellBoltzmann,
}

impl Statistic {
    /// Evaluate the phase space distribution, \\(f\\).  They are defined as:
    ///
    /// \\[
    ///   f_{\textsc{fd}} = \frac{1}{\exp[(E - \mu) \beta] + 1};
    ///   f_{\textsc{be}} = \frac{1}{\exp[(E - \mu) \beta] - 1}; \quad \text{and}
    ///   f_{\textsc{mb}} = \exp[-(E - \mu) \beta].
    /// \\]
    pub fn phase_space(&self, e: f64, mu: f64, beta: f64) -> f64 {
        match *self {
            Statistic::FermiDirac => 1.0 / (f64::exp((e - mu) * beta) + 1.0),
            Statistic::BoseEinstein => 1.0 / (f64::exp((e - mu) * beta) - 1.0),
            Statistic::MaxwellBoltzmann => f64::exp(-(e - mu) * beta),
        }
    }

    /// Return number density for a particle following the specified statistic.
    ///
    /// \\[
    ///    n = \frac{1}{2 \pi\^2} \int_{m}\^{\infty} f(u, \mu, \beta) u \sqrt{u\^2 - m\^2} \dd u
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

                        self.phase_space(u, mu, beta) * u * f64::sqrt(u.powi(2) - mass.powi(2))
                            * dudt
                    },
                    0.0,
                    1.0,
                    1e-8,
                ).integral;
                integral / (2.0 * PI_2)
            }
            Statistic::BoseEinstein => {
                let integral = integrate(
                    |t| {
                        let u = mass + t / (1.0 - t);
                        let dudt = (t - 1.0).powi(-2);

                        self.phase_space(u, mu, beta) * u * f64::sqrt(u.powi(2) - mass.powi(2))
                            * dudt
                    },
                    0.0,
                    1.0,
                    1e-8,
                ).integral;
                integral / (2.0 * PI_2)
            }
            Statistic::MaxwellBoltzmann => {
                // Since K₂(x) exp(x) → 0 as x → ∞, we evaluate the bessel
                // function first and check if it is zero
                let k = besselk_2(mass * beta);
                if k == 0.0 {
                    0.0
                } else {
                    mass.powi(2) * k * f64::exp(beta * mu) / (2.0 * PI_2 * beta)
                }
            }
        }
    }

    /// Return number density for a massless particle following the specified
    /// statistic.
    ///
    /// \\[
    ///    n = \frac{1}{2 \pi\^2} \int_{0}\^{\infty} f(u, \mu, \beta) u\^2 \dd u
    /// \\]
    ///
    /// This is theoretically equivalent to setting `mass = 0` in
    /// [`Statistic::number_density`]; however, as there are analytic closed
    /// forms of the above integral for all statistics, this method will be much
    /// faster and more precise.
    pub fn massless_number_density(&self, mu: f64, beta: f64) -> f64 {
        debug_assert!(beta >= 0.0, "β must be positive.");

        match *self {
            Statistic::FermiDirac => beta.powi(-3) * polylog3_fd(mu * beta) / PI_2,
            Statistic::BoseEinstein => {
                debug_assert!(
                    mu <= 0.0,
                    "Bose–Einstein condensates (μ > 0) are not supported."
                );
                beta.powi(-3) * polylog3_be(mu * beta) / PI_2
            }
            Statistic::MaxwellBoltzmann => f64::exp(mu * beta) / (PI_2 * beta.powi(3)),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Approximations of special functions
////////////////////////////////////////////////////////////////////////////////

/// Approximation of polylogarithm appearing in the Bose–Einstein statistics.
/// Specifically, this approximates the function \\(\Li_{3} e\^x\\) for \\(x
/// \leq 0\\).
///
/// The approximation is split in three regimes:
///
/// - \\(|x| \ll 1\\), where the Taylor series expansion around \\(x = 0\\) is
///   used;
/// - \\(|x| \approx 1\\), where a mini-max polynomial is used which ensures the
///   error is spread over the whole interval and overall bias is minimized.
///   (This is unlike a Taylor series where the error is minimized at a single
///   point and increases (sometimes rapidly) everywhere else);
/// - \\(|x| \gg 1\\), where the Taylor series expansion around \\(x = \infty\\)
///   is used.
fn polylog3_be(x: f64) -> f64 {
    debug_assert!(x <= 0.0, "Argument must be negative");

    let x = -x;

    if x == 0.0 {
        ZETA_3
    } else if x < 0.3 {
        // Use the Taylor series expansion around x = 0
        ZETA_3 - 1.6449340668482264365 * x + x.powi(2) * (0.75 - 0.5 * x.ln()) + x.powi(3) / 12.0
            - x.powi(4) / 288.0 + x.powi(6) / 86_400.0
    } else if x < 2.9 {
        // Use an optimal rational approximation over the interval [0.3, 2.9].
        let num =
            1.2017351736193496426 + 11.130386039166276983 * x + 16.763142740004630592 * x.powi(2)
                - 3.5866918478417107386 * x.powi(3) - 2.5677885464330382624 * x.powi(4)
                + 1.3990498418553116275 * x.powi(5) - 0.31987635479827042139 * x.powi(6)
                + 0.041496218824634329175 * x.powi(7)
                - 0.0030484455959885731970 * x.powi(8)
                + 0.00010057186312115009437 * x.powi(9);
        let denom =
            1.0000000000000000000 + 10.610210615418754674 * x + 26.459498162787458568 * x.powi(2)
                + 19.636027801659040900 * x.powi(3) + 4.2282059978048909782 * x.powi(4)
                + 0.23347307007870895240 * x.powi(5);

        num / denom
    } else {
        // Use the Taylor series expansion around x = +∞
        let ex = (-x).exp();

        ex + ex.powi(2) / 8.0 + ex.powi(3) / 27.0 + ex.powi(3) / 64.0 + ex.powi(4) / 64.0
            + ex.powi(5) / 125.0 + ex.powi(6) / 216.0 + ex.powi(7) / 343.0
    }
}

/// Approximation of polylogarithm appearing in the Fermi–Dirac statistics.
/// Specifically, this approximates the function \\(-\Li_{3} (-e\^x)\\) for all
/// values of \\(x\\).
///
/// The approximation is split in three regimes:
///
/// - \\(|x| \ll 1\\), where the Taylor series expansion around \\(x = 0\\) is
///   used;
/// - \\(|x| \approx 1\\), where a mini-max polynomial is used which ensures the
///   error is spread over the whole interval and overall bias is minimized.
///   (This is unlike a Taylor series where the error is minimized at a single
///   point and increases (sometimes rapidly) everywhere else);
/// - \\(|x| \gg 1\\), where the Taylor series expansion around \\(x = \infty\\)
///   is used.
fn polylog3_fd(x: f64) -> f64 {
    if x < -2.0 {
        let ex = x.exp();

        ex - ex.powi(2) / 8.0 + ex.powi(3) / 27.0 - ex.powi(4) / 64.0 + ex.powi(5) / 125.0
            - ex.powi(6) / 216.0 + ex.powi(7) / 343.0
    } else if x < 2.0 {
        let num = 0.90154267737269372427 + 0.79013622598610500367 * x
            + 0.38420797837830254983 * x.powi(2)
            + 0.13046328601539294844 * x.powi(3)
            + 0.032584861964205293045 * x.powi(4)
            + 0.0059836402495576932393 * x.powi(5)
            + 0.00077875163435983431872 * x.powi(6)
            + 0.000065184512549694586562 * x.powi(7)
            + 2.6706623808079665948e-6 * x.powi(8);
        let denom = 1.0000000000000000000 - 0.035861649682456530419 * x
            + 0.074460604357617095904 * x.powi(2)
            - 0.0018665084270511311858 * x.powi(3)
            + 0.00098246043925023765414 * x.powi(4)
            - 9.9993514636981945645e-6 * x.powi(5);

        num / denom
    } else {
        let ex = (-x).exp();

        1.6449340668482264365 * x + x.powi(3) / 6.0 + ex - ex.powi(2) / 8.0 + ex.powi(3) / 27.0
            - ex.powi(4) / 64.0 + ex.powi(5) / 125.0 - ex.powi(6) / 216.0
            + ex.powi(7) / 343.0
    }
}

/// Approximation of modified Bessel function \\(K_2(x)\\).
///
/// The approximation is split in three regimes:
///
/// - \\(|x| \ll 1\\), where the Taylor series expansion around \\(x = 0\\) is
///   used;
/// - \\(|x| \approx 1\\), where a mini-max polynomial is used which ensures the
///   error is spread over the whole interval and overall bias is minimized.
///   (This is unlike a Taylor series where the error is minimized at a single
///   point and increases (sometimes rapidly) everywhere else);
/// - \\(|x| \gg 1\\), where the Taylor series expansion around \\(x = \infty\\)
///   is used.
fn besselk_2(x: f64) -> f64 {
    debug_assert!(x >= 0.0, "Argument of BesselK must be positive.");

    if x == 0.0 {
        return f64::INFINITY;
    }

    // The approximation is done in the log-transformed variable (that is, the
    // interpolation is of the function K₂(exp(x))).
    let xln = x.ln();

    if xln < 0.3 {
        let x2 = x.powi(2);

        -0.50000000000000000000 + x2 * (0.10824143945730155610 - 0.12500000000000000000 * xln)
            + 2.0 * x2.powi(-1)
            + x2.powi(2) * (0.015964564399219574120 - 0.010416666666666666667 * xln)
            + x2.powi(3) * (0.00062096294997561169124 - 0.00032552083333333333333 * xln)
            + x2.powi(4) * (0.000011796141758852787447 - 5.4253472222222222222e-6 * xln)
            + x2.powi(5) * (1.3465023364738628899e-7 - 5.6514033564814814815e-8 * xln)
            + x2.powi(6) * (1.0309882406219202048e-9 - 4.0367166832010582011e-10 * xln)
            + x2.powi(7) * (5.6763386749232758866e-12 - 2.1024566058338844797e-12 * xln)
    } else if xln < 3.4 {
        let num = -0.0000180238 * xln.powi(8) - 0.0000205089 * xln.powi(7)
            + 0.00064349 * xln.powi(6) - 0.0178769 * xln.powi(5)
            + 0.072644 * xln.powi(4) - 0.467429 * xln.powi(3)
            + 0.78293 * xln.powi(2) - 2.57649 * xln + 0.485409;
        let denom = 0.0000176583 * xln.powi(6) - 0.000523792 * xln.powi(5)
            + 0.00663565 * xln.powi(4) - 0.045622 * xln.powi(3)
            + 0.177564 * xln.powi(2) - 0.424493 * xln + 1.0;

        (num / denom).exp()
    } else {
        let ex = x.exp();
        if ex == f64::INFINITY {
            0.0
        } else {
            (1.4133050937925706925 - 0.64608232859088945941 * x + 0.39758912528670120579 * x.powi(2)
                - 0.38554096997498298743 * x.powi(3)
                + 1.0281092532666212998 * x.powi(4) + 2.3499640074665629710 * x.powi(5)
                + 1.2533141373155002512 * x.powi(6)) / (6.5 * xln).exp() / ex
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::f64;
    use utilities::test::*;

    // Pre-calculated values for the number densities, ordered as: (m, μ, β,
    // n(FD), n(BE), n(MB))
    pub(crate) const N_MASSIVE: [(f64, f64, f64, f64, f64, f64); 48] = [
        (
            0.,
            0.,
            0.001,
            9.134537117517981e7,
            1.2179382823357308e8,
            1.0132118364233777e8,
        ),
        (
            0.,
            0.,
            1.,
            0.09134537117517981,
            0.12179382823357308,
            0.10132118364233778,
        ),
        (
            0.,
            0.,
            100000.,
            9.13453711751798e-17,
            1.217938282335731e-16,
            1.0132118364233779e-16,
        ),
        (
            0.,
            0.,
            1e10,
            9.134537117517982e-32,
            1.2179382823357307e-31,
            1.0132118364233776e-31,
        ),
        (
            0.,
            -1.,
            0.001,
            9.126207294865051e7,
            1.216275875172078e8,
            1.0121991310240461e8,
        ),
        (
            0.,
            -1.,
            1.,
            0.03572229088117184,
            0.03921083444514606,
            0.0372739804171723,
        ),
        (0., -1., 100000., 0., 0., 0.),
        (0., -1., 1e10, 0., 0., 0.),
        (
            0.,
            -100.,
            0.001,
            8.335485239905512e7,
            1.07061981931697e8,
            9.16791981992802e7,
        ),
        (
            0.,
            -100.,
            1.,
            3.769225011298561e-45,
            3.7692250112985615e-45,
            3.769225011298561e-45,
        ),
        (0., -100., 100000., 0., 0., 0.),
        (0., -100., 1e10, 0., 0., 0.),
        (
            0.,
            0.5,
            0.001,
            9.138704662171356e7,
            f64::NAN,
            1.0137185690141802e8,
        ),
        (
            0.,
            0.5,
            1.,
            0.1429119703211247,
            f64::NAN,
            0.1670503906436362,
        ),
        (
            0.,
            0.5,
            100000.,
            0.0021108580008820372,
            f64::NAN,
            f64::INFINITY,
        ),
        (
            0.,
            0.5,
            1e10,
            0.0021108579925487032,
            f64::NAN,
            f64::INFINITY,
        ),
        (
            1.,
            0.,
            0.001,
            9.134535361757061e7,
            1.2179362303531149e8,
            1.013211583120911e8,
        ),
        (
            1.,
            0.,
            1.,
            0.07674843297292647,
            0.09007599694230539,
            0.08231530021891435,
        ),
        (1., 0., 100000., 0., 0., 0.),
        (1., 0., 1e10, 0., 0., 0.),
        (
            1.,
            -1.,
            0.001,
            9.126205540370326e7,
            1.2162741742152852e8,
            1.0121988779747552e8,
        ),
        (
            1.,
            -1.,
            1.,
            0.029460205682984904,
            0.03120816941577316,
            0.03028210664439372,
        ),
        (1., -1., 100000., 0., 0., 0.),
        (1., -1., 1e10, 0., 0., 0.),
        (
            1.,
            -100.,
            0.001,
            8.335483607631086e7,
            1.0706192235376576e8,
            9.16791752795252e7,
        ),
        (
            1.,
            -100.,
            1.,
            3.062191708033259e-45,
            3.062191708033259e-45,
            3.0621917080332595e-45,
        ),
        (1., -100., 100000., 0., 0., 0.),
        (1., -100., 1e10, 0., 0., 0.),
        (
            1.,
            0.5,
            0.001,
            9.1387029057771e7,
            1.2187702569412377e8,
            1.0137183155850305e8,
        ),
        (
            1.,
            0.5,
            1.,
            0.12172746331013472,
            0.16067639370683204,
            0.13571498637499102,
        ),
        (1., 0.5, 100000., 0., 0., 0.),
        (1., 0.5, 1e10, 0., 0., 0.),
        (1e8, 0., 0.001, 0., 0., 0.),
        (1e8, 0., 1., 0., 0., 0.),
        (1e8, 0., 100000., 0., 0., 0.),
        (1e8, 0., 1e10, 0., 0., 0.),
        (1e8, -1., 0.001, 0., 0., 0.),
        (1e8, -1., 1., 0., 0., 0.),
        (1e8, -1., 100000., 0., 0., 0.),
        (1e8, -1., 1e10, 0., 0., 0.),
        (1e8, -100., 0.001, 0., 0., 0.),
        (1e8, -100., 1., 0., 0., 0.),
        (1e8, -100., 100000., 0., 0., 0.),
        (1e8, -100., 1e10, 0., 0., 0.),
        (1e8, 0.5, 0.001, 0., 0., 0.),
        (1e8, 0.5, 1., 0., 0., 0.),
        (1e8, 0.5, 100000., 0., 0., 0.),
        (1e8, 0.5, 1e10, 0., 0., 0.),
    ];

    // Pre-calculated values for the number densities, ordered as: (m, μ, β,
    // n(FD), n(BE), n(MB))
    pub(crate) const N_MASSLESS: [(f64, f64, f64, f64, f64, f64); 16] = [
        (
            0.,
            0.,
            0.001,
            9.134537117517981e7,
            1.2179382823357308e8,
            1.0132118364233777e8,
        ),
        (
            0.,
            0.,
            1.,
            0.09134537117517981,
            0.12179382823357308,
            0.10132118364233778,
        ),
        (
            0.,
            0.,
            100000.,
            9.13453711751798e-17,
            1.217938282335731e-16,
            1.0132118364233779e-16,
        ),
        (
            0.,
            0.,
            1e10,
            9.134537117517982e-32,
            1.2179382823357307e-31,
            1.0132118364233776e-31,
        ),
        (
            0.,
            -1.,
            0.001,
            9.126207294865051e7,
            1.216275875172078e8,
            1.0121991310240461e8,
        ),
        (
            0.,
            -1.,
            1.,
            0.03572229088117184,
            0.03921083444514606,
            0.0372739804171723,
        ),
        (0., -1., 100000., 0., 0., 0.),
        (0., -1., 1e10, 0., 0., 0.),
        (
            0.,
            -100.,
            0.001,
            8.335485239905512e7,
            1.07061981931697e8,
            9.16791981992802e7,
        ),
        (
            0.,
            -100.,
            1.,
            3.769225011298561e-45,
            3.7692250112985615e-45,
            3.769225011298561e-45,
        ),
        (0., -100., 100000., 0., 0., 0.),
        (0., -100., 1e10, 0., 0., 0.),
        (
            0.,
            0.5,
            0.001,
            9.138704662171356e7,
            f64::NAN,
            1.0137185690141802e8,
        ),
        (
            0.,
            0.5,
            1.,
            0.1429119703211247,
            f64::NAN,
            0.1670503906436362,
        ),
        (
            0.,
            0.5,
            100000.,
            0.0021108580008820372,
            f64::NAN,
            f64::INFINITY,
        ),
        (
            0.,
            0.5,
            1e10,
            0.0021108579925487032,
            f64::NAN,
            f64::INFINITY,
        ),
    ];

    #[test]
    fn massive() {
        let fd = Statistic::FermiDirac;
        let be = Statistic::BoseEinstein;
        let mb = Statistic::MaxwellBoltzmann;

        for &(m, mu, beta, n_fd, n_be, n_mb) in N_MASSIVE.iter() {
            debug!("(m, μ, β) = ({}, {}, {})", m, mu, beta);

            if !n_fd.is_nan() {
                let n_fd_calc = fd.number_density(m, mu, beta);
                approx_eq(n_fd, n_fd_calc, 9.0, 1e-40);
            }

            if !n_be.is_nan() {
                let n_be_calc = be.number_density(m, mu, beta);
                approx_eq(n_be, n_be_calc, 9.0, 1e-40);
            }

            if !n_mb.is_nan() {
                let n_mb_calc = mb.number_density(m, mu, beta);
                approx_eq(n_mb, n_mb_calc, 9.0, 1e-40);
            }
        }
    }

    #[test]
    fn massless() {
        let fd = Statistic::FermiDirac;
        let be = Statistic::BoseEinstein;
        let mb = Statistic::MaxwellBoltzmann;

        for &(m, mu, beta, n_fd, n_be, n_mb) in N_MASSLESS.iter() {
            debug!("(m, μ, β) = ({}, {}, {})", m, mu, beta);

            if !n_fd.is_nan() {
                let n_fd_calc = fd.massless_number_density(mu, beta);
                approx_eq(n_fd, n_fd_calc, 9.0, 0.0);
            }

            if !n_be.is_nan() {
                let n_be_calc = be.massless_number_density(mu, beta);
                approx_eq(n_be, n_be_calc, 9.0, 0.0);
            }

            if !n_mb.is_nan() {
                let n_mb_calc = mb.massless_number_density(mu, beta);
                approx_eq(n_mb, n_mb_calc, 9.0, 0.0);
            }
        }
    }
}

#[cfg(feature = "nightly")]
#[cfg(test)]
mod bench {
    use super::*;
    use super::test::{N_MASSIVE, N_MASSLESS};
    use test::Bencher;
    use utilities::test::*;

    #[bench]
    fn fermi_dirac_massive(b: &mut Bencher) {
        let fd = Statistic::FermiDirac;

        b.iter(|| {
            for &(m, mu, beta, n_fd, _, _) in N_MASSIVE.iter() {
                if n_fd.is_nan() {
                    continue;
                }
                let n_fd_calc = fd.number_density(m, mu, beta);
                approx_eq(n_fd, n_fd_calc, 3.0, 1e-15);
            }
        });
    }

    #[bench]
    fn bose_einstein_massive(b: &mut Bencher) {
        let be = Statistic::BoseEinstein;

        b.iter(|| {
            for &(m, mu, beta, _, n_be, _) in N_MASSIVE.iter() {
                if n_be.is_nan() {
                    continue;
                }
                let n_be_calc = be.number_density(m, mu, beta);
                approx_eq(n_be, n_be_calc, 3.0, 1e-15);
            }
        });
    }

    #[bench]
    fn maxwell_boltzmann_massive(b: &mut Bencher) {
        let mb = Statistic::MaxwellBoltzmann;

        b.iter(|| {
            for &(m, mu, beta, _, _, n_mb) in N_MASSIVE.iter() {
                if n_mb.is_nan() {
                    continue;
                }
                let n_mb_calc = mb.number_density(m, mu, beta);
                approx_eq(n_mb, n_mb_calc, 3.0, 0.0);
            }
        });
    }

    #[bench]
    fn fermi_dirac_massless(b: &mut Bencher) {
        let fd = Statistic::FermiDirac;

        b.iter(|| {
            for &(_, mu, beta, n_fd, _, _) in N_MASSLESS.iter() {
                if n_fd.is_nan() {
                    continue;
                }
                let n_fd_calc = fd.massless_number_density(mu, beta);
                approx_eq(n_fd, n_fd_calc, 3.0, 1e-15);
            }
        });
    }

    #[bench]
    fn bose_einstein_massless(b: &mut Bencher) {
        let be = Statistic::BoseEinstein;

        b.iter(|| {
            for &(_, mu, beta, _, n_be, _) in N_MASSLESS.iter() {
                if n_be.is_nan() {
                    continue;
                }
                let n_be_calc = be.massless_number_density(mu, beta);
                approx_eq(n_be, n_be_calc, 3.0, 1e-15);
            }
        });
    }

    #[bench]
    fn maxwell_boltzmann_massless(b: &mut Bencher) {
        let mb = Statistic::MaxwellBoltzmann;

        b.iter(|| {
            for &(_, mu, beta, _, _, n_mb) in N_MASSLESS.iter() {
                if n_mb.is_nan() {
                    continue;
                }
                let n_mb_calc = mb.massless_number_density(mu, beta);
                approx_eq(n_mb, n_mb_calc, 3.0, 0.0);
            }
        });
    }
}
