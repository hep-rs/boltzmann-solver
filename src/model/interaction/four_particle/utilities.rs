mod integrate;

use super::M_BETA_THRESHOLD;
use crate::{constants::PI_5, prelude::ParticleData};
use special_functions::particle_physics::kallen_lambda_sqrt;
use std::f64;

/// Return the minimum and maximum value of the Mandelstam variable `$t$` based
/// on the four particle masses with particles 1 and 2 being initial state and
/// particles 3 and 4 being the final state particles.
///
/// Explicitly, the range of `$t$` values are:
///
/// ```math
/// \begin{aligned}
///   t &= \frac{1}{2} \left[
///       (m_1^2 + m_2^2 + m_3^2 + m_4^2) - s - \frac{(m_1^2 - m_2^2)(m_3^2 - m_4^2)}{s}
///     \right] \\
///     &\quad + \frac{\lambda^{\frac{1}{2}}(s, m_1^2, m_2^2) \lambda^{\frac{1}{2}}(s, m_3^2, m_4^2)}{2 s} \cos \theta
/// \end{aligned}
/// ```
#[must_use]
pub fn t_range(
    s: f64,
    p1: &ParticleData,
    p2: &ParticleData,
    p3: &ParticleData,
    p4: &ParticleData,
) -> (f64, f64) {
    // debug_assert!(
    //     s * (1.0 + 10.0 * f64::EPSILON) >= (p1.mass + p2.mass).powi(2),
    //     "s cannot be smaller than (m1 + m2)² (s = {:e}, (m1 + m2)² = {:e}).",
    //     s,
    //     (p1.mass + p2.mass).powi(2)
    // );
    // debug_assert!(
    //     s * (1.0 + 10.0 * f64::EPSILON) >= (p3.mass + p4.mass - 5.0 * f64::EPSILON).powi(2),
    //     "s cannot be smaller than (m3 + m4)² (s = {:e}, (m3 + m4)² = {:e}).",
    //     s,
    //     (p3.mass + p4.mass).powi(2)
    // );

    if s == 0.0 {
        return (0.0, 0.0);
    }

    let (x1, x2, x3, x4) = (p1.mass2 / s, p2.mass2 / s, p3.mass2 / s, p4.mass2 / s);

    let baseline = p1.mass2 + p3.mass2 - s / 2.0 * (1.0 + x1 - x2) * (1.0 + x3 - x4);
    let cosine = s / 2.0 * kallen_lambda_sqrt(1.0, x1, x2) * kallen_lambda_sqrt(1.0, x3, x4);

    (baseline - cosine, baseline + cosine)
}

/// Integrate the function of `$t$` over its range.
///
/// This function subdivides the integration domain to focus around the particle
/// masses where resonances are likely to appear.
#[must_use]
pub fn integrate_t<F>(
    integrand: F,
    s: f64,
    p1: &ParticleData,
    p2: &ParticleData,
    p3: &ParticleData,
    p4: &ParticleData,
) -> f64
where
    F: Fn(f64) -> f64,
{
    let (ta, tb) = t_range(s, p1, p2, p3, p4);

    let (integral, err) = integrate::legendre(integrand, ta, tb);

    log::trace!(
        "Mandelstam t Integration Result: {:e} ± {:e}",
        integral,
        err
    );

    integral
}

/// Integrate the amplitude with respect to the Mandelstam variable `$s$` and
/// `$t$`:
///
/// ```math
/// \int_{s_\text{min}}^{\infty} \int_{t_\text{min}}^{t_\text{max}}
///     \abs{\scM(s, t)}^2 \frac{K_1(\sqrt{s} \beta)}{\sqrt{s}}
/// \dd t \dd s
/// ```
///
/// where `$s_\text{min} = \mathop{\text{max}}((m_1 + m_2)^2, (m_3 + m_4)^2)$`
/// and `$t_{\text{min},\text{max}}$` are determined from [`t_range`].
#[must_use]
pub fn integrate_st<F>(
    amplitude: F,
    beta: f64,
    p1: &ParticleData,
    p2: &ParticleData,
    p3: &ParticleData,
    p4: &ParticleData,
) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    let s_min = (p1.mass + p2.mass).max(p3.mass + p4.mass).powi(2);
    let x_min = s_min.sqrt() * beta;

    let integrand = |x: f64| {
        let sqrt_s = x / beta;
        let s = sqrt_s.powi(2);

        let t_integrand = |t: f64| amplitude(s, t);
        integrate_t(t_integrand, s, p1, p2, p3, p4)
    };

    let (integral, err) = integrate::bessel(integrand, x_min);

    log::trace!(
        "Mandelstam s Integration Result: {:e} ± {:e}",
        integral,
        err
    );

    integral / (512.0 * PI_5)
}

/// Integrate the amplitude with respect to the Mandelstam variable `$s$` and
/// `$t$`, pre-divided by the number density of heavy particles.
///
/// A particle is deemed heavy when `$b \beta$` is greater than a particular
/// threshold.
#[must_use]
pub fn integrate_st_on_n<F>(
    amplitude: F,
    beta: f64,
    p1: &ParticleData,
    p2: &ParticleData,
    p3: &ParticleData,
    p4: &ParticleData,
) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    match (
        p1.mass * beta >= M_BETA_THRESHOLD,
        p2.mass * beta >= M_BETA_THRESHOLD,
        p3.mass * beta >= M_BETA_THRESHOLD,
        p4.mass * beta >= M_BETA_THRESHOLD,
    ) {
        // No heavy particle
        (false, false, false, false) => integrate_st(amplitude, beta, p1, p2, p3, p4),
        // One heavy particle
        (true, false, false, false) => integrate_st_on_n1(amplitude, beta, p1, p2, p3, p4, p1),
        (false, true, false, false) => integrate_st_on_n1(amplitude, beta, p1, p2, p3, p4, p2),
        (false, false, true, false) => integrate_st_on_n1(amplitude, beta, p1, p2, p3, p4, p3),
        (false, false, false, true) => integrate_st_on_n1(amplitude, beta, p1, p2, p3, p4, p4),
        // Two heavy particles
        (true, true, false, false) => integrate_st_on_n2(amplitude, beta, p1, p2, p3, p4, p1, p2),
        (true, false, true, false) => integrate_st_on_n2(amplitude, beta, p1, p2, p3, p4, p1, p3),
        (true, false, false, true) => integrate_st_on_n2(amplitude, beta, p1, p2, p3, p4, p1, p4),
        (false, true, true, false) => integrate_st_on_n2(amplitude, beta, p1, p2, p3, p4, p2, p3),
        (false, true, false, true) => integrate_st_on_n2(amplitude, beta, p1, p2, p3, p4, p2, p4),
        (false, false, true, true) => integrate_st_on_n2(amplitude, beta, p1, p2, p3, p4, p3, p4),
        // Three heavy particles
        (true, true, true, false) => {
            integrate_st_on_n3(amplitude, beta, p1, p2, p3, p4, p1, p2, p3)
        }
        (true, true, false, true) => {
            integrate_st_on_n3(amplitude, beta, p1, p2, p3, p4, p1, p2, p4)
        }
        (true, false, true, true) => {
            integrate_st_on_n3(amplitude, beta, p1, p2, p3, p4, p1, p3, p4)
        }
        (false, true, true, true) => {
            integrate_st_on_n3(amplitude, beta, p1, p2, p3, p4, p2, p3, p4)
        }
        // Four heavy particles
        (true, true, true, true) => integrate_st_on_n4(amplitude, beta, p1, p2, p3, p4),
    }
}

/// Integrate the over `$s$` and `$t$` divided by the heavy particle `h1`.
///
/// Uses the approximation
///
/// ```math
/// \frac{K_1(\sqrt s \beta)}{\sqrt s \beta} \frac{1}{n_a^{(0)}}
/// = \left( \sqrt{\frac{8}{\pi}} \zeta(3) \right) \frac{\sqrt{\pi / 2}}{g_a}
///   \frac{e^{\beta (m_a - \sqrt{s})}}{(\sqrt{m_a} \sqrt[4]{s} \beta)^3}
/// ```
#[must_use]
fn integrate_st_on_n1<F>(
    amplitude: F,
    beta: f64,
    p1: &ParticleData,
    p2: &ParticleData,
    p3: &ParticleData,
    p4: &ParticleData,
    h1: &ParticleData,
) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    // For values above of √s β above 700, the factor of K₁(√s β)/√s β evaluates
    // to 0 within machine precision.  So we use this as the upper bound instead
    // of infinity.
    let s_min = (p1.mass + p2.mass).max(p3.mass + p4.mass).powi(2);
    let s_max = (1e3 / beta.powi(2)).max(1e1 * s_min);

    let integrand = |s: f64| {
        let sqrt_s = s.sqrt();
        // 2 ζ(3) ≅ 2.4041138063191885
        let s_factor = (2.404_113_806_319_188_5 / h1.degrees_of_freedom())
            * f64::exp((h1.mass - sqrt_s) * beta)
            / ((h1.mass * sqrt_s).sqrt() * beta).powi(3);

        let t_integrand = |t: f64| amplitude(s, t);
        integrate_t(t_integrand, s, p1, p2, p3, p4) * s_factor
    };

    let (integral, err) = integrate::legendre(integrand, s_min, s_max);

    log::trace!(
        "Mandelstam t Integration Result: {:e} ± {:e}",
        integral,
        err
    );

    integral / (512.0 * PI_5)
}

/// Integrate the over `$s$` and `$t$` divided by the heavy particles `h1` and `h2`.
///
/// Uses the approximation
///
/// ```math
/// \frac{K_1(\sqrt s \beta)}{\sqrt s \beta} \frac{1}{n_a^{(0)} n_b^{(0)}}
/// = \left( \sqrt{\frac{8}{\pi}} \zeta(3) \right)^2 \frac{\sqrt{\pi / 2}}{g_a g_b}
///   \frac{e^{\beta (m_a + m_b - \sqrt{s})}}{\left(\sqrt{m_a m_b} \sqrt[4]{s} \beta^\frac{3}{2}\right)^3} \\
/// ```
#[allow(clippy::too_many_arguments)]
#[must_use]
fn integrate_st_on_n2<F>(
    amplitude: F,
    beta: f64,
    p1: &ParticleData,
    p2: &ParticleData,
    p3: &ParticleData,
    p4: &ParticleData,
    h1: &ParticleData,
    h2: &ParticleData,
) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    // For values above of √s β above 700, the factor of K₁(√s β)/√s β evaluates
    // to 0 within machine precision.  So we use this as the upper bound instead
    // of infinity.
    let s_min = (p1.mass + p2.mass).max(p3.mass + p4.mass).powi(2);
    let s_max = (1e3 / beta.powi(2)).max(1e1 * s_min);

    let integrand = |s: f64| {
        let sqrt_s = s.sqrt();
        // 4 √(2 / π) ζ(3)² ≅ 4.611583817377448
        let s_factor = (4.611_583_817_377_448
            / (h1.degrees_of_freedom() * h2.degrees_of_freedom()))
            * f64::exp((h1.mass + h2.mass - sqrt_s) * beta)
            / ((h1.mass * h2.mass * sqrt_s * beta.powi(3)).sqrt()).powi(3);

        let t_integrand = |t: f64| amplitude(s, t);
        integrate_t(t_integrand, s, p1, p2, p3, p4) * s_factor
    };

    let (integral, err) = integrate::legendre(integrand, s_min, s_max);

    log::trace!(
        "Mandelstam t Integration Result: {:e} ± {:e}",
        integral,
        err
    );

    integral / (512.0 * PI_5)
}

/// Integrate the over `$s$` and `$t$` divided by the heavy particles `h1`, `h2`
/// and `h3`.
///
/// Uses the approximation
///
/// ```math
/// \frac{K_1(\sqrt s \beta)}{\sqrt s \beta} \frac{1}{n_a^{(0)} n_b^{(0)} n_c^{(0)}}
/// = \left( \sqrt{\frac{8}{\pi}} \zeta(3) \right)^3 \frac{\sqrt{\pi / 2}}{g_a g_b g_c}
///   \frac{e^{\beta (m_a + m_b + m_c - \sqrt{s})}}{\left(\sqrt{m_a m_b m_c} \sqrt[4]{s} \beta^2\right)^3} \\
/// ```
#[allow(clippy::too_many_arguments)]
#[must_use]
fn integrate_st_on_n3<F>(
    amplitude: F,
    beta: f64,
    p1: &ParticleData,
    p2: &ParticleData,
    p3: &ParticleData,
    p4: &ParticleData,
    h1: &ParticleData,
    h2: &ParticleData,
    h3: &ParticleData,
) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    // For values above of √s β above 700, the factor of K₁(√s β)/√s β evaluates
    // to 0 within machine precision.  So we use this as the upper bound instead
    // of infinity.
    let s_min = (p1.mass + p2.mass).max(p3.mass + p4.mass).powi(2);
    let s_max = (1e3 / beta.powi(2)).max(1e1 * s_min);

    let integrand = |s: f64| {
        let sqrt_s = s.sqrt();
        // 16 ζ(3)³ / π ≅ 8.845964466739566
        let s_factor = (8.845_964_466_739_566
            / (h1.degrees_of_freedom() * h2.degrees_of_freedom() * h3.degrees_of_freedom()))
            * f64::exp((h1.mass + h2.mass + h3.mass - sqrt_s) * beta)
            / ((h1.mass * h2.mass * h3.mass * sqrt_s).sqrt() * beta.powi(2)).powi(3);

        let t_integrand = |t: f64| amplitude(s, t);
        integrate_t(t_integrand, s, p1, p2, p3, p4) * s_factor
    };

    let (integral, err) = integrate::legendre(integrand, s_min, s_max);

    log::trace!(
        "Mandelstam t Integration Result: {:e} ± {:e}",
        integral,
        err
    );

    integral / (512.0 * PI_5)
}

/// Integrate the over `$s$` and `$t$` divided by all four particles.
///
/// Uses the approximation
///
/// ```math
/// \frac{K_1(\sqrt s \beta)}{\sqrt s \beta} \frac{1}{n_a^{(0)} n_b^{(0)} n_c^{(0)} n_d^{(0)}}
/// = \left( \sqrt{\frac{8}{\pi}} \zeta(3) \right)^4 \frac{\sqrt{\pi / 2}}{g_a g_b g_c g_d}
///   \frac{e^{\beta (m_a + m_b + m_c + m_d - \sqrt{s})}}{\left(\sqrt{m_a m_b m_c m_d} \sqrt[4]{s} \beta^\frac{5}{2} \right)^3}
/// ```
#[must_use]
fn integrate_st_on_n4<F>(
    amplitude: F,
    beta: f64,
    p1: &ParticleData,
    p2: &ParticleData,
    p3: &ParticleData,
    p4: &ParticleData,
) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    // For values above of √s β above 700, the factor of K₁(√s β)/√s β evaluates
    // to 0 within machine precision.  So we use this as the upper bound instead
    // of infinity.
    let s_min = (p1.mass + p2.mass).max(p3.mass + p4.mass).powi(2);
    let s_max = (1e3 / beta.powi(2)).max(1e1 * s_min);

    let integrand = |s: f64| {
        let sqrt_s = s.sqrt();
        // 32 √2 ζ(3)⁴ / π^(3/2) ≅ 16.968375821762574
        let s_factor = (16.968_375_821_762_574
            / (p1.degrees_of_freedom()
                * p2.degrees_of_freedom()
                * p3.degrees_of_freedom()
                * p4.degrees_of_freedom()))
            * f64::exp((p1.mass + p2.mass + p3.mass + p4.mass - sqrt_s) * beta)
            / ((p1.mass * p2.mass * p3.mass * p4.mass * sqrt_s * beta.powi(5)).sqrt()).powi(3);

        let t_integrand = |t: f64| amplitude(s, t);
        integrate_t(t_integrand, s, p1, p2, p3, p4) * s_factor
    };

    let (integral, err) = integrate::legendre(integrand, s_min, s_max);

    log::trace!(
        "Mandelstam t Integration Result: {:e} ± {:e}",
        integral,
        err
    );

    integral / (512.0 * PI_5)
}

#[cfg(test)]
mod tests {
    use crate::{model::particle::SCALAR, prelude::ParticleData, utilities::test::approx_eq};
    use ndarray::Array1;
    use serde::{Deserialize, Serialize};
    use std::{env::temp_dir, error, fs, path::Path};

    /// Shorthand to create the CSV file in the appropriate directory and with
    /// headers.
    fn create_csv<P: AsRef<Path>>(p: P) -> Result<csv::Writer<fs::File>, Box<dyn error::Error>> {
        let dir = temp_dir()
            .join("boltzmann-solver")
            .join("interaction")
            .join("four_particle");
        fs::create_dir_all(&dir)?;

        let csv = csv::Writer::from_path(dir.join(p))?;

        Ok(csv)
    }

    /// Test Mandelstam t domain for all massless particles
    #[test]
    fn t_range_massless() -> Result<(), Box<dyn error::Error>> {
        let p: ParticleData = ParticleData::new(SCALAR, 0.0, 0.0);

        for &s in &Array1::geomspace(1e-10, 1e10, 1000).unwrap() {
            let (t1, t2) = super::t_range(s, &p, &p, &p, &p);
            approx_eq(t1, -s, 8.0, 0.0)?;
            approx_eq(t2, 0.0, 8.0, s)?;
        }

        Ok(())
    }

    /// Test Mandelstam t domain with a single mass scale (whereby the domain
    /// simplifies to a nice closed form).
    #[test]
    fn t_range_massive() -> Result<(), Box<dyn error::Error>> {
        let p0: ParticleData = ParticleData::new(SCALAR, 0.0, 0.0);

        for &m in &Array1::geomspace(1e-10_f64, 1e10, 100).unwrap() {
            let pm = ParticleData::new(SCALAR, m, 0.0);

            for &s in &Array1::geomspace(pm.mass2, pm.mass2 * 1e20, 1000).unwrap() {
                // Due to minor rounding issues, it's possible that s is small
                // than `pm.mass2`
                let s = s.max(pm.mass2);

                let (t1, t2) = super::t_range(s, &pm, &p0, &p0, &p0);
                approx_eq(t1, pm.mass2 - s, 8.0, s)?;
                approx_eq(t2, 0.0, 8.0, s)?;

                let (t1, t2) = super::t_range(s, &pm, &p0, &pm, &p0);
                approx_eq(t1, -(pm.mass2 - s).powi(2) / s, 8.0, s)?;
                approx_eq(t2, 0.0, 8.0, s)?;

                // Minimal value of s is larger in this case
                if s >= 4.0 * pm.mass2 {
                    let (t1, t2) = super::t_range(s, &pm, &pm, &p0, &p0);
                    approx_eq(
                        t1,
                        pm.mass2 - 0.5 * (s + f64::sqrt(s * (s - 4.0 * pm.mass2).abs())),
                        8.0,
                        s,
                    )?;
                    approx_eq(
                        t2,
                        pm.mass2 - 0.5 * (s - f64::sqrt(s * (s - 4.0 * pm.mass2).abs())),
                        8.0,
                        s,
                    )?;

                    let (t1, t2) = super::t_range(s, &pm, &pm, &pm, &pm);
                    approx_eq(t1, 4.0 * pm.mass2 - s, 8.0, s)?;
                    approx_eq(t2, 0.0, 8.0, s)?;
                }
            }
        }

        Ok(())
    }

    /// Test Mandelstam t domainfor a few random values.
    #[test]
    fn t_range_random() -> Result<(), Box<dyn error::Error>> {
        #[derive(Debug, Deserialize, Serialize)]
        struct Row {
            m1: f64,
            m2: f64,
            m3: f64,
            m4: f64,
            s: f64,
            t1: f64,
            t2: f64,
        }

        // Data generated randomly
        let mut csv = csv::Reader::from_reader(zstd::Decoder::new(fs::File::open(
            "tests/data/t_range_random.csv.zst",
        )?)?);

        for record in csv.deserialize() {
            let row: Row = record?;
            let p1: ParticleData = ParticleData::new(SCALAR, row.m1, 0.0);
            let p2 = ParticleData::new(SCALAR, row.m2, 0.0);
            let p3 = ParticleData::new(SCALAR, row.m3, 0.0);
            let p4 = ParticleData::new(SCALAR, row.m4, 0.0);

            let (t1, t2) = super::t_range(row.s, &p1, &p2, &p3, &p4);
            approx_eq(row.t1, t1, 0.5, row.s / 1e12)?;
            approx_eq(row.t2, t2, 0.5, row.s / 1e12)?;

            // Simultaneous interchange of 1 ↔ 2 and 3 ↔ 4 leaves the t variable
            // unchanged
            let (t1, t2) = super::t_range(row.s, &p2, &p1, &p4, &p3);
            approx_eq(row.t1, t1, 0.5, row.s / 1e12)?;
            approx_eq(row.t2, t2, 0.5, row.s / 1e12)?;
        }

        Ok(())
    }

    #[test]
    fn integrate_t_massless() -> Result<(), Box<dyn error::Error>> {
        let p: ParticleData = ParticleData::new(SCALAR, 0.0, 0.0);

        for &s in &Array1::geomspace(1e-10, 1e10, 1000).unwrap() {
            approx_eq(super::integrate_t(|_t| 1.0, s, &p, &p, &p, &p), s, 8.0, 0.0)?;
        }

        for &s in &Array1::geomspace(1e-10, 1e10, 1000).unwrap() {
            approx_eq(
                super::integrate_t(|t| t, s, &p, &p, &p, &p),
                -s.powi(2) / 2.0,
                8.0,
                0.0,
            )?;
        }

        Ok(())
    }

    #[test]
    fn integrate_st_massless() -> Result<(), Box<dyn error::Error>> {
        let mut rdr = csv::Reader::from_reader(zstd::Decoder::new(fs::File::open(
            "tests/data/st_integral_massless.csv.zst",
        )?)?);

        let functions = [
            |s, _t| s * s,
            |s, t| s * t,
            |_s, t| t * t,
            |s, _t| s * s / (s + 1.0),
            |s, t| s * t / (s + 1.0),
            |s, t| s * s * t * t / (s + 1.0) / (t + 1.0),
        ];

        for (row, result) in rdr.deserialize().enumerate() {
            let data: [f64; 5 + 6] = result.unwrap();

            let beta = data[0];
            let p1 = ParticleData::new(SCALAR, data[1], 0.0);
            let p2 = ParticleData::new(SCALAR, data[2], 0.0);
            let p3 = ParticleData::new(SCALAR, data[3], 0.0);
            let p4 = ParticleData::new(SCALAR, data[4], 0.0);
            let y = &data[5..];

            for (i, function) in functions.iter().enumerate() {
                let yi = y[i];
                let nyi = super::integrate_st(function, beta, &p1, &p2, &p3, &p4);
                approx_eq(yi, nyi, 11.0, 10_f64.powi(-200)).map_err(|err| {
                    println!(
                        "[{}] ∫({}, {:e}, {:e}, {:e}, {:e}, {:e}) = {:e} but expected {:e}.",
                        row, i, beta, p1.mass, p2.mass, p3.mass, p4.mass, nyi, yi
                    );
                    err
                })?;
            }
        }

        Ok(())
    }

    #[allow(clippy::shadow_unrelated)]
    #[test]
    fn integrate_st_massive() -> Result<(), Box<dyn error::Error>> {
        const BETA: f64 = 1e-2;
        let p1 = ParticleData::new(SCALAR, 0.2, 0.0);
        let p2 = ParticleData::new(SCALAR, 0.5, 0.0);
        let p3 = ParticleData::new(SCALAR, 2.0, 0.0);
        let p4 = ParticleData::new(SCALAR, 5.0, 0.0);

        let m2 = |_, _| 1.0;
        approx_eq(
            super::integrate_st(m2, BETA, &p1, &p2, &p3, &p4),
            2539.02,
            3.0,
            1e-20,
        )?;

        let m2 = |s: f64, _| 1.0 / (s.powi(2) + 1.0);
        approx_eq(
            super::integrate_st(m2, BETA, &p1, &p2, &p3, &p4),
            0.000_802_015,
            3.0,
            1e-20,
        )?;

        let m2 = |_, t: f64| 1.0 / (t.powi(2) + 1.0);
        approx_eq(
            super::integrate_st(m2, BETA, &p1, &p2, &p3, &p4),
            0.439_373,
            1.5,
            1e-20,
        )?;

        Ok(())
    }

    #[test]
    fn integrate_st_on_n1() -> Result<(), Box<dyn error::Error>> {
        #[derive(Debug, serde::Serialize)]
        struct Row {
            beta: f64,
            n: f64,
            gamma: f64,
            gamma_on_n: f64,
            gamma_auto: f64,
            gamma_tilde: f64,
        }

        let p0 = ParticleData::new(SCALAR, 0.0, 0.0);
        let pm = ParticleData::new(SCALAR, 1e10, 0.0);

        let mut csv = create_csv("integrate_st_on_n1.csv")?;

        let m2 = |_, _| 1.0;
        for &beta in &Array1::geomspace(1e-12, 1e-6, 1000).unwrap() {
            let n = pm.normalized_number_density(beta, 0.0);
            let gamma = super::integrate_st(m2, beta, &pm, &p0, &p0, &p0);
            let row = Row {
                beta,
                n,
                gamma,
                gamma_on_n: gamma / n,
                gamma_auto: super::integrate_st_on_n(m2, beta, &pm, &p0, &p0, &p0),
                gamma_tilde: super::integrate_st_on_n1(m2, beta, &pm, &p0, &p0, &p0, &pm),
            };

            csv.serialize(row)?;
        }

        Ok(())
    }

    #[test]
    fn integrate_st_on_n2() -> Result<(), Box<dyn error::Error>> {
        #[derive(Debug, serde::Serialize)]
        struct Row {
            beta: f64,
            n: f64,
            gamma: f64,
            gamma_on_n: f64,
            gamma_auto: f64,
            gamma_tilde: f64,
        }

        let p0 = ParticleData::new(SCALAR, 0.0, 0.0);
        let pm = ParticleData::new(SCALAR, 1e10, 0.0);

        let mut csv = create_csv("integrate_st_on_n2.csv")?;

        let m2 = |_, _| 1.0;
        for &beta in &Array1::geomspace(1e-12, 1e-6, 1000).unwrap() {
            let n = pm.normalized_number_density(beta, 0.0);
            let gamma = super::integrate_st(m2, beta, &pm, &pm, &p0, &p0);
            let row = Row {
                beta,
                n,
                gamma,
                gamma_on_n: gamma / n.powi(2),
                gamma_auto: super::integrate_st_on_n(m2, beta, &pm, &pm, &p0, &p0),
                gamma_tilde: super::integrate_st_on_n2(m2, beta, &pm, &pm, &p0, &p0, &pm, &pm),
            };

            csv.serialize(row)?;
        }

        Ok(())
    }

    #[test]
    fn integrate_st_on_n3() -> Result<(), Box<dyn error::Error>> {
        #[derive(Debug, serde::Serialize)]
        struct Row {
            beta: f64,
            n: f64,
            gamma: f64,
            gamma_on_n: f64,
            gamma_auto: f64,
            gamma_tilde: f64,
        }

        let p0 = ParticleData::new(SCALAR, 0.0, 0.0);
        let pm = ParticleData::new(SCALAR, 1e10, 0.0);

        let mut csv = create_csv("integrate_st_on_n3.csv")?;

        let m2 = |_, _| 1.0;
        for &beta in &Array1::geomspace(1e-12, 1e-6, 1000).unwrap() {
            let n = pm.normalized_number_density(beta, 0.0);
            let gamma = super::integrate_st(m2, beta, &pm, &pm, &pm, &p0);
            let row = Row {
                beta,
                n,
                gamma,
                gamma_on_n: gamma / n.powi(3),
                gamma_auto: super::integrate_st_on_n(m2, beta, &pm, &pm, &pm, &p0),
                gamma_tilde: super::integrate_st_on_n3(m2, beta, &pm, &pm, &pm, &p0, &pm, &pm, &pm),
            };

            csv.serialize(row)?;
        }

        Ok(())
    }

    #[test]
    fn integrate_st_on_n4() -> Result<(), Box<dyn error::Error>> {
        #[derive(Debug, serde::Serialize)]
        struct Row {
            beta: f64,
            n: f64,
            gamma: f64,
            gamma_on_n: f64,
            gamma_auto: f64,
            gamma_tilde: f64,
        }

        let pm = ParticleData::new(SCALAR, 1e10, 0.0);

        let mut csv = create_csv("integrate_st_on_n4.csv")?;

        let m2 = |_, _| 1.0;
        for &beta in &Array1::geomspace(1e-12, 1e-6, 1000).unwrap() {
            let n = pm.normalized_number_density(beta, 0.0);
            let gamma = super::integrate_st(m2, beta, &pm, &pm, &pm, &pm);
            let row = Row {
                beta,
                n,
                gamma,
                gamma_on_n: gamma / n.powi(4),
                gamma_auto: super::integrate_st_on_n(m2, beta, &pm, &pm, &pm, &pm),
                gamma_tilde: super::integrate_st_on_n4(m2, beta, &pm, &pm, &pm, &pm),
            };

            csv.serialize(row)?;
        }

        Ok(())
    }
}
