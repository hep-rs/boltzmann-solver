use crate::{
    model::{
        interaction::{
            checked_div, checked_mul, Interaction, PartialWidth, Particles, RateDensity,
            M_BETA_THRESHOLD,
        },
        Model,
    },
    solver::Context,
};
use special_functions::{bessel, particle_physics::kallen_lambda_sqrt};
use std::fmt;

/// Three particle interaction, all determined from the underlying squared amplitude.
pub struct ThreeParticle<M> {
    particles: Particles,
    /// Squared amplitude as a function of the model.
    squared_amplitude: Box<dyn Fn(&M) -> f64 + Sync>,
    /// Asymmetry between the amplitude and its `$\CP$` conjugate.
    asymmetry: Option<Box<dyn Fn(&M) -> f64 + Sync>>,
    gamma_enabled: bool,
    width_enabled: bool,
}

impl<M> ThreeParticle<M>
where
    M: Model,
{
    /// Create a new three-particle interaction.
    ///
    /// The order in which the interaction takes place is determined by the
    /// masses of the particles such that the heaviest is (inverse) decaying
    /// from/to the lighter two.
    ///
    /// Particle are indicated by positive integers and the corresponding
    /// antiparticles are indicated by negative integers.
    ///
    /// If the process differs from its `$\CP$`-conjugate process, the asymmetry
    /// can be specified with [`ThreeParticle::asymmetry`].
    pub fn new<F>(squared_amplitude: F, p1: isize, p2: isize, p3: isize) -> Self
    where
        F: Fn(&M) -> f64 + Sync + 'static,
    {
        let particles = Particles::new(&[p1], &[p2, p3]);

        Self {
            particles,
            squared_amplitude: Box::new(squared_amplitude),
            asymmetry: None,
            width_enabled: true,
            gamma_enabled: true,
        }
    }

    /// Create the set of related three particle interactions.
    ///
    /// The interactions are all related through crossing symmetry:
    ///
    /// - `$p_1 \leftrightarrow p_2 p_3$`,
    /// - `$\overline{p_2} \leftrightarrow \overline{p_1} p_3$`, and
    /// - `$\overline{p_3} \leftrightarrow \overline{p_1} p_2$`.
    pub fn new_all<F>(squared_amplitude: F, p1: isize, p2: isize, p3: isize) -> Vec<Self>
    where
        F: Fn(&M) -> f64 + Sync + Copy + 'static,
    {
        let mut v = vec![Self::new(squared_amplitude, p1, p2, p3)];

        // Avoid doubling up interactions if they have the same particles
        if p1 != -p2 {
            v.push(Self::new(squared_amplitude, -p2, -p1, p3));
        }
        if p2 != p3 {
            v.push(Self::new(squared_amplitude, -p3, -p1, p2));
        }

        v
    }

    /// Specify the asymmetry between this process and its `$\CP$`-conjugate.
    ///
    /// This asymmetry is specified in terms of the asymmetry in the squared
    /// amplitudes:
    ///
    /// ```math
    /// \delta \abs{\scM}^2
    ///   \defeq \abs{\scM(p_1 \to p_2 p_3)}^2 - \abs{\scM(\overline{p_1} \to \overline{p_2} \overline{p_3})}^2
    /// ```
    ///
    /// This asymmetry is subsequently used to compute the asymmetric
    /// interaction rate given by [`Interaction::delta_gamma`].
    pub fn asymmetry<F>(mut self, asymmetry: F) -> Self
    where
        F: Fn(&M) -> f64 + Sync + 'static,
    {
        self.set_asymmetry(asymmetry);
        self
    }

    /// Same as [`ThreeParticle::asymmetry`] but modifies the interaction by reference.
    pub fn set_asymmetry<F>(&mut self, asymmetry: F) -> &Self
    where
        F: Fn(&M) -> f64 + Sync + 'static,
    {
        self.asymmetry = Some(Box::new(asymmetry));
        self
    }

    /// Adjust whether this interaction will calculate decay widths.
    #[must_use]
    pub fn enable_width(mut self, v: bool) -> Self {
        self.width_enabled = v;
        self
    }

    /// Adjust whether this interaction will calculate decay widths.
    #[must_use]
    pub fn enable_gamma(mut self, v: bool) -> Self {
        self.gamma_enabled = v;
        self
    }
}

impl<M> fmt::Debug for ThreeParticle<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.asymmetry.is_some() {
            write!(
                f,
                "ThreeParticle {{ \
                 particles: {:?}, \
                 squared_amplitude: Box<Fn>, \
                 asymmetric: Some(Box<Fn>) \
                 }}",
                self.particles,
            )
        } else {
            write!(
                f,
                "ThreeParticle {{ \
                 particles: {:?}, \
                 squared_amplitude: Box<Fn>, \
                 asymmetric: None \
                 }}",
                self.particles
            )
        }
    }
}

impl<M> Interaction<M> for ThreeParticle<M>
where
    M: Model,
{
    fn particles(&self) -> &Particles {
        &self.particles
    }

    fn width_enabled(&self) -> bool {
        self.width_enabled
    }

    fn width(&self, context: &Context<M>) -> Option<PartialWidth> {
        if !self.width_enabled() {
            return None;
        }

        let ptcl = context.model.particles();

        // Get the *squared* masses
        let p1 = &ptcl[self.particles.incoming_idx[0]];
        let p2 = &ptcl[self.particles.outgoing_idx[0]];
        let p3 = &ptcl[self.particles.outgoing_idx[1]];

        if p1.mass > p2.mass + p3.mass {
            let p = kallen_lambda_sqrt(p1.mass2, p2.mass2, p3.mass2) / (2.0 * p1.mass);

            // 1 / 8 π ≅ 0.039788735772973836
            let width = 0.039_788_735_772_973_836 * p / p1.mass2
                * (self.squared_amplitude)(context.model).abs();

            debug_assert!(
                width.is_finite(),
                "[{}.{:02}|{:>10.4e}] Computed a non-finit width in interaction {}: {}",
                context.step,
                context.substep,
                context.beta,
                self.particles()
                    .display(context.model)
                    .unwrap_or_else(|_| self.particles().short_display()),
                width
            );

            Some(PartialWidth {
                width,
                parent: self.particles.incoming_signed[0],
                daughters: self.particles.outgoing_signed.clone(),
            })
        } else {
            None
        }
    }

    fn gamma_enabled(&self) -> bool {
        self.gamma_enabled
    }

    /// Unlike other function, this is computed in one of two ways:
    ///
    /// 1. If incoming particle's mass is greater than the temperature by a
    ///    specific threshold, then the result is divided analytically by the
    ///    normalized equilibrium number density of the particle such that
    ///    multiplying the result by the equilibrium number density will yield
    ///    the correct interaction density).  This is done in order to avoid a
    ///    `0 / 0` division as the reaction rate is propoertional `$K_1(m
    ///    \beta)$`, while the number density of the decaying particle
    ///    proportional to `$K_2(m \beta)$`, and the ratio of Bessel function
    ///    tends to 1 (while each individually they both tend to 0 very
    ///    quickly).
    /// 2. Otherwise, calculate the usual reaction rate.
    ///
    /// Note that in order to obtain the analytical normalized equilibrium
    /// density, we must assume that `$m \beta \gg 1 $` so that the number
    /// density approximates a Maxwell--Boltzmann distribution (irrespective of
    /// the original statistic), hence why we must use a threshold to switch
    /// between the two methods.
    fn gamma(&self, context: &Context<M>, real: bool) -> Option<f64> {
        if !self.gamma_enabled() {
            return None;
        }

        let ptcl = context.model.particles();

        // Get the particles
        let p1 = &ptcl[self.particles.incoming_idx[0]];
        let p2 = &ptcl[self.particles.outgoing_idx[0]];
        let p3 = &ptcl[self.particles.outgoing_idx[1]];

        // If the decay is kinematically forbidden, return 0.
        if p1.mass < p2.mass + p3.mass {
            return Some(0.0);
        }

        let z = p1.mass * context.beta;
        let gamma = if real || z < M_BETA_THRESHOLD {
            // 1 / 32 π³ ≅ 0.001007860451037484
            0.001_007_860_451_037_484
                * (self.squared_amplitude)(context.model).abs()
                * kallen_lambda_sqrt(p1.mass2, p2.mass2, p3.mass2)
                * (bessel::k1(z) / z)
        } else {
            // ζ(3) / 16 π³ ≅ 0.0024230112251823
            0.002_423_011_225_182_3
                * (self.squared_amplitude)(context.model).abs()
                * kallen_lambda_sqrt(p1.mass2, p2.mass2, p3.mass2)
                * (bessel::k1_on_k2(z) / z.powi(3))
                / p1.degrees_of_freedom()
        };

        debug_assert!(
            gamma.is_finite(),
            "[{}.{:02}|{:>10.4e}] Computed a non-finite value for γ in interaction {}: {}",
            context.step,
            context.substep,
            context.beta,
            self.particles()
                .display(context.model)
                .unwrap_or_else(|_| self.particles().short_display()),
            gamma
        );

        Some(gamma)
    }

    fn delta_gamma(&self, context: &Context<M>, real: bool) -> Option<f64> {
        let asymmetry = self.asymmetry.as_ref()?;

        let ptcl = context.model.particles();

        // Get the particles
        let p1 = &ptcl[self.particles.incoming_idx[0]];
        let p2 = &ptcl[self.particles.outgoing_idx[0]];
        let p3 = &ptcl[self.particles.outgoing_idx[1]];

        // If the decay is kinematically forbidden, return 0.
        if p1.mass < p2.mass + p3.mass {
            return Some(0.0);
        }

        let z = p1.mass * context.beta;
        let delta_gamma = if real || z < M_BETA_THRESHOLD {
            // 1 / 32 π³ ≅ 0.001007860451037484
            0.001_007_860_451_037_484
                * asymmetry(context.model).abs()
                * kallen_lambda_sqrt(p1.mass2, p2.mass2, p3.mass2)
                * (bessel::k1(z) / z)
        } else {
            // ζ(3) / 16 π³ ≅ 0.0024230112251823
            0.002_423_011_225_182_3
                * asymmetry(context.model).abs()
                * kallen_lambda_sqrt(p1.mass2, p2.mass2, p3.mass2)
                * (bessel::k1_on_k2(z) / z.powi(3))
                / p1.degrees_of_freedom()
        };

        debug_assert!(
            delta_gamma.is_finite(),
            "[{}.{:02}|{:>10.4e}] Computed a non-finite value for δγ in interaction {}: {}",
            context.step,
            context.substep,
            context.beta,
            self.particles()
                .display(context.model)
                .unwrap_or_else(|_| self.particles().short_display()),
            delta_gamma
        );

        Some(delta_gamma)
    }

    /// Override the default implementation of [`Interaction::rate`] to make use
    /// of the adjusted reaction rate density.
    fn rate(&self, context: &Context<M>) -> Option<RateDensity> {
        let gamma = self.gamma(context, false)?;
        let delta_gamma = self.delta_gamma(context, false);

        // If both rates are 0, there's no need to adjust it to the particles'
        // number densities.
        if gamma == 0.0 && delta_gamma.unwrap_or_default() == 0.0 {
            return None;
        }

        let ptcl = context.model.particles();

        // Get the various quantities associated with each particle.
        let [i0, i1, i2] = [
            self.particles.incoming_idx[0],
            self.particles.outgoing_idx[0],
            self.particles.outgoing_idx[1],
        ];
        let p1 = &ptcl[i0];
        let [n1, n2, n3] = [context.n[i0], context.n[i1], context.n[i2]];
        let [na1, na2, na3] = [
            self.particles.incoming_sign[0] * context.na[i0],
            self.particles.outgoing_sign[0] * context.na[i1],
            self.particles.outgoing_sign[1] * context.na[i2],
        ];
        let [eq1, eq2, eq3] = [context.eq[i0], context.eq[i1], context.eq[i2]];

        if eq1 == 0.0 && eq2 * eq3 == 0.0 {
            return None;
        }

        let mut rate = RateDensity::zero();

        // Since we get the 'false' interaction rate, we can always store that
        // in the rate density now.
        rate.gamma_tilde = gamma;
        rate.delta_gamma_tilde = delta_gamma;

        if p1.mass * context.beta < M_BETA_THRESHOLD {
            // Below the M_BETA_THRESHOLD, `gamma` is the usual rate which must
            // be scaled by factors of `n / eq` to get the actual forward and
            // backward rates.
            rate.gamma = gamma;
            rate.delta_gamma = delta_gamma;
            let delta_gamma = delta_gamma.unwrap_or_default();

            let symmetric_prefactor =
                checked_div(n1, eq1) - checked_div(n2, eq2) * checked_div(n3, eq3);
            rate.symmetric = checked_mul(gamma, symmetric_prefactor);
            rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                + checked_mul(
                    gamma,
                    checked_div(na1, eq1) - checked_div(na2 * n3 + na3 * n2, eq2 * eq3),
                );
        } else {
            // Above the M_BETA_THRESHOLD, `gamma` is already divided by eq1, so
            // we need not divide by `eq1` to calculate the forward rates, and we
            // have to multiply by `eq1` to get the backward rate.
            rate.gamma = gamma * eq1;
            rate.delta_gamma = delta_gamma.map(|v| v * eq1);
            let delta_gamma = delta_gamma.unwrap_or_default();

            // If eq1 is zero, the remaining n1 density is trying to decay and
            // the inverse decay should be negligible.
            if eq1 == 0.0 {
                rate.symmetric = checked_mul(gamma, n1);
                rate.asymmetric = checked_mul(delta_gamma, n1) + checked_mul(gamma, na1);
            } else {
                let symmetric_prefactor = n1 - eq1 * checked_div(n2, eq2) * checked_div(n3, eq3);
                rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                    + checked_mul(
                        gamma,
                        na1 - eq1 * checked_div(na2 * n3 + na3 * n2, eq2 * eq3),
                    );
            }
        }

        Some(rate * context.normalization)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        model::{interaction, Empty},
        prelude::*,
        statistic::Statistic,
    };
    use ndarray::prelude::*;
    use serde::{Deserialize, Serialize};
    use std::{env::temp_dir, error, fs, path::Path};

    const BETA_START: f64 = 1e-20;
    const BETA_END: f64 = 1e0;

    #[derive(Debug, Serialize, Deserialize)]
    struct CsvRow {
        beta: f64,
        hubble_rate: f64,
        n1: f64,
        eq1: f64,
        eq2: f64,
        eq3: f64,
        width: Option<f64>,
        gamma_true: Option<f64>,
        gamma_false: Option<f64>,
        delta_gamma_true: Option<f64>,
        delta_gamma_false: Option<f64>,
        symmetric_prefactor: f64,
        asymmetric_prefactor: f64,
        rate_symmetric: Option<f64>,
        rate_asymmetric: Option<f64>,
        adjusted_rate_symmetric: Option<f64>,
        adjusted_rate_asymmetric: Option<f64>,
    }

    /// Shorthand to create the CSV file in the appropriate directory and with
    /// headers.
    fn create_csv<P: AsRef<Path>>(p: P) -> Result<csv::Writer<fs::File>, Box<dyn error::Error>> {
        let dir = temp_dir()
            .join("boltzmann-solver")
            .join("interaction")
            .join("three_particle");
        fs::create_dir_all(&dir)?;

        let csv = csv::Writer::from_path(dir.join(p))?;

        Ok(csv)
    }

    /// Unit amplitude
    #[test]
    fn unit_amplitude() -> Result<(), Box<dyn error::Error>> {
        let mut model = Empty::default();
        model.extend_particles(
            [1e10, 1e5, 1e2]
                .iter()
                .enumerate()
                .map(|(i, &m)| Particle::new(0, m, m / 100.0).name(format!("{}", i + 1))),
        );
        let interaction = interaction::ThreeParticle::new(|_m| 1.0, 1, 2, 3);
        let mut csv = create_csv("unit_amplitude.csv")?;

        for &beta in &Array1::geomspace(BETA_START, BETA_END, 1000).unwrap() {
            model.set_beta(beta);
            let c = model.as_context();

            csv.serialize(CsvRow {
                beta,
                hubble_rate: c.hubble_rate,
                n1: Statistic::BoseEinstein.number_density(beta, 0.0, 0.0),
                eq1: c.eq[1],
                eq2: c.eq[2],
                eq3: c.eq[3],
                width: interaction.width(&c).map(|w| w.width),
                gamma_true: interaction.gamma(&c, true),
                gamma_false: interaction.gamma(&c, false),
                delta_gamma_true: interaction.delta_gamma(&c, true),
                delta_gamma_false: interaction.delta_gamma(&c, false),
                symmetric_prefactor: interaction.symmetric_prefactor(&c),
                asymmetric_prefactor: interaction.asymmetric_prefactor(&c),
                rate_symmetric: interaction.rate(&c).map(|r| r.symmetric),
                rate_asymmetric: interaction.rate(&c).map(|r| r.asymmetric),
                adjusted_rate_symmetric: interaction.adjusted_rate(&c).map(|r| r.symmetric),
                adjusted_rate_asymmetric: interaction.adjusted_rate(&c).map(|r| r.asymmetric),
            })?;

            csv.flush()?;
        }

        Ok(())
    }
}
