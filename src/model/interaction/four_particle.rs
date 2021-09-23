mod utilities;

use crate::{
    model::{
        interaction,
        interaction::{
            checked_div, checked_mul, Interaction, PartialWidth, RateDensity, M_BETA_THRESHOLD,
        },
        Model,
    },
    solver::Context,
    utilities::spline::Linear as Spline,
};
use std::{fmt, iter, sync::RwLock};

// TODO: Define a trait alias for MandelstamFn once it is stabilised
// (https://github.com/rust-lang/rust/issues/41517)

/// Three particle interaction, all determined from the underlying squared
/// amplitude.
pub struct FourParticle<M> {
    particles: interaction::Particles,

    /// Squared amplitude of the interaction.  This is a function of the model
    /// and the Mandelstam variables `$s$`, `$t$` and `$u$`.
    squared_amplitude: Box<dyn Fn(&M, f64, f64, f64) -> f64 + Sync>,
    /// Squared amplitude asymmetry for this interaction.  This is a function of
    /// the model and the Mandelstam variables `$s$`, `$t$` and `$u$`.
    #[allow(clippy::type_complexity)]
    asymmetry: Option<Box<dyn Fn(&M, f64, f64, f64) -> f64 + Sync>>,

    gamma_enabled: bool,
    width_enabled: bool,

    /// Since computing `$\tilde\gamma$` can be computationally expensive, we
    /// (optionally) precompute it and store it in a spline.  Since this
    /// includes the division by a varying number of heavy particle number
    /// densities, the discontinuities can cause issues with the spline.  To
    /// account for this, we store multiple splines based on the number of heavy
    /// particles: `[0]` for no heavy particles, `[1]` for one heavy particles,
    /// etc.
    gamma_tilde: RwLock<Vec<Spline>>,
    /// Spline for `$\delta\tilde\gamma$`.
    delta_gamma_tilde: RwLock<Vec<Spline>>,
}

impl<M> FourParticle<M> {
    /// Create a new four-particle interaction.
    ///
    /// The interaction will result in the annihilation of `p1` and `p2`, and
    /// the creation of `p3` and `p4`.  Particles are indicated by positive
    /// integers and the corresponding antiparticles are indicates by negative
    /// integers.
    ///
    /// The squared amplitude is defined of the form
    ///
    /// ```math
    /// \abs{\scM(p_1 p_2 \leftrightarrow p_3 p_4)}^2(s, t, u)
    /// ```
    ///
    /// where `$s$`, `$t$` and `$u$` are the usual Mandelstam variables.
    /// Crossing symmetry is then used in order to compute the other processes.
    ///
    /// The three-body decays are not currently computed.
    pub fn new<F>(squared_amplitude: F, p1: isize, p2: isize, p3: isize, p4: isize) -> Self
    where
        F: Fn(&M, f64, f64, f64) -> f64 + Sync + 'static,
    {
        let particles = interaction::Particles::new(&[p1, p2], &[p3, p4]);

        Self {
            particles,
            squared_amplitude: Box::new(squared_amplitude),
            asymmetry: None,
            width_enabled: false,
            gamma_enabled: true,
            gamma_tilde: RwLock::new(iter::repeat(Spline::empty()).take(5).collect()),
            delta_gamma_tilde: RwLock::new(iter::repeat(Spline::empty()).take(5).collect()),
        }
    }

    /// Create a set of related four-particle interactions.
    ///
    /// This functions creates all three possible interactions from the same
    /// underlying squared amplitude, related through crossing symmetry:
    ///
    /// ```math
    /// \\begin{aligned}
    ///   p_1 p_2 \leftrightarrow p_3 p_4, \\\\
    ///   p_1 \overline{p_3} \leftrightarrow \overline{p_2} p_4, \\\\
    ///   p_1 \overline{p_4} \leftrightarrow \overline{p_2} p_3. \\\\
    /// \\end{aligned}
    /// ```
    ///
    /// The squared amplitude is defined of the form:
    ///
    /// ```math
    /// \abs{\scM(p_1 p_2 \leftrightarrow p_3 p_4)}^2(s, t, u)
    /// ```
    ///
    /// where `$s$`, `$t$` and `$u$` are the usual Mandelstam variables.
    /// Crossing symmetry is then used in order to compute the other processes.
    pub fn new_all<F>(squared_amplitude: F, p1: isize, p2: isize, p3: isize, p4: isize) -> Vec<Self>
    where
        F: Fn(&M, f64, f64, f64) -> f64 + Sync + Copy + 'static,
    {
        let mut v = Vec::with_capacity(3);
        v.push(Self::new(squared_amplitude, p1, p2, p3, p4));

        // Avoid doubling up interactions if they have the same particles
        if p2 != -p3 {
            v.push(Self::new(
                move |c, s, t, u| squared_amplitude(c, t, s, u),
                p1,
                -p3,
                -p2,
                p4,
            ));
        }
        if p3 != p4 {
            v.push(Self::new(
                move |c, s, t, u| squared_amplitude(c, u, t, s),
                p1,
                -p4,
                p3,
                -p2,
            ));
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
    ///   \defeq \abs{\scM(p_1 p_2 \to p_3 p_4)}^2 - \abs{\scM(\overline{p_1} \overline{p_2} \to \overline{p_3} \overline{p_4})}^2
    ///   = \abs{\scM(p_1 p_2 \to p_3 p_4)}^2 - \abs{\scM(p_3 p_4 \to p_1 p_2)}^2
    /// ```
    ///
    /// This asymmetry is subsequently used to compute the asymmetric
    /// interaction rate given by [`Interaction::delta_gamma`].
    pub fn set_asymmetry<F>(&mut self, asymmetry: F) -> &Self
    where
        F: Fn(&M, f64, f64, f64) -> f64 + Sync + 'static,
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

impl<M: Model> FourParticle<M> {
    /// Count the number of heavy  particles
    pub fn count_heavy(&self, context: &Context<M>) -> usize {
        let particles = context.model.particles();
        self.particles
            .particle_counts
            .keys()
            .filter(|&&idx| particles[idx].mass * context.beta > M_BETA_THRESHOLD)
            .count()
    }
}

impl<M> fmt::Debug for FourParticle<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FourParticle {{ \
             particles: {:?}, \
             squared_amplitude: Box<Fn>, \
             }}",
            self.particles
        )
    }
}

impl<M> Interaction<M> for FourParticle<M>
where
    M: Model,
{
    fn particles(&self) -> &interaction::Particles {
        &self.particles
    }

    fn width_enabled(&self) -> bool {
        self.width_enabled
    }

    fn width(&self, _context: &Context<M>) -> Option<PartialWidth> {
        // TODO: Implement three-body decays
        None
    }

    fn gamma_enabled(&self) -> bool {
        self.gamma_enabled
    }

    fn gamma(&self, context: &Context<M>, real: bool) -> Option<f64> {
        if !self.gamma_enabled {
            return None;
        }

        // If we need the 'false' gamma (i.e. γ̃), then we use the precomputed
        // value provided the interval is deemed to be accurate.
        if let (false, Ok(spline)) = (real, self.gamma_tilde.read()) {
            let spline = &spline[self.count_heavy(context)];
            let log_beta = f64::ln(context.beta);
            if spline.contains(log_beta) {
                return Some(spline.sample(log_beta));
            }
        }

        let ptcl = context.model.particles();

        // Get the particles involved
        let p1 = &ptcl[self.particles.incoming_idx[0]];
        let p2 = &ptcl[self.particles.incoming_idx[1]];
        let p3 = &ptcl[self.particles.outgoing_idx[0]];
        let p4 = &ptcl[self.particles.outgoing_idx[1]];

        let integrand = |s, t| {
            let u = p1.mass2 + p2.mass2 + p3.mass2 + p4.mass2 - s - t;
            (self.squared_amplitude)(context.model, s, t, u).abs()
        };

        if real {
            Some(utilities::integrate_st(
                integrand,
                context.beta,
                p1,
                p2,
                p3,
                p4,
            ))
        } else {
            let gamma_tilde = utilities::integrate_st_on_n(integrand, context.beta, p1, p2, p3, p4);
            if let Ok(mut spline) = self.gamma_tilde.write() {
                let spline = &mut spline[self.count_heavy(context)];
                spline.add(f64::ln(context.beta), gamma_tilde);
            }
            Some(gamma_tilde)
        }
    }

    fn delta_gamma(&self, context: &Context<M>, real: bool) -> Option<f64> {
        let asymmetry = self.asymmetry.as_ref()?;

        // If we need the 'false' gamma (i.e. γ̃), then we use the precomputed
        // value provided the interval is deemed to be accurate.
        if let (false, Ok(spline)) = (real, self.delta_gamma_tilde.read()) {
            let spline = &spline[self.count_heavy(context)];
            let log_beta = f64::ln(context.beta);
            if spline.contains(log_beta) {
                return Some(spline.sample(log_beta));
            }
        }

        let ptcl = context.model.particles();

        let p1 = &ptcl[self.particles.incoming_idx[0]];
        let p2 = &ptcl[self.particles.incoming_idx[1]];
        let p3 = &ptcl[self.particles.outgoing_idx[0]];
        let p4 = &ptcl[self.particles.outgoing_idx[1]];

        let integrand = |s, t| {
            let u = p1.mass2 + p2.mass2 + p3.mass2 + p4.mass2 - s - t;
            asymmetry(context.model, s, t, u).abs()
        };

        if real {
            Some(utilities::integrate_st(
                integrand,
                context.beta,
                p1,
                p2,
                p3,
                p4,
            ))
        } else {
            let delta_gamma_tilde =
                utilities::integrate_st_on_n(integrand, context.beta, p1, p2, p3, p4);
            if let Ok(mut spline) = self.delta_gamma_tilde.write() {
                let spline = &mut spline[self.count_heavy(context)];
                spline.add(f64::ln(context.beta), delta_gamma_tilde);
            }
            Some(delta_gamma_tilde)
        }
    }

    #[allow(clippy::too_many_lines)]
    fn rate(&self, context: &Context<M>) -> Option<RateDensity> {
        let gamma = self.gamma(context, false)?;
        let delta_gamma = self.delta_gamma(context, false);

        // If both rates are 0, there's no need to adjust it to the particles'
        // number densities.
        if gamma == 0.0 && delta_gamma.unwrap_or_default() == 0.0 {
            return None;
        }

        let ptcl = context.model.particles();

        let [i1, i2, i3, i4] = [
            self.particles.incoming_idx[0],
            self.particles.incoming_idx[1],
            self.particles.outgoing_idx[0],
            self.particles.outgoing_idx[1],
        ];
        let [p1, p2, p3, p4] = [&ptcl[i1], &ptcl[i2], &ptcl[i3], &ptcl[i4]];
        let [n1, n2, n3, n4] = [context.n[i1], context.n[i2], context.n[i3], context.n[i4]];
        let [na1, na2, na3, na4] = [
            self.particles.incoming_sign[0] * context.na[i1],
            self.particles.incoming_sign[1] * context.na[i2],
            self.particles.outgoing_sign[0] * context.na[i3],
            self.particles.outgoing_sign[1] * context.na[i4],
        ];
        let [eq1, eq2, eq3, eq4] = [
            context.eq[i1],
            context.eq[i2],
            context.eq[i3],
            context.eq[i4],
        ];

        let mut rate = RateDensity::zero();

        // Since we get the 'false' interaction rate, we can always store that
        // in the rate density now.
        rate.gamma_tilde = gamma;
        rate.delta_gamma_tilde = delta_gamma;

        // We check for how many heavy particles we have.  In each case, we also
        // have to check if the equilibrium number density of the heavy
        // particle(s) is exactly 0 as it may result in NaN appearing if one of
        // the other divisors is also 0.
        #[allow(clippy::unnested_or_patterns)]
        match (
            p1.mass * context.beta > M_BETA_THRESHOLD,
            p2.mass * context.beta > M_BETA_THRESHOLD,
            p3.mass * context.beta > M_BETA_THRESHOLD,
            p4.mass * context.beta > M_BETA_THRESHOLD,
        ) {
            // No heavy particle
            (false, false, false, false) => {
                rate.gamma = gamma;
                rate.delta_gamma = delta_gamma;
                let delta_gamma = delta_gamma.unwrap_or_default();
                let symmetric_prefactor = checked_div(n1, eq1) * checked_div(n2, eq2)
                    - checked_div(n3, eq3) * checked_div(n4, eq4);
                rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                    + checked_mul(
                        gamma,
                        checked_div(na1 * n2 + na2 * n1, eq1 * eq2)
                            - checked_div(na3 * n4 + na4 * n3, eq3 * eq4),
                    );
            }

            // One heavy particles
            (true, false, false, false) => {
                rate.gamma = gamma * eq1;
                rate.delta_gamma = delta_gamma.map(|v| v * eq1);
                let delta_gamma = delta_gamma.unwrap_or_default();
                if eq1 == 0.0 {
                    let symmetric_prefactor = n1 * checked_div(n2, eq2);
                    rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                    rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                        + checked_mul(gamma, checked_div(na1 * n2 + na2 * n1, eq2));
                } else {
                    rate.symmetric = checked_mul(
                        gamma,
                        n1 * checked_div(n2, eq2)
                            - eq1 * checked_div(n3, eq3) * checked_div(n4, eq4),
                    );
                    rate.asymmetric = checked_mul(
                        delta_gamma,
                        n1 * checked_div(n2, eq2)
                            + eq1 * checked_div(n3, eq3) * checked_div(n4, eq4),
                    ) + checked_mul(
                        gamma,
                        checked_div(na1 * n2 + na2 * n1, eq2)
                            - eq1 * checked_div(na3 * n4 + na4 * n3, eq3 * eq4),
                    );
                }
            }
            (false, true, false, false) => {
                rate.gamma = gamma * eq2;
                rate.delta_gamma = delta_gamma.map(|v| v * eq2);
                let delta_gamma = delta_gamma.unwrap_or_default();
                if eq2 == 0.0 {
                    let symmetric_prefactor = checked_div(n1, eq1) * n2;
                    rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                    rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                        + checked_mul(gamma, checked_div(na1 * n2 + na2 * n1, eq1));
                } else {
                    let symmetric_prefactor = checked_div(n1, eq1) * n2
                        - eq2 * checked_div(n3, eq3) * checked_div(n4, eq4);
                    rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                    rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                        + checked_mul(
                            gamma,
                            checked_div(na1 * n2 + na2 * n1, eq1)
                                - eq2 * checked_div(na3 * n4 + na4 * n3, eq3 * eq4),
                        );
                }
            }
            (false, false, true, false) => {
                rate.gamma = gamma * eq3;
                rate.delta_gamma = delta_gamma.map(|v| v * eq3);
                let delta_gamma = delta_gamma.unwrap_or_default();
                if eq3 == 0.0 {
                    let symmetric_prefactor = -n3 * checked_div(n4, eq4);
                    rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                    rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                        - gamma * checked_div(na3 * n4 + na4 * n3, eq4);
                } else {
                    let symmetric_prefactor = eq3 * checked_div(n1, eq1) * checked_div(n2, eq2)
                        - n3 * checked_div(n4, eq4);
                    rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                    rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                        + checked_mul(
                            gamma,
                            eq3 * checked_div(na1 * n2 + na2 * n1, eq1 * eq2)
                                - checked_div(na3 * n4 + na4 * n3, eq4),
                        );
                };
            }
            (false, false, false, true) => {
                rate.gamma = gamma * eq4;
                rate.delta_gamma = delta_gamma.map(|v| v * eq4);
                let delta_gamma = delta_gamma.unwrap_or_default();
                if eq4 == 0.0 {
                    let symmetric_prefactor = -checked_div(n3, eq3) * n4;
                    rate.symmetric = gamma;
                    rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                        - gamma * checked_div(na3 * n4 + na4 * n3, eq3);
                } else {
                    let symmetric_prefactor = eq4 * checked_div(n1, eq1) * checked_div(n2, eq2)
                        - checked_div(n3, eq3) * n4;
                    rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                    rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                        + checked_mul(
                            gamma,
                            eq4 * checked_div(na1 * n2 + na2 * n1, eq1 * eq2)
                                - checked_div(na3 * n4 + na4 * n3, eq3),
                        );
                }
            }

            // Two heavy particles
            (true, true, false, false) => {
                rate.gamma = gamma * eq1 * eq2;
                rate.delta_gamma = delta_gamma.map(|v| v * eq1 * eq2);
                let delta_gamma = delta_gamma.unwrap_or_default();
                if eq1 * eq2 == 0.0 {
                    let symmetric_prefactor = n1 * n2;
                    rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                    rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                        + checked_mul(gamma, na1 * n2 + na2 * n1);
                } else {
                    let symmetric_prefactor =
                        n1 * n2 - eq1 * eq2 * checked_div(n3, eq3) * checked_div(n4, eq4);
                    rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                    rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                        + checked_mul(
                            gamma,
                            (na1 * n2 + na2 * n1)
                                - eq1 * eq2 * checked_div(na3 * n4 + na4 * n3, eq3 * eq4),
                        );
                }
            }
            (true, false, true, false) => {
                rate.gamma = gamma * eq1 * eq3;
                rate.delta_gamma = delta_gamma.map(|v| v * eq1 * eq3);
                let delta_gamma = delta_gamma.unwrap_or_default();
                match (eq1 == 0.0, eq3 == 0.0) {
                    (true, true) => {
                        rate.symmetric = 0.0;
                        rate.asymmetric = 0.0;
                    }
                    (true, false) => {
                        let symmetric_prefactor = eq3 * n1 * checked_div(n2, eq2);
                        rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                        rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                            + checked_mul(gamma, eq3 * checked_div(na1 * n2 + na2 * n1, eq2));
                    }
                    (false, true) => {
                        let symmetric_prefactor = -eq1 * n3 * checked_div(n4, eq4);
                        rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                        rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                            - gamma * eq1 * checked_div(na3 * n4 + na4 * n3, eq4);
                    }
                    (false, false) => {
                        let symmetric_prefactor =
                            eq3 * n1 * checked_div(n2, eq2) - eq1 * n3 * checked_div(n4, eq4);
                        rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                        rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                            + checked_mul(
                                gamma,
                                eq3 * checked_div(na1 * n2 + na2 * n1, eq2)
                                    - eq1 * checked_div(na3 * n4 + na4 * n3, eq4),
                            );
                    }
                }
            }
            (true, false, false, true) => {
                rate.gamma = gamma * eq1 * eq4;
                rate.delta_gamma = delta_gamma.map(|v| v * eq1 * eq4);
                let delta_gamma = delta_gamma.unwrap_or_default();
                match (eq1 == 0.0, eq4 == 0.0) {
                    (true, true) => {
                        rate.symmetric = 0.0;
                        rate.asymmetric = 0.0;
                    }
                    (true, false) => {
                        let symmetric_prefactor = eq4 * n1 * checked_div(n2, eq2);
                        rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                        rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                            + checked_mul(gamma, eq4 * checked_div(na1 * n2 + na2 * n1, eq2));
                    }
                    (false, true) => {
                        let symmetric_prefactor = -eq1 * checked_div(n3, eq3) * n4;
                        rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                        rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                            - gamma * eq1 * checked_div(na3 * n4 + na4 * n3, eq3);
                    }
                    (false, false) => {
                        let symmetric_prefactor =
                            eq4 * n1 * checked_div(n2, eq2) - eq1 * checked_div(n3, eq3) * n4;
                        rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                        rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                            + checked_mul(
                                gamma,
                                eq4 * checked_div(na1 * n2 + na2 * n1, eq2)
                                    - eq1 * checked_div(na3 * n4 + na4 * n3, eq3),
                            );
                    }
                }
            }
            (false, true, true, false) => {
                rate.gamma = gamma * eq2 * eq3;
                rate.delta_gamma = delta_gamma.map(|v| v * eq2 * eq3);
                let delta_gamma = delta_gamma.unwrap_or_default();
                match (eq2 == 0.0, eq3 == 0.0) {
                    (true, true) => {
                        rate.symmetric = 0.0;
                        rate.asymmetric = 0.0;
                    }
                    (true, false) => {
                        let symmetric_prefactor = eq3 * checked_div(n1, eq1) * n2;
                        rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                        rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                            + checked_mul(gamma, eq3 * checked_div(na1 * n2 + na2 * n1, eq1));
                    }
                    (false, true) => {
                        let symmetric_prefactor = -eq2 * n3 * checked_div(n4, eq4);
                        rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                        rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                            - gamma * eq2 * checked_div(na3 * n4 + na4 * n3, eq4);
                    }
                    (false, false) => {
                        let symmetric_prefactor =
                            eq3 * checked_div(n1, eq1) * n2 - eq2 * n3 * checked_div(n4, eq4);
                        rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                        rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                            + checked_mul(
                                gamma,
                                eq3 * checked_div(na1 * n2 + na2 * n1, eq1)
                                    - eq2 * checked_div(na3 * n4 + na4 * n3, eq4),
                            );
                    }
                }
            }
            (false, true, false, true) => {
                rate.gamma = gamma * eq2 * eq4;
                rate.delta_gamma = delta_gamma.map(|v| v * eq2 * eq4);
                let delta_gamma = delta_gamma.unwrap_or_default();
                match (eq2 == 0.0, eq4 == 0.0) {
                    (true, true) => {
                        rate.symmetric = 0.0;
                        rate.asymmetric = 0.0;
                    }
                    (true, false) => {
                        let symmetric_prefactor = eq4 * checked_div(n1, eq1) * n2;
                        rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                        rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                            + checked_mul(gamma, eq4 * checked_div(na1 * n2 + na2 * n1, eq1));
                    }
                    (false, true) => {
                        let symmetric_prefactor = -eq2 * checked_div(n3, eq3) * n4;
                        rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                        rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                            - gamma * eq2 * checked_div(na3 * n4 + na4 * n3, eq3);
                    }
                    (false, false) => {
                        let symmetric_prefactor =
                            eq4 * checked_div(n1, eq1) * n2 - eq2 * checked_div(n3, eq3) * n4;
                        rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                        rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                            + checked_mul(
                                gamma,
                                eq4 * checked_div(na1 * n2 + na2 * n1, eq1)
                                    - eq2 * checked_div(na3 * n4 + na4 * n3, eq3),
                            );
                    }
                }
            }
            (false, false, true, true) => {
                rate.gamma = gamma * eq3 * eq4;
                rate.delta_gamma = delta_gamma.map(|v| v * eq3 * eq4);
                let delta_gamma = delta_gamma.unwrap_or_default();
                if eq3 * eq4 == 0.0 {
                    let symmetric_prefactor = -n3 * n4;
                    rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                    rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                        - gamma * (na3 * n4 + na4 * n3);
                } else {
                    let symmetric_prefactor =
                        eq3 * eq4 * checked_div(n1, eq1) * checked_div(n2, eq2) - n3 * n4;
                    rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                    rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                        + checked_mul(
                            gamma,
                            eq3 * eq4 * checked_div(na1 * n2 + na2 * n1, eq1 * eq2)
                                - (na3 * n4 + na4 * n3),
                        );
                }
            }

            // Three heaving particles
            (true, true, true, false) => {
                rate.gamma = gamma * eq1 * eq2 * eq3;
                rate.delta_gamma = delta_gamma.map(|v| v * eq1 * eq2 * eq3);
                let delta_gamma = delta_gamma.unwrap_or_default();
                match (eq1 == 0.0, eq2 == 0.0, eq3 == 0.0) {
                    (true, true, true) | (true, false, true) | (false, true, true) => {
                        rate.symmetric = 0.0;
                        rate.asymmetric = 0.0;
                    }
                    (true, true, false) | (true, false, false) | (false, true, false) => {
                        let symmetric_prefactor = eq3 * n1 * n2;
                        rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                        rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                            + checked_mul(gamma, eq3 * (na1 * n2 + na2 * n1));
                    }
                    (false, false, true) => {
                        let symmetric_prefactor = eq1 * eq2 * n3 * checked_div(n4, eq4);
                        rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                        rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                            + checked_mul(gamma, eq1 * eq2 * checked_div(na3 * n4 + na4 * n3, eq4));
                    }
                    (false, false, false) => {
                        let symmetric_prefactor =
                            eq3 * n1 * n2 - eq1 * eq2 * n3 * checked_div(n4, eq4);
                        rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                        rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                            + checked_mul(
                                gamma,
                                eq3 * (na1 * n2 + na2 * n1)
                                    - eq1 * eq2 * checked_div(na3 * n4 + na4 * n3, eq4),
                            );
                    }
                }
            }
            (true, true, false, true) => {
                rate.gamma = gamma * eq1 * eq2 * eq4;
                rate.delta_gamma = delta_gamma.map(|v| v * eq1 * eq2 * eq4);
                let delta_gamma = delta_gamma.unwrap_or_default();
                match (eq1 == 0.0, eq2 == 0.0, eq4 == 0.0) {
                    (true, true, true) | (true, false, true) | (false, true, true) => {
                        rate.symmetric = 0.0;
                        rate.asymmetric = 0.0;
                    }
                    (true, true, false) | (true, false, false) | (false, true, false) => {
                        let symmetric_prefactor = eq4 * n1 * n2;
                        rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                        rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                            + checked_mul(gamma, eq4 * (na1 * n2 + na2 * n1));
                    }
                    (false, false, true) => {
                        let symmetric_prefactor = eq1 * eq2 * checked_div(n3, eq3) * n4;
                        rate.symmetric = -checked_mul(gamma, symmetric_prefactor);
                        rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                            - gamma * eq1 * eq2 * checked_div(na3 * n4 + na4 * n3, eq3);
                    }
                    (false, false, false) => {
                        let symmetric_prefactor =
                            eq4 * n1 * n2 - eq1 * eq2 * checked_div(n3, eq3) * n4;
                        rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                        rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                            + checked_mul(
                                gamma,
                                eq4 * (na1 * n2 + na2 * n1)
                                    - eq1 * eq2 * checked_div(na3 * n4 + na4 * n3, eq3),
                            );
                    }
                }
            }
            (true, false, true, true) => {
                rate.gamma = gamma * eq1 * eq3 * eq4;
                rate.delta_gamma = delta_gamma.map(|v| v * eq1 * eq3 * eq4);
                let delta_gamma = delta_gamma.unwrap_or_default();
                match (eq1 == 0.0, eq3 == 0.0, eq4 == 0.0) {
                    (true, true, true) | (true, true, false) | (true, false, true) => {
                        rate.symmetric = 0.0;
                        rate.asymmetric = 0.0;
                    }
                    (false, true, true) | (false, true, false) | (false, false, true) => {
                        let symmetric_prefactor = -eq1 * n3 * n4;
                        rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                        rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                            - gamma * eq1 * (na3 * n4 + na4 * n3);
                    }
                    (true, false, false) => {
                        let symmetric_prefactor = eq3 * eq4 * n1 * checked_div(n2, eq2);
                        rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                        rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                            + checked_mul(gamma, eq3 * eq4 * checked_div(na1 * n2 + na2 * n1, eq2));
                    }
                    (false, false, false) => {
                        let symmetric_prefactor =
                            eq3 * eq4 * n1 * checked_div(n2, eq2) - eq1 * n3 * n4;
                        rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                        rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                            + checked_mul(
                                gamma,
                                eq3 * eq4 * checked_div(na1 * n2 + na2 * n1, eq2)
                                    - eq1 * (na3 * n4 + na4 * n3),
                            );
                    }
                }
            }
            (false, true, true, true) => {
                rate.gamma = gamma * eq2 * eq3 * eq4;
                rate.delta_gamma = delta_gamma.map(|v| v * eq2 * eq3 * eq4);
                let delta_gamma = delta_gamma.unwrap_or_default();
                match (eq2 == 0.0, eq3 == 0.0, eq4 == 0.0) {
                    (true, true, true) | (true, true, false) | (true, false, true) => {
                        rate.symmetric = 0.0;
                        rate.asymmetric = 0.0;
                    }
                    (false, true, true) | (false, true, false) | (false, false, true) => {
                        let symmetric_prefactor = -eq2 * n3 * n4;
                        rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                        rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                            - gamma * eq2 * (na3 * n4 + na4 * n3);
                    }
                    (true, false, false) => {
                        let symmetric_prefactor = eq3 * eq4 * checked_div(n1, eq1) * n2;
                        rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                        rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                            + checked_mul(gamma, eq3 * eq4 * checked_div(na1 * n2 + na2 * n1, eq2));
                    }
                    (false, false, false) => {
                        let symmetric_prefactor =
                            eq3 * eq4 * checked_div(n1, eq1) * n2 - eq2 * n3 * n4;
                        rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                        rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                            + checked_mul(
                                gamma,
                                eq3 * eq4 * checked_div(na1 * n2 + na2 * n1, eq2)
                                    - eq2 * (na3 * n4 + na4 * n3),
                            );
                    }
                }
            }
            // Four heavy particles
            (true, true, true, true) => {
                rate.gamma = gamma * eq1 * eq2 * eq3 * eq4;
                rate.delta_gamma = delta_gamma.map(|v| v * eq1 * eq2 * eq3 * eq4);
                let delta_gamma = delta_gamma.unwrap_or_default();
                if rate.gamma == 0.0 {
                    rate.symmetric = 0.0;
                    rate.asymmetric = 0.0;
                } else {
                    let symmetric_prefactor = eq3 * eq4 * n1 * n2 - eq1 * eq2 * n3 * n4;
                    rate.symmetric = checked_mul(gamma, symmetric_prefactor);
                    rate.asymmetric = checked_mul(delta_gamma, symmetric_prefactor)
                        + checked_mul(
                            gamma,
                            eq3 * eq4 * (na1 * n2 + na2 * n1) - eq1 * eq2 * (na3 * n4 + na4 * n3),
                        );
                }
            }
        };

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
        be_n: f64,
        eq1: f64,
        eq2: f64,
        eq3: f64,
        eq4: f64,
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
            .join("four_particle");
        fs::create_dir_all(&dir)?;

        let csv = csv::Writer::from_path(dir.join(p))?;

        Ok(csv)
    }

    #[test]
    fn massless() -> Result<(), Box<dyn error::Error>> {
        let mut model = Empty::default();
        model.extend_particles(
            [0.0, 0.0, 0.0, 0.0]
                .iter()
                .enumerate()
                .map(|(i, &m)| Particle::new(0, m, m / 100.0).name(format!("{}", i + 1))),
        );
        let interaction = interaction::FourParticle::new(|_m, _s, _t, _u| 1.0, 1, 2, 3, 4);
        let mut csv = create_csv("massless.csv")?;

        for &beta in &Array1::geomspace(BETA_START, BETA_END, 1000).unwrap() {
            // println!("beta = {}", beta);
            model.set_beta(beta);
            let c = model.as_context();

            let rate = interaction.rate(&c);
            csv.serialize(CsvRow {
                beta,
                hubble_rate: c.hubble_rate,
                be_n: Statistic::BoseEinstein.number_density(beta, 0.0, 0.0),
                eq1: c.eq[1],
                eq2: c.eq[2],
                eq3: c.eq[3],
                eq4: c.eq[4],
                width: interaction.width(&c).map(|w| w.width),
                gamma_true: interaction.gamma(&c, true),
                gamma_false: interaction.gamma(&c, false),
                delta_gamma_true: interaction.delta_gamma(&c, true),
                delta_gamma_false: interaction.delta_gamma(&c, false),
                symmetric_prefactor: interaction.symmetric_prefactor(&c),
                asymmetric_prefactor: interaction.asymmetric_prefactor(&c),
                rate_symmetric: rate.map(|r| r.symmetric),
                rate_asymmetric: rate.map(|r| r.asymmetric),
                adjusted_rate_symmetric: interaction.adjusted_rate(&c).map(|r| r.symmetric),
                adjusted_rate_asymmetric: interaction.adjusted_rate(&c).map(|r| r.asymmetric),
            })?;

            csv.flush()?;
        }

        Ok(())
    }

    #[test]
    fn massive_m000() -> Result<(), Box<dyn error::Error>> {
        let mut model = Empty::default();
        model.extend_particles(
            [1e10, 0.0, 0.0, 0.0]
                .iter()
                .enumerate()
                .map(|(i, &m)| Particle::new(0, m, m / 100.0).name(format!("{}", i + 1))),
        );
        let interaction = interaction::FourParticle::new(|_m, _s, _t, _u| 1.0, 1, 2, 3, 4);
        let mut csv = create_csv("massive_m000.csv")?;

        for &beta in &Array1::geomspace(BETA_START, BETA_END, 1000).unwrap() {
            // println!("beta = {}", beta);
            model.set_beta(beta);
            let c = model.as_context();

            csv.serialize(CsvRow {
                beta,
                hubble_rate: c.hubble_rate,
                be_n: Statistic::BoseEinstein.number_density(beta, 0.0, 0.0),
                eq1: c.eq[1],
                eq2: c.eq[2],
                eq3: c.eq[3],
                eq4: c.eq[4],
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

    #[test]
    fn massive_m0m0() -> Result<(), Box<dyn error::Error>> {
        let mut model = Empty::default();
        model.extend_particles(
            [1e10, 0.0, 1e10, 0.0]
                .iter()
                .enumerate()
                .map(|(i, &m)| Particle::new(0, m, m / 100.0).name(format!("{}", i + 1))),
        );
        let interaction = interaction::FourParticle::new(|_m, _s, _t, _u| 1.0, 1, 2, 3, 4);
        let mut csv = create_csv("massive_m0m0.csv")?;

        for &beta in &Array1::geomspace(BETA_START, BETA_END, 1000).unwrap() {
            // println!("beta = {}", beta);
            model.set_beta(beta);
            let c = model.as_context();

            csv.serialize(CsvRow {
                beta,
                hubble_rate: c.hubble_rate,
                be_n: Statistic::BoseEinstein.number_density(beta, 0.0, 0.0),
                eq1: c.eq[1],
                eq2: c.eq[2],
                eq3: c.eq[3],
                eq4: c.eq[4],
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

    #[test]
    fn massive_mm00() -> Result<(), Box<dyn error::Error>> {
        let mut model = Empty::default();
        model.extend_particles(
            [1e10, 1e10, 0.0, 0.0]
                .iter()
                .enumerate()
                .map(|(i, &m)| Particle::new(0, m, m / 100.0).name(format!("{}", i + 1))),
        );
        let interaction = interaction::FourParticle::new(|_m, _s, _t, _u| 1.0, 1, 2, 3, 4);
        let mut csv = create_csv("massive_mm00.csv")?;

        for &beta in &Array1::geomspace(BETA_START, BETA_END, 1000).unwrap() {
            // println!("beta = {}", beta);
            model.set_beta(beta);
            let c = model.as_context();

            csv.serialize(CsvRow {
                beta,
                hubble_rate: c.hubble_rate,
                be_n: Statistic::BoseEinstein.number_density(beta, 0.0, 0.0),
                eq1: c.eq[1],
                eq2: c.eq[2],
                eq3: c.eq[3],
                eq4: c.eq[4],
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

    /// Unit amplitude
    #[test]
    fn massive_mmmm() -> Result<(), Box<dyn error::Error>> {
        let mut model = Empty::default();
        model.extend_particles(
            [1e10, 1e10, 1e10, 1e10]
                .iter()
                .enumerate()
                .map(|(i, &m)| Particle::new(0, m, m / 100.0).name(format!("{}", i + 1))),
        );
        let interaction = interaction::FourParticle::new(|_m, _s, _t, _u| 1.0, 1, 2, 3, 4);
        let mut csv = create_csv("massive_mmmm.csv")?;

        for &beta in &Array1::geomspace(BETA_START, BETA_END, 1000).unwrap() {
            // println!("beta = {}", beta);
            model.set_beta(beta);
            let c = model.as_context();

            csv.serialize(CsvRow {
                beta,
                hubble_rate: c.hubble_rate,
                be_n: Statistic::BoseEinstein.number_density(beta, 0.0, 0.0),
                eq1: c.eq[1],
                eq2: c.eq[2],
                eq3: c.eq[3],
                eq4: c.eq[4],
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
