mod utilities;

use crate::{
    model::{
        interaction::{checked_div, Interaction, InteractionParticles, RateDensity},
        Model,
    },
    solver::Context,
    utilities::spline::Spline,
};
use std::{fmt, sync::RwLock};

pub(crate) const M_BETA_THRESHOLD: f64 = 1e1;

// TODO: Define a trait alias for MandelstamFn once it is stabilised
// (https://github.com/rust-lang/rust/issues/41517)

/// Three particle interaction, all determined from the underlying squared
/// amplitude.
pub struct FourParticle<M> {
    particles: InteractionParticles,
    /// Squared amplitude as a function of the model.
    squared_amplitude: Box<dyn Fn(&M, f64, f64, f64) -> f64 + Sync>,
    // gamma_spline_real: RwLock<Spline>,
    // gamma_spline_on_n: RwLock<Spline>,
    #[allow(clippy::type_complexity)]
    asymmetry: Option<Box<dyn Fn(&M, f64, f64, f64) -> f64 + Sync>>,
    asymmetry_spline_true: RwLock<Spline>,
    asymmetry_spline_false: RwLock<Spline>,
    gamma_enabled: bool,
    width_enabled: bool,
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
        let particles = InteractionParticles::new(&[p1, p2], &[p3, p4]);

        Self {
            particles,
            squared_amplitude: Box::new(squared_amplitude),
            // gamma_spline_real: RwLock::new(Spline::empty()),
            // gamma_spline_on_n: RwLock::new(Spline::empty()),
            asymmetry: None,
            asymmetry_spline_true: RwLock::new(Spline::empty()),
            asymmetry_spline_false: RwLock::new(Spline::empty()),
            width_enabled: false,
            gamma_enabled: true,
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
            ))
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
    pub fn enable_width(mut self, v: bool) -> Self {
        self.width_enabled = v;
        self
    }

    /// Adjust whether this interaction will calculate decay widths.
    pub fn enable_gamma(mut self, v: bool) -> Self {
        self.gamma_enabled = v;
        self
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
    fn particles(&self) -> &InteractionParticles {
        &self.particles
    }

    fn width_enabled(&self) -> bool {
        self.width_enabled
    }

    // TODO: Implement three-body decays
    // fn width(&self, c: &Context<M>) -> Option<PartialWidth> {}

    fn gamma_enabled(&self) -> bool {
        self.gamma_enabled
    }

    fn gamma(&self, c: &Context<M>, real: bool) -> Option<f64> {
        if !self.gamma_enabled {
            return None;
        }

        // let ln_beta = c.beta.ln();
        // if let Ok(spline) = if real {
        //     self.gamma_spline_real.read()
        // } else {
        //     self.gamma_spline_on_n.read()
        // } {
        //     if spline.accurate(ln_beta) {
        //         return Some(spline.sample(ln_beta));
        //     }
        // }

        let ptcl = c.model.particles();

        // Get the particles involved
        let p1 = &ptcl[self.particles.incoming_idx[0]];
        let p2 = &ptcl[self.particles.incoming_idx[1]];
        let p3 = &ptcl[self.particles.outgoing_idx[0]];
        let p4 = &ptcl[self.particles.outgoing_idx[1]];

        let integrand = |s, t| {
            let u = p1.mass2 + p2.mass2 + p3.mass2 + p4.mass2 - s - t;
            (self.squared_amplitude)(&c.model, s, t, u).abs()
        };
        let gamma = if real {
            utilities::integrate_st(integrand, c.beta, p1, p2, p3, p4)
        } else {
            utilities::integrate_st_on_n(integrand, c.beta, p1, p2, p3, p4)
        };

        // debug_assert!(
        //     gamma.is_finite(),
        //     "[{}.{:02}|{:>9.3e}] Computed a non-finit value for γ in interaction {}: {}",
        //     c.step,
        //     c.substep,
        //     c.beta,
        //     self.particles()
        //         .display(c.model)
        //         .unwrap_or_else(|_| self.particles().short_display()),
        //     gamma
        // );

        // if let Ok(mut spline) = if real {
        //     self.gamma_spline_real.write()
        // } else {
        //     self.gamma_spline_on_n.write()
        // } {
        //     spline.add(ln_beta, gamma);
        // }

        Some(gamma)
    }

    fn delta_gamma(&self, c: &Context<M>, real: bool) -> Option<f64> {
        let asymmetry = self.asymmetry.as_ref()?;

        let ln_beta = c.beta.ln();
        if let Ok(spline) = if real {
            self.asymmetry_spline_true.read()
        } else {
            self.asymmetry_spline_false.read()
        } {
            if spline.accurate(ln_beta) {
                return Some(spline.sample(ln_beta));
            }
        }

        let ptcl = c.model.particles();

        let p1 = &ptcl[self.particles.incoming_idx[0]];
        let p2 = &ptcl[self.particles.incoming_idx[1]];
        let p3 = &ptcl[self.particles.outgoing_idx[0]];
        let p4 = &ptcl[self.particles.outgoing_idx[1]];

        let integrand = |s, t| {
            let u = p1.mass2 + p2.mass2 + p3.mass2 + p4.mass2 - s - t;
            asymmetry(&c.model, s, t, u).abs()
        };
        let delta_gamma = if real {
            utilities::integrate_st(integrand, c.beta, p1, p2, p3, p4)
        } else {
            utilities::integrate_st_on_n(integrand, c.beta, p1, p2, p3, p4)
        };

        // debug_assert!(
        //     delta_gamma.is_finite(),
        //     "[{}.{:02}|{:>9.3e}] Computed a non-finit value for δγ in interaction {}: {}",
        //     c.step,
        //     c.substep,
        //     c.beta,
        //     self.particles()
        //         .display(c.model)
        //         .unwrap_or_else(|_| self.particles().short_display()),
        //     delta_gamma
        // );

        if let Ok(mut spline) = if real {
            self.asymmetry_spline_true.write()
        } else {
            self.asymmetry_spline_false.write()
        } {
            spline.add(ln_beta, delta_gamma);
        }

        Some(delta_gamma)
    }

    #[allow(clippy::too_many_lines)]
    fn rate(&self, c: &Context<M>) -> Option<RateDensity> {
        let gamma = self.gamma(c, false).unwrap_or_default();
        let delta_gamma = self.delta_gamma(c, false);

        // If both rates are 0, there's no need to adjust it to the particles'
        // number densities.
        if gamma == 0.0 && (delta_gamma.is_none() || delta_gamma.unwrap() == 0.0) {
            return None;
        }

        let ptcl = c.model.particles();

        let [i1, i2, i3, i4] = [
            self.particles.incoming_idx[0],
            self.particles.incoming_idx[1],
            self.particles.outgoing_idx[0],
            self.particles.outgoing_idx[1],
        ];
        let [p1, p2, p3, p4] = [&ptcl[i1], &ptcl[i2], &ptcl[i3], &ptcl[i4]];
        let [n1, n2, n3, n4] = [c.n[i1], c.n[i2], c.n[i3], c.n[i4]];
        let [na1, na2, na3, na4] = [
            self.particles.incoming_sign[0] * c.na[i1],
            self.particles.incoming_sign[1] * c.na[i2],
            self.particles.outgoing_sign[0] * c.na[i3],
            self.particles.outgoing_sign[1] * c.na[i4],
        ];
        let [eq1, eq2, eq3, eq4] = [c.eq[i1], c.eq[i2], c.eq[i3], c.eq[i4]];

        let mut rate = RateDensity::zero();

        // We check for how many heavy particles we have.  In each case, we also
        // have to check if the equilibrium number density of the heavy
        // particle(s) is exactly 0 as it may result in NaN appearing if one of
        // the other divisors is also 0.
        #[allow(clippy::unnested_or_patterns)]
        match (
            p1.mass * c.beta > M_BETA_THRESHOLD,
            p2.mass * c.beta > M_BETA_THRESHOLD,
            p3.mass * c.beta > M_BETA_THRESHOLD,
            p4.mass * c.beta > M_BETA_THRESHOLD,
        ) {
            // No heavy particle
            (false, false, false, false) => {
                rate.gamma = gamma;
                rate.delta_gamma = delta_gamma;
                let delta_gamma = delta_gamma.unwrap_or_default();
                rate.symmetric = gamma
                    * (checked_div(n1, eq1) * checked_div(n2, eq2)
                        - checked_div(n3, eq3) * checked_div(n4, eq4));
                rate.asymmetric = delta_gamma
                    * (checked_div(n1, eq1) * checked_div(n2, eq2)
                        + checked_div(n3, eq3) * checked_div(n4, eq4))
                    + gamma
                        * (checked_div(na1 * n2 + na2 * n1, eq1 * eq2)
                            - checked_div(na3 * n4 + na4 * n3, eq3 * eq4));
            }

            // One heavy particles
            (true, false, false, false) => {
                rate.gamma = gamma * eq1;
                rate.delta_gamma = delta_gamma.map(|v| v * eq1);
                let delta_gamma = delta_gamma.unwrap_or_default();
                if eq1 == 0.0 {
                    rate.symmetric = gamma * (n1 * checked_div(n2, eq2));
                    rate.asymmetric = delta_gamma * (n1 * checked_div(n2, eq2))
                        + gamma * (checked_div(na1 * n2 + na2 * n1, eq2));
                } else {
                    rate.symmetric = gamma
                        * (n1 * checked_div(n2, eq2)
                            - eq1 * checked_div(n3, eq3) * checked_div(n4, eq4));
                    rate.asymmetric = delta_gamma
                        * (n1 * checked_div(n2, eq2)
                            + eq1 * checked_div(n3, eq3) * checked_div(n4, eq4))
                        + gamma
                            * (checked_div(na1 * n2 + na2 * n1, eq2)
                                - eq1 * checked_div(na3 * n4 + na4 * n3, eq3 * eq4));
                }
            }
            (false, true, false, false) => {
                rate.gamma = gamma * eq2;
                rate.delta_gamma = delta_gamma.map(|v| v * eq2);
                let delta_gamma = delta_gamma.unwrap_or_default();
                if eq2 == 0.0 {
                    rate.symmetric = gamma * checked_div(n1, eq1) * n2;
                    rate.asymmetric = delta_gamma * checked_div(n1, eq1) * n2
                        + gamma * checked_div(na1 * n2 + na2 * n1, eq1);
                } else {
                    rate.symmetric = gamma
                        * (checked_div(n1, eq1) * n2
                            - eq2 * checked_div(n3, eq3) * checked_div(n4, eq4));
                    rate.asymmetric = delta_gamma
                        * (checked_div(n1, eq1) * n2
                            + eq2 * checked_div(n3, eq3) * checked_div(n4, eq4))
                        + gamma
                            * (checked_div(na1 * n2 + na2 * n1, eq1)
                                - eq2 * checked_div(na3 * n4 + na4 * n3, eq3 * eq4));
                }
            }
            (false, false, true, false) => {
                rate.gamma = gamma * eq3;
                rate.delta_gamma = delta_gamma.map(|v| v * eq3);
                let delta_gamma = delta_gamma.unwrap_or_default();
                if eq3 == 0.0 {
                    rate.symmetric = -gamma * n3 * checked_div(n4, eq4);
                    rate.asymmetric = delta_gamma * n3 * checked_div(n4, eq4)
                        - gamma * checked_div(na3 * n4 + na4 * n3, eq4);
                } else {
                    rate.symmetric = gamma
                        * (eq3 * checked_div(n1, eq1) * checked_div(n2, eq2)
                            - n3 * checked_div(n4, eq4));
                    rate.asymmetric = delta_gamma
                        * (eq3 * checked_div(n1, eq1) * checked_div(n2, eq2)
                            + n3 * checked_div(n4, eq4))
                        + gamma
                            * (eq3 * checked_div(na1 * n2 + na2 * n1, eq1 * eq2)
                                - checked_div(na3 * n4 + na4 * n3, eq4))
                };
            }
            (false, false, false, true) => {
                rate.gamma = gamma * eq4;
                rate.delta_gamma = delta_gamma.map(|v| v * eq4);
                let delta_gamma = delta_gamma.unwrap_or_default();
                if eq4 == 0.0 {
                    rate.symmetric = -gamma * checked_div(n3, eq3) * n4;
                    rate.asymmetric = delta_gamma * checked_div(n3, eq3) * n4
                        - gamma * checked_div(na3 * n4 + na4 * n3, eq3);
                } else {
                    rate.symmetric = gamma
                        * (eq4 * checked_div(n1, eq1) * checked_div(n2, eq2)
                            - checked_div(n3, eq3) * n4);
                    rate.asymmetric = delta_gamma
                        * (eq4 * checked_div(n1, eq1) * checked_div(n2, eq2)
                            + checked_div(n3, eq3) * n4)
                        + gamma
                            * (eq4 * checked_div(na1 * n2 + na2 * n1, eq1 * eq2)
                                - checked_div(na3 * n4 + na4 * n3, eq3));
                }
            }

            // Two heavy particles
            (true, true, false, false) => {
                rate.gamma = gamma * eq1 * eq2;
                rate.delta_gamma = delta_gamma.map(|v| v * eq1 * eq2);
                let delta_gamma = delta_gamma.unwrap_or_default();
                if eq1 * eq2 == 0.0 {
                    rate.symmetric = gamma * n1 * n2;
                    rate.asymmetric = delta_gamma * n1 * n2 + gamma * (na1 * n2 + na2 * n1);
                } else {
                    rate.symmetric =
                        gamma * (n1 * n2 - eq1 * eq2 * checked_div(n3, eq3) * checked_div(n4, eq4));
                    rate.asymmetric = delta_gamma
                        * (n1 * n2 + eq1 * eq2 * checked_div(n3, eq3) * checked_div(n4, eq4))
                        + gamma
                            * ((na1 * n2 + na2 * n1)
                                - eq1 * eq2 * checked_div(na3 * n4 + na4 * n3, eq3 * eq4));
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
                        rate.symmetric = gamma * eq3 * n1 * checked_div(n2, eq2);
                        rate.asymmetric = delta_gamma * eq3 * n1 * checked_div(n2, eq2)
                            + gamma * eq3 * checked_div(na1 * n2 + na2 * n1, eq2);
                    }
                    (false, true) => {
                        rate.symmetric = -gamma * eq1 * n3 * checked_div(n4, eq4);
                        rate.asymmetric = delta_gamma * eq1 * n3 * checked_div(n4, eq4)
                            - gamma * eq1 * checked_div(na3 * n4 + na4 * n3, eq4);
                    }
                    (false, false) => {
                        rate.symmetric = gamma
                            * (eq3 * n1 * checked_div(n2, eq2) - eq1 * n3 * checked_div(n4, eq4));
                        rate.asymmetric = delta_gamma
                            * (eq3 * n1 * checked_div(n2, eq2) + eq1 * n3 * checked_div(n4, eq4))
                            + gamma
                                * (eq3 * checked_div(na1 * n2 + na2 * n1, eq2)
                                    - eq1 * checked_div(na3 * n4 + na4 * n3, eq4));
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
                        rate.symmetric = gamma * eq4 * n1 * checked_div(n2, eq2);
                        rate.asymmetric = delta_gamma * eq4 * n1 * checked_div(n2, eq2)
                            + gamma * eq4 * checked_div(na1 * n2 + na2 * n1, eq2);
                    }
                    (false, true) => {
                        rate.symmetric = -gamma * eq1 * checked_div(n3, eq3) * n4;
                        rate.asymmetric = delta_gamma * eq1 * checked_div(n3, eq3) * n4
                            - gamma * eq1 * checked_div(na3 * n4 + na4 * n3, eq3);
                    }
                    (false, false) => {
                        rate.symmetric = gamma
                            * (eq4 * n1 * checked_div(n2, eq2) - eq1 * checked_div(n3, eq3) * n4);
                        rate.asymmetric = delta_gamma
                            * (eq4 * n1 * checked_div(n2, eq2) + eq1 * checked_div(n3, eq3) * n4)
                            + gamma
                                * (eq4 * checked_div(na1 * n2 + na2 * n1, eq2)
                                    - eq1 * checked_div(na3 * n4 + na4 * n3, eq3));
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
                        rate.symmetric = gamma * eq3 * checked_div(n1, eq1) * n2;
                        rate.asymmetric = delta_gamma * eq3 * checked_div(n1, eq1) * n2
                            + gamma * eq3 * checked_div(na1 * n2 + na2 * n1, eq1);
                    }
                    (false, true) => {
                        rate.symmetric = -gamma * eq2 * n3 * checked_div(n4, eq4);
                        rate.asymmetric = delta_gamma * eq2 * n3 * checked_div(n4, eq4)
                            - gamma * eq2 * checked_div(na3 * n4 + na4 * n3, eq4);
                    }
                    (false, false) => {
                        rate.symmetric = gamma
                            * (eq3 * checked_div(n1, eq1) * n2 - eq2 * n3 * checked_div(n4, eq4));
                        rate.asymmetric = delta_gamma
                            * (eq3 * checked_div(n1, eq1) * n2 + eq2 * n3 * checked_div(n4, eq4))
                            + gamma
                                * (eq3 * checked_div(na1 * n2 + na2 * n1, eq1)
                                    - eq2 * checked_div(na3 * n4 + na4 * n3, eq4));
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
                        rate.symmetric = gamma * eq4 * checked_div(n1, eq1) * n2;
                        rate.asymmetric = delta_gamma * eq4 * checked_div(n1, eq1) * n2
                            + gamma * eq4 * checked_div(na1 * n2 + na2 * n1, eq1);
                    }
                    (false, true) => {
                        rate.symmetric = -gamma * eq2 * checked_div(n3, eq3) * n4;
                        rate.asymmetric = delta_gamma * eq2 * checked_div(n3, eq3) * n4
                            - gamma * eq2 * checked_div(na3 * n4 + na4 * n3, eq3);
                    }
                    (false, false) => {
                        rate.symmetric = gamma
                            * (eq4 * checked_div(n1, eq1) * n2 - eq2 * checked_div(n3, eq3) * n4);
                        rate.asymmetric = delta_gamma
                            * (eq4 * checked_div(n1, eq1) * n2 + eq2 * checked_div(n3, eq3) * n4)
                            + gamma
                                * (eq4 * checked_div(na1 * n2 + na2 * n1, eq1)
                                    - eq2 * checked_div(na3 * n4 + na4 * n3, eq3));
                    }
                }
            }
            (false, false, true, true) => {
                rate.gamma = gamma * eq3 * eq4;
                rate.delta_gamma = delta_gamma.map(|v| v * eq3 * eq4);
                let delta_gamma = delta_gamma.unwrap_or_default();
                if eq3 * eq4 == 0.0 {
                    rate.symmetric = -gamma * n3 * n4;
                    rate.asymmetric = delta_gamma * n3 * n4 - gamma * (na3 * n4 + na4 * n3);
                } else {
                    rate.symmetric =
                        gamma * (eq3 * eq4 * checked_div(n1, eq1) * checked_div(n2, eq2) - n3 * n4);
                    rate.asymmetric = delta_gamma
                        * (eq3 * eq4 * checked_div(n1, eq1) * checked_div(n2, eq2) + n3 * n4)
                        + gamma
                            * (eq3 * eq4 * checked_div(na1 * n2 + na2 * n1, eq1 * eq2)
                                - (na3 * n4 + na4 * n3));
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
                        rate.symmetric = gamma * (eq3 * n1 * n2);
                        rate.asymmetric =
                            delta_gamma * (eq3 * n1 * n2) + gamma * (eq3 * (na1 * n2 + na2 * n1));
                    }
                    (false, false, true) => {
                        rate.symmetric = gamma * eq1 * eq2 * n3 * checked_div(n4, eq4);
                        rate.asymmetric = delta_gamma * eq1 * eq2 * n3 * checked_div(n4, eq4)
                            + gamma * eq1 * eq2 * checked_div(na3 * n4 + na4 * n3, eq4);
                    }
                    (false, false, false) => {
                        rate.symmetric =
                            gamma * (eq3 * n1 * n2 - eq1 * eq2 * n3 * checked_div(n4, eq4));
                        rate.asymmetric = delta_gamma
                            * (eq3 * n1 * n2 + eq1 * eq2 * n3 * checked_div(n4, eq4))
                            + gamma
                                * (eq3 * (na1 * n2 + na2 * n1)
                                    - eq1 * eq2 * checked_div(na3 * n4 + na4 * n3, eq4));
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
                        rate.symmetric = gamma * eq4 * n1 * n2;
                        rate.asymmetric =
                            delta_gamma * eq4 * n1 * n2 + gamma * eq4 * (na1 * n2 + na2 * n1);
                    }
                    (false, false, true) => {
                        rate.symmetric = -gamma * eq1 * eq2 * checked_div(n3, eq3) * n4;
                        rate.asymmetric = delta_gamma * eq1 * eq2 * checked_div(n3, eq3) * n4
                            - gamma * eq1 * eq2 * checked_div(na3 * n4 + na4 * n3, eq3);
                    }
                    (false, false, false) => {
                        rate.symmetric =
                            gamma * (eq4 * n1 * n2 - eq1 * eq2 * checked_div(n3, eq3) * n4);
                        rate.asymmetric = delta_gamma
                            * (eq4 * n1 * n2 + eq1 * eq2 * checked_div(n3, eq3) * n4)
                            + gamma
                                * (eq4 * (na1 * n2 + na2 * n1)
                                    - eq1 * eq2 * checked_div(na3 * n4 + na4 * n3, eq3));
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
                        rate.symmetric = -gamma * eq1 * n3 * n4;
                        rate.asymmetric =
                            delta_gamma * eq1 * n3 * n4 - gamma * eq1 * (na3 * n4 + na4 * n3);
                    }
                    (true, false, false) => {
                        rate.symmetric = gamma * eq3 * eq4 * n1 * checked_div(n2, eq2);
                        rate.asymmetric = delta_gamma * eq3 * eq4 * n1 * checked_div(n2, eq2)
                            + gamma * eq3 * eq4 * checked_div(na1 * n2 + na2 * n1, eq2);
                    }
                    (false, false, false) => {
                        rate.symmetric =
                            gamma * (eq3 * eq4 * n1 * checked_div(n2, eq2) - eq1 * n3 * n4);
                        rate.asymmetric = delta_gamma
                            * (eq3 * eq4 * n1 * checked_div(n2, eq2) + eq1 * n3 * n4)
                            + gamma
                                * (eq3 * eq4 * checked_div(na1 * n2 + na2 * n1, eq2)
                                    - eq1 * (na3 * n4 + na4 * n3));
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
                        rate.symmetric = -gamma * eq2 * n3 * n4;
                        rate.asymmetric =
                            delta_gamma * eq2 * n3 * n4 - gamma * eq2 * (na3 * n4 + na4 * n3);
                    }
                    (true, false, false) => {
                        rate.symmetric = gamma * (eq3 * eq4 * checked_div(n1, eq1) * n2);
                        rate.asymmetric = delta_gamma * (eq3 * eq4 * checked_div(n1, eq1) * n2)
                            + gamma * (eq3 * eq4 * checked_div(na1 * n2 + na2 * n1, eq2));
                    }
                    (false, false, false) => {
                        rate.symmetric =
                            gamma * (eq3 * eq4 * checked_div(n1, eq1) * n2 - eq2 * n3 * n4);
                        rate.asymmetric = delta_gamma
                            * (eq3 * eq4 * checked_div(n1, eq1) * n2 + eq2 * n3 * n4)
                            + gamma
                                * (eq3 * eq4 * checked_div(na1 * n2 + na2 * n1, eq2)
                                    - eq2 * (na3 * n4 + na4 * n3));
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
                    rate.symmetric = gamma * (eq3 * eq4 * n1 * n2 - eq1 * eq2 * n3 * n4);
                    rate.asymmetric = delta_gamma * (eq3 * eq4 * n1 * n2 + eq1 * eq2 * n3 * n4)
                        + gamma
                            * (eq3 * eq4 * (na1 * n2 + na2 * n1)
                                - eq1 * eq2 * (na3 * n4 + na4 * n3));
                }
            }
        };

        if rate.symmetric.is_nan() || rate.symmetric.is_infinite() {
            rate.symmetric = 0.0;
        }
        if rate.asymmetric.is_nan() || rate.asymmetric.is_infinite() {
            rate.asymmetric = 0.0;
        }

        Some(rate * c.normalization)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        model::{interaction, EmptyModel},
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
        let mut model = EmptyModel::default();
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
                be_n: Statistic::BoseEinstein.massless_number_density(0.0, beta),
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
        let mut model = EmptyModel::default();
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
                be_n: Statistic::BoseEinstein.massless_number_density(0.0, beta),
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
        let mut model = EmptyModel::default();
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
                be_n: Statistic::BoseEinstein.massless_number_density(0.0, beta),
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
        let mut model = EmptyModel::default();
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
                be_n: Statistic::BoseEinstein.massless_number_density(0.0, beta),
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
        let mut model = EmptyModel::default();
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
                be_n: Statistic::BoseEinstein.massless_number_density(0.0, beta),
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
