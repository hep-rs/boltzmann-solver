use crate::{
    model::{
        interaction::{Interaction, InteractionParticles},
        Model,
    },
    solver::Context,
    utilities::{integrate_st, spline::Spline},
};
use std::{fmt, sync::RwLock};

// TODO: Define a trait alias for MandelstamFn once it is stabilised
// (https://github.com/rust-lang/rust/issues/41517)

/// Three particle interaction, all determined from the underlying squared
/// amplitude.
pub struct FourParticle<M> {
    particles: InteractionParticles,
    /// Squared amplitude as a function of the model.
    squared_amplitude: Box<dyn Fn(&M, f64, f64, f64) -> f64 + Sync>,
    gamma_spline: RwLock<Spline>,
    #[allow(clippy::type_complexity)]
    asymmetry: Option<Box<dyn Fn(&M, f64, f64, f64) -> f64 + Sync>>,
    asymmetry_spline: RwLock<Spline>,
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
            gamma_spline: RwLock::new(Spline::empty()),
            asymmetry: None,
            asymmetry_spline: RwLock::new(Spline::empty()),
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
    /// This asymmetry is subsequently used to compute the asymmetry in the
    /// interaction rate given by [`Interaction::asymmetry`].
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

    fn gamma(&self, c: &Context<M>, _real: bool) -> Option<f64> {
        if !self.gamma_enabled {
            return None;
        }

        let ln_beta = c.beta.ln();

        if let Ok(gamma_spline) = self.gamma_spline.read() {
            if gamma_spline.accurate(ln_beta) {
                return Some(gamma_spline.sample(ln_beta).exp());
            }
        }
        let ptcl = c.model.particles();

        // Get the *squared* masses
        let p0 = &ptcl[self.particles.incoming_idx[0]];
        let p1 = &ptcl[self.particles.incoming_idx[1]];
        let p2 = &ptcl[self.particles.outgoing_idx[0]];
        let p3 = &ptcl[self.particles.outgoing_idx[1]];

        let gamma = integrate_st(
            |s, t| {
                let u = p0.mass2 + p1.mass2 + p2.mass2 + p3.mass2 - s - t;
                (self.squared_amplitude)(&c.model, s, t, u).abs()
            },
            c.beta,
            p0.mass,
            p1.mass,
            p2.mass,
            p3.mass,
        )
        // FIXME: Should we take the absolute value?
        .abs();

        debug_assert!(
            gamma.is_finite(),
            "Computed a non-finit value for γ at step {} in interaction {}: {}",
            c.step,
            self.particles()
                .display(c.model)
                .unwrap_or_else(|_| self.particles().short_display()),
            gamma
        );

        if let Ok(mut gamma_spline) = self.gamma_spline.write() {
            gamma_spline.add(ln_beta, gamma.ln());
        }

        Some(gamma)
    }

    fn delta_gamma(&self, c: &Context<M>, _real: bool) -> Option<f64> {
        let asymmetry = self.asymmetry.as_ref()?;

        let ln_beta = c.beta.ln();

        if let Ok(asymmetry_spline) = self.asymmetry_spline.read() {
            if asymmetry_spline.accurate(ln_beta) {
                return Some(asymmetry_spline.sample(ln_beta).exp());
            }
        }
        let ptcl = c.model.particles();

        // Get the *squared* masses
        let p0 = &ptcl[self.particles.incoming_idx[0]];
        let p1 = &ptcl[self.particles.incoming_idx[1]];
        let p2 = &ptcl[self.particles.outgoing_idx[0]];
        let p3 = &ptcl[self.particles.outgoing_idx[1]];

        let delta_gamma = integrate_st(
            |s, t| {
                let u = p0.mass2 + p1.mass2 + p2.mass2 + p3.mass2 - s - t;
                asymmetry(&c.model, s, t, u).abs()
            },
            c.beta,
            p0.mass,
            p1.mass,
            p2.mass,
            p3.mass,
        )
        // FIXME: Should we take the absolute value?
        .abs();

        debug_assert!(
            delta_gamma.is_finite(),
            "Computed a non-finit value for δγ at step {} in interaction {}: {}",
            c.step,
            self.particles()
                .display(c.model)
                .unwrap_or_else(|_| self.particles().short_display()),
            delta_gamma
        );

        if let Ok(mut asymmetry_spline) = self.asymmetry_spline.write() {
            asymmetry_spline.add(ln_beta, delta_gamma.ln());
        }

        Some(delta_gamma)
    }
}
