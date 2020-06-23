use crate::{
    model::{
        interaction::{
            Interaction, InteractionParticleIndices, InteractionParticleSigns, InteractionParticles,
        },
        Model,
    },
    solver::Context,
    utilities::{integrate_st, spline::CubicHermiteSpline},
};
use std::{fmt, sync::RwLock};

// TODO: Define a trait alias for MandelstamFn once it is stabilised
// (https://github.com/rust-lang/rust/issues/41517)

/// Three particle interaction, all determined from the underlying squared
/// amplitude.
pub struct FourParticle<M> {
    particles: InteractionParticles,
    particles_idx: InteractionParticleIndices,
    particles_sign: InteractionParticleSigns,
    /// Squared amplitude as a function of the model.
    squared_amplitude: Box<dyn Fn(&M, f64, f64, f64) -> f64 + Sync>,
    gamma_spline: RwLock<CubicHermiteSpline>,
    #[allow(clippy::type_complexity)]
    asymmetry: Option<Box<dyn Fn(&M, f64, f64, f64) -> f64 + Sync>>,
    asymmetry_spline: RwLock<CubicHermiteSpline>,
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
    /// \\begin{equation}
    ///   \abs{\mathcal{M}(p_1 p_2 \leftrightarrow p_3 p_4)}^2(s, t, u)
    /// \\end{equation}
    ///
    /// where \\(s\\), \\(t\\) and \\(u\\) are the usual Mandelstam variables.
    /// Crossing symmetry is then used in order to compute the other processes.
    ///
    /// The three-body decays are not currently computed.
    pub fn new<F>(squared_amplitude: F, p1: isize, p2: isize, p3: isize, p4: isize) -> Self
    where
        F: Fn(&M, f64, f64, f64) -> f64 + Sync + 'static,
    {
        let particles = InteractionParticles {
            incoming: vec![p1, p2],
            outgoing: vec![p3, p4],
        };
        let particles_idx = particles.as_idx();
        let particles_sign = particles.as_sign();

        Self {
            particles,
            particles_idx,
            particles_sign,
            squared_amplitude: Box::new(squared_amplitude),
            gamma_spline: RwLock::new(CubicHermiteSpline::empty()),
            asymmetry: None,
            asymmetry_spline: RwLock::new(CubicHermiteSpline::empty()),
            width_enabled: false,
            gamma_enabled: true,
        }
    }

    /// Create a set of related four-particle interactions.
    ///
    /// This functions creates all three possible interactions from the same
    /// underlying squared amplitude, related through crossing symmetry:
    ///
    /// \\begin{equation} \\begin{aligned} p_1 p_2 \leftrightarrow p_3 p_4, \\\\
    ///   p_1 \overline{p_3} \leftrightarrow \overline{p_2} p_4, \\\\
    ///   p_1 \overline{p_4} \leftrightarrow \overline{p_2} p_3. \\\\
    /// \\end{aligned} \\end{equation}
    ///
    /// The squared amplitude is defined of the form:
    ///
    /// \\begin{equation}
    ///    \abs{\mathcal{M}(p_1 p_2 \leftrightarrow p_3 p_4)}^2(s, t, u)
    /// \\end{equation}
    ///
    /// where \\(s\\), \\(t\\) and \\(u\\) are the usual Mandelstam variables.
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

    /// Specify the asymmetry between this process and its CP-conjugate.
    ///
    /// This asymmetry is specified in terms of the asymmetry in the squared
    /// amplitudes:
    ///
    /// \\begin{equation}
    ///   \delta \abs{\mathcal{M}}^2
    ///     \defeq \abs{\mathcal{M}(p_1 p_2 \to p_3 p_4)}^2 - \abs{\mathcal{M}(\overline{p_1} \overline{p_2} \to \overline{p_3} \overline{p_4})}^2
    ///     = \abs{\mathcal{M}(p_1 p_2 \to p_3 p_4)}^2 - \abs{\mathcal{M}(p_3 p_4 \to p_1 p_2)}^2
    /// \\end{equation}
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
             particles_idx: {:?}, \
             particles_sign: {:?}, \
             squared_amplitude: Box<Fn>, \
             }}",
            self.particles, self.particles_idx, self.particles_sign
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

    fn particles_idx(&self) -> &InteractionParticleIndices {
        &self.particles_idx
    }

    fn particles_sign(&self) -> &InteractionParticleSigns {
        &self.particles_sign
    }

    fn width_enabled(&self) -> bool {
        self.width_enabled
    }

    // TODO: Implement three-body decays
    // fn width(&self, c: &Context<M>) -> Option<PartialWidth> {}

    fn gamma_enabled(&self) -> bool {
        self.gamma_enabled
    }

    fn gamma(&self, c: &Context<M>) -> Option<f64> {
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
        let m0 = ptcl[self.particles_idx.incoming[0]].mass2;
        let m1 = ptcl[self.particles_idx.incoming[1]].mass2;
        let m2 = ptcl[self.particles_idx.outgoing[0]].mass2;
        let m3 = ptcl[self.particles_idx.outgoing[1]].mass2;

        let gamma = integrate_st(
            |s, t| {
                let u = m0 + m1 + m2 + m3 - s - t;
                (self.squared_amplitude)(&c.model, s, t, u).abs()
            },
            c.beta,
            m0,
            m1,
            m2,
            m3,
        )
        // FIXME: Should we take the absolute value?
        .abs();

        if let Ok(mut gamma_spline) = self.gamma_spline.write() {
            gamma_spline.add(ln_beta, gamma.ln());
        }

        Some(gamma)
    }

    fn asymmetry(&self, c: &Context<M>) -> Option<f64> {
        let asymmetry = self.asymmetry.as_ref()?;

        let ln_beta = c.beta.ln();

        if let Ok(asymmetry_spline) = self.asymmetry_spline.read() {
            if asymmetry_spline.accurate(ln_beta) {
                return Some(asymmetry_spline.sample(ln_beta).exp());
            }
        }
        let ptcl = c.model.particles();

        // Get the *squared* masses
        let m0 = ptcl[self.particles_idx.incoming[0]].mass2;
        let m1 = ptcl[self.particles_idx.incoming[1]].mass2;
        let m2 = ptcl[self.particles_idx.outgoing[0]].mass2;
        let m3 = ptcl[self.particles_idx.outgoing[1]].mass2;

        let delta_gamma = integrate_st(
            |s, t| {
                let u = m0 + m1 + m2 + m3 - s - t;
                asymmetry(&c.model, s, t, u).abs()
            },
            c.beta,
            m0,
            m1,
            m2,
            m3,
        )
        // FIXME: Should we take the absolute value?
        .abs();

        if let Ok(mut asymmetry_spline) = self.asymmetry_spline.write() {
            asymmetry_spline.add(ln_beta, delta_gamma.ln());
        }

        Some(delta_gamma)
    }
}
