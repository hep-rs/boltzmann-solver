use crate::{
    model::{
        interaction::{
            checked_div, Interaction, InteractionParticleIndices, InteractionParticleSigns,
            InteractionParticles, PartialWidth, RateDensity,
        },
        Model,
    },
    solver::Context,
    utilities::kallen_lambda_sqrt,
};
use special_functions::bessel;
use std::fmt;

const M_BETA_THRESHOLD: f64 = 1e1;

/// Three particle interaction, all determined from the underlying squared amplitude.
pub struct ThreeParticle<M> {
    particles: InteractionParticles,
    particles_idx: InteractionParticleIndices,
    particles_sign: InteractionParticleSigns,
    /// Squared amplitude as a function of the model.
    squared_amplitude: Box<dyn Fn(&M) -> f64 + Sync>,
    /// Asymmetry between the amplitude and its \\(\mathcal{CP}\\) conjugate.
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
    /// If the process differs from its CP-conjugate process, the asymmetry can
    /// be specified with [`ThreeParticle::asymmetry`].
    pub fn new<F>(squared_amplitude: F, p1: isize, p2: isize, p3: isize) -> Self
    where
        F: Fn(&M) -> f64 + Sync + 'static,
    {
        let particles = InteractionParticles {
            incoming: vec![p1],
            outgoing: vec![p2, p3],
        };
        let particles_idx = particles.as_idx();
        let particles_sign = particles.as_sign();

        Self {
            particles,
            particles_idx,
            particles_sign,
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
    /// - \\(p_1 \leftrightarrow p_2 p_3\\),
    /// - \\(\overline{p_2} \leftrightarrow \overline{p_1} p_3\\), and
    /// - \\(\overline{p_3} \leftrightarrow \overline{p_1} p_2\\).
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

    /// Specify the asymmetry between this process and its CP-conjugate.
    pub fn set_asymmetry<F>(mut self, asymmetry: F) -> Self
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
                 particles_idx: {:?}, \
                 particles_sign: {:?}, \
                 squared_amplitude: Box<Fn>, \
                 asymmetric: Some(Box<Fn>) \
                 }}",
                self.particles, self.particles_idx, self.particles_sign
            )
        } else {
            write!(
                f,
                "ThreeParticle {{ \
                 particles: {:?}, \
                 particles_idx: {:?}, \
                 particles_sign: {:?}, \
                 squared_amplitude: Box<Fn>, \
                 asymmetric: None \
                 }}",
                self.particles, self.particles_idx, self.particles_sign
            )
        }
    }
}

impl<M> Interaction<M> for ThreeParticle<M>
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

    fn width(&self, c: &Context<M>) -> Option<PartialWidth> {
        if !self.width_enabled {
            return None;
        }

        let ptcl = c.model.particles();

        // Get the *squared* masses
        let p0 = &ptcl[self.particles_idx.incoming[0]];
        let p1 = &ptcl[self.particles_idx.outgoing[0]];
        let p2 = &ptcl[self.particles_idx.outgoing[1]];

        if p0.mass > p1.mass + p2.mass {
            let p = kallen_lambda_sqrt(p0.mass2, p1.mass2, p2.mass2) / (2.0 * p0.mass);

            // 1 / 8 π ≅ 0.039788735772973836
            let width =
                0.039_788_735_772_973_836 * p / p0.mass * (self.squared_amplitude)(c.model).abs();

            Some(PartialWidth {
                width,
                parent: self.particles.incoming[0],
                daughters: self.particles.outgoing.clone(),
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
    ///    normalized equilibrium number density of the particle (and
    ///    multiplying the result by the equilibrium number density will yield
    ///    the correct interaction density).  This is done in order to avoid a
    ///    `0 / 0` division to the reaction rate being proportional to \\(K_1(m
    ///    \beta)\\), while the number density is proportional to \\(K_2(m
    ///    \beta)\\).
    /// 2. Otherwise, calculate the usual reaction rate.
    ///
    /// Note that in order to obtain the analytical normalized equilibrium
    /// density, we must assume that \\(m \beta \gg 1 \\) so that it
    /// approaximates Maxwell--Boltzmann distribution, hence why we must use a
    /// threshold to switch between the two methods.
    fn gamma(&self, c: &Context<M>) -> Option<f64> {
        if !self.gamma_enabled {
            return None;
        }

        let ptcl = c.model.particles();

        // Get the particles
        let p0 = &ptcl[self.particles_idx.incoming[0]];
        let p1 = &ptcl[self.particles_idx.outgoing[0]];
        let p2 = &ptcl[self.particles_idx.outgoing[1]];

        if p0.mass * c.beta > M_BETA_THRESHOLD {
            // ζ(3) / 16 π³ ≅ 0.0024230112251823
            let z = p0.mass * c.beta;
            Some(
                0.002_423_011_225_182_3
                    * (self.squared_amplitude)(&c.model).abs()
                    * kallen_lambda_sqrt(p0.mass2, p1.mass2, p2.mass2)
                    / z.powi(3)
                    * bessel::k1_on_k2(z)
                    / p0.degrees_of_freedom(),
            )
        } else {
            // 1 / 32 π³ ≅ 0.001007860451037484
            let z = p0.mass * c.beta;
            Some(
                0.001_007_860_451_037_484
                    * (self.squared_amplitude)(&c.model).abs()
                    * kallen_lambda_sqrt(p0.mass2, p1.mass2, p2.mass2)
                    / z
                    * bessel::k1(z),
            )
        }
    }

    fn asymmetry(&self, c: &Context<M>) -> Option<f64> {
        self.asymmetry.as_ref().map(|a| a(c.model))
    }

    /// Override the default implementation of [`Interaction::rate`] to make use
    /// of the adjusted reaction rate density.
    fn rate(&self, gamma: Option<f64>, c: &Context<M>) -> Option<RateDensity> {
        // If there's no interaction rate or it is 0 to begin with, there's no
        // need to adjust it to the particles' number densities.
        if gamma.map_or(true, |gamma| gamma == 0.0) {
            return None;
        }

        // let ptcl = c.model.particles();
        let gamma = gamma.unwrap();
        let particles_sign = self.particles_sign();
        let ptcl = c.model.particles();

        // Get the particles
        let p0 = &ptcl[self.particles_idx.incoming[0]];
        let p1 = &ptcl[self.particles_idx.outgoing[0]];
        let p2 = &ptcl[self.particles_idx.outgoing[1]];

        // Get the indices
        let i0 = self.particles_idx.incoming[0];
        let i1 = self.particles_idx.outgoing[0];
        let i2 = self.particles_idx.outgoing[1];
        let [n0, n1, n2] = [c.n[i0], c.n[i1], c.n[i2]];
        let [na0, na1, na2] = [c.na[i0], c.na[i1], c.na[i2]];
        let [eq0, eq1, eq2] = [c.eq[i0], c.eq[i1], c.eq[i2]];
        let [a0, a1, a2] = [
            particles_sign.incoming[0],
            particles_sign.outgoing[0],
            particles_sign.outgoing[1],
        ];

        if p0.mass * c.beta > M_BETA_THRESHOLD {
            let (forward, asymmetric_forward) = if p0.mass > p1.mass + p2.mass {
                (gamma * n0, gamma * a0 * na0)
            } else {
                (0.0, 0.0)
            };

            let gamma = gamma * eq0;
            let backward = gamma * checked_div(n1, eq1) * checked_div(n2, eq2);
            let asymmetric_backward = gamma
                * (checked_div(a1 * na1, eq1) * checked_div(n2, eq2)
                    + checked_div(a2 * na2, eq2) * checked_div(n1, eq1)
                    - checked_div(a1 * na1, eq1) * checked_div(a2 * na2, eq2));

            Some(RateDensity {
                forward,
                backward,
                asymmetric_forward,
                asymmetric_backward,
            })
        } else {
            let (forward, asymmetric_forward) = if p0.mass > p1.mass + p2.mass {
                (
                    gamma * checked_div(n0, eq0),
                    gamma * checked_div(a0 * na0, eq0),
                )
            } else {
                (0.0, 0.0)
            };

            let backward = gamma * checked_div(n1, eq1) * checked_div(n2, eq2);
            let asymmetric_backward = gamma
                * (checked_div(a1 * na1, eq1) * checked_div(n2, eq2)
                    + checked_div(a2 * na2, eq2) * checked_div(n1, eq1)
                    - checked_div(a1 * na1, eq1) * checked_div(a2 * na2, eq2));

            Some(RateDensity {
                forward,
                backward,
                asymmetric_forward,
                asymmetric_backward,
            })
        }
    }
}
