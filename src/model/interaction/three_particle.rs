use crate::{
    model::{
        interaction::{
            Interaction, InteractionParticleIndices, InteractionParticleSigns,
            InteractionParticles, PartialWidth, RateDensity,
        },
        Model,
    },
    solver::Context,
    utilities::kallen_lambda_sqrt,
};
use special_functions::bessel;

/// Three particle interaction, all determined from the underlying squared amplitude.
pub struct ThreeParticle<M> {
    particles: InteractionParticles,
    particles_idx: InteractionParticleIndices,
    particles_sign: InteractionParticleSigns,
    /// Squared amplitude as a function of the model.
    squared_amplitude: Box<dyn Fn(&M) -> f64 + Sync>,
    /// Asymmetry between the amplitude and its \\(\mathcal{CP}\\) conjugate.
    asymmetry: Option<Box<dyn Fn(&M) -> f64 + Sync>>,
}

impl<M: Model> ThreeParticle<M> {
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
            ingoing: vec![p1],
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
        vec![
            Self::new(squared_amplitude, p1, p2, p3),
            Self::new(squared_amplitude, -p2, -p1, p3),
            Self::new(squared_amplitude, -p3, -p1, p2),
        ]
    }

    pub fn asymmetry<F>(mut self, asymmetry: F) -> Self
    where
        F: Fn(&M) -> f64 + Sync + 'static,
    {
        self.asymmetry = Some(Box::new(asymmetry));
        self
    }
}

impl<M> std::fmt::Debug for ThreeParticle<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

impl<M: Model> Interaction<M> for ThreeParticle<M> {
    fn particles(&self) -> &InteractionParticles {
        &self.particles
    }

    fn particles_idx(&self) -> &InteractionParticleIndices {
        &self.particles_idx
    }

    fn particles_sign(&self) -> &InteractionParticleSigns {
        &self.particles_sign
    }

    fn width(&self, c: &Context<M>) -> Option<PartialWidth> {
        let ptcl = c.model.particles();

        // Get the *squared* masses
        let m0 = ptcl[self.particles_idx.ingoing[0]].mass2;
        let m1 = ptcl[self.particles_idx.outgoing[0]].mass2;
        let m2 = ptcl[self.particles_idx.outgoing[1]].mass2;

        if m0 > m1 + m2 {
            let p = kallen_lambda_sqrt(m0, m1, m2) / (2.0 * m0);

            // 1 / 8 π ≅ 0.039788735772973836
            let width =
                0.039_788_735_772_973_836 * p / m0 * (self.squared_amplitude)(c.model).abs();

            Some(PartialWidth {
                width,
                parent: self.particles.ingoing[0],
                daughters: self.particles.outgoing.clone(),
            })
        } else {
            None
        }
    }

    /// Note that for three particle interactions, this does *not* return the
    /// reaction rate density, but can be obtained multiplied by the decaying
    /// particle's number density.
    ///
    /// The reason for this is that it prevents the computation of \\(\gamma / n
    /// \\) which becomes numerically unstable when \\(n\\) becomes very small.
    fn gamma(&self, c: &Context<M>) -> f64 {
        let ptcl = c.model.particles();

        // Get the masses
        let m0 = ptcl[self.particles_idx.ingoing[0]].mass;
        let m1 = ptcl[self.particles_idx.outgoing[0]].mass;
        let m2 = ptcl[self.particles_idx.outgoing[1]].mass;

        // The interaction rate goes from heaviest ↔ lighter two.
        //
        // TODO: What happens when the decaying particle's mass is not
        // greater than the sum of the masses of the daughter particles?
        if m0 < m1 || m0 < m2 {
            0.0
        } else {
            // 0.002_423_011_225_182_300_4 = ζ(3) / 16 π³
            0.002_423_011_225_182_300_4
                * (self.squared_amplitude)(&c.model).abs()
                * bessel::k1_on_k2(m0 * c.beta)
                / c.beta.powi(3)
                / m0
                * c.normalization
        }
    }

    /// Override the default implementation of calculate_rate to make use of the
    /// adjusted reaction rate density.
    fn calculate_rate(&self, gamma_tilde: f64, c: &Context<M>) -> RateDensity {
        if gamma_tilde == 0.0 {
            return RateDensity {
                forward: 0.0,
                backward: 0.0,
                asymmetric_forward: 0.0,
                asymmetric_backward: 0.0,
            };
        }

        // Get the masses
        let p0 = self.particles_idx.ingoing[0];
        let p1 = self.particles_idx.outgoing[0];
        let p2 = self.particles_idx.outgoing[1];

        let forward = gamma_tilde * c.n[p0];
        let backward =
            gamma_tilde * c.eq[p0] * nan_to_zero((c.n[p1] / c.eq[p1]) * (c.n[p2] / c.eq[p2]));

        let particles_sign = self.particles_sign();

        let asymmetric_forward = gamma_tilde * particles_sign.ingoing[0] * c.na[p0];
        let asymmetric_backward = gamma_tilde
            * c.eq[p0]
            * nan_to_zero(
                (c.n[p1] * particles_sign.outgoing[1] * c.na[p2]
                    + c.n[p2] * particles_sign.outgoing[0] * c.na[p1]
                    - particles_sign.outgoing[0]
                        * particles_sign.outgoing[1]
                        * c.na[p1]
                        * c.na[p2])
                    / (c.eq[p1] * c.eq[p2]),
            );

        RateDensity {
            forward,
            backward,
            asymmetric_forward,
            asymmetric_backward,
        }
    }
}

/// Converts NaN floating points to 0
fn nan_to_zero(v: f64) -> f64 {
    if v.is_nan() {
        0.0
    } else {
        v
    }
}
