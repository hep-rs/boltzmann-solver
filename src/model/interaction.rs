use crate::{
    model::Model,
    solver::Context,
    utilities::{integrate_st, kallen_lambda, spline::CubicHermiteSpline},
};
use ndarray::prelude::*;
use special_functions::bessel;
use std::{convert::TryFrom, sync::RwLock};
// use std::ops;

/// Interaction between particles.
///
/// This can be either a three-body or four-body interaction.  See the
/// documentation of [`Interaction::three_particle`] and [`Interaction::four_particle`]
/// for more details as to their implementations.
#[allow(clippy::large_enum_variant)]
pub enum Interaction<M: Model> {
    TwoParticle {
        signed_particles: [isize; 2],
        particles: [usize; 2],
        antiparticles: [f64; 2],
        m2: Box<dyn Fn(&M) -> f64>,
    },
    ThreeParticle {
        signed_particles: [[isize; 3]; 3],
        particles: [[usize; 3]; 3],
        antiparticles: [[f64; 3]; 3],
        m2: Box<dyn Fn(&M) -> f64>,
        asymmetry: Option<Box<dyn Fn(&M) -> f64>>,
    },
    FourParticle {
        signed_particles: [[isize; 4]; 3],
        particles: [[usize; 4]; 3],
        antiparticles: [[f64; 4]; 3],
        m2: Box<dyn Fn(&M, f64, f64, f64) -> f64>,
        gamma: [RwLock<CubicHermiteSpline>; 3],
    },
}

impl<M: Model> Interaction<M> {
    /// Return `true` if the interaction is a two-particle interaction.
    pub fn is_two_particle(&self) -> bool {
        match &self {
            Interaction::TwoParticle { .. } => true,
            _ => false,
        }
    }

    /// Return `true` if the interaction is a three-particle interaction.
    pub fn is_three_particle(&self) -> bool {
        match &self {
            Interaction::ThreeParticle { .. } => true,
            _ => false,
        }
    }

    /// Return `true` if the interaction is a four-particle interaction.
    pub fn is_four_particle(&self) -> bool {
        match &self {
            Interaction::FourParticle { .. } => true,
            _ => false,
        }
    }

    /// Create a new two-particle interaction.
    ///
    /// This will calculate the interaction between the particles \\(p_1\\) and
    /// \\(p_2\\).
    ///
    /// # Panics
    ///
    /// Panics if `p1 == p2`.  For a mass interaction, make sure that `p1 == -
    /// p2`.
    pub fn two_particle<F>(m2: F, p1: isize, p2: isize) -> Self
    where
        F: Fn(&M) -> f64 + 'static,
    {
        debug_assert!(p1 != p2, "cannot have {} ↔ {} interaction", p1, p2);
        let u1 = usize::try_from(p1.abs()).unwrap();
        let u2 = usize::try_from(p2.abs()).unwrap();
        Interaction::TwoParticle {
            particles: [u1, u2],
            signed_particles: [p1, p2],
            antiparticles: [p1.signum() as f64, p2.signum() as f64],
            m2: Box::new(m2),
        }
    }

    /// Create a new three-particle interaction.
    ///
    /// This will calculate the interaction between the three particles
    /// \\(p_1\\), \\(p_2\\) and \\(p_3\\).  The order in which the interaction
    /// takes place is determined by the masses of the particles such that the
    /// heaviest is (inverse) decaying from/to the lighter two.
    ///
    /// Particle are indicated by positive integers and the corresponding
    /// antiparticles are indicated by negative integers.
    ///
    /// If the process differs from its CP-conjugate process, the asymmetry can
    /// be specified with [`Interaction::asymmetry`].
    ///
    /// Only the interaction between the heavier particle and the lighter two is
    /// computed.
    pub fn three_particle<F>(m2: F, p1: isize, p2: isize, p3: isize) -> Self
    where
        F: Fn(&M) -> f64 + 'static,
    {
        let u1 = usize::try_from(p1.abs()).unwrap();
        let u2 = usize::try_from(p2.abs()).unwrap();
        let u3 = usize::try_from(p3.abs()).unwrap();
        Interaction::ThreeParticle {
            particles: [[u1, u2, u3], [u2, u1, u3], [u3, u1, u2]],
            signed_particles: [[p1, p2, p3], [-p2, -p1, p3], [-p3, -p1, p2]],
            antiparticles: [
                [p1.signum() as f64, p2.signum() as f64, p3.signum() as f64],
                [-p2.signum() as f64, -p1.signum() as f64, p3.signum() as f64],
                [-p3.signum() as f64, -p1.signum() as f64, p2.signum() as f64],
            ],
            m2: Box::new(m2),
            asymmetry: None,
        }
    }

    /// Add a new four-particle interaction.
    ///
    /// These are used to compute the various \\(2 \leftrightarrow 2\\)
    /// scatterings between the particles:
    ///
    /// \\begin{equation}
    /// \\begin{aligned}
    ///   p_1 p_2 \leftrightarrow p_3 p_4, \\\\
    ///   p_1 \overline{p_3} \leftrightarrow \overline{p_2} p_4, \\\\
    ///   p_1 \overline{p_4} \leftrightarrow \overline{p_2} p_3. \\\\
    /// \\end{aligned}
    /// \\end{equation}
    ///
    /// Particles are indicated by positive integers and the corresponding
    /// antiparticles are indicates by negative integers.
    ///
    /// If the process differs from its CP-conjugate process, the asymmetry can
    /// be specified with [`Interaction::asymmetry`].
    ///
    /// The squared amplitude is defined of the form
    ///
    /// \\begin{equation}
    ///   \abs{\mathcal{M}(p_1 p_2 \leftrightarrow p_3 p_4)}(s, t, u)
    /// \\end{equation}
    ///
    /// where \\(s\\), \\(t\\) and \\(u\\) are the usual Mandelstam variables.
    /// Crossing symmetry is then used in order to compute the other processes.
    ///
    /// The three-body decays are not currently computed.
    pub fn four_particle<F>(m2: F, p1: isize, p2: isize, p3: isize, p4: isize) -> Self
    where
        F: Fn(&M, f64, f64, f64) -> f64 + 'static,
    {
        let [u1, u2, u3, u4] = [
            usize::try_from(p1.abs()).unwrap(),
            usize::try_from(p2.abs()).unwrap(),
            usize::try_from(p3.abs()).unwrap(),
            usize::try_from(p4.abs()).unwrap(),
        ];
        Interaction::FourParticle {
            particles: [[u1, u2, u3, u4], [u1, u3, u2, u4], [u1, u4, u2, u3]],
            signed_particles: [[p1, p2, p3, p4], [p1, -p3, -p2, p4], [p1, -p4, -p2, p3]],
            antiparticles: [
                [
                    p1.signum() as f64,
                    p2.signum() as f64,
                    p3.signum() as f64,
                    p4.signum() as f64,
                ],
                [
                    p1.signum() as f64,
                    -p3.signum() as f64,
                    -p2.signum() as f64,
                    p4.signum() as f64,
                ],
                [
                    p1.signum() as f64,
                    -p4.signum() as f64,
                    -p2.signum() as f64,
                    p3.signum() as f64,
                ],
            ],
            m2: Box::new(m2),
            gamma: [
                RwLock::new(CubicHermiteSpline::empty()),
                RwLock::new(CubicHermiteSpline::empty()),
                RwLock::new(CubicHermiteSpline::empty()),
            ],
        }
    }

    /// Set the asymmetry between this interaction and its CP-conjugate.
    ///
    /// For three-body interactions, this is defined as:
    ///
    /// \\begin{equation}
    ///   \varepsilon \defeq \abs{\mathcal{M}(p_1 \leftrightarrow p_2 p_3)}^2
    ///     - \abs{\mathcal{M}(\overline{p_1} \leftrightarrow \overline{p_2} \overline{p_3})}^2.
    /// \\end{equation}
    ///
    /// If an asymmetry is defined, then the decay of the heavier particle will
    /// generate an asymmetry in the lighter two.
    ///
    /// Asymmetries for four-particle interactions are currently unsupported.
    pub fn asymmetry<F>(mut self, asym: F) -> Self
    where
        F: Fn(&M) -> f64 + 'static,
    {
        match &mut self {
            Interaction::ThreeParticle { asymmetry, .. } => *asymmetry = Some(Box::new(asym)),
            _ => {
                log::error!("asymmetry only applies to three-particle interactions");
                panic!("asymmetry only applies to three-particle interactions");
            }
        }

        self
    }

    /// Return the particles involved in this interaction.
    ///
    /// The resulting vector is a `Vec<Vec<isize>>` where the outer list are for
    /// the different possible interactions (calculated using crossing
    /// symmetry), and the inner list as the signed particle numbers involved in
    /// the specific interaction.
    pub fn particles(&self) -> Vec<Vec<isize>> {
        match self {
            Interaction::TwoParticle {
                signed_particles, ..
            } => vec![signed_particles.to_vec()],
            Interaction::ThreeParticle {
                signed_particles, ..
            } => signed_particles.iter().map(|ps| ps.to_vec()).collect(),
            Interaction::FourParticle {
                signed_particles, ..
            } => signed_particles.iter().map(|ps| ps.to_vec()).collect(),
        }
    }

    /// Analogue of [`Interaction::particles`], but instead of returning the
    /// signed particle numbers, return purely the particle index which does not
    /// distinguish between particles and anti-particles.
    pub fn particles_idx(&self) -> Vec<Vec<usize>> {
        match self {
            Interaction::TwoParticle { particles, .. } => vec![particles.to_vec()],
            Interaction::ThreeParticle { particles, .. } => {
                particles.iter().map(|ps| ps.to_vec()).collect()
            }
            Interaction::FourParticle { particles, .. } => {
                particles.iter().map(|ps| ps.to_vec()).collect()
            }
        }
    }

    /// Calculate the value(s) of `gamma`.
    pub fn gamma(&self, c: &Context<M>) -> Vec<f64> {
        let mut gammas = Vec::with_capacity(3);

        match self {
            Interaction::TwoParticle { .. } => {
                // let gamma = m2(&c.model).abs() / c.beta.powi(2) * c.normalization;
                // gammas.push(gamma);
                unimplemented!()
            }
            Interaction::ThreeParticle { particles, m2, .. } => {
                let ptcl = c.model.particles();
                let max_m = ptcl[particles[0][0]]
                    .mass
                    .max(ptcl[particles[0][1]].mass)
                    .max(ptcl[particles[0][2]].mass);

                // Iterate over the three possible configurations (though only 1
                // will be non-zero)
                for &[p0, _, _] in particles {
                    // TODO: What happens when the decaying particle's mass is
                    // not greater than the sum of the masses of the daughter
                    // particles?
                    #[allow(clippy::float_cmp)]
                    let decaying = ptcl[p0].mass == max_m;
                    if decaying {
                        // 0.002_423_011_225_182_300_4 = ζ(3) / 16 π³
                        let gamma_tilde = 0.002_423_011_225_182_300_4
                            * m2(&c.model).abs()
                            * bessel::k1_on_k2(ptcl[p0].mass * c.beta)
                            / c.beta.powi(3)
                            / ptcl[p0].mass
                            * c.normalization;

                        gammas.push(gamma_tilde);
                    } else {
                        gammas.push(0.0);
                    }
                }
            }
            Interaction::FourParticle {
                particles,
                m2,
                gamma,
                ..
            } => {
                let ptcl = c.model.particles();
                let mass2_sum: f64 = particles[0].iter().map(|&pi| ptcl[pi].mass2).sum();

                let ln_beta = c.beta.ln();

                // Iterator over the three possible configurations
                for i in 0..3 {
                    {
                        let spline = gamma[i].read().unwrap();
                        if spline.accurate(ln_beta) {
                            gammas.push(spline.sample(ln_beta).exp());
                            continue;
                        }
                    }

                    let [p0, p1, p2, p3] = particles[i];

                    let m2: Box<dyn Fn(&M, f64, f64, f64) -> f64> = match i {
                        0 => Box::new(|c, s, t, u| m2(c, s, t, u)),
                        1 => Box::new(|c, s, t, u| m2(c, t, s, u)),
                        2 => Box::new(|c, s, t, u| m2(c, u, t, s)),
                        _ => unreachable!(),
                    };

                    let value = integrate_st(
                        |s, t| {
                            let u = mass2_sum - s - t;
                            m2(&c.model, s, t, u)
                        },
                        c.beta,
                        ptcl[p0].mass2,
                        ptcl[p1].mass2,
                        ptcl[p2].mass2,
                        ptcl[p3].mass2,
                    ) * c.normalization;
                    // if value < 0.0 {
                    //     log::error!("value: {:e}", value);
                    // } else {
                    //     log::warn!("value: {:e}", value);
                    // }
                    let value = value.abs();
                    gammas.push(value);

                    let mut spline = gamma[i].write().unwrap();
                    spline.add(ln_beta, value.ln());
                }
            }
        }

        gammas
    }

    /// Calculate the decay width associated with a particular interaction.
    ///
    /// This returns two values:
    ///
    /// - a vector of particles involved, with the first particle being the one
    ///   decaying and the remaining being the daughter particles; and
    /// - the width of this process in GeV.
    pub fn width(&self, c: &Context<M>) -> (Vec<isize>, f64) {
        let mut daughters = Vec::new();
        let mut width = 0.0;

        match self {
            Interaction::TwoParticle { .. } => (),
            Interaction::ThreeParticle {
                particles,
                signed_particles,
                m2,
                ..
            } => {
                let ptcl = c.model.particles();
                let max_m = ptcl[particles[0][0]]
                    .mass
                    .max(ptcl[particles[0][1]].mass)
                    .max(ptcl[particles[0][2]].mass);

                // Iterate over the three possible configurations (though only 1
                // will be non-zero)
                for i in 0..3 {
                    let [p0, p1, p2] = particles[i];
                    let [s0, s1, s2] = signed_particles[i];

                    #[allow(clippy::float_cmp)]
                    let heaviest = ptcl[p0].mass == max_m;
                    if !heaviest || ptcl[p0].mass < ptcl[p1].mass + ptcl[p2].mass {
                        continue;
                    }

                    let p = kallen_lambda(ptcl[p0].mass2, ptcl[p1].mass2, ptcl[p2].mass2).sqrt()
                        / (2.0 * ptcl[p0].mass);

                    // 1 / 8 π ≅ 0.039788735772973836
                    width = 0.039_788_735_772_973_836 * p / ptcl[p0].mass2 * m2(c.model).abs();
                    daughters.push(s0);
                    daughters.push(s1);
                    daughters.push(s2);
                }
            }
            Interaction::FourParticle {
                // particles,
                // signed_particles,
                // m2,
                ..
            } => {
                unimplemented!()
            }
        }

        (daughters, width)
    }

    /// Calculate the interaction rates.
    ///
    /// This returns a vector of
    /// ```
    /// [(rate_forward, rate_backward), (arate_forward, arate_backward)]
    /// ```(rate_forward, rate_backward,
    /// arate_forward, arate_backward)`, where `arate` is the rate of the
    /// asymmetries.  For three-particle interactions, the result is a vector of
    /// length 3 with only 1 being non-zero, while four-body interactions will
    /// produce an array of length 3.
    fn calculate_rate(&self, c: &Context<M>) -> Vec<[(f64, f64); 2]> {
        let mut rates = Vec::with_capacity(3);
        let gamma = self.gamma(c);

        match self {
            Interaction::TwoParticle {
                particles,
                antiparticles,
                ..
            } => {
                let &[p0, p1] = particles;
                let &[a0, a1] = antiparticles;
                let gamma = gamma[0] * c.step_size;

                let (rate_forward, rate_backward) = if p0 != p1 {
                    (
                        gamma * nan_to_zero(c.n[p0] / c.eq[p0]),
                        gamma * nan_to_zero(c.n[p1] / c.eq[p1]),
                    )
                } else {
                    (0.0, 0.0)
                };

                let arate_forward = gamma * nan_to_zero(a0 * c.na[p0] / c.eq[p0]);
                let arate_backward = gamma * nan_to_zero(a1 * c.na[p1] / c.eq[p1]);

                rates.push([
                    (rate_forward, rate_backward),
                    (arate_forward, arate_backward),
                ]);
            }
            Interaction::ThreeParticle {
                particles,
                antiparticles,
                ..
            } => {
                let ptcl = c.model.particles();
                let max_m = ptcl[particles[0][0]]
                    .mass
                    .max(ptcl[particles[0][1]].mass)
                    .max(ptcl[particles[0][2]].mass);

                // Iterate over the three possible configurations (though only 1
                // will be non-zero)
                for i in 0..3 {
                    let [p0, p1, p2] = particles[i];
                    let [a0, a1, a2] = antiparticles[i];

                    #[allow(clippy::float_cmp)]
                    let decaying = ptcl[p0].mass == max_m;
                    if decaying {
                        let gamma_tilde = gamma[i] * c.step_size;

                        let rate_forward = gamma_tilde * c.n[p0];
                        let rate_backward = gamma_tilde
                            * c.eq[p0]
                            * nan_to_zero((c.n[p1] / c.eq[p1]) * (c.n[p2] / c.eq[p2]));

                        let arate_forward = gamma_tilde * a0 * c.na[p0];
                        let arate_backward = gamma_tilde
                            * c.eq[p0]
                            * nan_to_zero(
                                (c.n[p1] * a2 * c.na[p2] + c.n[p2] * a1 * c.na[p1]
                                    - a1 * a2 * c.na[p1] * c.na[p2])
                                    / (c.eq[p1] * c.eq[p2]),
                            );

                        rates.push([
                            (rate_forward, rate_backward),
                            (arate_forward, arate_backward),
                        ]);
                    } else {
                        rates.push([(0.0, 0.0), (0.0, 0.0)]);
                    }
                }
            }
            Interaction::FourParticle {
                particles,
                antiparticles,
                ..
            } => {
                // Iterator over the three possible configurations
                for i in 0..3 {
                    let [p0, p1, p2, p3] = particles[i];
                    let [a0, a1, a2, a3] = antiparticles[i];
                    let gamma = gamma[i] * c.step_size;

                    // A NaN can occur from `0.0 / 0.0`, in which case the
                    // correct value ought to be 0.
                    let forward = nan_to_zero((c.n[p0] / c.eq[p0]) * (c.n[p1] / c.eq[p1]));
                    let backward = nan_to_zero((c.n[p2] / c.eq[p2]) * (c.n[p3] / c.eq[p3]));
                    let aforward = nan_to_zero(
                        (c.n[p0] * a1 * c.na[p1] + c.n[p1] * a0 * c.na[p0]
                            - a0 * a1 * c.na[p0] * c.na[p1])
                            / (c.eq[p0] * c.eq[p1]),
                    );
                    let abackward = nan_to_zero(
                        (c.n[p2] * a3 * c.na[p3] + c.n[p3] * a2 * c.na[p2]
                            - a2 * a3 * c.na[p2] * c.na[p3])
                            / (c.eq[p2] * c.eq[p3]),
                    );

                    rates.push([
                        (forward * gamma, backward * gamma),
                        (aforward * gamma, abackward * gamma),
                    ]);
                }
            }
        }

        self.adjust_rate_overshoot(c, rates)
    }

    /// Adjust the backward and forward rates such that they do not
    /// overshoot the equilibrium number densities.
    fn adjust_rate_overshoot(
        &self,
        c: &Context<M>,
        mut rates: Vec<[(f64, f64); 2]>,
    ) -> Vec<[(f64, f64); 2]> {
        // If an overshoot of the interaction rate is detected, the rate is
        // adjusted such that `dn` satisfies:
        //
        // ```text
        // n + dn = (eq + (ALPHA - 1) * n) / ALPHA
        // ```
        //
        // Large values means that the correction towards equilibrium are
        // weaker, while smaller values make the move towards equilibrium
        // stronger.  Values of ALPHA less than 1 will overshoot the
        // equilibrium.
        const ALPHA_N: f64 = 1.1;
        const ALPHA_NA: f64 = 1.1;

        match self {
            Interaction::TwoParticle {
                particles,
                antiparticles,
                ..
            } => {
                let &[p0, p1] = particles;
                let &[a0, a1] = antiparticles;
                let [rate, arate] = &mut rates[0];

                if p0 != p1 {
                    let mut net_rate = rate.0 - rate.1;
                    if overshoots(c, p0, -net_rate) {
                        rate.0 = (c.n[p0] - c.eq[p0]) / ALPHA_N + rate.1;
                        net_rate = rate.0 - rate.1;
                    }
                    if overshoots(c, p1, net_rate) {
                        rate.1 = (c.n[p1] - c.eq[p1]) / ALPHA_N + rate.0;
                    }

                    let mut net_arate = arate.0 - arate.1;
                    if asymmetry_overshoots(c, p0, -a0 * net_arate) {
                        arate.0 = a0 * c.na[p0] / ALPHA_NA + arate.1;
                        net_arate = arate.0 - arate.1;
                    }
                    if asymmetry_overshoots(c, p1, a1 * net_arate) {
                        arate.1 = a1 * c.na[p1] / ALPHA_NA + arate.0;
                    }
                } else {
                    // When p0 == p1, if either the forward or backward rate go
                    // to infinity, the net rate will be calculated as NaN
                    let net_arate = arate.0 - arate.1;
                    if net_arate.is_nan() || asymmetry_overshoots(c, p0, -a0 * net_arate) {
                        arate.0 = a0 * c.na[p0] / 2.0;
                        arate.1 = 0.0;
                    }
                }
            }
            Interaction::ThreeParticle {
                particles,
                antiparticles,
                ..
            } => {
                for i in 0..3 {
                    let [p0, p1, p2] = particles[i];
                    let [a0, a1, a2] = antiparticles[i];
                    let [rate, arate] = &mut rates[i];

                    let mut net_rate = rate.0 - rate.1;
                    if overshoots(c, p0, -net_rate) {
                        rate.0 = (c.n[p0] - c.eq[p0]) / ALPHA_N + rate.1;
                        net_rate = rate.0 - rate.1;
                    }
                    if overshoots(c, p1, net_rate) {
                        rate.1 = (c.n[p1] - c.eq[p1]) / ALPHA_N + rate.0;
                        net_rate = rate.0 - rate.1;
                    }
                    if overshoots(c, p2, net_rate) {
                        rate.1 = (c.n[p2] - c.eq[p2]) / ALPHA_N + rate.0;
                    }

                    let mut net_arate = arate.0 - arate.1;
                    if asymmetry_overshoots(c, p0, -a0 * net_arate) {
                        arate.0 = a0 * c.na[p0] / ALPHA_NA + arate.1;
                        net_arate = arate.0 - arate.1;
                    }
                    if asymmetry_overshoots(c, p1, a1 * net_arate) {
                        arate.1 = a1 * c.na[p1] / ALPHA_NA + arate.0;
                        net_arate = arate.0 - arate.1;
                    }
                    if asymmetry_overshoots(c, p2, a2 * net_arate) {
                        arate.1 = a2 * c.na[p2] / ALPHA_NA + arate.0;
                    }
                }
            }
            Interaction::FourParticle {
                particles,
                antiparticles,
                ..
            } => {
                for i in 0..3 {
                    let [p0, p1, p2, p3] = particles[i];
                    let [a0, a1, a2, a3] = antiparticles[i];
                    let [rate, arate] = &mut rates[i];

                    let mut net_rate = rate.0 - rate.1;
                    if overshoots(c, p0, -net_rate) {
                        rate.0 = (c.n[p0] - c.eq[p0]) / ALPHA_N + rate.1;
                        net_rate = rate.0 - rate.1;
                    }
                    if overshoots(c, p0, -net_rate) {
                        rate.0 = (c.n[p0] - c.eq[p1]) / ALPHA_N + rate.1;
                        net_rate = rate.0 - rate.1;
                    }
                    if overshoots(c, p1, net_rate) {
                        rate.1 = (c.n[p2] - c.eq[p2]) / ALPHA_N + rate.0;
                        net_rate = rate.0 - rate.1;
                    }
                    if overshoots(c, p2, net_rate) {
                        rate.1 = (c.n[p3] - c.eq[p3]) / ALPHA_N + rate.0;
                    }

                    let mut net_arate = arate.0 - arate.1;
                    if asymmetry_overshoots(c, p0, -a0 * net_arate) {
                        arate.0 = c.na[p0] / ALPHA_NA + arate.1;
                        net_arate = arate.0 - arate.1;
                    }
                    if asymmetry_overshoots(c, p1, -a1 * net_arate) {
                        arate.0 = c.na[p1] / ALPHA_NA + arate.1;
                        net_arate = arate.0 - arate.1;
                    }
                    if asymmetry_overshoots(c, p2, a2 * net_arate) {
                        arate.1 = c.na[p2] / ALPHA_NA + arate.0;
                        net_arate = arate.0 - arate.1;
                    }
                    if asymmetry_overshoots(c, p3, a3 * net_arate) {
                        arate.1 = c.na[p3] / ALPHA_NA + arate.0;
                    }
                }
            }
        }

        rates
    }

    /// Add this interaction to the `dn` array.
    ///
    /// The changes in `dn` should contain only the effect from the integrated
    /// collision operator and not take into account the normalization to
    /// inverse-temperature evolution nor the dilation factor from expansion of
    /// the Universe.  These factors are handled separately and automatically.
    ///
    /// This function automatically adjusts the rate so that overshooting is
    /// avoided.
    pub fn change(
        &self,
        mut dn: Array1<f64>,
        mut dna: Array1<f64>,
        c: &Context<M>,
    ) -> (Array1<f64>, Array1<f64>) {
        let rates = self.calculate_rate(c);

        match self {
            Interaction::TwoParticle {
                particles,
                antiparticles,
                ..
            } => {
                let &[p0, p1] = particles;
                let &[a0, a1] = antiparticles;
                let [rate, arate] = rates[0];
                let net_rate = rate.0;
                let net_arate = arate.0 - arate.1;
                log::trace!("γ({} ↔ {}) = {:<10.3e}", p0, p1, net_rate);
                log::trace!("γ'({} ↔ {}) = {:<10.3e}", p0, p1, net_arate);

                dn[p0] -= net_rate;
                dn[p1] += net_rate;

                dna[p0] -= a0 * net_arate;
                dna[p1] += a1 * net_arate;
            }
            Interaction::ThreeParticle {
                particles,
                antiparticles,
                asymmetry,
                ..
            } => {
                for i in 0..3 {
                    let [p0, p1, p2] = particles[i];
                    let [a0, a1, a2] = antiparticles[i];
                    let [rate, arate] = rates[i];
                    let net_rate = rate.0 - rate.1;
                    let net_arate = arate.0 - arate.1;
                    log::trace!("γ({} ↔ {}, {}) = {:<10.3e}", p0, p1, p2, net_rate);
                    log::trace!("γ'({} ↔ {}, {}) = {:<10.3e}", p0, p1, p2, net_arate);

                    dn[p0] -= net_rate;
                    dn[p1] += net_rate;
                    dn[p2] += net_rate;

                    dna[p0] -= a0 * net_arate;
                    dna[p1] += a1 * net_arate;
                    dna[p2] += a2 * net_arate;

                    if let Some(asymmetry) = asymmetry {
                        let source = net_rate * asymmetry(&c.model);
                        dna[p1] += a1 * source;
                        dna[p2] += a2 * source;
                    }
                }
            }
            Interaction::FourParticle {
                particles,
                antiparticles,
                ..
            } => {
                for i in 0..3 {
                    let [p0, p1, p2, p3] = particles[i];
                    let [a0, a1, a2, a3] = antiparticles[i];
                    let [rate, arate] = rates[i];
                    let net_rate = rate.0 - rate.1;
                    let net_arate = arate.0 - arate.1;
                    log::trace!("γ({}, {} ↔ {}, {}) = {:<10.3e}", p0, p1, p2, p3, net_rate);
                    log::trace!("γ'({}, {} ↔ {}, {}) = {:<10.3e}", p0, p1, p2, p3, net_arate);

                    dn[p0] -= net_rate;
                    dn[p1] -= net_rate;
                    dn[p2] += net_rate;
                    dn[p3] += net_rate;

                    dna[p0] -= a0 * net_arate;
                    dna[p1] -= a1 * net_arate;
                    dna[p2] += a2 * net_arate;
                    dna[p3] += a3 * net_arate;
                }
            }
        }

        (dn, dna)
    }
}

/// Check whether particle `i` from the model with the given rate change will
/// overshoot equilibrium.
fn overshoots<M: Model>(c: &Context<M>, i: usize, rate: f64) -> bool {
    (c.n[i] > c.eq[i] && c.n[i] + rate < c.eq[i]) || (c.n[i] < c.eq[i] && c.n[i] + rate > c.eq[i])
}

/// Check whether particle asymmetry `i` from the model with the given rate
/// change will overshoot 0.
fn asymmetry_overshoots<M: Model>(c: &Context<M>, i: usize, rate: f64) -> bool {
    (c.na[i] > 0.0 && c.na[i] + rate < 0.0) || (c.na[i] < 0.0 && c.na[i] + rate > 0.0)
}

fn nan_to_zero(v: f64) -> f64 {
    if v.is_nan() {
        0.0
    } else {
        v
    }
}

unsafe impl<M: Model> Sync for Interaction<M> {}
unsafe impl<M: Model> Send for Interaction<M> {}
