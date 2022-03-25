pub mod data;

use crate::model::{
    particle::{LEFT_WEYL_SPINOR, SCALAR, TENSOR},
    Model, ParticleData,
};
use ndarray::{array, prelude::*};
use num::{Complex, One, Zero};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
// use special_functions::particle_physics::pave_absorptive;
use std::{f64, f64::consts::SQRT_2};

/// The Standard Model of particle physics.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Standard {
    /// Inverse temperature in GeV`$^{-1}$`
    pub beta: f64,

    /// Particles
    pub particles: Vec<ParticleData>,

    // Gauge couplings
    /// Hypercharge gauge coupling
    pub g1: f64,
    /// Weak gauge coupling
    pub g2: f64,
    /// Strong gauge coupling
    pub g3: f64,

    // Yukawa couplings
    /// Up-quark Yukawa
    pub yu: Array2<f64>,
    /// Down-quark Yukawa
    pub yd: Array2<f64>,
    /// Electron Yukawa
    pub ye: Array2<f64>,
    /// CKM matrix
    pub ckm: Array2<Complex<f64>>,
    /// PMNS matrix
    pub pmns: Array2<Complex<f64>>,

    // Scalar potential
    /// 0-temperature mass of the Higgs
    pub mh: f64,
    /// Vacuum expectation value of the Higgs
    pub vev: f64,
    /// Quadratic coupling of the Higgs
    pub mu2: f64,
    /// Quartic term in scalar potential
    pub lambda: f64,
}

impl Model for Standard {
    fn zero() -> Self {
        let particles = vec![
            ParticleData::new(SCALAR, 0.0, 0.0)
                .name("none")
                .dof(0.0)
                .own_antiparticle(), // [0] dummy particle so particles start at index `1`
            ParticleData::new(TENSOR, 0.0, 0.0)
                .name("A")
                .own_antiparticle(), // [1] Hypercharge gauge boson
            ParticleData::new(TENSOR, 0.0, 0.0)
                .name("W")
                .dof(3.0)
                .own_antiparticle(), // [2] SU(2) gauge boson
            ParticleData::new(TENSOR, 0.0, 0.0)
                .name("G")
                .dof(8.0)
                .own_antiparticle(), // [3] SU(3) gauge boson
            ParticleData::new(SCALAR, 0.0, 0.0).name("H").dof(2.0 * 2.0), // [4] Higgs
            ParticleData::new(LEFT_WEYL_SPINOR, 0.0, 0.0)
                .name("L1")
                .dof(2.0), // [5]
            ParticleData::new(LEFT_WEYL_SPINOR, 0.0, 0.0)
                .name("L2")
                .dof(2.0), // [6]
            ParticleData::new(LEFT_WEYL_SPINOR, 0.0, 0.0)
                .name("L3")
                .dof(2.0), // [7]
            ParticleData::new(LEFT_WEYL_SPINOR, 0.0, 0.0)
                .name("e1")
                .dof(1.0), // [8]
            ParticleData::new(LEFT_WEYL_SPINOR, 0.0, 0.0)
                .name("e2")
                .dof(1.0), // [9]
            ParticleData::new(LEFT_WEYL_SPINOR, 0.0, 0.0)
                .name("e3")
                .dof(1.0), // [10]
            ParticleData::new(LEFT_WEYL_SPINOR, 0.0, 0.0)
                .name("Q1")
                .dof(2.0 * 3.0), // [11]
            ParticleData::new(LEFT_WEYL_SPINOR, 0.0, 0.0)
                .name("Q2")
                .dof(2.0 * 3.0), // [12]
            ParticleData::new(LEFT_WEYL_SPINOR, 0.0, 0.0)
                .name("Q3")
                .dof(2.0 * 3.0), // [13]
            ParticleData::new(LEFT_WEYL_SPINOR, 0.0, 0.0)
                .name("u1")
                .dof(3.0), // [14]
            ParticleData::new(LEFT_WEYL_SPINOR, 0.0, 0.0)
                .name("u2")
                .dof(3.0), // [15]
            ParticleData::new(LEFT_WEYL_SPINOR, 0.0, 0.0)
                .name("u3")
                .dof(3.0), // [16]
            ParticleData::new(LEFT_WEYL_SPINOR, 0.0, 0.0)
                .name("d1")
                .dof(3.0), // [17]
            ParticleData::new(LEFT_WEYL_SPINOR, 0.0, 0.0)
                .name("d2")
                .dof(3.0), // [18]
            ParticleData::new(LEFT_WEYL_SPINOR, 0.0, 0.0)
                .name("d3")
                .dof(3.0), // [19]
        ];

        let mh: f64 = data::MASS_H;
        let vev: f64 = data::VEV;
        let mu2 = -mh.powi(2) / 2.0;
        let lambda = (mu2 / (2.0 * vev)).powi(2);

        Standard {
            beta: f64::INFINITY,
            g1: 3.585e-01,
            g2: 6.476e-01,
            g3: 1.164e+00,
            ckm: generate_ckm(),
            pmns: generate_pmns(),
            yu: SQRT_2 / data::VEV
                * array![
                    [data::MASS_UP, 0.0, 0.0],
                    [0.0, data::MASS_CH, 0.0],
                    [0.0, 0.0, data::MASS_TO],
                ],
            yd: SQRT_2 / data::VEV
                * array![
                    [data::MASS_DO, 0.0, 0.0],
                    [0.0, data::MASS_ST, 0.0],
                    [0.0, 0.0, data::MASS_BO],
                ],
            ye: SQRT_2 / data::VEV
                * array![
                    [data::MASS_EL, 0.0, 0.0],
                    [0.0, data::MASS_MU, 0.0],
                    [0.0, 0.0, data::MASS_TA],
                ],
            mh,
            vev,
            mu2,
            lambda,
            particles,
        }
    }

    /// Set inverse temperature for the model.
    ///
    /// This does **not** compute the widths as it is assumed that most
    /// implementation will add more particles, thus making any Standard Model
    /// specific computation redundant.
    ///
    /// This does update the thermal masses of all Standard Model particles
    /// given the Standard Model interactions.  If other interactions are
    /// included, the contributions of these interactions should also be
    /// included.
    fn set_beta(&mut self, beta: f64) {
        debug_assert!(
            beta.is_finite(),
            "The inverse temperature should always be finite.\
             For the zero-temperature case, use Model::zero."
        );
        self.beta = beta;

        // Update the values of the gauge coupling for the running
        self.g1 = data::G1_RUNNING.sample(beta);
        self.g2 = data::G2_RUNNING.sample(beta);
        self.g3 = data::G3_RUNNING.sample(beta);

        // Precompute the squared couplings to be used in the thermal masses
        let g1 = self.g1.powi(2) / 8.0;
        let g2 = self.g2.powi(2) / 8.0;
        let g3 = self.g3.powi(2) / 8.0;
        let yu = self.yu.dot(&self.yu).into_diag() / 16.0;
        let yd = self.yd.dot(&self.yd).into_diag() / 16.0;
        let ye = self.ye.dot(&self.ye).into_diag() / 16.0;
        let mu2 = self.mu2;
        let lambda = self.lambda / 4.0;

        // Update the thermal masses
        self.particle_mut("H", 0).set_mass(
            std::f64::consts::SQRT_2
                * f64::sqrt(
                    -mu2 + lambda
                        + g1 / 4.0
                        + (3.0 / 4.0) * g2
                        + 2.0 * yu.sum()
                        + 2.0 * yd.sum()
                        + 2.0 * ye.sum(),
                )
                / beta,
        );
        self.particle_mut("A", 0)
            .set_mass(f64::sqrt((22.0 / 3.0) * g1) / beta);
        self.particle_mut("W", 0)
            .set_mass(f64::sqrt((22.0 / 3.0) * g2) / beta);

        for i in 0..3 {
            self.particle_mut("L", i)
                .set_mass(f64::sqrt(g1 / 4.0 + (3.0 / 4.0) * g2 + ye[i]) / beta);
            self.particle_mut("e", i)
                .set_mass(f64::sqrt(g1 + ye[i]) / beta);
            self.particle_mut("Q", i)
                .set_mass(f64::sqrt(g1 / 36.0 + (3.0 / 4.0) * g2 + g3 + yd[i] + yu[i]) / beta);
            self.particle_mut("u", i)
                .set_mass(f64::sqrt(g1 * 4.0 / 9.0 + g3 + yu[i]) / beta);
            self.particle_mut("d", i)
                .set_mass(f64::sqrt(g1 / 9.0 + g3 + yd[i]) / beta);
        }
    }

    fn get_beta(&self) -> f64 {
        self.beta
    }

    fn entropy_dof(&self, beta: f64) -> f64 {
        debug_assert!(beta > 0.0, "beta must be positive.");
        data::STANDARD_MODEL_GSTAR.sample(f64::ln(beta))
    }

    fn particles(&self) -> &[ParticleData] {
        &self.particles
    }

    fn particles_mut(&mut self) -> &mut [ParticleData] {
        &mut self.particles
    }

    fn static_particle_idx<S: AsRef<str>>(name: S, i: usize) -> Result<usize, (S, usize)> {
        match (name.as_ref(), i) {
            ("A", _) => Ok(1),
            ("W", _) => Ok(2),
            ("G", _) => Ok(3),
            ("H", _) => Ok(4),
            ("L", i) if i < 3 => Ok(5 + i),
            ("e", i) if i < 3 => Ok(8 + i),
            ("Q", i) if i < 3 => Ok(11 + i),
            ("u", i) if i < 3 => Ok(14 + i),
            ("d", i) if i < 3 => Ok(17 + i),
            (_, i) => Err((name, i)),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn self_energy_absorptive(&self, p: &ParticleData, momentum: f64) -> f64 {
        use crate::constants::PI_2;
        use special_functions::particle_physics::pave_absorptive::b;

        let ptcls = self.particles();
        let pa = &ptcls[1];
        let pw = &ptcls[2];
        let pg = &ptcls[3];
        let ph = &ptcls[4];
        let pl = &ptcls[5..8];
        let pe = &ptcls[8..11];
        let pq = &ptcls[11..14];
        let pu = &ptcls[14..17];
        let pd = &ptcls[17..20];

        #[allow(clippy::match_same_arms)]
        match p.name.as_str() {
            "A" => 0.0,
            "W" => 0.0,
            "G" => 0.0,
            "H" => {
                (64.0 * PI_2).recip()
                    * (-4.0
                        * (0..3)
                            .map(|i| {
                                (0..3)
                                    .map(|j| {
                                        3.0 * self.yd[[j, i]].powi(2)
                                            * (-momentum + pd[j].mass2 + pq[i].mass2)
                                            * b(0, 0, momentum, pq[i].mass, pd[j].mass)
                                            + 3.0
                                                * self.yu[[j, i]].powi(2)
                                                * (-momentum + pu[j].mass2 + pq[i].mass2)
                                                * b(0, 0, momentum, pu[i].mass, pd[j].mass)
                                            + self.ye[[j, i]].powi(2)
                                                * (-momentum + pe[j].mass2 + pl[i].mass2)
                                                * b(0, 0, momentum, pl[i].mass, pe[j].mass)
                                    })
                                    .sum::<f64>()
                            })
                            .sum::<f64>()
                        + self.g1.powi(2)
                            * (-2.0 * momentum + pa.mass2 - 2.0 * ph.mass2)
                            * b(0, 0, momentum, ph.mass, pa.mass)
                        + 3.0
                            * self.g2.powi(2)
                            * (-2.0 * momentum + pw.mass2 - 2.0 * ph.mass2)
                            * b(0, 0, momentum, ph.mass, pw.mass))
            }
            "L1" => {
                let i = 0;
                -(32.0 * PI_2).recip()
                    * (self.g1.powi(2) * b(0, 1, momentum, pl[i].mass, pa.mass)
                        + 3.0 * self.g2.powi(2) * b(0, 1, momentum, pl[i].mass, pw.mass)
                        + 2.0
                            * (0..3)
                                .map(|j| {
                                    self.ye[[j, i]].powi(2) * b(0, 1, momentum, pe[j].mass, ph.mass)
                                })
                                .sum::<f64>())
            }
            "L2" => {
                let i = 1;
                -(32.0 * PI_2).recip()
                    * (self.g1.powi(2) * b(0, 1, momentum, pl[i].mass, pa.mass)
                        + 3.0 * self.g2.powi(2) * b(0, 1, momentum, pl[i].mass, pw.mass)
                        + 2.0
                            * (0..3)
                                .map(|j| {
                                    self.ye[[j, i]].powi(2) * b(0, 1, momentum, pe[j].mass, ph.mass)
                                })
                                .sum::<f64>())
            }
            "L3" => {
                let i = 2;
                -(32.0 * PI_2).recip()
                    * (self.g1.powi(2) * b(0, 1, momentum, pl[i].mass, pa.mass)
                        + 3.0 * self.g2.powi(2) * b(0, 1, momentum, pl[i].mass, pw.mass)
                        + 2.0
                            * (0..3)
                                .map(|j| {
                                    self.ye[[j, i]].powi(2) * b(0, 1, momentum, pe[j].mass, ph.mass)
                                })
                                .sum::<f64>())
            }
            "e1" => {
                let i = 0;
                -(8.0 * PI_2).recip()
                    * (self.g1.powi(2) * b(0, 1, momentum, pl[i].mass, pa.mass)
                        + (0..3)
                            .map(|j| {
                                self.ye[[i, j]].powi(2) * b(0, 1, momentum, pl[j].mass, ph.mass)
                            })
                            .sum::<f64>())
            }
            "e2" => {
                let i = 1;
                -(8.0 * PI_2).recip()
                    * (self.g1.powi(2) * b(0, 1, momentum, pl[i].mass, pa.mass)
                        + (0..3)
                            .map(|j| {
                                self.ye[[i, j]].powi(2) * b(0, 1, momentum, pl[j].mass, ph.mass)
                            })
                            .sum::<f64>())
            }
            "e3" => {
                let i = 2;
                -(8.0 * PI_2).recip()
                    * (self.g1.powi(2) * b(0, 1, momentum, pl[i].mass, pa.mass)
                        + (0..3)
                            .map(|j| {
                                self.ye[[i, j]].powi(2) * b(0, 1, momentum, pl[j].mass, ph.mass)
                            })
                            .sum::<f64>())
            }
            "Q1" => {
                let i = 0;
                -(96.0 * PI_2).recip()
                    * (self.g1.powi(2) * b(0, 1, momentum, pq[i].mass, pa.mass)
                        + 27.0 * self.g2.powi(2) * b(0, 1, momentum, pq[i].mass, pw.mass)
                        + 48.0 * self.g3.powi(2) * b(0, 1, momentum, pq[i].mass, pg.mass)
                        + 18.0
                            * (0..3)
                                .map(|j| {
                                    self.yd[[j, i]].powi(2) * b(0, 1, momentum, pd[j].mass, ph.mass)
                                })
                                .sum::<f64>()
                        + 18.0
                            * (0..3)
                                .map(|j| {
                                    self.yu[[j, i]].powi(2) * b(0, 1, momentum, pu[j].mass, ph.mass)
                                })
                                .sum::<f64>())
            }
            "Q2" => {
                let i = 1;
                -(96.0 * PI_2).recip()
                    * (self.g1.powi(2) * b(0, 1, momentum, pq[i].mass, pa.mass)
                        + 27.0 * self.g2.powi(2) * b(0, 1, momentum, pq[i].mass, pw.mass)
                        + 48.0 * self.g3.powi(2) * b(0, 1, momentum, pq[i].mass, pg.mass)
                        + 18.0
                            * (0..3)
                                .map(|j| {
                                    self.yd[[j, i]].powi(2) * b(0, 1, momentum, pd[j].mass, ph.mass)
                                })
                                .sum::<f64>()
                        + 18.0
                            * (0..3)
                                .map(|j| {
                                    self.yu[[j, i]].powi(2) * b(0, 1, momentum, pu[j].mass, ph.mass)
                                })
                                .sum::<f64>())
            }
            "Q3" => {
                let i = 2;
                -(96.0 * PI_2).recip()
                    * (self.g1.powi(2) * b(0, 1, momentum, pq[i].mass, pa.mass)
                        + 27.0 * self.g2.powi(2) * b(0, 1, momentum, pq[i].mass, pw.mass)
                        + 48.0 * self.g3.powi(2) * b(0, 1, momentum, pq[i].mass, pg.mass)
                        + 18.0
                            * (0..3)
                                .map(|j| {
                                    self.yd[[j, i]].powi(2) * b(0, 1, momentum, pd[j].mass, ph.mass)
                                })
                                .sum::<f64>()
                        + 18.0
                            * (0..3)
                                .map(|j| {
                                    self.yu[[j, i]].powi(2) * b(0, 1, momentum, pu[j].mass, ph.mass)
                                })
                                .sum::<f64>())
            }
            "u1" => {
                let i = 0;
                -(24.0 * PI_2).recip()
                    * (4.0 * self.g1.powi(2) * b(0, 1, momentum, pu[i].mass, pa.mass)
                        + 12.0 * self.g3.powi(2) * b(0, 1, momentum, pu[i].mass, pg.mass)
                        + 9.0
                            * (0..3)
                                .map(|j| {
                                    self.yu[[i, j]].powi(2) * b(0, 1, momentum, pq[j].mass, ph.mass)
                                })
                                .sum::<f64>())
            }
            "u2" => {
                let i = 1;
                -(24.0 * PI_2).recip()
                    * (4.0 * self.g1.powi(2) * b(0, 1, momentum, pu[i].mass, pa.mass)
                        + 12.0 * self.g3.powi(2) * b(0, 1, momentum, pu[i].mass, pg.mass)
                        + 9.0
                            * (0..3)
                                .map(|j| {
                                    self.yu[[i, j]].powi(2) * b(0, 1, momentum, pq[j].mass, ph.mass)
                                })
                                .sum::<f64>())
            }
            "u3" => {
                let i = 2;
                -(24.0 * PI_2).recip()
                    * (4.0 * self.g1.powi(2) * b(0, 1, momentum, pu[i].mass, pa.mass)
                        + 12.0 * self.g3.powi(2) * b(0, 1, momentum, pu[i].mass, pg.mass)
                        + 9.0
                            * (0..3)
                                .map(|j| {
                                    self.yu[[i, j]].powi(2) * b(0, 1, momentum, pq[j].mass, ph.mass)
                                })
                                .sum::<f64>())
            }
            "d1" => {
                let i = 0;
                -(24.0 * PI_2).recip()
                    * (self.g1.powi(2) * b(0, 1, momentum, pd[i].mass, pa.mass)
                        + 12.0 * self.g3.powi(2) * b(0, 1, momentum, pd[i].mass, pg.mass)
                        + 9.0
                            * (0..3)
                                .map(|j| {
                                    self.yd[[i, j]].powi(2) * b(0, 1, momentum, pq[j].mass, ph.mass)
                                })
                                .sum::<f64>())
            }
            "d2" => {
                let i = 1;
                -(24.0 * PI_2).recip()
                    * (self.g1.powi(2) * b(0, 1, momentum, pd[i].mass, pa.mass)
                        + 12.0 * self.g3.powi(2) * b(0, 1, momentum, pd[i].mass, pg.mass)
                        + 9.0
                            * (0..3)
                                .map(|j| {
                                    self.yd[[i, j]].powi(2) * b(0, 1, momentum, pq[j].mass, ph.mass)
                                })
                                .sum::<f64>())
            }
            "d3" => {
                let i = 2;
                -(24.0 * PI_2).recip()
                    * (self.g1.powi(2) * b(0, 1, momentum, pd[i].mass, pa.mass)
                        + 12.0 * self.g3.powi(2) * b(0, 1, momentum, pd[i].mass, pg.mass)
                        + 9.0
                            * (0..3)
                                .map(|j| {
                                    self.yd[[i, j]].powi(2) * b(0, 1, momentum, pq[j].mass, ph.mass)
                                })
                                .sum::<f64>())
            }
            _ => 0.0,
        }
    }
}

fn generate_ckm() -> Array2<Complex<f64>> {
    use crate::model::standard::data::{CKM_A, CKM_ETA, CKM_LAMBDA, CKM_RHO};

    let a_2 = CKM_A.powi(2);
    let lambda_2 = CKM_LAMBDA.powi(2);
    let rho_eta = Complex::new(CKM_RHO, CKM_ETA);

    // z = ρ + i η
    let z = f64::sqrt((1.0 - a_2 * CKM_LAMBDA.powi(4)) / (1.0 - lambda_2))
        * rho_eta
        * (1.0 - CKM_A.powi(2) * CKM_LAMBDA.powi(4) * rho_eta).finv();

    array![
        [
            Complex::new(
                1.0 + lambda_2
                    * (lambda_2 * (lambda_2 * (-5.0 / 128.0 * lambda_2 - 1.0 / 16.0) - 1.0 / 8.0)
                        - 1.0 / 2.0)
                    + 1.0 / 4.0 * a_2 * (lambda_2 - 2.0) * CKM_LAMBDA.powi(6) * z.norm_sqr(),
                0.0
            ),
            Complex::new(
                CKM_LAMBDA - 0.5 * a_2 * CKM_LAMBDA.powi(7) * z.norm_sqr(),
                0.0
            ),
            CKM_A * CKM_LAMBDA.powi(3) * z.conj()
        ],
        [
            0.5 * CKM_LAMBDA * (a_2 * CKM_LAMBDA.powi(4) * ((lambda_2 - 2.0) * z + 1.0) - 2.0),
            1.0 + lambda_2
                * (lambda_2
                    * (lambda_2
                        * (lambda_2 * (a_2 * (1.0 / 16.0 - a_2 / 8.0) - 5.0 / 128.0)
                            + a_2 * (1.0 / 5.0 - z)
                            - 1.0 / 16.0)
                        - a_2 / 2.0
                        - 1.0 / 8.0)
                    - 1.0 / 2.0),
            Complex::new(
                CKM_A * lambda_2 - CKM_A.powi(3) * CKM_LAMBDA.powi(8) * z.norm_sqr(),
                0.0
            )
        ],
        [
            CKM_LAMBDA.powi(3)
                * (lambda_2
                    * (lambda_2 * (CKM_A.powi(3) * z / 2.0 + CKM_A * z / 8.0) + CKM_A * z / 2.0)
                    - CKM_A * z
                    + CKM_A),
            lambda_2
                * (lambda_2
                    * (lambda_2
                        * (lambda_2 * (CKM_A.powi(3) * z / 2.0 + CKM_A / 16.0) + CKM_A / 8.0)
                        - CKM_A * z
                        + CKM_A / 2.0)
                    - CKM_A),
            Complex::new(
                1.0 + CKM_LAMBDA.powi(4)
                    * (lambda_2 * (-0.5 * a_2 * z.norm_sqr() - CKM_A.powi(4) * lambda_2)
                        - a_2 / 2.0),
                0.0
            )
        ]
    ]
}

fn generate_pmns() -> Array2<Complex<f64>> {
    use crate::model::standard::data::{PMNS_DELTA, PMNS_T12, PMNS_T13, PMNS_T23};

    let r23 = {
        let (sin, cos) = PMNS_T23.sin_cos();
        array![
            [Complex::one(), Complex::zero(), Complex::zero()],
            [
                Complex::zero(),
                Complex::new(cos, 0.0),
                Complex::new(sin, 0.0)
            ],
            [
                Complex::zero(),
                Complex::new(-sin, 0.0),
                Complex::new(cos, 0.0)
            ]
        ]
    };
    let r13 = {
        let (sin, cos) = PMNS_T13.sin_cos();
        array![
            [
                Complex::new(cos, 0.0),
                Complex::zero(),
                Complex::from_polar(sin, -PMNS_DELTA),
            ],
            [Complex::zero(), Complex::one(), Complex::zero(),],
            [
                Complex::from_polar(-sin, PMNS_DELTA),
                Complex::zero(),
                Complex::new(cos, 0.0),
            ]
        ]
    };
    let r12 = {
        let (sin, cos) = PMNS_T12.sin_cos();
        array![
            [
                Complex::new(cos, 0.0),
                Complex::new(sin, 0.0),
                Complex::zero()
            ],
            [
                Complex::new(-sin, 0.0),
                Complex::new(cos, 0.0),
                Complex::zero()
            ],
            [Complex::zero(), Complex::zero(), Complex::one()]
        ]
    };

    r23.dot(&r13).dot(&r12)
}

impl std::default::Default for Standard {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::Standard;
    use crate::{model::Model, utilities::test::complex_approx_eq};
    use num::Complex;
    use std::error;

    /// Check that the particle index is really corresponding to the correct
    /// particle.
    #[test]
    fn particle_indices() {
        let model = Standard::zero();

        for (i, p) in model.particles().iter().enumerate() {
            let name = &p.name;
            if name.len() == 1 {
                assert_eq!(Ok(i), Standard::static_particle_idx(name, 0));
                assert_eq!(Ok(i), model.particle_idx(name, 0));
            } else if name.len() == 2 {
                let mut chars = name.chars();
                let head = chars.next().unwrap();
                let idx = chars.next().unwrap() as usize - 49;
                assert_eq!(Ok(i), Standard::static_particle_idx(&head.to_string(), idx));
                assert_eq!(Ok(i), model.particle_idx(&head.to_string(), idx));
            }
        }
    }

    /// Check the values of the CKM matrix
    #[test]
    fn ckm() -> Result<(), Box<dyn error::Error>> {
        let ckm = super::generate_ckm();

        complex_approx_eq(ckm[[0, 0]], Complex::new(9.7439e-1, 0.0), 4.0, 0.0)?;
        complex_approx_eq(ckm[[0, 1]], Complex::new(2.2484e-1, 0.0), 4.0, 0.0)?;
        complex_approx_eq(ckm[[0, 2]], Complex::new(1.5042e-3, -3.3600e-3), 4.0, 0.0)?;
        complex_approx_eq(ckm[[1, 0]], Complex::new(-2.2470e-1, -1.3634e-4), 4.0, 0.0)?;
        complex_approx_eq(ckm[[1, 1]], Complex::new(9.7354e-1, -3.1449e-5), 4.0, 0.0)?;
        complex_approx_eq(ckm[[1, 2]], Complex::new(4.1628e-2, 0.0), 4.0, 0.0)?;
        complex_approx_eq(ckm[[2, 0]], Complex::new(7.8954e-3, -3.2711e-3), 4.0, 0.0)?;
        complex_approx_eq(ckm[[2, 1]], Complex::new(-4.0901e-2, -7.5479e-4), 4.0, 0.0)?;
        complex_approx_eq(ckm[[2, 2]], Complex::new(9.9913e-1, 0.0), 4.0, 0.0)?;

        Ok(())
    }
}
