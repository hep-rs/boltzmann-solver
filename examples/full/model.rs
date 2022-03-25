//! Construct the [`Model`] instance

// We allow `dead_code` in this module as we prefer to define all parameters in
// case we need them.  If certain parameters are quite computationally costly to
// compute and ultimately not used, it would be preferable to comment these out.
#![allow(dead_code)]

pub mod interaction;

use boltzmann_solver::{model::LEFT_WEYL_SPINOR, prelude::*};
use ndarray::{array, prelude::*};
use num::{Complex, One, Zero};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::{f64, f64::consts};

// //////////////////////////////////////////////////////////////////////////////
// Model
// //////////////////////////////////////////////////////////////////////////////

/// Full Leptogenesis
///
/// This model is based on the Standard Model and augments it with some new free
/// parameters.
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct LeptogenesisModel {
    /// Standard Model
    ///
    /// We could (re)define all the parameters of the Standard Model, but it is
    /// easier declare it as sub-struct of our model.
    pub sm: StandardModel,
    /// Lightest SM neutrino mass in GeV.
    pub m1: f64,
    /// Heavy neutrino masses.
    pub mn: Array1<f64>,
    /// Casas-Ibara parameterization.  These are entirely unconstrained, though
    /// their real parts are angles (thus `$\Re(\omega_i) \in [-\pi, \pi]$`).
    pub omega: Array1<Complex<f64>>,
    /// The neutrino Majorana phase differences `$\alpha_{12}$` and
    /// `$\alpha_{13}$`.  These are constrained to `$\Re(\alpha_{1i}) \in
    /// [-2\pi, 2\pi]$`.
    pub alpha: Array1<f64>,
    /// Coupling between H L(i1) and N(i2)
    ///
    /// This is fully determined by the Casas-Ibara parameterization:
    ///
    /// ```math
    ///   y_\nu = \frac{1}{v / \sqrt{2}} U_{PMNS} \sqrt{\hat m} R^T \sqrt{\hat M}
    /// ```
    ///
    /// where `$U_{PMNS}$` is the usual PMNS matrix, `$\hat m$` is the
    /// diagonalized light-neutrino mass-matrix, `$\hat M$` is the diagonalized
    /// heavy-neutrino mass matrix and `$R$` is the complex rotation matrix.
    pub yv: Array2<Complex<f64>>,
    /// Interactions
    ///
    /// In this instance, we do *not* define any interactions by default as we
    /// want to see the effects of individual interactions and how specific
    /// combinations of interactions interact.
    ///
    ///
    /// The code features two separate implementations based on whether parallel
    /// evaluations are enabled.  This would generally not be done, but is used
    /// here for testing purposes.
    #[cfg(feature = "parallel")]
    #[cfg_attr(feature = "serde", serde(skip))]
    pub interactions: Vec<Box<dyn Interaction<Self> + Sync>>,
    #[cfg(not(feature = "parallel"))]
    #[cfg_attr(feature = "serde", serde(skip))]
    pub interactions: Vec<Box<dyn Interaction<Self>>>,
}

impl LeptogenesisModel {
    /// Generate the `$R$` matrix from the complex Casas-Ibara parameters:
    ///
    /// ```math
    /// R = \begin{pmatrix}
    ///   1 &  0   & 0 \\
    ///   0 &  c_1 & s_1 \\
    ///   0 & -s_1 & c_1
    /// \end{pmatrix} \begin{pmatrix}
    ///    c_2 & 0 & s_2 \\
    ///    0   & 1 & 0 \\
    ///   -s_2 & 0 & c_2
    /// \end{pmatrix} \begin{pmatrix}
    ///    c_3 & s_3 & 0 \\
    ///   -s_3 & c_3 & 0 \\
    ///    0   & 0   & 1
    /// \end{pmatrix}
    /// ```
    fn generate_r(&self) -> Array2<Complex<f64>> {
        let r1 = {
            let sin = self.omega[0].sin();
            let cos = self.omega[0].cos();

            array![
                [Complex::one(), Complex::zero(), Complex::zero()],
                [Complex::zero(), cos, sin],
                [Complex::zero(), -sin, cos],
            ]
        };
        let r2 = {
            let sin = self.omega[1].sin();
            let cos = self.omega[1].cos();

            array![
                [cos, Complex::zero(), sin],
                [Complex::zero(), Complex::one(), Complex::zero()],
                [-sin, Complex::zero(), cos],
            ]
        };
        let r3 = {
            let sin = self.omega[2].sin();
            let cos = self.omega[2].cos();

            array![
                [cos, sin, Complex::zero()],
                [-sin, cos, Complex::zero()],
                [Complex::zero(), Complex::zero(), Complex::one(),],
            ]
        };

        r1.dot(&r2).dot(&r3)
    }

    /// Calculates the value of [`$y_\nu$`](LeptogenesisModel::yv) and sets it
    /// internally.
    ///
    /// ```math
    ///   y_\nu = \frac{1}{v / \sqrt{2}} U_{PMNS} \sqrt{\hat m} R^T \sqrt{\hat M}
    /// ```
    fn calculate_yv(&mut self) {
        use boltzmann_solver::sm_data::{DELTA_M21, DELTA_M3L, VEV};

        let alpha = array![
            [Complex::one(), Complex::zero(), Complex::zero()],
            [
                Complex::zero(),
                Complex::from_polar(1.0, self.alpha[0] / 2.0),
                Complex::zero()
            ],
            [
                Complex::zero(),
                Complex::zero(),
                Complex::from_polar(1.0, self.alpha[1] / 2.0)
            ],
        ];
        let r = self.generate_r();
        let m_light = Array2::from_diag(&array![self.m1, self.m1 + DELTA_M21, self.m1 + DELTA_M3L])
            .map(|v| v.sqrt().into());
        let m_heavy = Array2::from_diag(&self.mn).map(|v| v.sqrt().into());

        self.yv = self
            .sm
            .pmns
            .dot(&alpha)
            .dot(&m_light)
            .dot(&r.t())
            .dot(&m_heavy)
            / (VEV / consts::SQRT_2);

        log::debug!("PMNS:\n{:.3e}", self.sm.pmns.dot(&alpha));
        log::debug!("mν:\n{:.3e}", m_light);
        log::debug!("R:\n{:.3e}", r);
        log::debug!("mN:\n{:.3e}", m_heavy);
        log::debug!("Yν:\n{:.3e}", self.yv);
    }
}

impl Model for LeptogenesisModel {
    fn zero() -> Self {
        let mut sm = StandardModel::zero();

        let mn = array![1e6, 1e8, 1e10];
        sm.particles.push(
            ParticleData::new(LEFT_WEYL_SPINOR, mn[0], 0.0)
                .name("N1")
                .own_antiparticle(),
        );
        sm.particles.push(
            ParticleData::new(LEFT_WEYL_SPINOR, mn[1], 0.0)
                .name("N2")
                .own_antiparticle(),
        );
        sm.particles.push(
            ParticleData::new(LEFT_WEYL_SPINOR, mn[2], 0.0)
                .name("N3")
                .own_antiparticle(),
        );

        let mut model = LeptogenesisModel {
            sm,
            m1: 1e-11,
            mn,
            omega: array![
                Complex::new(0.174533, 0.698132),
                Complex::new(1.22173, 1.74533),
                Complex::new(2.44346, 3.31613),
            ],
            // omega: array![Complex::zero(), Complex::zero(), Complex::zero()],
            alpha: array![0.0, 0.0],
            yv: Array2::zeros((3, 3)),
            interactions: Vec::new(),
        };

        model.calculate_yv();
        model.update_widths();

        model
    }

    fn set_beta(&mut self, beta: f64) {
        self.sm.set_beta(beta);

        // Precompute the squared couplings to be used in the thermal masses
        let g1 = self.sm.g1.powi(2) / 8.0;
        let g2 = self.sm.g2.powi(2) / 8.0;
        let yu = self.sm.yu.dot(&self.sm.yu).into_diag() / 16.0;
        let yd = self.sm.yd.dot(&self.sm.yd).into_diag() / 16.0;
        let ye = self.sm.ye.dot(&self.sm.ye).into_diag() / 16.0;
        let yv = self
            .yv
            .dot(&self.yv.t().map(|y| y.conj()))
            .into_diag()
            .map(|y| y.re)
            / 16.0;
        let lambda = 0.0 * self.sm.lambda / 4.0;

        self.particle_mut("H", 0).set_mass(
            f64::consts::SQRT_2
                * f64::sqrt(
                    g1 / 4.0
                        + (3.0 / 4.0) * g2
                        + 2.0 * yu.sum()
                        + 2.0 * yd.sum()
                        + 2.0 * ye.sum()
                        + 2.0 * yv.sum()
                        + lambda,
                )
                / beta,
        );

        for i in 0..3 {
            let mi = self.mn[i] + f64::sqrt(yv[i]) / beta;

            self.particle_mut("N", i).set_mass(mi);
            self.particle_mut("L", i)
                .set_mass(f64::sqrt(g1 / 4.0 + (3.0 / 4.0) * g2 + ye[i] + yv[i]) / beta);
        }

        self.update_widths();

        // for i in 0..3 {
        //     self.particle_mut("L", i).self_energy = Some(Box::new(move |s, m| {
        //         let lepton = m.particle("L", i);
        //         let higgs = m.particle("H", 0);

        //         let mut sum = Complex::zero();
        //         for j in 0..3 {
        //             let electron = m.particle("e", j);
        //             let neutrino = m.particle("N", j);
        //             sum += m.sm.ye[[j, i]]
        //                 * m.sm.ye[[j, i]]
        //                 * pave::b(0, 1, s, electron.mass, higgs.mass);

        //             sum += m.yv[[j, i]]
        //                 * m.yv[[j, i]].conj()
        //                 * pave::b(0, 1, s, neutrino.mass, higgs.mass);
        //         }
        //         sum *= 2.0;

        //         sum += m.sm.g1.powi(2) * pave::b(0, 1, s, lepton.mass, m.particle("A", 0).mass);
        //         sum +=
        //             3.0 * m.sm.g2.powi(2) * pave::b(0, 1, s, lepton.mass, m.particle("W", 0).mass);

        //         sum.re / (32.0 * boltzmann_solver::constants::PI_2)
        //     }))
        // }
    }

    fn get_beta(&self) -> f64 {
        self.sm.get_beta()
    }

    fn entropy_dof(&self, beta: f64) -> f64 {
        self.sm.entropy_dof(beta)
            + self.particles()[20..]
                .iter()
                .map(|p| p.entropy_dof(self.get_beta()))
                .sum::<f64>()
    }

    fn particles(&self) -> &[ParticleData] {
        &self.sm.particles
    }

    fn particles_mut(&mut self) -> &mut [ParticleData] {
        &mut self.sm.particles
    }

    fn static_particle_idx<S: AsRef<str>>(name: S, i: usize) -> Result<usize, (S, usize)> {
        let idx = match (name.as_ref(), i) {
            ("N", i) if i < 3 => Ok(20 + i),
            (_, i) => Err((name, i)),
        };

        idx.or_else(|(name, i)| StandardModel::static_particle_idx(name, i))
    }

    #[allow(clippy::too_many_lines)]
    fn self_energy_absorptive(&self, p: &ParticleData, momentum: f64) -> f64 {
        use boltzmann_solver::constants::PI_2;
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
        let pn = &ptcls[20..23];

        match p.name.as_str() {
            "A" => 0.0,
            "W" => 0.0,
            "G" => 0.0,
            "H" => {
                (64.0 * PI_2).recip()
                    * (4.0
                        * (0..3)
                            .map(|i| {
                                (0..3)
                                    .map(|j| {
                                        3.0 * self.sm.yd[[j, i]].powi(2)
                                            * (momentum - (pd[j].mass2 + pq[i].mass2))
                                            * b(0, 0, momentum, pq[i].mass, pd[j].mass)
                                            + 3.0
                                                * self.sm.yu[[j, i]].powi(2)
                                                * (momentum - (pu[j].mass2 + pq[i].mass2))
                                                * b(0, 0, momentum, pu[i].mass, pd[j].mass)
                                            + self.sm.ye[[j, i]].powi(2)
                                                * (momentum - (pe[j].mass2 + pl[i].mass2))
                                                * b(0, 0, momentum, pl[i].mass, pe[j].mass)
                                            + self.yv[[j, i]].norm_sqr()
                                                * (momentum - (pn[j].mass2 + pl[i].mass2))
                                                * b(0, 0, momentum, pn[i].mass, pl[j].mass)
                                    })
                                    .sum::<f64>()
                            })
                            .sum::<f64>()
                        + self.sm.g1.powi(2)
                            * (-2.0 * momentum + pa.mass2 - 2.0 * ph.mass2)
                            * b(0, 0, momentum, ph.mass, pa.mass)
                        + 3.0
                            * self.sm.g2.powi(2)
                            * (-2.0 * momentum + pw.mass2 - 2.0 * ph.mass2)
                            * b(0, 0, momentum, ph.mass, pw.mass))
            }
            "L1" => {
                let i = 0;

                (32.0 * PI_2).recip()
                    * (self.sm.g1.powi(2) * b(0, 1, momentum, pl[i].mass, pa.mass)
                        + 3.0 * self.sm.g2.powi(2) * b(0, 1, momentum, pl[i].mass, pw.mass)
                        + pl[i].mass
                        + 2.0
                            * (0..3)
                                .map(|j| {
                                    self.sm.ye[[j, i]].powi(2)
                                        * b(0, 1, momentum, pe[j].mass, ph.mass)
                                        + self.yv[[j, i]].norm_sqr()
                                            * b(0, 1, momentum, pn[j].mass, ph.mass)
                                })
                                .sum::<f64>())
            }
            "L2" => {
                let i = 1;
                (32.0 * PI_2).recip()
                    * (self.sm.g1.powi(2) * b(0, 1, momentum, pl[i].mass, pa.mass)
                        + 3.0 * self.sm.g2.powi(2) * b(0, 1, momentum, pl[i].mass, pw.mass)
                        + pl[i].mass
                        + 2.0
                            * (0..3)
                                .map(|j| {
                                    self.sm.ye[[j, i]].powi(2)
                                        * b(0, 1, momentum, pe[j].mass, ph.mass)
                                        + self.yv[[j, i]].norm_sqr()
                                            * b(0, 1, momentum, pn[j].mass, ph.mass)
                                })
                                .sum::<f64>())
            }
            "L3" => {
                let i = 2;
                (32.0 * PI_2).recip()
                    * (self.sm.g1.powi(2) * b(0, 1, momentum, pl[i].mass, pa.mass)
                        + 3.0 * self.sm.g2.powi(2) * b(0, 1, momentum, pl[i].mass, pw.mass)
                        + pl[i].mass
                        + 2.0
                            * (0..3)
                                .map(|j| {
                                    self.sm.ye[[j, i]].powi(2)
                                        * b(0, 1, momentum, pe[j].mass, ph.mass)
                                        + self.yv[[j, i]].norm_sqr()
                                            * b(0, 1, momentum, pn[j].mass, ph.mass)
                                })
                                .sum::<f64>())
            }
            "e1" => {
                let i = 0;
                (8.0 * PI_2).recip()
                    * (self.sm.g1.powi(2) * b(0, 1, momentum, pl[i].mass, pa.mass)
                        + (0..3)
                            .map(|j| {
                                self.sm.ye[[i, j]].powi(2) * b(0, 1, momentum, pl[j].mass, ph.mass)
                            })
                            .sum::<f64>())
            }
            "e2" => {
                let i = 1;
                (8.0 * PI_2).recip()
                    * (self.sm.g1.powi(2) * b(0, 1, momentum, pl[i].mass, pa.mass)
                        + (0..3)
                            .map(|j| {
                                self.sm.ye[[i, j]].powi(2) * b(0, 1, momentum, pl[j].mass, ph.mass)
                            })
                            .sum::<f64>())
            }
            "e3" => {
                let i = 2;
                (8.0 * PI_2).recip()
                    * (self.sm.g1.powi(2) * b(0, 1, momentum, pl[i].mass, pa.mass)
                        + (0..3)
                            .map(|j| {
                                self.sm.ye[[i, j]].powi(2) * b(0, 1, momentum, pl[j].mass, ph.mass)
                            })
                            .sum::<f64>())
            }
            "N1" => {
                let i = 0;
                (8.0 * PI_2).recip()
                    * ((0..3)
                        .map(|j| {
                            self.yv[[i, j]].norm_sqr() * b(0, 1, momentum, pl[j].mass, ph.mass)
                        })
                        .sum::<f64>())
            }
            "N2" => {
                let i = 1;
                (8.0 * PI_2).recip()
                    * ((0..3)
                        .map(|j| {
                            self.yv[[i, j]].norm_sqr() * b(0, 1, momentum, pl[j].mass, ph.mass)
                        })
                        .sum::<f64>())
            }
            "N3" => {
                let i = 2;
                (8.0 * PI_2).recip()
                    * ((0..3)
                        .map(|j| {
                            self.yv[[i, j]].norm_sqr() * b(0, 1, momentum, pl[j].mass, ph.mass)
                        })
                        .sum::<f64>())
            }
            "Q1" => {
                let i = 0;
                (96.0 * PI_2).recip()
                    * (self.sm.g1.powi(2) * b(0, 1, momentum, pq[i].mass, pa.mass)
                        + 27.0 * self.sm.g2.powi(2) * b(0, 1, momentum, pq[i].mass, pw.mass)
                        + 48.0 * self.sm.g3.powi(2) * b(0, 1, momentum, pq[i].mass, pg.mass)
                        + 18.0
                            * (0..3)
                                .map(|j| {
                                    self.sm.yd[[j, i]].powi(2)
                                        * b(0, 1, momentum, pd[j].mass, ph.mass)
                                })
                                .sum::<f64>()
                        + 18.0
                            * (0..3)
                                .map(|j| {
                                    self.sm.yu[[j, i]].powi(2)
                                        * b(0, 1, momentum, pu[j].mass, ph.mass)
                                })
                                .sum::<f64>())
            }
            "Q2" => {
                let i = 1;
                (96.0 * PI_2).recip()
                    * (self.sm.g1.powi(2) * b(0, 1, momentum, pq[i].mass, pa.mass)
                        + 27.0 * self.sm.g2.powi(2) * b(0, 1, momentum, pq[i].mass, pw.mass)
                        + 48.0 * self.sm.g3.powi(2) * b(0, 1, momentum, pq[i].mass, pg.mass)
                        + 18.0
                            * (0..3)
                                .map(|j| {
                                    self.sm.yd[[j, i]].powi(2)
                                        * b(0, 1, momentum, pd[j].mass, ph.mass)
                                })
                                .sum::<f64>()
                        + 18.0
                            * (0..3)
                                .map(|j| {
                                    self.sm.yu[[j, i]].powi(2)
                                        * b(0, 1, momentum, pu[j].mass, ph.mass)
                                })
                                .sum::<f64>())
            }
            "Q3" => {
                let i = 2;
                (96.0 * PI_2).recip()
                    * (self.sm.g1.powi(2) * b(0, 1, momentum, pq[i].mass, pa.mass)
                        + 27.0 * self.sm.g2.powi(2) * b(0, 1, momentum, pq[i].mass, pw.mass)
                        + 48.0 * self.sm.g3.powi(2) * b(0, 1, momentum, pq[i].mass, pg.mass)
                        + 18.0
                            * (0..3)
                                .map(|j| {
                                    self.sm.yd[[j, i]].powi(2)
                                        * b(0, 1, momentum, pd[j].mass, ph.mass)
                                })
                                .sum::<f64>()
                        + 18.0
                            * (0..3)
                                .map(|j| {
                                    self.sm.yu[[j, i]].powi(2)
                                        * b(0, 1, momentum, pu[j].mass, ph.mass)
                                })
                                .sum::<f64>())
            }
            "u1" => {
                let i = 0;
                (24.0 * PI_2).recip()
                    * (4.0 * self.sm.g1.powi(2) * b(0, 1, momentum, pu[i].mass, pa.mass)
                        + 12.0 * self.sm.g3.powi(2) * b(0, 1, momentum, pu[i].mass, pg.mass)
                        + 9.0
                            * (0..3)
                                .map(|j| {
                                    self.sm.yu[[i, j]].powi(2)
                                        * b(0, 1, momentum, pq[j].mass, ph.mass)
                                })
                                .sum::<f64>())
            }
            "u2" => {
                let i = 1;
                (24.0 * PI_2).recip()
                    * (4.0 * self.sm.g1.powi(2) * b(0, 1, momentum, pu[i].mass, pa.mass)
                        + 12.0 * self.sm.g3.powi(2) * b(0, 1, momentum, pu[i].mass, pg.mass)
                        + 9.0
                            * (0..3)
                                .map(|j| {
                                    self.sm.yu[[i, j]].powi(2)
                                        * b(0, 1, momentum, pq[j].mass, ph.mass)
                                })
                                .sum::<f64>())
            }
            "u3" => {
                let i = 2;
                (24.0 * PI_2).recip()
                    * (4.0 * self.sm.g1.powi(2) * b(0, 1, momentum, pu[i].mass, pa.mass)
                        + 12.0 * self.sm.g3.powi(2) * b(0, 1, momentum, pu[i].mass, pg.mass)
                        + 9.0
                            * (0..3)
                                .map(|j| {
                                    self.sm.yu[[i, j]].powi(2)
                                        * b(0, 1, momentum, pq[j].mass, ph.mass)
                                })
                                .sum::<f64>())
            }
            "d1" => {
                let i = 0;
                (24.0 * PI_2).recip()
                    * (self.sm.g1.powi(2) * b(0, 1, momentum, pd[i].mass, pa.mass)
                        + 12.0 * self.sm.g3.powi(2) * b(0, 1, momentum, pd[i].mass, pg.mass)
                        + 9.0
                            * (0..3)
                                .map(|j| {
                                    self.sm.yd[[i, j]].powi(2)
                                        * b(0, 1, momentum, pq[j].mass, ph.mass)
                                })
                                .sum::<f64>())
            }
            "d2" => {
                let i = 1;
                (24.0 * PI_2).recip()
                    * (self.sm.g1.powi(2) * b(0, 1, momentum, pd[i].mass, pa.mass)
                        + 12.0 * self.sm.g3.powi(2) * b(0, 1, momentum, pd[i].mass, pg.mass)
                        + 9.0
                            * (0..3)
                                .map(|j| {
                                    self.sm.yd[[i, j]].powi(2)
                                        * b(0, 1, momentum, pq[j].mass, ph.mass)
                                })
                                .sum::<f64>())
            }
            "d3" => {
                let i = 2;
                (24.0 * PI_2).recip()
                    * (self.sm.g1.powi(2) * b(0, 1, momentum, pd[i].mass, pa.mass)
                        + 12.0 * self.sm.g3.powi(2) * b(0, 1, momentum, pd[i].mass, pg.mass)
                        + 9.0
                            * (0..3)
                                .map(|j| {
                                    self.sm.yd[[i, j]].powi(2)
                                        * b(0, 1, momentum, pq[j].mass, ph.mass)
                                })
                                .sum::<f64>())
            }
            _ => 0.0,
        }
    }
}

#[cfg(not(feature = "parallel"))]
impl ModelInteractions for LeptogenesisModel {
    // type Iter = &Vec<Self::Item>;
    type Item = Box<dyn Interaction<Self>>;

    fn interactions(&self) -> &[Self::Item] {
        &self.interactions
    }
}

#[cfg(feature = "parallel")]
impl ModelInteractions for LeptogenesisModel {
    // type Iter = &'data Vec<Self::Item>;
    type Item = Box<dyn Interaction<Self> + Sync>;

    fn interactions(&self) -> &[Self::Item] {
        &self.interactions
    }
}

impl std::default::Default for LeptogenesisModel {
    fn default() -> Self {
        Self::zero()
    }
}
