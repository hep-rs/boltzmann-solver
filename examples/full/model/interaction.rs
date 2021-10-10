//! Define the various interactions involved in leptogenesis.

use super::LeptogenesisModel;
use boltzmann_solver::{constants::PI_2, pave, prelude::*};
use itertools::iproduct;
use num::{Complex, Zero};

/// Interaction `$H \leftrightarrow L \overline e$`
pub fn hle() -> Vec<interaction::ThreeParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, i3) in iproduct!(0..1, 0..3, 0..3) {
        if i2 != i3 {
            continue;
        }

        let squared_amplitude = move |m: &LeptogenesisModel| {
            let p1 = m.particle("H", i1);
            let p2 = m.particle("L", i2);
            let p3 = m.particle("e", i3);

            2.0 * m.sm.ye[[i3, i2]].powi(2) * (p1.mass2 - p2.mass2 - p3.mass2).abs()
        };

        interactions.append(&mut interaction::ThreeParticle::new_all(
            squared_amplitude,
            LeptogenesisModel::static_particle_num("H", i1).unwrap(),
            LeptogenesisModel::static_particle_num("L", i2).unwrap(),
            -LeptogenesisModel::static_particle_num("e", i3).unwrap(),
        ));
    }

    interactions
        .drain(..)
        .filter(|i| {
            i.particles()
                .particle_counts
                .values()
                .any(|&(c, ca)| c != 0.0 && ca != 0.0)
        })
        .collect()
}

/// Interaction `$H \leftrightarrow Q \overline u$`
pub fn hqu() -> Vec<interaction::ThreeParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, i3) in iproduct!(0..1, 0..3, 0..3) {
        if i2 != i3 {
            continue;
        }

        let squared_amplitude = move |m: &LeptogenesisModel| {
            let p1 = m.particle("H", i1);
            let p2 = m.particle("Q", i2);
            let p3 = m.particle("u", i3);

            2.0 * 3.0 * m.sm.yu[[i3, i2]].powi(2) * (p1.mass2 - p2.mass2 - p3.mass2).abs()
        };

        interactions.append(&mut interaction::ThreeParticle::new_all(
            squared_amplitude,
            LeptogenesisModel::static_particle_num("H", i1).unwrap(),
            LeptogenesisModel::static_particle_num("Q", i2).unwrap(),
            -LeptogenesisModel::static_particle_num("u", i3).unwrap(),
        ));
    }

    interactions
        .drain(..)
        .filter(|i| {
            i.particles()
                .particle_counts
                .values()
                .any(|&(c, ca)| c != 0.0 && ca != 0.0)
        })
        .collect()
}

/// Interaction `$H \leftrightarrow Q \overline d$`
pub fn hqd() -> Vec<interaction::ThreeParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, i3) in iproduct!(0..1, 0..3, 0..3) {
        if i2 != i3 {
            continue;
        }

        let squared_amplitude = move |m: &LeptogenesisModel| {
            let p1 = m.particle("H", i1);
            let p2 = m.particle("Q", i2);
            let p3 = m.particle("d", i3);

            2.0 * 3.0 * m.sm.yd[[i3, i2]].powi(2) * (p1.mass2 - p2.mass2 - p3.mass2).abs()
        };

        interactions.append(&mut interaction::ThreeParticle::new_all(
            squared_amplitude,
            LeptogenesisModel::static_particle_num("H", i1).unwrap(),
            LeptogenesisModel::static_particle_num("Q", i2).unwrap(),
            -LeptogenesisModel::static_particle_num("d", i3).unwrap(),
        ));
    }

    interactions
        .drain(..)
        .filter(|i| {
            i.particles()
                .particle_counts
                .values()
                .any(|&(c, ca)| c != 0.0 && ca != 0.0)
        })
        .collect()
}

/// Interaction `$H \leftrightarrow \overline L N$`
pub fn hln() -> Vec<interaction::ThreeParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, i3) in iproduct!(0..1, 0..3, 0..3) {
        let squared_amplitude = move |m: &LeptogenesisModel| {
            let p1 = m.particle("H", i1);
            let p2 = m.particle("L", i2);
            let p3 = m.particle("N", i3);

            2.0 * m.yv[[i3, i2]].norm_sqr() * (p1.mass2 - p2.mass2 - p3.mass2).abs()
        };

        interactions.append(&mut interaction::ThreeParticle::new_all(
            squared_amplitude,
            LeptogenesisModel::static_particle_num("H", i1).unwrap(),
            -LeptogenesisModel::static_particle_num("L", i2).unwrap(),
            LeptogenesisModel::static_particle_num("N", i3).unwrap(),
        ));

        let len = interactions.len();
        interactions[len - 3..].iter_mut().for_each(|i| {
            i.set_asymmetry(move |m: &LeptogenesisModel| {
                let p1 = m.particle("H", i1);
                let p2 = m.particle("L", i2);
                let p3 = m.particle("N", i3);

                let mass_correction = iproduct!(0..3, 0..3, 0..1)
                    .filter(|(i4, _, _)| i4 != &i3)
                    .map(|(i4, i5, i6)| {
                        let p4 = m.particle("N", i4);
                        let p5 = m.particle("L", i5);
                        let p6 = m.particle("H", i6);

                        let numerator = (m.yv[[i4, i5]] * m.yv[[i3, i5]].conj())
                            * (p2.mass2 + p3.mass2 - p1.mass2)
                            * ((m.yv[[i3, i2]] * m.yv[[i4, i2]].conj())
                                * ((p5.mass2 + p3.mass2 - p6.mass2)
                                    * pave::b(0, 0, p3.mass2, p5.mass, p6.mass)
                                    + p3.mass2 * pave::b(0, 1, p3.mass2, p5.mass, p6.mass)
                                    - pave::a(0, p5.mass)
                                    + pave::a(0, p6.mass))
                                + (p3.mass * p4.mass)
                                    * (m.yv[[i4, i2]] * m.yv[[i3, i2]].conj())
                                    * pave::b(0, 1, p3.mass2, p5.mass, p6.mass));

                        let denominator = p3.mass2 - p4.mass2;

                        2.0 * -2.0 * numerator.im / denominator
                    })
                    .sum::<f64>();

                let vertex_correction = iproduct!(0..3, 0..3, 0..1)
                    .map(|(i4, i5, i6)| {
                        let p4 = m.particle("N", i4);
                        let p5 = m.particle("L", i5);
                        let p6 = m.particle("H", i6);

                        (p3.mass * p4.mass)
                            * 2.0
                            * (m.yv[[i3, i2]]
                                * m.yv[[i3, i5]]
                                * m.yv[[i4, i2]].conj()
                                * m.yv[[i4, i5]].conj())
                            .im
                            * (-pave::b(0, 0, p1.mass2, p4.mass, p5.mass)
                                + pave::b(0, 0, p3.mass2, p6.mass, p5.mass)
                                + (p3.mass2 + p4.mass2 - p1.mass2 - p6.mass2)
                                    * pave::c(
                                        0, 0, 0, p2.mass2, p3.mass2, p1.mass2, p4.mass, p6.mass,
                                        p5.mass,
                                    ))
                    })
                    .sum::<f64>();

                (mass_correction + vertex_correction) / (16.0 * PI_2)
            });
        });
    }

    interactions
        .drain(..)
        .filter(|i| {
            i.particles()
                .particle_counts
                .values()
                .any(|&(c, ca)| c != 0.0 && ca != 0.0)
        })
        .collect()
}

/// Interaction `$f \leftrightarrow f A$`
pub fn ffa() -> Vec<interaction::ThreeParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (fermion, factor) in &[
        ("Q", 2.0 * (1.0 / 6.0)),
        ("d", 2.0 / 3.0),
        ("u", 8.0 / 3.0),
        ("L", 2.0 * (1.0 / 2.0)),
        ("e", 2.0),
    ] {
        for (i1, i2, i3) in iproduct!(0..3, 0..3, 0..1) {
            if i1 != i2 {
                continue;
            }

            let squared_amplitude = move |m: &LeptogenesisModel| {
                let p1 = m.particle(fermion, i1);
                // let p2 = m.particle(fermion, i2);
                let p3 = m.particle("A", i3);

                factor * m.sm.g1.powi(2) * (p3.mass2 - p1.mass2)
            };

            interactions.extend(interaction::ThreeParticle::new_all(
                squared_amplitude,
                LeptogenesisModel::static_particle_num(fermion, i1).unwrap(),
                LeptogenesisModel::static_particle_num(fermion, i2).unwrap(),
                LeptogenesisModel::static_particle_num("A", i3).unwrap(),
            ));
        }
    }

    interactions
        .drain(..)
        .filter(|i| {
            i.particles()
                .particle_counts
                .values()
                .any(|&(c, ca)| c != 0.0 && ca != 0.0)
        })
        .collect()
}

/// Interaction `$f \leftrightarrow f W$`
pub fn ffw() -> Vec<interaction::ThreeParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (fermion, factor) in &[("Q", 3.0), ("L", 1.0)] {
        for (i1, i2, i3) in iproduct!(0..3, 0..3, 0..1) {
            if i1 != i2 {
                continue;
            }

            let squared_amplitude = move |m: &LeptogenesisModel| {
                let p1 = m.particle(fermion, i1);
                // let p2 = m.particle(fermion, i2);
                let p3 = m.particle("W", i3);

                factor * m.sm.g2.powi(2) * (p3.mass2 - p1.mass2)
            };

            interactions.extend(interaction::ThreeParticle::new_all(
                squared_amplitude,
                LeptogenesisModel::static_particle_num(fermion, i1).unwrap(),
                LeptogenesisModel::static_particle_num(fermion, i2).unwrap(),
                LeptogenesisModel::static_particle_num("W", i3).unwrap(),
            ));
        }
    }

    interactions
        .drain(..)
        .filter(|i| {
            i.particles()
                .particle_counts
                .values()
                .any(|&(c, ca)| c != 0.0 && ca != 0.0)
        })
        .collect()
}

/// Interaction `$f \leftrightarrow f G$`
pub fn ffg() -> Vec<interaction::ThreeParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (fermion, factor) in &[("Q", 2.0), ("u", 1.0), ("d", 1.0)] {
        for (i1, i2, i3) in iproduct!(0..3, 0..3, 0..1) {
            if i1 != i2 {
                continue;
            }

            let squared_amplitude = move |m: &LeptogenesisModel| {
                let p1 = m.particle(fermion, i1);
                // let p2 = m.particle(fermion, i2);
                let p3 = m.particle("G", i3);

                factor * 48.0 * m.sm.g3.powi(2) * (p3.mass2 - p1.mass2)
            };

            interactions.extend(interaction::ThreeParticle::new_all(
                squared_amplitude,
                LeptogenesisModel::static_particle_num(fermion, i1).unwrap(),
                LeptogenesisModel::static_particle_num(fermion, i2).unwrap(),
                LeptogenesisModel::static_particle_num("G", i3).unwrap(),
            ));
        }
    }

    interactions
        .drain(..)
        .filter(|i| {
            i.particles()
                .particle_counts
                .values()
                .any(|&(c, ca)| c != 0.0 && ca != 0.0)
        })
        .collect()
}

/// Interaction `$H \leftrightarrow H A$`
pub fn hha() -> Vec<interaction::ThreeParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, i3) in iproduct!(0..1, 0..1, 0..1) {
        if i1 != i2 {
            continue;
        }

        let squared_amplitude = move |m: &LeptogenesisModel| {
            let p1 = m.particle("H", i1);
            // let p2 = m.particle("H", i2);
            let p3 = m.particle("A", i3);

            2.0 * (1.0 / 4.0) * m.sm.g1.powi(2) * (p3.mass2 - 4.0 * p1.mass2)
        };

        interactions.append(&mut interaction::ThreeParticle::new_all(
            squared_amplitude,
            LeptogenesisModel::static_particle_num("H", i1).unwrap(),
            LeptogenesisModel::static_particle_num("H", i2).unwrap(),
            LeptogenesisModel::static_particle_num("A", i3).unwrap(),
        ));
    }

    interactions
        .drain(..)
        .filter(|i| {
            i.particles()
                .particle_counts
                .values()
                .any(|&(c, ca)| c != 0.0 && ca != 0.0)
        })
        .collect()
}

/// Interaction `$H H \leftrightarrow A A$`
pub fn hhaa() -> Vec<interaction::FourParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, i3, i4) in iproduct!(0..1, 0..1, 0..1, 0..1) {
        if i1 != i2 {
            continue;
        }

        let squared_amplitude = move |m: &LeptogenesisModel, s: f64, _t, _u| {
            // let p1 = m.particle("H", i1);
            // let p2 = m.particle("H", i2);
            let p3 = m.particle("A", i3);
            // let p4 = m.particle("A", i4);

            2.0 * (1.0 / 16.0)
                * m.sm.g1.powi(4)
                * (s.powi(2) - 4.0 * s * p3.mass2 + 12.0 * p3.mass2.powi(2))
                / p3.mass2.powi(2)
        };

        interactions.append(&mut interaction::FourParticle::new_all(
            squared_amplitude,
            LeptogenesisModel::static_particle_num("H", i1).unwrap(),
            -LeptogenesisModel::static_particle_num("H", i2).unwrap(),
            LeptogenesisModel::static_particle_num("A", i3).unwrap(),
            LeptogenesisModel::static_particle_num("A", i4).unwrap(),
        ));
    }

    interactions
        .drain(..)
        .filter(|i| {
            i.particles()
                .particle_counts
                .values()
                .any(|&(c, ca)| c != 0.0 && ca != 0.0)
        })
        .collect()
}

/// Interaction `$H \leftrightarrow H W$`
pub fn hhw() -> Vec<interaction::ThreeParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, _i3) in iproduct!(0..1, 0..1, 0..1) {
        if i1 != i2 {
            continue;
        }

        let squared_amplitude = move |m: &LeptogenesisModel| {
            let p1 = m.particle("H", i1);
            // let p2 = m.particle("H", i2);
            let p3 = m.particle("W", 0);

            2.0 * (1.0 / 4.0) * m.sm.g2.powi(2) * (p3.mass2 - 4.0 * p1.mass2)
        };

        interactions.append(&mut interaction::ThreeParticle::new_all(
            squared_amplitude,
            LeptogenesisModel::static_particle_num("H", i1).unwrap(),
            LeptogenesisModel::static_particle_num("H", i2).unwrap(),
            LeptogenesisModel::static_particle_num("W", 0).unwrap(),
        ));
    }

    interactions
        .drain(..)
        .filter(|i| {
            i.particles()
                .particle_counts
                .values()
                .any(|&(c, ca)| c != 0.0 && ca != 0.0)
        })
        .collect()
}

/// Interaction `$H H \leftrightarrow W W$`
pub fn hhww() -> Vec<interaction::FourParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, i3, i4) in iproduct!(0..1, 0..1, 0..1, 0..1) {
        if i1 != i2 {
            continue;
        }

        let squared_amplitude = move |m: &LeptogenesisModel, s: f64, _, _| {
            // let p1 = m.particle("H", i1);
            // let p2 = m.particle("H", i2);
            let p3 = m.particle("W", i3);
            // let p4 = m.particle("W", i4);

            2.0 * (3.0 / 16.0)
                * m.sm.g2.powi(4)
                * (s.powi(2) - 4.0 * s * p3.mass2 + 12.0 * p3.mass2.powi(2))
                / p3.mass2.powi(2)
        };

        interactions.append(&mut interaction::FourParticle::new_all(
            squared_amplitude,
            LeptogenesisModel::static_particle_num("H", i1).unwrap(),
            -LeptogenesisModel::static_particle_num("H", i2).unwrap(),
            LeptogenesisModel::static_particle_num("W", i3).unwrap(),
            LeptogenesisModel::static_particle_num("W", i4).unwrap(),
        ));
    }

    interactions
        .drain(..)
        .filter(|i| {
            i.particles()
                .particle_counts
                .values()
                .any(|&(c, ca)| c != 0.0 && ca != 0.0)
        })
        .collect()
}

/// Interaction `$H H \leftrightarrow A W$`
pub fn hhaw() -> Vec<interaction::FourParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, i3, i4) in iproduct!(0..1, 0..1, 0..1, 0..1) {
        if i1 != i2 {
            continue;
        }

        let squared_amplitude = move |m: &LeptogenesisModel, s: f64, _, _| {
            // let p1 = m.particle("H", i1);
            // let p2 = m.particle("H", i2);
            let p3 = m.particle("A", i3);
            let p4 = m.particle("W", i4);

            (2.0 / 16.0 + 1.0 / 8.0)
                * m.sm.g1.powi(2)
                * m.sm.g2.powi(2)
                * (s.powi(2) - 4.0 * p3.mass2 * (s - 2.0 * p4.mass2) + 4.0 * p3.mass2.powi(2))
                / (p3.mass2 * p4.mass2)
        };

        interactions.append(&mut interaction::FourParticle::new_all(
            squared_amplitude,
            LeptogenesisModel::static_particle_num("H", i1).unwrap(),
            -LeptogenesisModel::static_particle_num("H", i2).unwrap(),
            LeptogenesisModel::static_particle_num("A", i3).unwrap(),
            LeptogenesisModel::static_particle_num("W", i4).unwrap(),
        ));
    }

    interactions
        .drain(..)
        .filter(|i| {
            i.particles()
                .particle_counts
                .values()
                .any(|&(c, ca)| c != 0.0 && ca != 0.0)
        })
        .collect()
}

/// Interaction `$H H \leftrightarrow \overline L \overline L$`
#[allow(dead_code)]
pub fn hhll1() -> Vec<interaction::FourParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, i3, i4) in iproduct!(0..1, 0..1, 0..3, 0..3).filter(|(_, _, i3, i4)| i4 >= i3) {
        let squared_amplitude = move |m: &LeptogenesisModel, s, t, u| {
            // let p1 = m.particle("H", i1);
            // let p2 = m.particle("H", i2);
            let p3 = m.particle("L", i3);
            // let p4 = m.particle("L", i4);

            let mut result = Complex::zero();
            for (i5, ic5) in iproduct!(0..3, 0..3) {
                let p5 = m.particle("N", i5);
                let pc5 = m.particle("N", ic5);

                let prefactor = (p5.mass * pc5.mass)
                    * (m.yv[[ic5, i3]]
                        * m.yv[[ic5, i4]]
                        * m.yv[[i5, i3]].conj()
                        * m.yv[[i5, i4]].conj())
                    * (s - 2.0 * p3.mass2);

                // H H -> vv and ee
                let m11 = prefactor * (pc5.propagator(t) * p5.propagator(t).conj());
                let m12 = prefactor * (pc5.propagator(u) * p5.propagator(t).conj());
                let m22 = prefactor * (pc5.propagator(u) * p5.propagator(u).conj());
                result += 2.0 * (m11 + 2.0 * m12 + m22);

                // HH -> ve
                let m11 = prefactor * (pc5.propagator(t) * p5.propagator(t).conj());
                result += m11;
            }

            result.re
        };

        interactions.append(&mut interaction::FourParticle::new_all(
            squared_amplitude,
            LeptogenesisModel::static_particle_num("H", i1).unwrap(),
            LeptogenesisModel::static_particle_num("H", i2).unwrap(),
            -LeptogenesisModel::static_particle_num("L", i3).unwrap(),
            -LeptogenesisModel::static_particle_num("L", i4).unwrap(),
        ));
    }

    interactions
        .drain(..)
        .filter(|i| {
            i.particles()
                .particle_counts
                .values()
                .any(|&(c, ca)| c != 0.0 && ca != 0.0)
        })
        .collect()
}

/// Interaction `$H \overline H \leftrightarrow \overline L L$`
#[allow(dead_code)]
pub fn hhll2() -> Vec<interaction::FourParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, i3, i4) in iproduct!(0..1, 0..1, 0..3, 0..3) {
        let squared_amplitude = move |m: &LeptogenesisModel, s, t, u| {
            // let p1 = m.particle("H", i1);
            let p2 = m.particle("H", i2);
            let p3 = m.particle("L", i3);
            let p4 = m.particle("L", i4);

            let mut result = Complex::zero();
            for (i5, ic5) in iproduct!(0..3, 0..3) {
                let p5 = m.particle("N", i5);
                let pc5 = m.particle("N", ic5);

                // H -H -> -v v, -e e
                let m11 = (m.yv[[i5, i4]]
                    * m.yv[[ic5, i3]]
                    * m.yv[[i5, i3]].conj()
                    * m.yv[[ic5, i4]].conj())
                    * (pc5.propagator(t) * p5.propagator(t).conj())
                    * (-p2.mass2 * (s + p4.mass2 + t + u) + p4.mass2 * (s + u)
                        - p3.mass2 * (3.0 * p4.mass2 - 3.0 * p2.mass2 + t)
                        + p2.mass2.powi(2)
                        + t * u);
                result += 2.0 * m11;

                // H -H -> e -v
                let m11 = (m.yv[[i5, i3]]
                    * m.yv[[ic5, i4]]
                    * m.yv[[i5, i4]].conj()
                    * m.yv[[ic5, i3]].conj())
                    * (pc5.propagator(u) * p5.propagator(u).conj())
                    * (p3.mass2 * (s - p4.mass2 + p2.mass2 + t)
                        - p2.mass2 * (s + t + u)
                        - 2.0 * p3.mass2.powi(2)
                        + p4.mass2 * (p2.mass2 - u)
                        + p2.mass2.powi(4)
                        + t * u);
                result += m11;
            }

            result.re
        };

        interactions.append(&mut interaction::FourParticle::new_all(
            squared_amplitude,
            LeptogenesisModel::static_particle_num("H", i1).unwrap(),
            -LeptogenesisModel::static_particle_num("H", i2).unwrap(),
            -LeptogenesisModel::static_particle_num("L", i3).unwrap(),
            LeptogenesisModel::static_particle_num("L", i4).unwrap(),
        ));
    }

    interactions
        .drain(..)
        .filter(|i| {
            i.particles()
                .particle_counts
                .values()
                .any(|&(c, ca)| c != 0.0 && ca != 0.0)
        })
        .collect()
}

/// Interaction `$H \overline H \leftrightarrow \overline e N$`
#[allow(dead_code)]
pub fn hhen() -> Vec<interaction::FourParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, i3, i4) in iproduct!(0..1, 0..1, 0..3, 0..3) {
        let squared_amplitude = move |m: &LeptogenesisModel, s, t, u| {
            // let p1 = m.particle("H", i1);
            let p2 = m.particle("H", i2);
            let p3 = m.particle("e", i3);
            let p4 = m.particle("N", i4);

            let m11: Complex<f64> = (m.yv[[i4, i3]].norm_sqr() * m.sm.ye[[i3, i3]].powi(2))
                * (p3.propagator(t) * p3.propagator(t).conj())
                * (-p2.mass2 * (s + p4.mass2 + t + u)
                    + p4.mass2 * (s + u)
                    + p3.mass2 * (3.0 * p4.mass2 - 3.0 * p2.mass2 + t)
                    + p2.mass2.powi(2)
                    + t * u);

            let m12: Complex<f64> = (m.yv[[i4, i3]].norm_sqr() * m.sm.ye[[i3, i3]].powi(2))
                * (p3.propagator(u) * p3.propagator(t).conj())
                * (-p2.mass2 * (s + t + u)
                    + p3.mass2 * (p4.mass2 + 2.0 * p2.mass2)
                    + p4.mass2.powi(2)
                    + t * u);

            let m22: Complex<f64> = (m.yv[[i4, i3]].norm_sqr() * m.sm.ye[[i3, i3]].powi(2))
                * (p3.propagator(u) * p3.propagator(u).conj())
                * (p3.mass2 * (s - p4.mass2 + p2.mass2 + t)
                    - p2.mass2 * (s + t + u)
                    - 2.0 * p3.mass2.powi(2)
                    + p4.mass2 * (p2.mass2 - u)
                    + p2.mass2.powi(2)
                    + t * u);

            (m11 + 2.0 * m12 + m22).re
        };

        interactions.append(&mut interaction::FourParticle::new_all(
            squared_amplitude,
            LeptogenesisModel::static_particle_num("H", i1).unwrap(),
            LeptogenesisModel::static_particle_num("H", i2).unwrap(),
            -LeptogenesisModel::static_particle_num("e", i3).unwrap(),
            LeptogenesisModel::static_particle_num("N", i4).unwrap(),
        ));
    }

    interactions
        .drain(..)
        .filter(|i| {
            i.particles()
                .particle_counts
                .values()
                .any(|&(c, ca)| c != 0.0 && ca != 0.0)
        })
        .collect()
}

/// Interaction `$N \overline H \leftrightarrow \overline L A$`
#[allow(dead_code)]
pub fn nhla() -> Vec<interaction::FourParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, i3, i4) in iproduct!(0..3, 0..1, 0..3, 0..1) {
        let squared_amplitude = move |m: &LeptogenesisModel, s: f64, t, u| {
            let p1 = m.particle("N", i1);
            let p2 = m.particle("H", i2);
            let p3 = m.particle("L", i3);
            let p4 = m.particle("A", i4);

            let m11: Complex<f64> = m.sm.g1.powi(2)
                * m.yv[[i1, i3]].norm_sqr()
                * (p3.propagator(p4.mass2 - p3.mass2 + s)
                    * p3.propagator(p4.mass2 - p3.mass2 + s).conj())
                * (p1.mass2
                    * (s.powi(2) - 5.0 * p3.mass2 * (s + p4.mass2) + 4.0 * s * p4.mass2
                        - p4.mass2.powi(2)
                        + 6.0 * p3.mass2.powi(2))
                    - s.powi(2) * t
                    + p3.mass2
                        * (-p3.mass2 * (4.0 * (s + t) + p4.mass2 + 2.0 * u)
                            + p4.mass2 * (s + 3.0 * t + 2.0 * u)
                            + s * (s + 4.0 * t + u)
                            - 3.0 * p4.mass2.powi(2)
                            + 4.0 * p3.mass2.powi(2))
                    - 2.0 * s * p4.mass2 * (t + u)
                    + p4.mass2.powi(2) * (2.0 * s + t))
                / (4.0 * p4.mass2);

            let m12: Complex<f64> = m.sm.g1.powi(2)
                * m.yv[[i1, i3]].norm_sqr()
                * (p2.propagator(t) * p3.propagator(p4.mass2 - p3.mass2 + s).conj())
                * (
                    // -4.0 * Complex::i() * epsilon
                    p4.mass2 * (s.powi(2) - s * t - u * (2.0 * t + u))
                        + p3.mass2
                            * (p2.mass2 * (s - 3.0 * p4.mass2 + 2.0 * t + u)
                                + p4.mass2 * (s + 5.0 * t + 4.0 * u)
                                - t * (s + 2.0 * t + u)
                                - 3.0 * p1.mass2 * (2.0 * p4.mass2 + p2.mass2 - t)
                                - 2.0 * p4.mass2.powi(2))
                        - p2.mass2
                            * (p1.mass2 * (3.0 * p4.mass2 - s)
                                + p4.mass2 * (s - 2.0 * t - u)
                                + s * t
                                + p4.mass2.powi(2))
                        + p1.mass2 * (3.0 * p4.mass2 * u - s * t)
                        + s * t.powi(2)
                        - 2.0 * p3.mass2.powi(2) * (2.0 * p4.mass2 + p2.mass2 - t)
                        + p4.mass2.powi(2) * u
                )
                / (4.0 * p4.mass2);

            let m22: Complex<f64> = -m.sm.g1.powi(2)
                * m.yv[[i1, i3]].norm_sqr()
                * (p2.propagator(t) * p2.propagator(t).conj())
                * (-p3.mass2 - p1.mass2 + t)
                * (-2.0 * p2.mass2 * (p4.mass2 + t) + (p4.mass2 + t).powi(2) + p2.mass2.powi(2))
                / (4.0 * p4.mass2);

            2.0 * (m11 + 2.0 * m12 * m22).re
        };

        interactions.append(&mut interaction::FourParticle::new_all(
            squared_amplitude,
            LeptogenesisModel::static_particle_num("N", i1).unwrap(),
            LeptogenesisModel::static_particle_num("H", i2).unwrap(),
            -LeptogenesisModel::static_particle_num("L", i3).unwrap(),
            LeptogenesisModel::static_particle_num("A", i4).unwrap(),
        ));
    }

    interactions
        .drain(..)
        .filter(|i| {
            i.particles()
                .particle_counts
                .values()
                .any(|&(c, ca)| c != 0.0 && ca != 0.0)
        })
        .collect()
}

/// Interaction `$N H \leftrightarrow \overline L W$`
#[allow(dead_code)]
pub fn nhlw() -> Vec<interaction::FourParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, i3, i4) in iproduct!(0..3, 0..1, 0..3, 0..1) {
        let squared_amplitude = move |m: &LeptogenesisModel, s: f64, t, u| {
            let p1 = m.particle("N", i1);
            let p2 = m.particle("H", i2);
            let p3 = m.particle("L", i3);
            let p4 = m.particle("W", i4);

            let m11: Complex<f64> = m.sm.g2.powi(2)
                * m.yv[[i1, i3]].norm_sqr()
                * (p3.propagator(p4.mass2 - p3.mass2 + s)
                    * p3.propagator(p4.mass2 - p3.mass2 + s).conj())
                * (p1.mass2 * (s.powi(2) + 4.0 * s * p4.mass2 - p4.mass2.powi(2)) - s.powi(2) * t
                    + p3.mass2
                        * (-5.0 * p1.mass2 * (s + p4.mass2)
                            + p4.mass2 * (s + 3.0 * t + 2.0 * u)
                            + s * (s + 4.0 * t + u)
                            - 3.0 * p4.mass2.powi(2))
                    - p3.mass2.powi(2) * (4.0 * s - 6.0 * p1.mass2 + p4.mass2 + 4.0 * t + 2.0 * u)
                    - 2.0 * s * p4.mass2 * (t + u)
                    + p4.mass2 * (2.0 * s + t)
                    + 4.0 * p3.mass2.powi(3))
                / (4.0 * p4.mass2);

            let m12: Complex<f64> = m.sm.g2.powi(2)
                * m.yv[[i1, i3]].norm_sqr()
                * (p2.propagator(t) * p3.propagator(p4.mass2 - p3.mass2 + s).conj())
                * (p4.mass2
                    * (
                        // -4.0 * Complex::i() * epsilon
                        s.powi(2)
                            * p3.mass2
                            * (s - 6.0 * p1.mass2 - 3.0 * p2.mass2 + 5.0 * t + 4.0 * u)
                            + p2.mass2 * (-s - 3.0 * p1.mass2 + 2.0 * t + u)
                            - s * t
                            - 4.0 * p3.mass2.powi(2)
                            + 3.0 * p1.mass2 * u
                            - u * (2.0 * t + u)
                    )
                    + (t + p2.mass2)
                        * (-p3.mass2 * (s - 3.0 * p1.mass2 + 2.0 * t + u)
                            + s * (t - p1.mass2)
                            + 2.0 * p4.mass2.powi(2))
                    + p4.mass2.powi(2) * (-2.0 * p3.mass2 - p2.mass2 + u))
                / (4.0 * p4.mass2);

            let m22: Complex<f64> = -m.sm.g2.powi(2)
                * m.yv[[i1, i3]].norm_sqr()
                * (p2.propagator(t) * p2.propagator(t).conj())
                * (-p3.mass2 - p1.mass2 + t)
                * (-2.0 * p2.mass2 * (p4.mass2 + t) + (t - p4.mass2).powi(2) + p2.mass2.powi(2))
                / (4.0 * p4.mass2);

            6.0 * (m11 + 2.0 * m12 * m22).re
        };

        interactions.append(&mut interaction::FourParticle::new_all(
            squared_amplitude,
            LeptogenesisModel::static_particle_num("N", i1).unwrap(),
            LeptogenesisModel::static_particle_num("H", i2).unwrap(),
            -LeptogenesisModel::static_particle_num("L", i3).unwrap(),
            LeptogenesisModel::static_particle_num("W", i4).unwrap(),
        ));
    }

    interactions
        .drain(..)
        .filter(|i| {
            i.particles()
                .particle_counts
                .values()
                .any(|&(c, ca)| c != 0.0 && ca != 0.0)
        })
        .collect()
}

/// Interaction `$Q \overline u \leftrightarrow L N$`
#[allow(dead_code)]
pub fn quln() -> Vec<interaction::FourParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, i3, i4) in iproduct!(0..3, 0..3, 0..3, 0..3) {
        if i1 != i2 {
            continue;
        }

        let squared_amplitude = move |m: &LeptogenesisModel, s: f64, _t, _u| {
            let p1 = m.particle("Q", i1);
            let p2 = m.particle("u", i2);
            let p3 = m.particle("L", i3);
            let p4 = m.particle("N", i4);

            let mut result = Complex::zero();

            for (i5, ic5) in iproduct!(0..1, 0..1) {
                let p5 = m.particle("H", i5);
                let pc5 = m.particle("H", ic5);

                let m11 = 3.0
                    * (m.sm.yu[[i1, i1]].powi(2) * m.yv[[i4, i3]].norm_sqr())
                    * (s - 2.0 * p3.mass2)
                    * (s - p1.mass2 - p2.mass2)
                    * p5.propagator(s - p3.mass2 + p4.mass2).conj()
                    * pc5.propagator(s - p3.mass2 + p4.mass2);

                result += 2.0 * m11;
            }

            result.re
        };

        interactions.append(&mut interaction::FourParticle::new_all(
            squared_amplitude,
            LeptogenesisModel::static_particle_num("Q", i1).unwrap(),
            -LeptogenesisModel::static_particle_num("u", i2).unwrap(),
            LeptogenesisModel::static_particle_num("L", i3).unwrap(),
            LeptogenesisModel::static_particle_num("N", i4).unwrap(),
        ));
    }

    interactions
        .drain(..)
        .filter(|i| {
            i.particles()
                .particle_counts
                .values()
                .any(|&(c, ca)| c != 0.0 && ca != 0.0)
        })
        .collect()
}

/// Interaction `$Q \overline d \leftrightarrow L N$`
#[allow(dead_code)]
pub fn qdln() -> Vec<interaction::FourParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, i3, i4) in iproduct!(0..3, 0..3, 0..3, 0..3) {
        if i1 != i2 {
            continue;
        }

        let squared_amplitude = move |m: &LeptogenesisModel, s: f64, _t, _u| {
            let p1 = m.particle("Q", i1);
            let p2 = m.particle("d", i2);
            let p3 = m.particle("L", i3);
            let p4 = m.particle("N", i4);

            let mut result = Complex::zero();

            for (i5, ic5) in iproduct!(0..1, 0..1) {
                let p5 = m.particle("H", i5);
                let pc5 = m.particle("H", ic5);

                let m11 = 3.0
                    * (m.sm.yd[[i1, i1]].powi(2) * m.yv[[i4, i3]].norm_sqr())
                    * (s - 2.0 * p3.mass2)
                    * (s - p1.mass2 - p2.mass2)
                    * p5.propagator(s - p3.mass2 + p4.mass2).conj()
                    * pc5.propagator(s - p3.mass2 + p4.mass2);

                result += 2.0 * m11;
            }

            result.re
        };

        interactions.append(&mut interaction::FourParticle::new_all(
            squared_amplitude,
            LeptogenesisModel::static_particle_num("Q", i1).unwrap(),
            -LeptogenesisModel::static_particle_num("d", i2).unwrap(),
            LeptogenesisModel::static_particle_num("L", i3).unwrap(),
            LeptogenesisModel::static_particle_num("N", i4).unwrap(),
        ));
    }

    interactions
        .drain(..)
        .filter(|i| {
            i.particles()
                .particle_counts
                .values()
                .any(|&(c, ca)| c != 0.0 && ca != 0.0)
        })
        .collect()
}

/// Interaction `$Q \overline e \leftrightarrow \overline L N$`
#[allow(dead_code)]
pub fn leln() -> Vec<interaction::FourParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, i3, i4) in iproduct!(0..3, 0..3, 0..3, 0..3) {
        if i1 != i2 {
            continue;
        }

        let squared_amplitude = move |m: &LeptogenesisModel, s: f64, t: f64, u| {
            let p1 = m.particle("L", i1);
            let p2 = m.particle("e", i2);
            let p3 = m.particle("L", i3);
            let p4 = m.particle("N", i4);

            let mut result = Complex::zero();

            for (i5, ic5) in iproduct!(0..1, 0..1) {
                let p5 = m.particle("H", i5);
                let pc5 = m.particle("H", ic5);

                let m11 = if i1 == i2 {
                    (m.sm.ye[[i1, i1]].powi(2) * m.yv[[i4, i3]].norm_sqr())
                        * (s - 2.0 * p3.mass2)
                        * (s - p1.mass2 - p2.mass2)
                        * p5.propagator(s - p3.mass2 + p4.mass2).conj()
                        * pc5.propagator(s - p3.mass2 + p4.mass2)
                } else {
                    Complex::zero()
                };

                let m12 = if i1 == i2 && i1 == i3 && i1 == i4 {
                    0.5 * (m.yv[[i4, i1]] * m.yv[[i4, i3]].conj() * m.sm.ye[[i1, i1]].powi(2))
                        * (pc5.propagator(u) * p5.propagator(s - p3.mass2 + p4.mass2).conj())
                        * (
                            // 4.0 * Complex::i() * epsilon
                            s.powi(2) + p2.mass2 * (-s + p3.mass2 + p4.mass2 + t - u)
                                - p1.mass2 * (s - 3.0 * p3.mass2 + p4.mass2 - t + u)
                                + p3.mass2 * (-2.0 * s + t - u)
                                + p4.mass2 * (t - u)
                                - t.powi(2)
                                + u.powi(2)
                        )
                } else {
                    Complex::zero()
                };

                let m22 = if i2 == i3 {
                    (m.sm.ye[[i2, i2]].powi(2) * m.yv[[i4, i1]].norm_sqr())
                        * (u - p2.mass - p1.mass2)
                        * (u - p1.mass2 - p4.mass2)
                        * p5.propagator(u).conj()
                        * pc5.propagator(u)
                } else {
                    Complex::zero()
                };

                result += 2.0 * (m11 + 2.0 * m12 + m22);
            }

            result.re
        };

        interactions.append(&mut interaction::FourParticle::new_all(
            squared_amplitude,
            LeptogenesisModel::static_particle_num("L", i1).unwrap(),
            -LeptogenesisModel::static_particle_num("e", i2).unwrap(),
            -LeptogenesisModel::static_particle_num("L", i3).unwrap(),
            LeptogenesisModel::static_particle_num("N", i4).unwrap(),
        ));
    }

    interactions
        .drain(..)
        .filter(|i| {
            i.particles()
                .particle_counts
                .values()
                .any(|&(c, ca)| c != 0.0 && ca != 0.0)
        })
        .collect()
}

/// Interaction `$L N \leftrightarrow L N$`
#[allow(dead_code)]
pub fn lnln() -> Vec<interaction::FourParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, i3, i4) in iproduct!(0..3, 0..3, 0..3, 0..3) {
        let squared_amplitude = move |m: &LeptogenesisModel, s: f64, t: f64, u| {
            let p1 = m.particle("L", i1);
            let p2 = m.particle("N", i2);
            let p3 = m.particle("L", i3);
            let p4 = m.particle("N", i4);

            let mut result = Complex::zero();

            for (i5, ic5) in iproduct!(0..1, 0..1) {
                let p5 = m.particle("H", i5);
                let pc5 = m.particle("H", ic5);

                let m11 = (m.yv[[i1, i2]].norm_sqr() * m.yv[[i4, i3]].norm_sqr())
                    * (s - 2.0 * p3.mass2)
                    * (s - p1.mass2 - p2.mass2)
                    * p5.propagator(s - p3.mass2 + p4.mass2).conj()
                    * pc5.propagator(s - p3.mass2 + p4.mass2);

                let m12 = (p2.mass * p4.mass)
                    * (m.yv[[i4, i1]] * m.yv[[i4, i3]] * (m.yv[[i2, i1]] * m.yv[[i2, i3]]).conj())
                    * (pc5.propagator(u) * p5.propagator(s - p3.mass2 + p4.mass2).conj())
                    * (p1.mass + p3.mass2 - t);

                let m22 = (m.yv[[i2, i3]].norm_sqr() * m.yv[[i4, i1]].norm_sqr())
                    * (u - p1.mass2 - p4.mass2)
                    * (u - p3.mass2 - p2.mass2)
                    * p5.propagator(u).conj()
                    * pc5.propagator(u);

                result += m11 + 2.0 * m12 + m22;
            }

            result.re
        };

        interactions.append(&mut interaction::FourParticle::new_all(
            squared_amplitude,
            LeptogenesisModel::static_particle_num("L", i1).unwrap(),
            LeptogenesisModel::static_particle_num("N", i2).unwrap(),
            LeptogenesisModel::static_particle_num("L", i3).unwrap(),
            LeptogenesisModel::static_particle_num("N", i4).unwrap(),
        ));
    }

    interactions
        .drain(..)
        .filter(|i| {
            i.particles()
                .particle_counts
                .values()
                .any(|&(c, ca)| c != 0.0 && ca != 0.0)
        })
        .collect()
}
