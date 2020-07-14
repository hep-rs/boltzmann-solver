//! Define the various interactions involved in leptogenesis.

use crate::LeptogenesisModel;
use boltzmann_solver::{constants::PI_2, model::interaction, pave, prelude::*};
use itertools::iproduct;

/// Interaction H ↔ L(i2), -e(i3)
pub fn hle() -> Vec<interaction::ThreeParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, i3) in iproduct!(0..1, 0..3, 0..3) {
        if i2 != i3 {
            continue;
        }

        let m2 = move |m: &LeptogenesisModel| {
            let p1 = m.particle("H", i1);
            let p2 = m.particle("L", i2);
            let p3 = m.particle("e", i3);

            2.0 * m.sm.ye[[i3, i2]].powi(2) * (p1.mass2 - p2.mass2 - p3.mass2)
        };

        interactions.append(&mut interaction::ThreeParticle::new_all(
            m2,
            LeptogenesisModel::particle_num("H", i1).unwrap(),
            LeptogenesisModel::particle_num("L", i2).unwrap(),
            -LeptogenesisModel::particle_num("e", i3).unwrap(),
        ));
    }

    interactions
}

/// Interaction H ↔ Q(i2), -u(i3)
pub fn hqu() -> Vec<interaction::ThreeParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, i3) in iproduct!(0..1, 0..3, 0..3) {
        if i2 != i3 {
            continue;
        }

        let m2 = move |m: &LeptogenesisModel| {
            let p1 = m.particle("H", i1);
            let p2 = m.particle("Q", i2);
            let p3 = m.particle("u", i3);

            2.0 * m.sm.yu[[i3, i2]].powi(2) * (p1.mass2 - p2.mass2 - p3.mass2)
        };

        interactions.append(&mut interaction::ThreeParticle::new_all(
            m2,
            LeptogenesisModel::particle_num("H", i1).unwrap(),
            LeptogenesisModel::particle_num("Q", i2).unwrap(),
            -LeptogenesisModel::particle_num("u", i3).unwrap(),
        ));
    }

    interactions
}

/// Interaction H ↔ Q(i2), -d(i3)
pub fn hqd() -> Vec<interaction::ThreeParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, i3) in iproduct!(0..1, 0..3, 0..3) {
        if i2 != i3 {
            continue;
        }

        let m2 = move |m: &LeptogenesisModel| {
            let p1 = m.particle("H", i1);
            let p2 = m.particle("Q", i2);
            let p3 = m.particle("d", i3);

            2.0 * m.sm.yd[[i3, i2]].powi(2) * (p1.mass2 - p2.mass2 - p3.mass2)
        };

        interactions.append(&mut interaction::ThreeParticle::new_all(
            m2,
            LeptogenesisModel::particle_num("H", i1).unwrap(),
            LeptogenesisModel::particle_num("Q", i2).unwrap(),
            -LeptogenesisModel::particle_num("d", i3).unwrap(),
        ));
    }

    interactions
}

/// Interaction H ↔ -L(i2), N(i3)
pub fn hln() -> Vec<interaction::ThreeParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, i3) in iproduct!(0..1, 0..3, 0..3) {
        let m2 = move |m: &LeptogenesisModel| {
            let p1 = m.particle("H", i1);
            let p2 = m.particle("L", i2);
            let p3 = m.particle("N", i3);

            2.0 * m.yv[[i3, i2]].norm_sqr() * (p1.mass2 - p2.mass2 - p3.mass2).abs()
        };

        interactions.extend(
            interaction::ThreeParticle::new_all(
                m2,
                LeptogenesisModel::particle_num("H", i1).unwrap(),
                -LeptogenesisModel::particle_num("L", i2).unwrap(),
                LeptogenesisModel::particle_num("N", i3).unwrap(),
            )
            .drain(..),
        );

        let len = interactions.len();

        interactions[len - 3..].iter_mut().for_each(|interaction| {
            interaction.set_asymmetry(move |m: &LeptogenesisModel| {
                let p1 = m.particle("H", i1);
                let p2 = m.particle("L", i2);
                let p3 = m.particle("N", i3);

                let mass_correction = iproduct!(0..3, 0..3)
                    .filter(|(i4, _)| i4 != &i3)
                    .map(|(i4, i5)| {
                        let p4 = m.particle("N", i4);
                        let p5 = m.particle("L", i5);

                        let numerator = p3.mass
                            * (m.yv[[i4, i5]] * m.yv[[i3, i5]].conj())
                            * (p2.mass2 + p3.mass2 - p1.mass2)
                            * (p3.mass * m.yv[[i3, i2]] * m.yv[[i4, i2]].conj()
                                - p4.mass * m.yv[[i4, i2]] * m.yv[[i3, i2]].conj())
                            * pave::b(0, 1, p3.mass2, p5.mass, p1.mass);
                        let denominator = p3.mass2 - p4.mass2;

                        2.0 * -2.0 * numerator.im / denominator
                    })
                    .sum::<f64>();

                let vertex_correction = iproduct!(0..3, 0..3)
                    .map(|(i4, i5)| {
                        let p4 = m.particle("L", i4);
                        let p5 = m.particle("N", i5);

                        (p3.mass * p5.mass)
                            * 2.0
                            * (m.yv[[i3, i2]]
                                * m.yv[[i3, i4]]
                                * m.yv[[i5, i2]].conj()
                                * m.yv[[i5, i4]].conj())
                            .im
                            * (-pave::b(0, 0, p1.mass2, p5.mass, p4.mass)
                                + pave::b(0, 0, p3.mass2, p1.mass, p4.mass)
                                + (p3.mass2 + p5.mass2 - 2.0 * p1.mass2)
                                    * pave::c(
                                        0, 0, 0, p2.mass2, p1.mass2, p3.mass2, p1.mass, p5.mass,
                                        p4.mass,
                                    ))
                    })
                    .sum::<f64>();

                (mass_correction + vertex_correction) / (16.0 * PI_2)
            });
        });
    }

    interactions
}
pub fn ffa() -> Vec<interaction::ThreeParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (fermion, factor) in &[
        ("Q", 1.0 / 6.0),
        ("d", 2.0 / 3.0),
        ("u", 8.0 / 3.0),
        ("L", 1.0 / 2.0),
        ("e", 2.0),
    ] {
        for (i1, i2, i3) in iproduct!(0..1, 0..3, 0..3) {
            if i2 != i3 {
                continue;
            }

            let m2 = move |m: &LeptogenesisModel| {
                let p1 = m.particle("A", i1);
                // let p2 = m.particle(fermion, i2);
                let p3 = m.particle(fermion, i3);

                factor * m.sm.g1.powi(2) * (p3.mass2 - p1.mass2)
            };

            interactions.extend(interaction::ThreeParticle::new_all(
                m2,
                LeptogenesisModel::particle_num("A", i1).unwrap(),
                LeptogenesisModel::particle_num(fermion, i2).unwrap(),
                -LeptogenesisModel::particle_num(fermion, i3).unwrap(),
            ));
        }
    }

    interactions
}

pub fn ffw() -> Vec<interaction::ThreeParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (fermion, factor) in &[("Q", 3.0 / 2.0), ("L", 1.0 / 2.0)] {
        for (i1, i2, i3) in iproduct!(0..1, 0..3, 0..3) {
            if i2 != i3 {
                continue;
            }

            let m2 = move |m: &LeptogenesisModel| {
                let p1 = m.particle("W", i1);
                let p2 = m.particle(fermion, i2);
                // let p3 = m.particle(fermion, i3);

                factor * m.sm.g1.powi(2) * (p1.mass2 - p2.mass2)
            };

            interactions.extend(interaction::ThreeParticle::new_all(
                m2,
                LeptogenesisModel::particle_num("W", i1).unwrap(),
                LeptogenesisModel::particle_num(fermion, i2).unwrap(),
                -LeptogenesisModel::particle_num(fermion, i3).unwrap(),
            ));
        }
    }

    interactions
}

pub fn hha() -> Vec<interaction::ThreeParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, i3) in iproduct!(0..1, 0..1, 0..1) {
        let m2 = move |m: &LeptogenesisModel| {
            let p1 = m.particle("H", i1);
            // let p2 = m.particle("H", i2);
            let p3 = m.particle("A", i3);

            0.25 * m.sm.g1.powi(2) * (p3.mass2 - 4.0 * p1.mass2)
        };

        interactions.append(&mut interaction::ThreeParticle::new_all(
            m2,
            LeptogenesisModel::particle_num("H", i1).unwrap(),
            LeptogenesisModel::particle_num("H", i2).unwrap(),
            -LeptogenesisModel::particle_num("A", i3).unwrap(),
        ));
    }

    interactions
}

pub fn hhaa() -> Vec<interaction::FourParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, i3, i4) in iproduct!(0..1, 0..1, 0..1, 0..1) {
        let m2 = move |m: &LeptogenesisModel, _, _, _| 16.0 * m.sm.g1.powi(4);

        interactions.append(&mut interaction::FourParticle::new_all(
            m2,
            LeptogenesisModel::particle_num("H", i1).unwrap(),
            -LeptogenesisModel::particle_num("H", i2).unwrap(),
            LeptogenesisModel::particle_num("A", i3).unwrap(),
            LeptogenesisModel::particle_num("A", i4).unwrap(),
        ));
    }

    interactions
}

pub fn hhw() -> Vec<interaction::ThreeParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, i3) in iproduct!(0..1, 0..1, 0..1) {
        let m2 = move |m: &LeptogenesisModel| {
            let p1 = m.particle("H", i1);
            // let p2 = m.particle("H", i2);
            let p3 = m.particle("W", i3);

            0.25 * m.sm.g2.powi(2) * (p3.mass2 - 4.0 * p1.mass2)
        };

        interactions.append(&mut interaction::ThreeParticle::new_all(
            m2,
            LeptogenesisModel::particle_num("H", i1).unwrap(),
            LeptogenesisModel::particle_num("H", i2).unwrap(),
            -LeptogenesisModel::particle_num("W", i3).unwrap(),
        ));
    }

    interactions
}

pub fn hhww() -> Vec<interaction::FourParticle<LeptogenesisModel>> {
    let mut interactions = Vec::new();

    for (i1, i2, i3, i4) in iproduct!(0..1, 0..1, 0..1, 0..1) {
        let m2 = move |m: &LeptogenesisModel, _, _, _| 48.0 * m.sm.g1.powi(4);

        interactions.append(&mut interaction::FourParticle::new_all(
            m2,
            LeptogenesisModel::particle_num("H", i1).unwrap(),
            -LeptogenesisModel::particle_num("H", i2).unwrap(),
            LeptogenesisModel::particle_num("W", i3).unwrap(),
            LeptogenesisModel::particle_num("W", i4).unwrap(),
        ));
    }

    interactions
}

/// Interaction:
/// H, H ↔ -L(i3), -L(i4)
pub fn hhll1() -> Vec<interaction::FourParticle<LeptogenesisModel>> {
    let mut interactions = Vec::with_capacity(6);

    for (i1, i2, i3, i4) in iproduct!(0..1, 0..1, 0..3, 0..3) {
        let m2 = move |m: &LeptogenesisModel, s: f64, t: f64, u: f64| {
            // let p1 = m.particle("H", i1);
            // let p2 = m.particle("H", i2);
            let p3 = m.particle("L", i3);
            let p4 = m.particle("L", i4);

            iproduct!(0..3, 0..3)
                .map(|(i5, i105)| {
                    let p5 = m.particle("N", i5);
                    let p105 = m.particle("N", i105);

                    let y: f64 = (m.yv[[i105, i3]]
                        * m.yv[[i105, i4]]
                        * m.yv[[i5, i3]].conj()
                        * m.yv[[i5, i4]].conj())
                    .re;

                    y * p5.mass
                        * p105.mass
                        * (s - p3.mass2 - p4.mass2)
                        * (2.0
                            * (p105.propagator(-t) + p105.propagator(-u))
                            * (p5.propagator(-t) + p5.propagator(-u)).conj()
                            + p105.propagator(-u) * p5.propagator(-u).conj())
                        .re
                })
                .sum()
        };

        interactions.append(&mut interaction::FourParticle::new_all(
            m2,
            LeptogenesisModel::particle_num("H", i1).unwrap(),
            LeptogenesisModel::particle_num("H", i2).unwrap(),
            -LeptogenesisModel::particle_num("L", i3).unwrap(),
            -LeptogenesisModel::particle_num("L", i4).unwrap(),
        ))
    }

    interactions
}

/// Interaction:
/// -H, H ↔ -L(i3), L(i4)
pub fn hhll2() -> Vec<interaction::FourParticle<LeptogenesisModel>> {
    let mut interactions = Vec::with_capacity(6);

    for (i1, _i2, i3, i4) in iproduct!(0..1, 0..1, 0..3, 0..3) {
        let m2 = move |m: &LeptogenesisModel, s: f64, t: f64, u: f64| {
            let p1 = m.particle("H", i1);
            // let p2 = m.particle("H", i2);
            let p3 = m.particle("L", i3);
            let p4 = m.particle("L", i4);

            iproduct!(0..3, 0..3)
                .map(|(i5, i105)| {
                    let p5 = m.particle("N", i5);
                    let p105 = m.particle("N", i105);

                    let y: f64 = (m.yv[[i105, i3]]
                        * m.yv[[i105, i4]]
                        * m.yv[[i5, i3]].conj()
                        * m.yv[[i5, i4]].conj())
                    .re;

                    y * (
                        // ee and vv final states
                        2.0 * (p1.mass2 * (t + u - s) - p4.mass2 * (u - s + p4.mass2)
                            + t * u
                            + p3.mass2 * (t + 2.0 * p1.mass2 - 2.0 * p4.mass2)
                            + p1.mass2.powi(2))
                            / (t + p105.mass2)
                            / (t + p5.mass2)
                            + (p1.mass2 * (t + u - s)
                                - p4.mass2 * (t - s + 2.0 * p3.mass2 + p4.mass2)
                                + t * u
                                + p3.mass2 * (u + 2.0 * p1.mass2)
                                + p1.mass2.powi(2))
                                / (t + p105.mass2)
                                / (t + p5.mass2)
                    )
                })
                .sum()
        };

        interactions.append(&mut interaction::FourParticle::new_all(
            m2,
            -LeptogenesisModel::particle_num("H", 0).unwrap(),
            LeptogenesisModel::particle_num("H", 0).unwrap(),
            -LeptogenesisModel::particle_num("L", i3).unwrap(),
            LeptogenesisModel::particle_num("L", i4).unwrap(),
        ))
    }

    interactions
}

/// Interaction:
/// N(i1), L(i2) ↔ -Q(i3), d(i4)
pub fn nlqd() -> Vec<interaction::FourParticle<LeptogenesisModel>> {
    let mut interactions = Vec::with_capacity(3 * 3 * 3);

    for (i1, i2, i3, i4) in iproduct!(0..3, 0..3, 0..3, 0..4) {
        if i3 != i4 {
            continue;
        }

        let m2 = move |m: &LeptogenesisModel, s: f64, _: f64, _: f64| {
            let p1 = m.particle("N", i1);
            let p2 = m.particle("L", i2);
            let p3 = m.particle("Q", i3);
            let p4 = m.particle("d", i4);
            let p5 = m.particle("H", 0);

            let y: f64 = (m.sm.yd[[i4, i3]] * m.yv[[i1, i2]]).norm_sqr();

            y * (s - p4.mass2 - p3.mass2) * (-s + p2.mass2 + p1.mass2)
                / ((p5.mass2 - s).powi(2) + p5.mass2 * p5.width2)
        };

        interactions.append(&mut interaction::FourParticle::new_all(
            m2,
            LeptogenesisModel::particle_num("N", i1).unwrap(),
            LeptogenesisModel::particle_num("L", i2).unwrap(),
            -LeptogenesisModel::particle_num("Q", i3).unwrap(),
            LeptogenesisModel::particle_num("d", i4).unwrap(),
        ))
    }

    interactions
}

/// Interaction:
/// N(i1), L(i2) ↔ -Q(i3), u(i4)
pub fn nlqu() -> Vec<interaction::FourParticle<LeptogenesisModel>> {
    let mut interactions = Vec::with_capacity(3 * 3 * 3);

    for (i1, i2, i3, i4) in iproduct!(0..3, 0..3, 0..3, 0..4) {
        if i3 != i4 {
            continue;
        }

        let m2 = move |m: &LeptogenesisModel, s: f64, _: f64, _: f64| {
            let p1 = m.particle("N", i1);
            let p2 = m.particle("L", i2);
            let p3 = m.particle("Q", i3);
            let p4 = m.particle("u", i4);
            let p5 = m.particle("H", 0);

            let y: f64 = (m.sm.yd[[i4, i3]] * m.yv[[i1, i2]]).norm_sqr();

            y * (s - p4.mass2 - p3.mass2) * (-s + p2.mass2 + p1.mass2)
                / ((p5.mass2 - s).powi(2) + p5.mass2 * p5.width2)
        };

        interactions.append(&mut interaction::FourParticle::new_all(
            m2,
            LeptogenesisModel::particle_num("N", i1).unwrap(),
            LeptogenesisModel::particle_num("L", i2).unwrap(),
            -LeptogenesisModel::particle_num("Q", i3).unwrap(),
            LeptogenesisModel::particle_num("u", i4).unwrap(),
        ))
    }

    interactions
}
