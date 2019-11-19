//! Define the various interactions involved in leptogenesis.

use crate::LeptogenesisModel;
use boltzmann_solver::prelude::*;
use itertools::iproduct;
use std::convert::TryFrom;

/// Interaction N(i1) ↔ N(i2)
pub fn nn() -> Vec<Interaction<LeptogenesisModel>> {
    let mut interactions = Vec::with_capacity(3);

    for i1 in 0..1 {
        let p1 = LeptogenesisModel::particle_idx("N", i1).unwrap();

        let m2 = move |m: &LeptogenesisModel| m.mn[i1].powi(2);

        interactions.push(Interaction::two_body(
            m2,
            isize::try_from(p1).unwrap(),
            -isize::try_from(p1).unwrap(),
        ));
    }

    interactions
}

/// Interaction H ↔ -L(i2), N(i3)
pub fn hln() -> Vec<Interaction<LeptogenesisModel>> {
    let mut interactions = Vec::with_capacity(3 * 3);

    for (i2, i3) in iproduct!(0..3, 0..3) {
        let p1 = LeptogenesisModel::particle_idx("H", 0).unwrap();
        let p2 = LeptogenesisModel::particle_idx("L", i2).unwrap();
        let p3 = LeptogenesisModel::particle_idx("N", i3).unwrap();

        let m2 = move |m: &LeptogenesisModel| {
            let ptcl = m.particles();
            2.0 * m.yv[[i3, i2]].norm_sqr() * (ptcl[p1].mass2 - ptcl[p2].mass2 - ptcl[p3].mass2)
        };

        interactions.push(
            Interaction::three_body(
                m2,
                isize::try_from(p1).unwrap(),
                -isize::try_from(p2).unwrap(),
                isize::try_from(p3).unwrap(),
            )
            .asymmetry(move |m: &LeptogenesisModel| m.epsilon[[i3, i2]]),
        );
    }

    interactions
}

/// Interaction:
/// H, H ↔ -L(i3), -L(i4)
pub fn hhll1() -> Vec<Interaction<LeptogenesisModel>> {
    let mut interactions = Vec::with_capacity(3 * 3);

    let i1 = LeptogenesisModel::particle_idx("H", 0).unwrap();
    let i2 = LeptogenesisModel::particle_idx("H", 0).unwrap();
    for (i3, i4) in iproduct!(0..1, 0..1) {
        let m2 = move |m: &LeptogenesisModel, s: f64, t: f64, u: f64| {
            // let p1 = m.particle("H", 0);
            // let p2 = m.particle("H", 0);
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
                        * (
                            // ee and vv final states
                            2.0 * (t + u + 2.0 * p105.mass2) * (t + u + 2.0 * p5.mass2)
                            / (t + p105.mass2)
                            / (t + p5.mass2)
                            / (u + p105.mass2)
                            / (u + p5.mass2)
                            // ev final states
                            + 1.0 / (u + p105.mass2) / (u + p5.mass2)
                        )
                })
                .sum()
        };

        interactions.push(Interaction::four_body(
            m2,
            isize::try_from(i1).unwrap(),
            isize::try_from(i2).unwrap(),
            -isize::try_from(LeptogenesisModel::particle_idx("L", i3).unwrap()).unwrap(),
            -isize::try_from(LeptogenesisModel::particle_idx("L", i4).unwrap()).unwrap(),
        ))
    }

    interactions
}

/// Interaction:
/// -H, H ↔ -L(i3), L(i4)
pub fn hhll2() -> Vec<Interaction<LeptogenesisModel>> {
    let mut interactions = Vec::with_capacity(3 * 3);

    let i1 = LeptogenesisModel::particle_idx("H", 0).unwrap();
    let i2 = LeptogenesisModel::particle_idx("H", 0).unwrap();
    for (i3, i4) in iproduct!(0..3, 0..3) {
        let m2 = move |m: &LeptogenesisModel, s: f64, t: f64, u: f64| {
            let p1 = m.particle("H", 0);
            // let p2 = m.particle("H", 0);
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

        interactions.push(Interaction::four_body(
            m2,
            isize::try_from(i1).unwrap(),
            isize::try_from(i2).unwrap(),
            -isize::try_from(LeptogenesisModel::particle_idx("L", i3).unwrap()).unwrap(),
            -isize::try_from(LeptogenesisModel::particle_idx("L", i4).unwrap()).unwrap(),
        ))
    }

    interactions
}

/// Interaction:
/// N(i1), L(i2) ↔ -Q(i3), d(i4)
pub fn nlqd() -> Vec<Interaction<LeptogenesisModel>> {
    let mut interactions = Vec::with_capacity(3 * 3 * 3);

    for (i1, i2, i3) in iproduct!(0..3, 0..3, 0..3) {
        let i4 = i3;
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

        interactions.push(Interaction::four_body(
            m2,
            isize::try_from(i1).unwrap(),
            isize::try_from(i2).unwrap(),
            -isize::try_from(i3).unwrap(),
            isize::try_from(i4).unwrap(),
        ))
    }

    interactions
}

/// Interaction:
/// N(i1), L(i2) ↔ -Q(i3), u(i4)
pub fn nlqu() -> Vec<Interaction<LeptogenesisModel>> {
    let mut interactions = Vec::with_capacity(3 * 3 * 3);

    for (i1, i2, i3) in iproduct!(0..3, 0..3, 0..3) {
        let i4 = i3;
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

        interactions.push(Interaction::four_body(
            m2,
            isize::try_from(i1).unwrap(),
            isize::try_from(i2).unwrap(),
            -isize::try_from(i3).unwrap(),
            isize::try_from(i4).unwrap(),
        ))
    }

    interactions
}