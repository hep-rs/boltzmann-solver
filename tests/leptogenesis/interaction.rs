//! Define the various interactions involved in leptogenesis.

use super::model::{p_i, LeptogenesisModel};
use boltzmann_solver::{
    constants::{PI_3, PI_5},
    solver::{number_density::NumberDensitySolver, Solver},
    utilities::{checked_div, integrate_st},
};
use itertools::iproduct;
use ndarray::prelude::*;
use special_functions::bessel;
use std::sync::RwLock;

/// Interactions that maintain certain particles in equilibrium number density.
///
/// We are keeping H, L1, L2 and L3 at equilibrium.
pub fn equilibrium(solver: &mut NumberDensitySolver<LeptogenesisModel>) {
    solver.add_interaction(|n, ref c| {
        let mut dn = Array1::zeros(n.dim());
        for &(p, i) in &[("H", 0)] {
            let pi = p_i(p, i);

            dn[pi] = 2.0 * (c.eq_n[pi] - n[pi]) / c.normalization / c.step_size;
        }

        dn
    });
}

/// Interaction H ↔ -L(i2), N(i3)
pub fn n_el_h(solver: &mut NumberDensitySolver<LeptogenesisModel>) {
    let output_dir = crate::output_dir();
    let csv = RwLock::new(csv::Writer::from_path(output_dir.join("n_el_h.csv")).unwrap());
    {
        let mut csv = csv.write().unwrap();
        csv.write_field("step").unwrap();
        csv.write_field("beta").unwrap();
        for i3 in 0..3 {
            csv.write_field(format!("p[{}]", i3)).unwrap();
            csv.write_field(format!("γ[{}]", i3)).unwrap();
            csv.write_field(format!("n[{}]", i3)).unwrap();
        }
        csv.write_record(None::<&[u8]>).unwrap();
    }

    solver.add_interaction(move |n, ref c| {
        {
            let mut csv = csv.write().unwrap();
            csv.write_field(format!("{}", c.step)).unwrap();
            csv.write_field(format!("{:.15e}", c.beta)).unwrap();
        }

        let mut dn = Array1::zeros(n.dim());

        // Create short hands to access the masses and couplings
        let mass = &c.model.mass;
        let mass2 = &c.model.mass2;
        let coupling = &c.model.coupling;

        // Iterate over all combinations of particles
        let i1 = 0;
        for i3 in 0..3 {
            let max_m = mass.h.max(mass.n[i3]);
            let (p1, p2);

            // Store the index of the parent and daughter particles in p1 and
            // p2, p3 respectively.
            #[allow(clippy::float_cmp)]
            match (max_m == mass.h, max_m == mass.n[i3]) {
                (true, false) => {
                    p1 = p_i("H", i1);
                    p2 = p_i("N", i3);
                }
                (false, true) => {
                    p1 = p_i("N", i3);
                    p2 = p_i("H", i1);
                }
                _ => unreachable!(),
            };

            // Amplitude squared and phase space
            let m2 = (0..3)
                .map(|i2| coupling.y_v[[i3, i2]].norm_sqr())
                .sum::<f64>()
                * (mass2.h - mass2.n[i3]);
            let phase_space = max_m * bessel::k1(max_m * c.beta) / (32.0 * PI_3 * c.beta);
            // Factor of 2 to account to both SU(2) processes
            let gamma = 2.0 * m2.abs() * phase_space;

            // Calculate the net decay rate
            let net_decay =
                (checked_div(n[p1], c.eq_n[p1]) - checked_div(n[p2], c.eq_n[p2])) * gamma;

            dn[p_i("BL", 0)] += c.model.epsilon * net_decay - n[p_i("BL", 0)] * gamma;

            dn[p1] -= net_decay;
            dn[p2] += net_decay;

            {
                let mut csv = csv.write().unwrap();
                csv.write_field(format!("{}", p1)).unwrap();
                csv.write_field(format!("{:.3e}", c.normalization * gamma))
                    .unwrap();
                csv.write_field(format!("{:.3e}", c.normalization * net_decay))
                    .unwrap();
            }
        }

        {
            let mut csv = csv.write().unwrap();
            csv.write_record(None::<&[u8]>).unwrap();
        }

        dn
    });
}

/// Scattering NL ↔ Qq, NQ ↔ Lq and Nq ↔ LQ (s- and t-channel)
pub fn n_el_ql_qr(solver: &mut NumberDensitySolver<LeptogenesisModel>) {
    let output_dir = crate::output_dir();
    let csv = RwLock::new(csv::Writer::from_path(output_dir.join("n_el_ql_qr.csv")).unwrap());
    {
        let mut csv = csv.write().unwrap();
        csv.serialize(["step", "beta", "N₁Q → Lq", "Lq → N₁Q"])
            .unwrap();
    }

    solver.add_interaction(move |n, ref c| {
        let mut dn = Array1::zeros(n.dim());
        let p_c = p_i("N", 0);
        let p_z = p_i("H", 0);

        let gamma = {
            let m2_prefactor: f64 = iproduct!(0..3, 0..3, 0..3)
                .map(|(a2, b1, b2)| {
                    9.0 * c.model.coupling.y_v[[0, a2]].norm_sqr()
                        * (c.model.coupling.y_d[[b1, b2]].norm_sqr()
                            + c.model.coupling.y_u[[b1, b2]].norm_sqr())
                })
                .sum();

            let m2_st = |s: f64, t: f64| {
                (
                    // s-channel
                    s * (s - c.model.particles[p_c].mass2)
                        * (s - c.model.particles[p_z].mass2).powi(2)
                        / ((s - c.model.particles[p_z].mass2).powi(2)
                            + c.model.particles[p_z].width2 * c.model.particles[p_z].mass2)
                            .powi(2)
                ) + (
                    // t-channel
                    2.0 * (t + c.model.particles[p_c].mass2)
                        * (t + c.model.particles[p_z].mass2).powi(2)
                        / ((t + c.model.particles[p_z].mass2).powi(2)
                            + c.model.particles[p_z].width2 * c.model.particles[p_z].mass2)
                            .powi(2)
                )
            };

            m2_prefactor * integrate_st(m2_st, c.beta, c.model.particles[p_c].mass, 0.0, 0.0, 0.0)
                / (512.0 * PI_5)
        };

        let forward = checked_div(n[1], c.eq_n[1]) * gamma;
        let net_forward = (checked_div(n[1], c.eq_n[1]) - 1.0) * gamma;
        let backward = gamma;

        dn[p_i("BL", 0)] -= n[p_i("BL", 0)] * backward;
        dn[1] -= net_forward;

        {
            let mut csv = csv.write().unwrap();
            csv.write_field(format!("{}", c.step)).unwrap();
            csv.write_field(format!("{:.15e}", c.beta)).unwrap();
            csv.write_record(&[format!("{:.3e}", forward), format!("{:.3e}", backward)])
                .unwrap();
        }

        dn
    });
}
