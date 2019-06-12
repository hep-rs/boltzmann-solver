//! Define the various interactions involved in leptogenesis.

use super::model::{p_i, LeptogenesisModel};
use boltzmann_solver::{
    constants::{PI_1, PI_5},
    solver::{number_density::NumberDensitySolver, Solver},
    utilities::{checked_div, integrate_st},
};
use itertools::iproduct;
use special_functions::bessel;
use std::cell::RefCell;

/// Interactions that maintain certain particles in equilibrium number density.
///
/// We are keeping H, L1, L2 and L3 at equilibrium.
pub fn equilibrium(solver: &mut NumberDensitySolver<LeptogenesisModel>) {
    solver.add_interaction(|mut s, n, ref c| {
        for &(p, i) in &[("H", 0), ("L", 0), ("L", 1), ("L", 2)] {
            let pi = p_i(p, i);

            s[pi] = 1e15 * (c.eq_n[pi] - n[pi]);
        }

        s
    });
}

/// Interaction H ↔ -L(i2), N(i3)
pub fn n_el_h(solver: &mut NumberDensitySolver<LeptogenesisModel>) {
    let csv = RefCell::new(csv::Writer::from_path("/tmp/leptogenesis_sp/decay.csv").unwrap());
    {
        let mut csv = csv.borrow_mut();
        csv.write_field("step").unwrap();
        csv.write_field("beta").unwrap();
        for (i2, i3) in iproduct!(0..3, 0..3) {
            csv.write_field(format!("p[{}.{}]", i2, i3)).unwrap();
            csv.write_field(format!("γ[{}.{}]", i2, i3)).unwrap();
            csv.write_field(format!("d[{}.{}]", i2, i3)).unwrap();
            csv.write_field(format!("i[{}.{}]", i2, i3)).unwrap();
            csv.write_field(format!("n[{}.{}]", i2, i3)).unwrap();
        }
        csv.write_record(None::<&[u8]>).unwrap();
    }

    solver.add_interaction(move |mut s, n, ref c| {
        let mut csv = csv.borrow_mut();

        csv.write_field(format!("{}", c.step)).unwrap();
        csv.write_field(format!("{:.15e}", c.beta)).unwrap();

        // Create short hands to access the masses and couplings
        let mass = &c.model.mass;
        let mass2 = &c.model.mass2;
        let coupling = &c.model.coupling;

        // Iterate over all combinations of particles
        for (i2, i3) in iproduct!(0..3, 0..3) {
            let max_m = mass.h.max(mass.n[i3]);
            let p1;
            let p2;
            let p3;

            // Store the index of the parent and daughter particles in p1 and p2,
            // p3 respectively.
            match (max_m == mass.h, max_m == mass.n[i3]) {
                (true, _) => {
                    p1 = p_i("H", 0);
                    p2 = p_i("L", i2);
                    p3 = p_i("N", i3);
                }
                (_, true) => {
                    p1 = p_i("N", i3);
                    p2 = p_i("H", 0);
                    p3 = p_i("L", i2);
                }
                (false, false) => unreachable!(),
            };

            // Amplitude squared
            let m2 = coupling.y_v[[i3, i2]].norm_sqr() * (mass2.h - mass2.n[i3]);
            // Phase space integration
            let phase_space =
                bessel::k_1_on_k_2(max_m * c.beta) / (c.hubble_rate * c.beta * 16.0 * PI_1 * max_m);
            // Factor of 2 to account to both SU(2) processes
            let gamma_tilde = -2.0 * m2 * phase_space;

            // Calculate the decay and inverse decay rates
            let decay = n[p1] * gamma_tilde;
            let inverse_decay =
                c.eq_n[p1] * checked_div(n[p2] * n[p3], c.eq_n[p2] * c.eq_n[p3]) * gamma_tilde;
            let net_decay = decay - inverse_decay;

            s[p_i("BL", 0)] += c.model.epsilon * net_decay - n[p_i("BL", 0)] * inverse_decay;

            s[p1] -= net_decay;
            s[p2] += net_decay;
            s[p3] += net_decay;

            csv.write_field(format!("{}", p1)).unwrap();
            csv.write_field(format!("{:.3e}", gamma_tilde)).unwrap();
            csv.write_field(format!("{:.3e}", decay / gamma_tilde))
                .unwrap();
            csv.write_field(format!("{:.3e}", inverse_decay / gamma_tilde))
                .unwrap();
            csv.write_field(format!("{:.3e}", net_decay / gamma_tilde))
                .unwrap();
        }

        csv.write_record(None::<&[u8]>).unwrap();

        s
    });
}

/// Scattering NL ↔ Qq, NQ ↔ Lq and Nq ↔ LQ (s- and t-channel)
pub fn n_el_ql_qr(solver: &mut NumberDensitySolver<LeptogenesisModel>) {
    let csv =
        RefCell::new(csv::Writer::from_path("/tmp/leptogenesis_sp/scattering_NLQq.csv").unwrap());
    csv.borrow_mut()
        .serialize(["step", "beta", "N₁Q → Lq", "Lq → N₁Q"])
        .unwrap();

    solver.add_interaction(move |mut s, n, ref c| {
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
                / (512.0 * PI_5 * c.hubble_rate * c.beta)
        };

        let forward = checked_div(n[1], c.eq_n[1]) * gamma;
        let net_forward = (checked_div(n[1], c.eq_n[1]) - 1.0) * gamma;
        let backward = gamma;

        s[0] -= n[0] * backward;
        s[1] -= net_forward;

        {
            let mut csv = csv.borrow_mut();
            csv.write_field(format!("{}", c.step)).unwrap();
            csv.write_field(format!("{:.15e}", c.beta)).unwrap();
            csv.write_record(&[format!("{:.3e}", forward), format!("{:.3e}", backward)])
                .unwrap();
        }

        s
    });
}
