//! Define the various interactions involved in leptogenesis.

use super::model::{p_i, LeptogenesisModel, NAMES};
use boltzmann_solver::{
    constants::{PI_1, PI_5},
    solver::{number_density::NumberDensitySolver, Solver},
    utilities::{checked_div, integrate_st},
};
use itertools::iproduct;
use special_functions::bessel;
use std::cell::RefCell;

/// Interaction N ↔ LH
pub fn n_el_h(solver: &mut NumberDensitySolver<LeptogenesisModel>) {
    let csv = RefCell::new(csv::Writer::from_path("/tmp/leptogenesis_sp/decay.csv").unwrap());
    csv.borrow_mut()
        .serialize(("step", "beta", NAMES[0], NAMES[1]))
        .unwrap();

    solver.add_interaction(move |mut s, n, ref c| {
        let ptcl_a = p_i("N", 0);
        let ptcl_b = p_i("H", 0);

        // Note that this is *not* exactly gamma and differs in the
        // normalization.  It actually just needs to be multiplied by n for
        // the decay, and eq_n for the inverse decay.
        let gamma_tilde = {
            let mut m2 = 0.0;
            for b in 0..3 {
                m2 += c.model.coupling.y_v[[0, b]].norm_sqr();
            }
            m2 *= (c.model.mass2[ptcl_a] - c.model.mass2[ptcl_b])
                * bessel::k_1_on_k_2(c.model.mass[ptcl_a] * c.beta);
            m2 /= c.hubble_rate * c.beta * 16.0 * PI_1 * c.model.mass[ptcl_a];
            m2
        };

        // The full γ(e H → N).  We are assuming e and H are always in
        // equilibrium thus their scaling factors are irrelevant.
        // let _decay = n[1] * gamma_tilde;
        let inverse_decay = c.eq_n[ptcl_a] * gamma_tilde;
        let net_decay = (n[ptcl_a] - c.eq_n[ptcl_a]) * gamma_tilde;

        let dbl = c.model.epsilon * net_decay - n[0] * inverse_decay;
        s[0] += dbl;
        s[ptcl_a] -= net_decay;

        {
            let mut csv = csv.borrow_mut();
            csv.write_field(format!("{}", c.step)).unwrap();
            csv.write_field(format!("{:.15e}", c.beta)).unwrap();
            csv.write_record(&[format!("{:.3e}", dbl), format!("{:.3e}", -net_decay)])
                .unwrap();
        }

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
        let ptcl_c = p_i("N", 0);
        let ptcl_h = p_i("H", 0);
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
                    s * (s - c.model.mass2[ptcl_c]) * (s - c.model.mass2[ptcl_h]).powi(2)
                        / ((s - c.model.mass2[ptcl_h]).powi(2)
                            + c.model.width2[ptcl_h] * c.model.mass2[ptcl_h])
                            .powi(2)
                ) + (
                    // t-channel
                    2.0 * (t + c.model.mass2[ptcl_c]) * (t + c.model.mass2[ptcl_h]).powi(2)
                        / ((t + c.model.mass2[ptcl_h]).powi(2)
                            + c.model.width2[ptcl_h] * c.model.mass2[ptcl_h])
                            .powi(2)
                )
            };

            m2_prefactor * integrate_st(m2_st, c.beta, c.model.mass[ptcl_c], 0.0, 0.0, 0.0)
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
