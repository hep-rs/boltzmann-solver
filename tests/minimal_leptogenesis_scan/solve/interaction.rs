use crate::model::VanillaLeptogenesisModel;
use boltzmann_solver::{
    constants::{PI_1, PI_5},
    solver::{number_density::NumberDensitySolver, Solver},
    utilities::checked_div,
};
use itertools::iproduct;
use quadrature::integrate;
use special_functions::bessel;

/// Interaction N ↔ LH
pub fn n_el_h(solver: &mut NumberDensitySolver<VanillaLeptogenesisModel>) {
    solver.add_interaction(move |mut s, n, ref c| {
        // Note that this is *not* exactly gamma and differs in the
        // normalization.  It actually just needs to be multiplied by n for
        // the decay, and eq_n for the inverse decay.
        let gamma_tilde = {
            let mut m2 = 0.0;
            for b in 0..3 {
                m2 += c.model.coupling.y_v[[0, b]].norm_sqr();
            }
            m2 *= (c.model.mass2.n[0] - c.model.mass2.h)
                * bessel::k_1_on_k_2(c.model.mass.n[0] * c.beta);
            m2 /= c.hubble_rate * c.beta * 16.0 * PI_1 * c.model.mass.n[0];
            m2
        };

        // The full γ(e H → N).  We are assuming e and H are always in
        // equilibrium thus their scaling factors are irrelevant.
        // let _decay = n[1] * gamma_tilde;
        let inverse_decay = c.eq_n[1] * gamma_tilde;
        let net_decay = (n[1] - c.eq_n[1]) * gamma_tilde;

        let dbl = c.model.epsilon * net_decay - n[0] * inverse_decay;
        s[0] += dbl;
        let dn1 = -net_decay;
        s[1] += dn1;

        s
    });
}

/// Scattering NL ↔ Qq, NQ ↔ Lq and Nq ↔ LQ (s- and t-channel)
pub fn n_el_ql_qr(solver: &mut NumberDensitySolver<VanillaLeptogenesisModel>) {
    solver.add_interaction(move |mut s, n, ref c| {
        let mut gamma = {
            let mut m2 = 0.0;
            for (a2, b1, b2) in iproduct!(0..3, 0..3, 0..3) {
                m2 += 9.0
                    * c.model.coupling.y_v[[0, a2]].norm_sqr()
                    * (c.model.coupling.y_d[[b1, b2]].norm_sqr()
                        + c.model.coupling.y_u[[b1, b2]].norm_sqr());
            }

            let s_integrand = |ss: f64| {
                let s = c.model.mass2.n[0] + (1.0 - ss) / ss;
                let dsdss = ss.powi(-2);
                let sqrt_s = s.sqrt();

                let t_integrand = |t: f64| {
                    (
                        // s-channel
                        s * (s - c.model.mass2.n[0]) * (s - c.model.mass2.h).powi(2)
                            / ((s - c.model.mass2.h).powi(2) + c.model.width2.h * c.model.mass2.h)
                                .powi(2)
                    ) + (
                        // t-channel
                        2.0 * (t + c.model.mass2.n[0]) * (t + c.model.mass2.h).powi(2)
                            / ((t + c.model.mass2.h).powi(2) + c.model.width2.h * c.model.mass2.h)
                                .powi(2)
                    )
                };

                integrate(t_integrand, c.model.mass2.n[0] - s, 0.0, 0.0).integral
                    * bessel::k_1(sqrt_s * c.beta)
                    / sqrt_s
                    * dsdss
            };
            m2 * integrate(s_integrand, 0.0, 1.0, 0.0).integral
        };
        gamma /= 512.0 * PI_5 * c.hubble_rate * c.beta;

        // let forward = checked_div(n[1], c.eq_n[1]) * gamma;
        let net_forward = (checked_div(n[1], c.eq_n[1]) - 1.0) * gamma;
        let backward = gamma;

        s[0] -= n[0] * backward;
        s[1] -= net_forward;

        s
    });
}
