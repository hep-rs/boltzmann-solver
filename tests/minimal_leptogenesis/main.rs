extern crate boltzmann_solver;
extern crate csv;
extern crate itertools;
extern crate ndarray;
extern crate num;
extern crate quadrature;
extern crate rgsl;
extern crate special_functions;

mod model;

use boltzmann_solver::{
    constants::{PI_1, PI_5},
    particle::Particle,
    solver::{number_density::NumberDensitySolver, InitialCondition, Model, Solver},
    universe::StandardModel,
    utilities::checked_div,
};
use itertools::iproduct;
use model::{VanillaLeptogenesisModel, PARTICLE_NAMES};
use quadrature::integrate;
use special_functions::bessel;
use std::cell::RefCell;

#[test]
fn minimal_leptogenesis() {
    // Setup the directory for CSV output
    ::std::fs::create_dir("/tmp/minimal_leptogenesis/").unwrap_or(());

    let model = VanillaLeptogenesisModel::new(1e-15);

    // Set up the universe in which we'll run the Boltzmann equations
    let universe = StandardModel::new();

    // Create the Solver and set integration parameters
    let mut solver: NumberDensitySolver<VanillaLeptogenesisModel> = NumberDensitySolver::new()
        .beta_range(1e-15, 1e0)
        .initialize();

    // Add the particles to the solver, using for initial condition either 0 or
    // equilibrium number density.
    solver.add_particle(
        Particle::new(PARTICLE_NAMES[0].to_string(), 0, 0.0).set_dof(0.0),
        InitialCondition::Zero,
    );

    solver.add_particle(
        Particle::new(PARTICLE_NAMES[1].to_string(), 1, model.mass.n[0]),
        InitialCondition::Equilibrium(0.0),
    );
    solver.add_particle(
        Particle::new(PARTICLE_NAMES[2].to_string(), 1, model.mass.n[1]),
        InitialCondition::Equilibrium(0.0),
    );
    solver.add_particle(
        Particle::new(PARTICLE_NAMES[3].to_string(), 1, model.mass.n[2]),
        InitialCondition::Equilibrium(0.0),
    );

    // Logging of number densities
    ////////////////////////////////////////////////////////////////////////////////
    let csv = RefCell::new(csv::Writer::from_path("/tmp/minimal_leptogenesis/n.csv").unwrap());

    {
        let mut csv = csv.borrow_mut();
        csv.write_field("step").unwrap();
        csv.write_field("beta").unwrap();

        for name in &PARTICLE_NAMES {
            csv.write_field(name).unwrap();
            csv.write_field(format!("({})", name)).unwrap();
            csv.write_field(format!("Δ{}", name)).unwrap();
        }

        csv.write_record(None::<&[u8]>).unwrap();
    }

    solver.set_logger(move |n, dn, c| {
        // if BETA_RANGE.0 < c.beta && c.beta < BETA_RANGE.1 {
        let mut csv = csv.borrow_mut();
        csv.write_field(format!("{}", c.step)).unwrap();
        csv.write_field(format!("{:.15e}", c.beta)).unwrap();

        for i in 0..n.len() {
            csv.write_field(format!("{:.3e}", n[i])).unwrap();
            csv.write_field(format!("{:.3e}", c.eq_n[i])).unwrap();
            csv.write_field(format!("{:.3e}", dn[i])).unwrap();
        }

        csv.write_record(None::<&[u8]>).unwrap();
    });

    // Interactions
    ////////////////////////////////////////////////////////////////////////////////

    interaction_n_el_h(&mut solver);
    interaction_n_el_ql_qr(&mut solver);

    // Run solver
    ////////////////////////////////////////////////////////////////////////////////

    let sol = solver.solve(&universe);

    assert!(1e-10 < sol[0].abs() && sol[0].abs() < 1e-5);
    assert!(sol[1] < 1e-20);
}

/// Interaction N ↔ LH
fn interaction_n_el_h(solver: &mut NumberDensitySolver<VanillaLeptogenesisModel>) {
    let csv = RefCell::new(csv::Writer::from_path("/tmp/minimal_leptogenesis/decay.csv").unwrap());
    csv.borrow_mut()
        .serialize(["step", "beta", "γ̃", "N₁ → HL", "HL → N₁"])
        .unwrap();

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
        let decay = n[1] * gamma_tilde;
        let inverse_decay = c.eq_n[1] * gamma_tilde;
        let net_decay = (n[1] - c.eq_n[1]) * gamma_tilde;

        s[0] += -c.model.epsilon * net_decay - n[0] * inverse_decay;
        s[1] -= net_decay;

        {
            let mut csv = csv.borrow_mut();
            csv.write_field(format!("{}", c.step)).unwrap();
            csv.write_field(format!("{:.15e}", c.beta)).unwrap();
            csv.write_record(&[
                format!("{:.3e}", gamma_tilde),
                format!("{:.3e}", decay),
                format!("{:.3e}", inverse_decay),
            ])
            .unwrap();
        }

        s
    });
}

/// Scattering NL ↔ Qq, NQ ↔ Lq and Nq ↔ LQ (s- and t-channel)
fn interaction_n_el_ql_qr(solver: &mut NumberDensitySolver<VanillaLeptogenesisModel>) {
    let csv = RefCell::new(
        csv::Writer::from_path("/tmp/minimal_leptogenesis/scattering_NLQq.csv").unwrap(),
    );
    csv.borrow_mut()
        .serialize(["step", "beta", "N₁Q → Lq", "Lq → N₁Q"])
        .unwrap();

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

        let forward = checked_div(n[1], c.eq_n[1]) * gamma;
        let backward = gamma;
        let net_forward = (checked_div(n[1], c.eq_n[1]) - 1.0) * gamma;

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
