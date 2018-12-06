extern crate boltzmann_solver;
extern crate csv;
extern crate itertools;
extern crate ndarray;
extern crate num;
extern crate quadrature;
extern crate rgsl;
extern crate special_functions;

use boltzmann_solver::{
    constants::{PI_1, PI_5},
    particle::Particle,
    solver::{number_density::NumberDensitySolver, InitialCondition, Solver},
    universe::StandardModel,
};
use itertools::iproduct;
use ndarray::{array, prelude::*};
use num::{complex::Complex, zero};
use quadrature::integrate;
use special_functions::bessel;
use std::cell::RefCell;

fn checked_div(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        0.0
    } else if b == 0.0 {
        1.0
    } else {
        a / b
    }
}

#[allow(non_snake_case)]
#[derive(Clone)]
struct Couplings {
    yukawa_HQu: Array2<Complex<f64>>,
    yukawa_HQd: Array2<Complex<f64>>,
    yukawa_HNL: Array2<Complex<f64>>,
    epsilon: f64,
}

impl Couplings {
    fn new() -> Self {
        Couplings {
            yukawa_HQu: array![
                [Complex::new(172.200, 0.0), zero(), zero()],
                [zero(), Complex::new(95e-3, 0.0), zero()],
                [zero(), zero(), Complex::new(2.2e-3, 0.0)],
            ] * 2.0f64.sqrt()
                / 246.0,
            yukawa_HQd: array![
                [Complex::new(4.2, 0.0), zero(), zero()],
                [zero(), Complex::new(1.25, 0.0), zero()],
                [zero(), zero(), Complex::new(5e-3, 0.0)],
            ] * 2.0f64.sqrt()
                / 246.0,
            yukawa_HNL: array![[
                Complex::new(1e-4, 0.0),
                Complex::new(1e-4, 1e-5),
                Complex::new(1e-4, 1e-5)
            ]],
            epsilon: 1e-6,
        }
    }
}

#[test]
fn minimal_leptogenesis() {
    // Setup the directory for CSV output
    ::std::fs::create_dir("/tmp/minimal_leptogenesis/").unwrap_or(());

    // Set up the universe in which we'll run the Boltzmann equations
    let universe = StandardModel::new();

    // Initialize particles we want to keep track of.  All other particles
    // are assumed to be always in equilibrium.
    let b_minus_l = Particle::new(0, 0.0).set_dof(0.0);
    let n = Particle::new(1, 1e10);

    // Thermal correction to the Higgs mass is given by:
    //    Π = (3 / 16) (g2)² T² + (1 / 16) (g')² T²
    let mass2_h = |beta: f64| 0.16 * beta.powi(-2);
    // RH neutrino mass
    let mass_n = n.mass;
    let mass2_n = mass_n.powi(2);

    let width2_h = |beta: f64| 0.01 * (0.16 * beta.powi(-2));

    // Create the Solver and set integration parameters
    let mut solver = NumberDensitySolver::new()
        .beta_range(1e-14, 1e0)
        .error_tolerance(1e-1, 1e-2)
        .initialize();

    // Add the particles to the solver, using for initial condition either 0 or
    // equilibrium number density.
    solver.add_particle(b_minus_l, InitialCondition::Zero);
    solver.add_particle(n, InitialCondition::Zero);
    // solver.add_particle(n, InitialCondition::Equilibrium(0.0));

    // Interaction N ↔ LH
    ////////////////////////////////////////////////////////////////////////////////
    let csv = RefCell::new(csv::Writer::from_path("/tmp/minimal_leptogenesis/decay.csv").unwrap());
    csv.borrow_mut()
        .serialize(("beta", "γ̃", "N₁ → HL", "HL → N₁"))
        .unwrap();

    let couplings = Couplings::new();
    solver.add_interaction(move |mut s, n, ref c| {
        // Note that this is *not* exactly gamma and differs in the
        // normalization.  It actually just needs to be multiplied by n for
        // the decay, and eq_n for the inverse decay.
        let gamma_tilde = {
            let mut m2 = 0.0;
            for b in 0..3 {
                m2 += couplings.yukawa_HNL[[0, b]].norm_sqr();
            }
            m2 *= (mass2_n - mass2_h(c.beta)) * bessel::k_1_on_k_2(mass_n * c.beta);
            m2 /= c.hubble_rate * c.beta * 16.0 * PI_1 * mass_n;
            m2
        };

        // The full γ(e H → N).  We are assuming e and H are always in
        // equilibrium thus their scaling factors are irrelevant.
        let decay = n[1] * gamma_tilde;
        let inverse_decay = c.eq_n[1] * gamma_tilde;
        let net_decay = decay - inverse_decay;

        s[0] += -couplings.epsilon * net_decay - n[0] * inverse_decay;
        s[1] -= net_decay;

        csv.borrow_mut()
            .serialize((c.beta, gamma_tilde, decay, inverse_decay))
            .unwrap();

        s
    });

    // Scattering NL ↔ Qq, NQ ↔ Lq and Nq ↔ LQ (s- and t-channel)
    ////////////////////////////////////////////////////////////////////////////////
    let csv = RefCell::new(
        csv::Writer::from_path("/tmp/minimal_leptogenesis/scattering_NLQq.csv").unwrap(),
    );
    csv.borrow_mut()
        .serialize(("beta", "N₁Q → Lq", "Lq → N₁Q"))
        .unwrap();

    let couplings = Couplings::new();
    solver.add_interaction(move |mut s, n, ref c| {
        let mut gamma = {
            let mass2_h = mass2_h(c.beta);
            let width2_h = width2_h(c.beta);

            let mut m2 = 0.0;
            for (a2, b1, b2) in iproduct!(0..3, 0..3, 0..3) {
                m2 += 9.0
                    * couplings.yukawa_HNL[[0, a2]].norm_sqr()
                    * (couplings.yukawa_HQd[[b1, b2]].norm_sqr()
                        + couplings.yukawa_HQu[[b1, b2]].norm_sqr());
            }

            let s_integrand = |ss: f64| {
                let s = mass2_n + (1.0 - ss) / ss;
                let dsdss = ss.powi(-2);
                let sqrt_s = s.sqrt();

                let t_integrand = |t: f64| {
                    ( // s-channel
                        s * (s - mass2_n) * (s - mass2_h).powi(2)
                            / ((s - mass2_h).powi(2) + width2_h * mass2_h).powi(2)
                    ) + ( // t-channel
                          2.0 * (t + mass2_n) * (t + mass2_h).powi(2)
                              / ((t + mass2_h).powi(2) + width2_h * mass2_h).powi(2)
                      )
                };

                integrate(t_integrand, mass2_n - s, 0.0, 0.0).integral
                    * bessel::k_1(sqrt_s * c.beta)
                    / sqrt_s
                    * dsdss
            };
            m2 * integrate(s_integrand, 0.0, 1.0, 0.0).integral
        };
        gamma /= 512.0 * PI_5 * c.hubble_rate * c.beta;

        let forward = checked_div(n[1], c.eq_n[1]) * gamma;
        let backward = gamma;
        let net_forward = forward - backward;

        s[0] -= n[0] * backward;
        s[1] -= net_forward;

        csv.borrow_mut()
            .serialize((c.beta, forward, backward))
            .unwrap();

        s
    });

    // Logging of number densities
    ////////////////////////////////////////////////////////////////////////////////
    let logger = RefCell::new(csv::Writer::from_path("/tmp/minimal_leptogenesis/n.csv").unwrap());
    logger
        .borrow_mut()
        .serialize(("beta", "B-L", "N₁", "(N₁)"))
        .unwrap();

    solver.set_logger(move |n, c| {
        logger
            .borrow_mut()
            .serialize((c.beta, n[0], n[1], c.eq_n[1]))
            .unwrap();
    });

    let sol = solver.solve(&universe);

    assert!(1e-10 < sol[0].abs() && sol[0].abs() < 1e-5);
    assert!(sol[1] < 1e-20);
}
