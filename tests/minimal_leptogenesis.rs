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
    solver::{number_density::NumberDensitySolver, InitialCondition, Model, Solver},
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

struct Masses {
    n: f64,
    h: f64,
}

impl Masses {
    fn new(beta: f64) -> Self {
        Masses {
            n: 1e10,
            h: 0.4 / beta,
        }
    }
}

struct SquaredMasses {
    n: f64,
    h: f64,
}

impl SquaredMasses {
    fn new(beta: f64) -> Self {
        let m = Masses::new(beta);
        SquaredMasses {
            n: m.n.powi(2),
            h: m.h.powi(2),
        }
    }
}

struct Widths {
    // n: f64,
    h: f64,
}

impl Widths {
    fn new(beta: f64) -> Self {
        let m = Masses::new(beta);
        Widths {
            // n: 0.0,
            h: 0.1 * m.h,
        }
    }
}

struct SquaredWidths {
    // n: f64,
    h: f64,
}

impl SquaredWidths {
    fn new(beta: f64) -> Self {
        let w = Widths::new(beta);
        SquaredWidths {
            // n: w.n.powi(2),
            h: w.h.powi(2),
        }
    }
}

#[allow(non_snake_case)]
struct LeptogenesisModel {
    mass: Masses,
    mass2: SquaredMasses,
    // width: Widths,
    width2: SquaredWidths,
    y_HQu: Array2<Complex<f64>>,
    y_HQd: Array2<Complex<f64>>,
    y_HNL: Array2<Complex<f64>>,
    epsilon: f64,
}

impl Model for LeptogenesisModel {
    fn new(beta: f64) -> Self {
        LeptogenesisModel {
            mass: Masses::new(beta),
            mass2: SquaredMasses::new(beta),
            // width: Widths::new(beta),
            width2: SquaredWidths::new(beta),
            y_HQu: array![
                [Complex::new(172.200, 0.0), zero(), zero()],
                [zero(), Complex::new(95e-3, 0.0), zero()],
                [zero(), zero(), Complex::new(2.2e-3, 0.0)],
            ] * 2.0f64.sqrt()
                / 246.0,
            y_HQd: array![
                [Complex::new(4.2, 0.0), zero(), zero()],
                [zero(), Complex::new(1.25, 0.0), zero()],
                [zero(), zero(), Complex::new(5e-3, 0.0)],
            ] * 2.0f64.sqrt()
                / 246.0,
            y_HNL: array![[
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

    let model = LeptogenesisModel::new(1e-15);

    // Set up the universe in which we'll run the Boltzmann equations
    let universe = StandardModel::new();

    // Initialize particles we want to keep track of.  All other particles
    // are assumed to be always in equilibrium.
    let b_minus_l = Particle::new(0, 0.0).set_dof(0.0);
    let n = Particle::new(1, model.mass.n);

    // Create the Solver and set integration parameters
    let mut solver: NumberDensitySolver<LeptogenesisModel> = NumberDensitySolver::new()
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

    solver.add_interaction(move |mut s, n, ref c| {
        // Note that this is *not* exactly gamma and differs in the
        // normalization.  It actually just needs to be multiplied by n for
        // the decay, and eq_n for the inverse decay.
        let gamma_tilde = {
            let mut m2 = 0.0;
            for b in 0..3 {
                m2 += c.model.y_HNL[[0, b]].norm_sqr();
            }
            m2 *= (c.model.mass2.n - c.model.mass2.h) * bessel::k_1_on_k_2(c.model.mass.n * c.beta);
            m2 /= c.hubble_rate * c.beta * 16.0 * PI_1 * c.model.mass.n;
            m2
        };

        // The full γ(e H → N).  We are assuming e and H are always in
        // equilibrium thus their scaling factors are irrelevant.
        let decay = n[1] * gamma_tilde;
        let inverse_decay = c.eq_n[1] * gamma_tilde;
        let net_decay = decay - inverse_decay;

        s[0] += -c.model.epsilon * net_decay - n[0] * inverse_decay;
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

    solver.add_interaction(move |mut s, n, ref c| {
        let mut gamma = {
            let mut m2 = 0.0;
            for (a2, b1, b2) in iproduct!(0..3, 0..3, 0..3) {
                m2 += 9.0
                    * c.model.y_HNL[[0, a2]].norm_sqr()
                    * (c.model.y_HQd[[b1, b2]].norm_sqr() + c.model.y_HQu[[b1, b2]].norm_sqr());
            }

            let s_integrand = |ss: f64| {
                let s = c.model.mass2.n + (1.0 - ss) / ss;
                let dsdss = ss.powi(-2);
                let sqrt_s = s.sqrt();

                let t_integrand = |t: f64| {
                    (
                        // s-channel
                        s * (s - c.model.mass2.n) * (s - c.model.mass2.h).powi(2)
                            / ((s - c.model.mass2.h).powi(2) + c.model.width2.h * c.model.mass2.h)
                                .powi(2)
                    ) + (
                        // t-channel
                        2.0 * (t + c.model.mass2.n) * (t + c.model.mass2.h).powi(2)
                            / ((t + c.model.mass2.h).powi(2) + c.model.width2.h * c.model.mass2.h)
                                .powi(2)
                    )
                };

                integrate(t_integrand, c.model.mass2.n - s, 0.0, 0.0).integral
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
