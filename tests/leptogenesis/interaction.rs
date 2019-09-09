//! Define the various interactions involved in leptogenesis.

use super::model::{p_i, LeptogenesisModel};
use boltzmann_solver::{
    constants::PI_3,
    solver::number_density::{Interacting::*, Interaction, SolverBuilder},
    // utilities::{four, integrate_st, three},
};
use special_functions::bessel;

/// Interaction H ↔ -L(i2), N(i3)
pub fn n_el_h(solver: &mut SolverBuilder<LeptogenesisModel>) {
    solver.add_interaction(|n, ref c| {
        let mut interactions = Vec::with_capacity(27);

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

            // Amplitude squared and phase space.  Factor of 2 to account for
            // both SU(2) processes
            let m2 = 2.0
                * (0..3)
                    .map(|i2| coupling.y_v[[i3, i2]].norm_sqr())
                    .sum::<f64>()
                * (mass2.h - mass2.n[i3]);
            let phase_space = max_m * bessel::k1(max_m * c.beta) / (32.0 * PI_3 * c.beta);
            let gamma = m2.abs() * phase_space;

            // Add this interaction to the list of collections
            interactions.push(
                Interaction::new(
                    vec![
                        Initial(p1),
                        Final(p2),
                        Other(0, c.model.epsilon, 0.0, 0.0, -n[0]),
                    ],
                    gamma,
                )
                .decay(m2.abs()),
            );
        }

        interactions
    });
}
