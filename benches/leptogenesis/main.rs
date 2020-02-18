#![cfg(feature = "nightly")]
#![cfg_attr(feature = "nightly", feature(test))]

extern crate test;

mod model;

use crate::model::{interaction, LeptogenesisModel};
use boltzmann_solver::prelude::*;
// use ndarray::prelude::*;

#[cfg(feature = "nightly")]
use test::{black_box, Bencher};

/// Box an interaction
#[cfg(feature = "parallel")]
fn into_interaction_box<I, M>(interaction: I) -> Box<dyn Interaction<M> + Sync>
where
    I: Interaction<M> + Sync + 'static,
    M: Model,
{
    Box::new(interaction)
}

/// Box an interaction
#[cfg(not(feature = "parallel"))]
fn into_interaction_box<I, M>(interaction: I) -> Box<dyn Interaction<M>>
where
    I: Interaction<M> + 'static,
    M: Model,
{
    Box::new(interaction)
}

/// Filter iteraction based whether they involve first-generation particles only
/// or not.
fn one_generation<I, M>(interaction: &I) -> bool
where
    I: Interaction<M>,
    M: Model,
{
    let ptcl = interaction.particles_idx();
    ptcl.incoming.iter().chain(&ptcl.outgoing).all(|i| match i {
        1 | 2 | 3 | 4 | 5 | 8 | 11 | 14 | 17 | 20 => true,
        _ => false,
    })
}

#[bench]
pub fn decay_only_1gen(b: &mut Bencher) {
    b.iter(|| {
        let mut model = LeptogenesisModel::zero();
        model.interactions.extend(
            interaction::hln()
                .drain(..)
                .filter(one_generation)
                .map(into_interaction_box),
        );
        let no_asymmetry: Vec<usize> = (0..3)
            .map(|i| LeptogenesisModel::particle_idx("N", i).unwrap())
            .collect();

        let builder = SolverBuilder::new()
            .no_asymmetry(no_asymmetry)
            .model(model)
            .beta_range(1e-17, 1e-3)
            .step_precision(1e-6, 1e-1);

        let mut solver = builder.build().unwrap();
        black_box(solver.solve());
    })
}

#[bench]
pub fn decay_only_3gen(b: &mut Bencher) {
    b.iter(|| {
        let mut model = LeptogenesisModel::zero();
        model
            .interactions
            .extend(interaction::hln().drain(..).map(into_interaction_box));
        let no_asymmetry: Vec<usize> = (0..3)
            .map(|i| LeptogenesisModel::particle_idx("N", i).unwrap())
            .collect();

        let builder = SolverBuilder::new()
            .no_asymmetry(no_asymmetry)
            .model(model)
            .beta_range(1e-17, 1e-3)
            .step_precision(1e-6, 1e-1);

        let mut solver = builder.build().unwrap();
        black_box(solver.solve());
    })
}
