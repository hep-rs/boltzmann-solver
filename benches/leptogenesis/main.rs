#![cfg(feature = "nightly")]
#![cfg_attr(feature = "nightly", feature(test))]

extern crate test;

mod model;

use crate::model::{interaction, LeptogenesisModel};
use boltzmann_solver::prelude::*;
// use ndarray::prelude::*;

#[cfg(feature = "nightly")]
use test::{black_box, Bencher};

#[bench]
pub fn decay_only_1gen(b: &mut Bencher) {
    b.iter(|| {
        let mut model = LeptogenesisModel::zero();
        model
            .interactions
            .push(Box::new(interaction::hln().remove(0)));
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
        model.interactions.extend(
            interaction::hln()
                .drain(..)
                .map(|i| Box::new(i) as Box<dyn Interaction<LeptogenesisModel> + Sync>),
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
