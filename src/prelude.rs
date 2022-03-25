//! Common imports for this crate.

pub use crate::{
    model::{
        interaction, interaction::Interaction, standard::data as sm_data, Model, ModelInteractions,
        ParticleData, Propagator, Standard as StandardModel, DIRAC_SPINOR, LEFT_WEYL_SPINOR,
        RIGHT_WEYL_SPINOR, SCALAR, TENSOR, VECTOR,
    },
    solver::{Context, Solver, SolverBuilder},
    statistic::Statistics,
};
