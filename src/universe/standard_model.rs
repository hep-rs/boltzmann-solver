use crate::{constants::STANDARD_MODEL_GSTAR, universe::Universe};
use special_functions::approximations::interpolation;

/// Implementation of [`Universe`] for the Standard Model.
///
/// At this stage, this only implements the evolution of the matter-dominated
/// epoch of the Universe, assuming only Standard Model contributions.  In
/// particular, it implements the evolution of \\(g_{*}\\) as described in [*On
/// Effective Degrees of Freedom in the Early Universe* by Lars
/// Husdal](https://arxiv.org/abs/1609.04979).
///
/// Due to the intricacies of the transition from the quark–gluon plasma to
/// hadrons, none of the individual particles of the Standard Model are included
/// here.  In the future, the interactions from the leptons, the Higgs doublet
/// and electromagnetic and weak bosons might be included.
#[derive(Debug, Default, Clone, Copy)]
pub struct StandardModel;

impl StandardModel {
    /// Create an instance of the Standard Model.
    pub fn new() -> Self {
        StandardModel::default()
    }
}

impl Universe for StandardModel {
    fn entropy_dof(&self, beta: f64) -> f64 {
        debug_assert!(beta > 0.0, "beta must be positive.");
        interpolation::linear(&STANDARD_MODEL_GSTAR, beta.ln())
    }
}

#[cfg(test)]
mod tests {
    use super::StandardModel;
    use crate::universe::Universe;
    use crate::utilities::test::*;

    #[test]
    fn dof() {
        let sm = StandardModel::new();
        let expected = [
            (0.000_1, 106.75),
            (0.000_251_189, 106.748),
            (0.000_630_957, 106.727),
            (0.001_584_89, 106.607),
            (0.003_981_07, 105.846),
            (0.01, 102.85),
            (0.025_118_9, 94.4434),
            (0.063_095_7, 87.4654),
            (0.158_489, 85.6251),
            (0.398_107, 82.6879),
            (1., 75.5),
            (2.511_89, 66.8536),
            (6.309_57, 27.0902),
            (15.848_9, 15.404),
            (39.810_7, 12.013_5),
            (100., 10.76),
            (251.189, 10.730_1),
            (630.957, 10.653),
            (1_584.89, 10.207_9),
            (3_981.07, 8.166_79),
            (10_000., 4.78),
        ];

        for &(beta, g) in &expected {
            approx_eq(g, sm.entropy_dof(beta), 5.0, 0.0);
        }
    }

    #[test]
    fn hubble() {
        let sm = StandardModel::new();
        let expected = [
            (0.000_1, 1.404_91e-10),
            (0.000_251_189, 2.226_61e-11),
            (0.000_630_957, 3.528_59e-12),
            (0.001_584_89, 5.589_3e-13),
            (0.003_981_07, 8.826_79e-14),
            (0.01, 1.379_01e-14),
            (0.025_118_9, 2.094_36e-15),
            (0.063_095_7, 3.194_35e-16),
            (0.158_489, 5.009_16e-17),
            (0.398_107, 7.801_64e-18),
            (1., 1.181_51e-18),
            (2.511_89, 1.762_09e-19),
            (6.309_57, 1.777_75e-20),
            (15.848_9, 2.124_62e-21),
            (39.810_7, 2.973_72e-22),
            (100., 4.460_37e-23),
            (251.189, 7.059_37e-24),
            (630.957, 1.114_81e-24),
            (1_584.89, 1.729_55e-25),
            (3_981.07, 2.451_83e-26),
            (10_000., 2.972_89e-27),
        ];

        for &(beta, h) in &expected {
            approx_eq(h, sm.hubble_rate(beta), 4.0, 0.0);
        }
    }
}
