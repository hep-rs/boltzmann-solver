use constants::STANDARD_MODEL_GSTAR;
use special_functions::interpolation;
use universe::Universe;

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
mod test {
    use super::StandardModel;
    use universe::Universe;
    use utilities::test::*;

    #[test]
    fn dof() {
        let sm = StandardModel::new();
        let expected = [
            (0.0001, 106.75),
            (0.000251189, 106.748),
            (0.000630957, 106.727),
            (0.00158489, 106.607),
            (0.00398107, 105.846),
            (0.01, 102.85),
            (0.0251189, 94.4434),
            (0.0630957, 87.4654),
            (0.158489, 85.6251),
            (0.398107, 82.6879),
            (1., 75.5),
            (2.51189, 66.8536),
            (6.30957, 27.0902),
            (15.8489, 15.404),
            (39.8107, 12.0135),
            (100., 10.76),
            (251.189, 10.7301),
            (630.957, 10.653),
            (1584.89, 10.2079),
            (3981.07, 8.16679),
            (10000., 4.78),
        ];

        for &(beta, g) in &expected {
            approx_eq(g, sm.entropy_dof(beta), 5.0, 0.0);
        }
    }

    #[test]
    fn hubble() {
        let sm = StandardModel::new();
        let expected = [
            (0.0001, 1.40491e-10),
            (0.000251189, 2.22661e-11),
            (0.000630957, 3.52859e-12),
            (0.00158489, 5.5893e-13),
            (0.00398107, 8.82679e-14),
            (0.01, 1.37901e-14),
            (0.0251189, 2.09436e-15),
            (0.0630957, 3.19435e-16),
            (0.158489, 5.00916e-17),
            (0.398107, 7.80164e-18),
            (1., 1.18151e-18),
            (2.51189, 1.76209e-19),
            (6.30957, 1.77775e-20),
            (15.8489, 2.12462e-21),
            (39.8107, 2.97372e-22),
            (100., 4.46037e-23),
            (251.189, 7.05937e-24),
            (630.957, 1.11481e-24),
            (1584.89, 1.72955e-25),
            (3981.07, 2.45183e-26),
            (10000., 2.97289e-27),
        ];

        for &(beta, h) in &expected {
            approx_eq(h, sm.hubble_rate(beta), 4.0, 0.0);
        }
    }
}
