// use common::{constants::{BOSON_GSTAR, FERMION_GSTAR, STANDARD_MODEL_GSTAR}, universe::Universe};
use common::{constants::STANDARD_MODEL_GSTAR, universe::Universe};
use utilities::interpolation;

/// Implementation of [`Universe`] for the Standard Model.
///
/// At this stage, this only implements the evolution of the matter-dominated
/// epoch of the Universe, assuming only Standard Model contributions.  In
/// particular, it implements the evolution of \\(g_{*}\\) as described in [*On
/// Effective Degrees of Freedom in the Early Universe* by Lars
/// Husdal](https://arxiv.org/abs/1609.04979).
///
/// Due to the intricacies of the transition from the quark--gluon plasma to
/// hadrons, none of the individual particles of the Standard Model are included
/// here.  In the future, the interactions from the leptons, the Higgs doublet
/// and electromagnetic and weak bosons might be included.
#[derive(Debug, Default)]
pub struct StandardModel;

impl StandardModel {
    pub fn new() -> Self {
        StandardModel::default()
    }
}

impl Universe for StandardModel {
    fn entropy_dof(&self, beta: f64) -> f64 {
        interpolation::linear(&STANDARD_MODEL_GSTAR, beta.ln())
    }
}

#[cfg(test)]
mod test {
    use super::StandardModel;
    use common::universe::Universe;
    use utilities::test::*;

    #[test]
    fn sm_plain() {
        let sm = StandardModel::default();
        let expected = [
            (10000., 4.78),
            (3981.0717055349724, 8.166790484506452),
            (1584.8931924611136, 10.20787562194194),
            (630.9573444801933, 10.653013986656847),
            (251.188643150958, 10.730051766378928),
            (100., 10.76),
            (39.81071705534972, 12.013526930417258),
            (15.848931924611136, 15.404034450702763),
            (6.309573444801933, 27.09018326257217),
            (2.51188643150958, 66.85356849949996),
            (1., 75.5),
            (0.3981071705534972, 82.68787985068066),
            (0.15848931924611134, 85.62506692957712),
            (0.06309573444801933, 87.46541509414472),
            (0.0251188643150958, 94.4433579979802),
            (0.01, 102.85),
            (0.003981071705534972, 105.84627054850046),
            (0.0015848931924611134, 106.60698601334316),
            (0.0006309573444801933, 106.72657542475909),
            (0.000251188643150958, 106.74751294159474),
            (0.0001, 106.75),
        ];

        for &(beta, g) in &expected {
            approx_eq(g, sm.entropy_dof(beta), 10.0, 0.0);
        }
    }
}
