//! Solver for the phase-space form of the Boltzmann equation.
//!
//! This solver allows for arbitrary phase spaces, though still assumes them to
//! be homogeneous and isotropic such that they only depending on the energy.
//!
//! The solver requires initial conditions to be provided and then evolves them
//! in time.  It discretizes the phase space in the energy-direction,
//! \\(f_{i}(\beta) \defeq f(E_i, \beta)\\), and then solves the differential
//! equation:
//!
//! \\begin{equation}
//!   \pfrac{f_i}{t} = \frac{\int \vt C\[f\] \dd \Omega}{4 \pi E_i}
//!                  + H \frac{E_i^2 - m^2}{E_i} \pfrac{f}{E}
//! \\end{equation}
//!
//! where the derivative with respective to \\(E\\) is done discretely; for example:
//!
//! \\begin{equation}
//! \pfrac{f}{E} \approx \left[ \frac{f_{i-1} - f_i}{E_{i-1} - E_i}
//!                           - \frac{f_{i+1} - f_{i-1}}{E_{i+1} - E_{i-1}}
//!                           + \frac{f_{i+1} - f_i}{E_{i+1} - E_i} \right]
//!              \stackrel{\mathrm{constant} \Delta E}{=} \frac{f_{i+1} - f_{i-1}}{2 \Delta E}.
//! \\end{equation}
//!
//! The collision operator is integrated over the angular components
//! analytically and the remaining energy integrals being done numerically over
//! the lattice:
//!
//! \\begin{align}
//!   \int \vt C[f_{a_{1}}(E_i, t_i)] \dd \Omega_{a_1}
//!     &= - \int \left( \prod_{\vt a \neq a_1, \vt b} \dd \Pi_i \right)
//!               (2 \pi)^4 \delta(p_{\vt a} - p_{\vt b}) \\\\
//!     &\quad \times \Bigl[ \abs{\mathcal M(\vt a | \vt b)}^2 \left( \prod_{\vt a} f_i \right) \left(\prod_{\vt b} 1 \pm f_i \right)
//!                        - \abs{\mathcal M(\vt b | \vt a)}^2 \left( \prod_{\vt b} f_i \right) \left(\prod_{\vt a} 1 \pm f_i \right) \Bigr],
//!     \\\\
//!     &= - \frac{1}{(\Delta E)^n} \sum_{E_{\vt a} \neq E_{a_1}, E_{\vt b}} (2\pi) \delta(E_{\vt a} - E_{\vt b})
//!        \int \left( \prod_{\vt a \neq a_1, \vt b} \dd \Omega_i \right) (2 \pi)^3 \delta(\Omega_{\vt a} - \Omega_{\vt b}) \\\\
//!     &\quad \times \Bigl[ \abs{\mathcal M(\vt a | \vt b)}^2 \left( \prod_{\vt a} f_i \right) \left(\prod_{\vt b} 1 \pm f_i \right)
//!                        - \abs{\mathcal M(\vt b | \vt a)}^2 \left( \prod_{\vt b} f_i \right) \left(\prod_{\vt a} 1 \pm f_i \right) \Bigr],
//! \end{align}
//!
//! where the integrals over \\(\int \dd \Omega_{\vt a, \vt b} (2\pi)^3
//! \delta(\Omega_{\vt a} - \Omega_{\vt b}) \abs{\mathcal M(\vt a | \vt b)}^2\\)
//! must be done analytically (and similarly for \\(\abs{\mathcal M(\vt b | \vt
//! a)}^2\\)) while leaving all dependence on \\(E_i\\) and \\(\abs{\vt p_i}\\)
//! explicit.

use super::{ErrorTolerance, InitialCondition, Solver, StepChange};
use ndarray::{prelude::*, Zip};
use particle::Particle;
use statistic::{
    Statistic::{BoseEinstein, FermiDirac},
    Statistics,
};
use std::f64;
use universe::Universe;

/// Context provided containing pre-computed values which might be useful when
/// evaluating interactions.
pub struct Context {
    /// Inverse temperature in GeV^{-1}
    pub beta: f64,
    /// Hubble rate, in GeV
    pub hubble_rate: f64,
    /// Equilibrium phase spaces for the particles.  The first axis has the
    /// particles and follows the same order as specified to the solver.  The
    /// second axis has the energies
    pub eq_f: Array2<f64>,
    /// Equilibrium phase space for massless bosons.  This is specified per
    /// degree of freedom (that is \\(g = 1\\))
    pub eq_boson: Array1<f64>,
    /// Equilibrium phase space for massless fermions.  This is specified per
    /// degree of freedom (that is \\(g = 1\\))
    pub eq_fermion: Array1<f64>,
}

/// The solver holding all the information.
pub struct PhaseSpaceSolver {
    initialized: bool,
    beta_range: (f64, f64),
    particles: Vec<Particle>,
    #[allow(clippy::type_complexity)]
    interactions: Vec<Box<Fn(Array2<f64>, &Array2<f64>, &Context) -> Array2<f64>>>,
    logger: Box<Fn(&Array2<f64>, &Context)>,
    energies: Array1<f64>,
    energy_steps: usize,
    energy_step_size: f64,
    step_change: StepChange,
    error_tolerance: ErrorTolerance,
}

impl PhaseSpaceSolver {
    /// Specify the number of energy steps to use in the energy lattice.
    ///
    /// By default, this is set to 2048 steps.
    ///
    /// # Panic
    ///
    /// Panics if the energy steps is zero.
    pub fn energy_steps(mut self, energy_steps: usize) -> Self {
        assert!(energy_steps > 0, "Energy steps must be greater than 0.");
        self.energy_steps = energy_steps;
        self
    }
}

impl Solver for PhaseSpaceSolver {
    /// The solution is a two-dimensional array.  The first axis (`Axis(0)`)
    /// corresponds to the particle species and has the same ordering as the
    /// order in which [`Solver::add_particle`] is invoked.  The second axis
    /// (`Axis(1)`) contains the energies.
    type Solution = Array2<f64>;

    type Context = Context;

    fn new() -> Self {
        Self {
            initialized: false,
            beta_range: (1e-20, 1e0),
            particles: Vec::with_capacity(20),
            interactions: Vec::with_capacity(100),
            logger: Box::new(|_, _| {}),
            energies: Array1::zeros(0),
            energy_steps: 2usize.pow(10),
            energy_step_size: 0.0,
            step_change: StepChange {
                increase: 1.1,
                decrease: 0.5,
            },
            error_tolerance: ErrorTolerance {
                upper: 1e-2,
                lower: 1e-5,
            },
        }
    }

    fn beta_range(mut self, start: f64, end: f64) -> Self {
        assert!(
            start < end,
            "The initial β must be smaller than the final β value."
        );
        self.beta_range = (start, end);
        self
    }

    fn temperature_range(mut self, start: f64, end: f64) -> Self {
        assert!(
            start > end,
            "The initial temperature must be larger than the final temperature."
        );
        self.beta_range = (start.recip(), end.recip());
        self
    }

    fn step_change(mut self, increase: f64, decrease: f64) -> Self {
        assert!(
            increase > 1.0,
            "The multiplicative factor to increase the step size must be greater
            than 1."
        );
        assert!(
            decrease < 1.0,
            "The multiplicative factor to decrease the step size must be greater
            than 1."
        );
        self.step_change = StepChange { increase, decrease };
        self
    }

    fn error_tolerance(mut self, upper: f64, lower: f64) -> Self {
        assert!(
            upper > lower,
            "The upper error tolerance must be greater than the lower tolerance"
        );
        self.error_tolerance = ErrorTolerance { upper, lower };
        self
    }

    fn initialize(mut self) -> Self {
        self.energies = Array1::linspace(0.0, 10.0 * self.beta_range.0.recip(), self.energy_steps);
        self.energy_step_size = self.energies[1] - self.energies[0];
        self.initialized = true;

        self
    }

    fn add_particle(&mut self, p: Particle, _initial_condition: InitialCondition) {
        self.particles.push(p);
    }

    fn add_particles<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = Particle>,
    {
        self.particles.extend(iter);
    }

    fn add_interaction<F: 'static>(&mut self, f: F) -> &mut Self
    where
        F: Fn(Self::Solution, &Self::Solution, &Self::Context) -> Self::Solution,
    {
        self.interactions.push(Box::new(f));
        self
    }

    fn set_logger<F: 'static>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&Self::Solution, &Self::Context),
    {
        self.logger = Box::new(f);
        self
    }

    fn solve<U>(&self, universe: &U) -> Self::Solution
    where
        U: Universe,
    {
        assert!(
            self.initialized,
            "The phase space solver has to be initialized first with the `initialize()` method."
        );

        let mut y = self.equilibrium_phase_space(self.beta_range.0);

        // Since the factor of (E² - m²) / E is constant, pre-compute it here once
        let ei_m_on_ei = Array2::from_shape_fn(y.dim(), |(si, ei)| {
            let e = self.energies[ei];
            if e == 0.0 {
                0.0
            } else {
                (e.powi(2) - self.particles[si].mass.powi(2)) / e.powi(2)
            }
        });

        let mut k1: Self::Solution;
        let mut k2: Self::Solution;
        let mut k3: Self::Solution;
        let mut k4: Self::Solution;
        let mut tmp: Self::Solution;

        let mut beta = self.beta_range.0;
        let mut h = beta / 10.0;

        let mut n_eval = 0;
        while beta < self.beta_range.1 {
            n_eval += 1;

            // Standard Runge-Kutta integration.
            let c = self.context(beta, universe);
            k1 = self
                .interactions
                .iter()
                .fold(Self::Solution::zeros(y.dim()), |s, f| f(s, &y, &c));
            Zip::from(&mut k1)
                .and(&self.derivative(&y))
                .and(&ei_m_on_ei)
                .apply(|k, &dy, &de| {
                    *k += dy * de * c.hubble_rate;
                    *k *= h;
                });

            let c = self.context(beta + 0.5 * h, universe);
            tmp = &y + &(&k1 * 0.5);
            k2 = self
                .interactions
                .iter()
                .fold(Self::Solution::zeros(y.dim()), |s, f| f(s, &tmp, &c));
            Zip::from(&mut k2)
                .and(&self.derivative(&tmp))
                .and(&ei_m_on_ei)
                .apply(|k, &dy, &de| {
                    *k += dy * de * c.hubble_rate;
                    *k *= h;
                });

            let c = self.context(beta + 0.5 * h, universe);
            tmp = &y + &(&k2 * 0.5);
            k3 = self
                .interactions
                .iter()
                .fold(Self::Solution::zeros(y.dim()), |s, f| f(s, &tmp, &c));
            Zip::from(&mut k3)
                .and(&self.derivative(&tmp))
                .and(&ei_m_on_ei)
                .apply(|k, &dy, &de| {
                    *k += dy * de * c.hubble_rate;
                    *k *= h;
                });

            let c = self.context(beta + h, universe);
            tmp = &y + &k3;
            k4 = self
                .interactions
                .iter()
                .fold(Self::Solution::zeros(y.dim()), |s, f| f(s, &tmp, &c));
            Zip::from(&mut k4)
                .and(&self.derivative(&tmp))
                .and(&ei_m_on_ei)
                .apply(|k, &dy, &de| {
                    *k += dy * de * c.hubble_rate;
                    *k *= h;
                });

            // Calculate dy.  Note that we consume k2, k3 and k4 here.  We use
            // k1 by reference since we need it later to get the error estimate.
            let dy = (k2 * 2.0 + k3 * 2.0 + k4 + &k1) / 6.0;

            // Check the error on the RK method vs the Euler method.  If it is
            // small enough, increase the step size.  We use the maximum error
            // for any given element of `dy`.
            let err = (k1 / &dy)
                .iter()
                .filter(|v| v.is_finite())
                .map(|v| (v - 1.0).abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);

            // Adjust the step size as needed based on the step size.
            if err < self.error_tolerance.lower {
                h *= self.step_change.increase;
                debug!(
                    "Step {:>7}, β = {:>9.2e} -> Increased h to {:.3e}",
                    n_eval, beta, h
                );
            } else if err > self.error_tolerance.upper {
                h *= self.step_change.decrease;
                debug!(
                    "Step {:>7}, β = {:>9.2e} -> Decreased h to {:.3e}",
                    n_eval, beta, h
                );

                // Prevent h from getting too small that it might make
                // integration take too long.  Use the result regardless even
                // though it is bigger than desired error.
                if beta / h > 1e5 {
                    warn!(
                        "Step {:>7}, β = {:>9.2e} -> Step size getting too small (β / h = {:.1e}).",
                        n_eval, beta, beta / h
                    );

                    while beta / h > 1e5 {
                        h *= self.step_change.increase;
                    }

                    y += &dy;
                    beta += h;
                }
                continue;
            }

            y += &dy;
            beta += h;
        }

        info!("Number of evaluations: {}", n_eval);

        y
    }
}

impl Default for PhaseSpaceSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl PhaseSpaceSolver {
    /// Create an array containing the initial conditions for all the particle
    /// species.
    ///
    /// All particles species are assumed to be in thermal equilibrium at this
    /// energy, with the distribution following either the Bose–Einstein or
    /// Fermi–Dirac distribution as determined by their spin.
    fn equilibrium_phase_space(&self, beta: f64) -> Array2<f64> {
        Array2::from_shape_fn((self.particles.len(), self.energies.dim()), |(si, ei)| {
            let p = &self.particles[si];
            let e = self.energies[ei];
            p.phase_space(e, 0.0, beta)
        })
    }

    /// Compute the energy derivatives.
    ///
    /// At the boundaries, the left- and right-sided derivatives are used.  For
    /// all other cases, a 2-sided derivative is used.
    ///
    /// All NaN and infinite values are excluded from the computation so as to
    /// avoid their propagation to nearby values.
    fn derivative(&self, phase_spaces: &Array2<f64>) -> Array2<f64> {
        Array2::from_shape_fn(phase_spaces.dim(), |(si, ei)| {
            let fm = if ei == 0 {
                &f64::INFINITY
            } else {
                phase_spaces.get([si, ei - 1]).unwrap_or(&f64::INFINITY)
            };
            let f = phase_spaces[[si, ei]];
            let fp = phase_spaces.get([si, ei + 1]).unwrap_or(&f64::INFINITY);

            match (fm.is_finite(), f.is_finite(), fp.is_finite()) {
                (true, _, true) => (fp - fm) / (2.0 * self.energy_step_size),
                (false, true, true) => (fp - f) / self.energy_step_size,
                (true, true, false) => (f - fm) / self.energy_step_size,
                _ => 0.0,
            }
        })
    }

    fn context<U: Universe>(&self, beta: f64, universe: &U) -> Context {
        Context {
            beta,
            hubble_rate: universe.hubble_rate(beta),
            eq_f: self.equilibrium_phase_space(beta),
            eq_boson: self
                .energies
                .map(|&e| BoseEinstein.phase_space(e, 0.0, 0.0, beta)),
            eq_fermion: self
                .energies
                .map(|&e| FermiDirac.phase_space(e, 0.0, 0.0, beta)),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use universe::StandardModel;

    #[test]
    fn no_interaction() {
        let phi = Particle::new(0, 5.0);
        let mut solver = PhaseSpaceSolver::new()
            .temperature_range(1e20, 1e-10)
            .error_tolerance(1e-1, 1e-2)
            .initialize();
        solver.add_particle(phi, InitialCondition::Equilibrium(0.0));

        solver.solve(&StandardModel::new());
    }
}
