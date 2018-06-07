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
//!   \pfrac{f_i}{t} = \frac{\int \vt C[f] \dd \Omega}{4 \pi E_i}
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

use ndarray::prelude::*;
use statistic::Statistic::{BoseEinstein, FermiDirac};
use std::f64;
use universe::Universe;

/// Species
pub struct Species {
    spin: u8,
    mass: f64,
}

impl Species {
    /// Create a new species with the specified spin and mass.
    pub fn new(spin: u8, mass: f64) -> Self {
        Self { spin, mass }
    }
}

/// The solver holding all the information.
#[allow(dead_code)]
pub struct PhaseSpaceSolver {
    beta_range: (f64, f64),
    species: Vec<Species>,
    energies: Array1<f64>,
    energy_step_size: f64,
}

impl PhaseSpaceSolver {
    /// Create a new instance of the phase space solver.
    ///
    /// The default range of temperatures span 1 GeV through to 10^{20} GeV.
    pub fn new() -> Self {
        Self {
            beta_range: (1e-20, 1e0),
            species: Vec::new(),
            energies: Array1::zeros(0),
            energy_step_size: 0.0,
        }
    }

    /// Set the range of inverse temperature values over which the phase space
    /// is evolved.
    ///
    /// Inverse temperature must be provided in units of GeV^{-1}.
    ///
    /// This function has a convenience alternative called
    /// [`PhaseSpaceSolver::temperature_range`] allowing for the limits to be
    /// specified as temperature in the units of GeV.
    ///
    /// # Panics
    ///
    /// Panics if the starting value is larger than the final value.
    pub fn beta_range(mut self, start: f64, end: f64) -> Self {
        assert!(
            start < end,
            "The initial β must be smaller than the final β value."
        );
        self.beta_range = (start, end);
        self
    }

    /// Set the range of temperature values over which the phase space is
    /// evolved.
    ///
    /// Temperature must be provided in units of GeV.
    ///
    /// This function is a convenience alternative to
    /// [`PhaseSpaceSolver::beta_range`].
    ///
    /// # Panics
    ///
    /// Panics if the starting value is smaller than the final value.
    pub fn temperature_range(mut self, start: f64, end: f64) -> Self {
        assert!(
            start > end,
            "The initial temperature must be larger than the final temperature."
        );
        self.beta_range = (start.recip(), end.recip());
        self
    }

    /// Specify the number of energy steps to use in the energy lattice.
    ///
    /// By default, this is set to 2048 steps.
    pub fn energy_steps(mut self, energy_steps: usize) -> Self {
        self.energies = Array1::linspace(0.0, self.beta_range.0.recip(), energy_steps);
        self.energy_step_size = self.energies[1] - self.energies[0];
        self
    }

    /// Add a particle species.
    ///
    /// The initial conditions for this particle are generated assuming the
    /// particle to be in thermal and chemical equilibrium at the initial
    /// temperature.
    ///
    /// If the particle and anti-particle are to be treated separately, the two
    /// species have to be added.
    pub fn add_species(&mut self, s: Species) -> &mut Self {
        self.species.push(s);
        self
    }

    /// Create an array containing the initial conditions for all the particle
    /// species.
    ///
    /// All particles species are assumed to be in thermal equilibrium at this
    /// energy, with the distribution following either the Bose–Einstein or
    /// Fermi–Dirac distribution as determined by their spin.
    fn create_initial_conditions(&self) -> Array2<f64> {
        Array2::from_shape_fn((self.species.len(), self.energies.dim()), |(si, ei)| {
            let s = &self.species[si];
            let e = self.energies[ei];

            if s.spin % 2 == 0 {
                BoseEinstein.phase_space(e, s.mass, 0.0, self.beta_range.0)
            } else {
                FermiDirac.phase_space(e, s.mass, 0.0, self.beta_range.0)
            }
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

    /// Evolve the initial conditions by solving the PDEs.
    ///
    /// Note that this is not a true PDE solver, but instead converts the PDE
    /// into an ODE in time (or inverse temperature in this case), with the
    /// derivative in energy being calculated solely from the previous
    /// time-step.
    pub fn solve<U>(&self, universe: &U) -> Array2<f64>
    where
        U: Universe,
    {
        let mut y = self.create_initial_conditions();
        for i in 0..y.shape()[0] {
            println!("{:>8} : {:>8.2e}", i, y.slice(s![i, ..]));
        }

        // Since the factor of (E² - m²) / E is constant, pre-compute it here once
        let ei_m_on_ei = Array2::from_shape_fn(y.dim(), |(si, ei)| {
            let e = self.energies[ei];
            if e == 0.0 {
                0.0
            } else {
                (e.powi(2) - self.species[si].mass.powi(2)) / e.powi(2)
            }
        });

        let mut beta = self.beta_range.0;
        let mut h = beta / 5.0;

        while beta < self.beta_range.1 {
            println!("beta = {:.3e}", beta);
            println!("{:>10} = {:.3e}", "H", universe.hubble_rate(beta));
            // Standard Runge-Kutta integration
            let k1 = self.derivative(&y) * &ei_m_on_ei * universe.hubble_rate(beta);
            let k2 = self.derivative(&(&y + &(&k1 * 0.5))) * &ei_m_on_ei
                * universe.hubble_rate(beta + 0.5 * h);
            let k3 = self.derivative(&(&y + &(&k2 * 0.5))) * &ei_m_on_ei
                * universe.hubble_rate(beta + 0.5 * h);
            let k4 = self.derivative(&(&y + &k3)) * &ei_m_on_ei * universe.hubble_rate(beta + h);
            // println!("k1 = {:.1e}", k1);
            // println!("k2 = {:.1e}", k2);
            // println!("k3 = {:.1e}", k3);
            // println!("k4 = {:.1e}", k4);

            let dy = (k2 * 2.0 + k3 * 3.0 + k4 + &k1) * h / 6.0;
            println!("{:>10} = {:.1e}", "dy", dy);
            y += &dy;
            println!("{:>10} = {:.1e}", "y", y);

            // Check the error on the RK method vs the Euler method.  If it is small enough, increase the step size.
            let err = ((dy - k1 * h) / &y).fold(0.0, |err, &v| {
                if v.is_finite() {
                    err + v.abs()
                } else {
                    err
                }
            }) / (y.len() as f64);
            println!("{:>10} = {:.3e}", "err", err);
            if err < 0.005 {
                h *= 1.05;
                continue;
            } else if err > 0.01 {
                h *= 0.95
            }
            beta += h;
        }

        y
    }
}

impl Default for PhaseSpaceSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use universe::StandardModel;

    #[test]
    fn no_interaction() {
        let phi = Species::new(0, 5.0);
        let mut solver = PhaseSpaceSolver::new()
            .energy_steps(16)
            .beta_range(1e-20, 1.0);
        solver.add_species(phi);

        println!("Energies : {:>8.2e}", solver.energies);
        for _ in 0..173 {
            print!("=");
        }
        println!("");

        let sol = solver.solve(&StandardModel::new());

        for i in 0..sol.shape()[0] {
            println!("{:>8} : {:>8.2e}", i, sol.slice(s![i, ..]));
        }
    }
}
