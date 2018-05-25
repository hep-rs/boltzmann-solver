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

/// The solver holding all the information.
#[allow(dead_code)]
pub struct PhaseSpaceSolver {
    initial_conditions: Array2<f64>,
    betas: Array1<f64>,
    energies: Array1<f64>,
}

impl PhaseSpaceSolver {
    /// Create a new instance of the phase space solver.
    ///
    /// The other attributes have to be set before it can be run.
    pub fn new() -> Self {
        Self {
            betas: Array1::zeros(0),
            energies: Array1::zeros(0),
            initial_conditions: Array2::zeros((0, 0)),
        }
    }
}

impl Default for PhaseSpaceSolver {
    fn default() -> Self {
        Self::new()
    }
}
