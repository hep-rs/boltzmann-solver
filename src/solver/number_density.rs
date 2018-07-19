//! Solver for the number density evolution given by integrating the Boltzmann
//! equation.
//!
//! Solving the Boltzmann equation in generality without making any assumption
//! about the phase space distribution \\(f\\) (other than specifying some
//! initial condition) is a difficult problem.  Furthermore, within the context
//! of baryogenesis and leptogenesis, we are only interested in the number
//! density (or more specifically, the difference in the number densities or a
//! particle and its corresponding antiparticle).
//!
//! # Assumptions
//!
//! A couple of assumptions can be made to simplify the Boltzmann equation so
//! that the number densities can be computed directly.
//!
//! - *Assume kinetic equilibrium.* If the rate at which a particle species is
//!   scattering is sufficiently fast, phase space distribution of this species
//!   will rapidly converge converge onto either the Bose–Einstein or
//!   Fermi–Dirac distributions:
//!
//!   \\begin{equation}
//!     f_{\textsc{BE}} = \frac{1}{\exp[(E - \mu) \beta] - 1}, \qquad
//!     f_{\textsc{FD}} = \frac{1}{\exp[(E - \mu) \beta] + 1}.
//!   \\end{equation}
//!
//!   For a particular that remains in kinetic equilibrium, the evolution of its
//!   phase space is entirely described by the evolution of \\(\mu\\) in time.
//!
//! - *Assume \\(\beta \gg E - \mu\\).* In the limit that the temperature is
//!   much less than \\(E - \mu\\) (or equivalently, that the inverse
//!   temperature \\(\beta\\) is greater than \\(E - \mu\\)), both the
//!   Fermi–Dirac and Bose–Einstein approach the Maxwell–Boltzmann distribution,
//!
//!   \\begin{equation}
//!     f_{\textsc{MB}} = \exp[-(E - \mu) \beta].
//!   \\end{equation}
//!
//!   This simplifies the expression for the number density to
//!
//!   \\begin{equation}
//!     n = \frac{g m^2 K_2(m \beta)}{2 \pi^2 \beta} e^{\mu \beta} = n^{(0)} e^{\mu \beta},
//!   \\end{equation}
//!
//!   where \\(n^{(0)}\\) is the equilibrium number density when \\(\mu = 0\\).
//!   and conversely allows for the equilibrium phase space distribution to be
//!   expressed in terms of the \\(\mu = 0\\) distribution:
//!
//!   \\begin{equation}
//!     f_{\textsc{MB}} = e^{\mu \beta} f_{\textsc{MB}}^{(0)} = \frac{n}{n^{(0)}} f_{\textsc{MB}}^{(0)}.
//!   \\end{equation}
//!
//!   Furthermore, the assumption that \\(\beta \gg E - \mu\\) implies that
//!   \\(f_{\textsc{BE}}, f_{\textsc{FD}} \ll 1\\).  Consequently, the Pauli
//!   suppression and Bose enhancement factors in the collision term can all be
//!   neglected resulting in:
//!
//!   \\begin{equation}
//!     \begin{aligned}
//!       g_{a_1} \int \vt C[f_{a_{1}}] \frac{\dd \vt p_{a_1}}{(2\pi)^3}
//!         &= - \int \left( \prod_{\vt a, \vt b} \dd \Pi_i \right)
//!                   (2 \pi)^4 \delta^4(p_{\vt a} - p_{\vt b}) \\\\
//!         &\quad \times \Biggl[ \abs{\mathcal M(\vt a | \vt b)}^2 \left( \prod_{\vt a} f_i \right)
//!                             - \abs{\mathcal M(\vt b | \vt a)}^2 \left( \prod_{\vt b} f_i \right) \Biggr] \\\\
//!         &= - \int \left( \prod_{\vt a, \vt b} \dd \Pi_i \right)
//!                   (2 \pi)^4 \delta^4(p_{\vt a} - p_{\vt b}) \\\\
//!         &\quad \times \Biggl[ \abs{\mathcal M(\vt a | \vt b)}^2 \left( \prod_{\vt a} \frac{n_i}{n_i^{(0)}} f_i^{(0)} \right)
//!                             - \abs{\mathcal M(\vt b | \vt a)}^2 \left( \prod_{\vt b} \frac{n_i}{n_i^{(0)}} f_i^{(0)} \right) \Biggr].
//!     \end{aligned}
//!   \\end{equation}
//!
//!   This is then commonly expressed as,
//!
//!   \\begin{equation}
//!       g_{a_1} \int \vt C[f_{a_{1}}] \frac{\dd \vt p_{a_1}}{(2\pi)^3}
//!         = - \left( \prod_{\vt a} \frac{n_i}{n_i^{(0)}} \right) \gamma(\vt a | \vt b)
//!           + \left( \prod_{\vt b} \frac{n_i}{n_i^{(0)}} \right) \gamma(\vt b | \vt a),
//!   \\end{equation}
//!
//!   where we have introduced the interaction density
//!
//!   \\begin{equation}
//!     \gamma(\vt a | \vt b)
//!       = \int \left( \prod_{\vt a, \vt b} \dd \Pi_i \right) (2 \pi)^4 \delta^4(p_{\vt a} - p_{\vt b})
//!         \abs{\mathcal M(\vt a | \vt b)}^2 \left( \prod_{\vt a} f^{(0)}_i \right).
//!   \\end{equation}
//!
// //! - *Assume \\(\mathcal{CP}\\) symmetry.* If \\(\mathcal{CP}\\) symmetry is
// //!   assumed, then \\(\abs{\mathcal M(\vt a | \vt b)}^2 \equiv \abs{\mathcal
// //!   M(\vt b | \vt a)}^2\\).  This simplification allows for the exponentials
// //!   of \\(\mu\\) to be taken out of the integral entirely:
// //!
// //!   \\begin{equation}
// //!     g_{a_1} \int \vt C[f_{a_{1}}] \frac{\dd \vt p_{a_1}}{(2\pi)^3}
// //!       = - \left[ e^{ \beta \sum_{\vt a} \mu_i } - e^{ \beta \sum_{\vt b} \mu_i } \right]
// //!            \int \left( \prod_{\vt a, \vt b} \dd \Pi_i \right) (2 \pi)^4 \delta^4(p_{\vt a} - p_{\vt b})
// //!            \abs{\mathcal M(\vt a | \vt b)}^2 e^{ - \beta \sum_{\vt a} E_i }.
// //!   \\end{equation}
// //!
// //!   The remaining integrand is then independent of time and can be
// //!   pre-calculated.  In the case of a \\(2 \leftrightarrow 2\\) interaction,
// //!   this integral is related to the thermally averaged cross-section
// //!   \\(\angles{\sigma v_\text{rel}}\\).
// //!
// //!   Solving the Boltzmann equation is generally required within the context of
// //!   baryogenesis and leptogenesis where the assumption of \\(\mathcal{CP}\\)
// //!   symmetry is evidently not correct.  In such cases, it is convenient to
// //!   define the parameter \\(\epsilon\\) to account for all of the
// //!   \\(\mathcal{CP}\\) asymmetry as.  That is:
// //!
// //!   \\begin{equation}
// //!     \abs{\mathcal M(\vt a | \vt b)}^2 = (1 + \epsilon) \abs{\mathcal{M^{(0)}(\vt a | \vt b)}}^2, \qquad
// //!     \abs{\mathcal M(\vt b | \vt a)}^2 = (1 - \epsilon) \abs{\mathcal{M^{(0)}(\vt a | \vt b)}}^2,
// //!   \\end{equation}
// //!
// //!   where \\(\abs{\mathcal{M^{(0)}}(\vt a | \vt b)}^2\\) is the
// //!   \\(\mathcal{CP}\\)-symmetric squared amplitude.  With \\(\epsilon\\)
// //!   defined as above, the collision term becomes:
// //!
// //!   \\begin{equation}
// //!     \begin{aligned}
// //!       g_{a_1} \int \vt C[f_{a_{1}}] \frac{\dd \vt p_{a_1}}{(2\pi)^3}
// //!         &= - \left[ e^{ \beta \sum_{\vt a} \mu_i } - e^{ \beta \sum_{\vt b} \mu_i } \right] \times
// //!              \int \left( \prod_{\vt a, \vt b} \dd \Pi_i \right) (2 \pi)^4 \delta^4(p_{\vt a} - p_{\vt b})
// //!              \abs{\mathcal M^{(0)}(\vt a | \vt b)}^2 e^{ - \beta \sum_{\vt a} E_i } \\\\
// //!         &\quad - \left[ e^{ \beta \sum_{\vt a} \mu_i } + e^{ \beta \sum_{\vt b} \mu_i } \right] \times
// //!              \int \left( \prod_{\vt a, \vt b} \dd \Pi_i \right) (2 \pi)^4 \delta^4(p_{\vt a} - p_{\vt b})
// //!              \epsilon \abs{\mathcal M^{(0)}(\vt a | \vt b)}^2 e^{ - \beta \sum_{\vt a} E_i }.
// //!     \\end{aligned}
// //!   \\end{equation}
//!
//! # Interactions
//!
//! The majority of interactions can be accounted for by looking at simply \\(1
//! \leftrightarrow 2\\) decays/inverse decays and \\(2 \leftrightarrow 2\\)
//! scatterings.  In the former case, the calculations can be done entirely
//! analytically and are presented below.
//!
//! ## Decays and Inverse Decays
//!
//! Considering the interaction \\(a \leftrightarrow b + c\\), the reaction is
//!
//! \\begin{equation}
//!   \begin{aligned}
//!     g_a \int \vt C\[f_a\] \frac{\dd \vt p_a}{(2\pi)^3}
//!       &= - \frac{n_a}{n^{(0)}_a} \gamma(a \to bc) + \frac{n_b n_c}{n^{(0)}_b n^{(0)}_c} \gamma(bc \to a)
//!   \end{aligned}
//! \\end{equation}
//!
//! In every case, the squared matrix element will be completely independent of
//! all initial and final state momenta (even \\(\varepsilon \cdot p\\) terms
//! vanish after averaging over both initial and final spins).
//!
//! ### Decay Term
//!
//! For the decay, the integration over the final state momenta is trivial as
//! the only final-state-momentum dependence appears within the Dirac delta
//! function.  Consequently, we need only integrate over the initial state
//! momentum:
//!
//! \\begin{equation}
//!   \begin{aligned}
//!     \gamma(a \to bc)
//!       &= - \abs{\mathcal M(a \to bc)}^2 \int \dd \Pi_a \dd \Pi_b \dd \Pi_c (2\pi)^4 \delta^4(p_a - p_b - p_c) f^{(0)}_a \\\\
//!       &= - \frac{\abs{\mathcal M(a \to bc)}^2}{16 \pi} \frac{m_a K_1(m_a \beta)}{2 \pi^2 \beta}
//!   \end{aligned}
//! \\end{equation}
//!
//! Combined with the number density scaling:
//!
//! \\begin{equation}
//!   \frac{n_a}{n^{(0)}_a} \gamma(a \to bc)
//!     = n_a \frac{\abs{\mathcal M(a \to bc)}^2}{16 \pi m_a} \frac{K_1(m_a \beta)}{K_2(m_a \beta)}
//! \\end{equation}
//!
//! where the analytic expression for \\(n^{(0)}_a\\) was used to introduce the
//! second Bessel function.
//!
//! The ratio of Bessel functions is generally referred to as the *time-dilation
//! factor* for the decaying particle.  When \\(m \beta \gg 1\\), the ratio of
//! Bessel functions approaches 1, and when \\(m \beta \ll 1\\), the ratio
//! approaches \\(0\\).
//!
//! If the final state particles are essentially massless in comparison to the
//! decaying particle, then the decay rate in the rest frame of the particle of
//! \\(\Gamma_\text{rest} = \abs{\mathcal M}^2 / 16 \pi m_a\\) and the above
//! expression can be simplified to
//!
//! \\begin{equation}
//!   \frac{n_a}{n^{(0)}\_a} \gamma(a \to bc)
//!   = n_a \Gamma_\text{rest} \frac{K_1(m_a \beta)}{K_2(m_a \beta)}.
//! \\end{equation}
//!
//! ### Inverse Decay
//!
//! The inverse decay rate is given by
//!
//! \\begin{equation}
//!   \gamma(bc \to a)
//!     = \abs{\mathcal M(bc \to a)}^2 \int \dd \Pi_a \dd \Pi_b \dd \Pi_c (2 \pi)^4 \delta^4(p_b + p_c - p_a) f^{(0)}_b f^{(0)}_c
//! \\end{equation}
//!
//! The Dirac delta enforces that \\(E_a = E_b + E_c\\) which implies that
//! \\(f^{(0)}_a = f^{(0)}_b f^{(0)}_c\\) thereby making the integral identical
//! to the decay scenario:
//!
//! \\begin{equation}
//!   \gamma(bc \to a)
//!     = \frac{\abs{\mathcal M(bc \to a)}^2}{16 \pi} \frac{m_a K_1(m_a \beta)}{2 \pi^2 \beta}
//!     = n^{(0)}_a \frac{\abs{\mathcal M(bc \to a)}^2}{16 \pi m_a} \frac{K_1(m_a \beta)}{K_2(m_a \beta)}
//! \\end{equation}
//!
//! The full expression including the number density scaling becomes:
//!
//! \\begin{equation}
//!   \frac{n_b n_c}{n^{(0)}_b n^{(0)}_c} \gamma(bc \to a)
//!     = \frac{n_b n_c}{n^{(0)}_b n^{(0)}_c} n^{(0)}_a \frac{\abs{\mathcal M(bc \to a)}^2}{16 \pi m_a} \frac{K_1(m_a \beta)}{K_2(m_a \beta)}
//! \\end{equation}

use super::{ErrorTolerance, Solver, StepChange};
use ndarray::{prelude::*, FoldWhile, Zip};
use particle::Particle;
use statistic::{
    Statistic::{BoseEinstein, FermiDirac}, Statistics,
};
use universe::Universe;

/// Context provided containing pre-computed values which might be useful when
/// evaluating interactions.
pub struct Context {
    /// Inverse temperature in GeV^{-1}
    pub beta: f64,
    /// Hubble rate, in GeV
    pub hubble_rate: f64,
    /// Equilibrium number densities for the particles.  This is provided in the
    /// same order as specified to the solver
    pub eq_n: Array1<f64>,
    /// Equilibrium number density for massless bosons.  This is specified per
    /// degree of freedom (that is \\(g = 1\\))
    pub eq_boson: f64,
    /// Equilibrium number density for massless fermions.  This is specified per
    /// degree of freedom (that is \\(g = 1\\))
    pub eq_fermion: f64,
}

/// Boltzmann equation solver for the number density
pub struct NumberDensitySolver {
    initialized: bool,
    beta_range: (f64, f64),
    particles: Vec<Particle>,
    #[cfg_attr(feature = "cargo-clippy", allow(type_complexity))]
    interactions: Vec<Box<Fn(Array1<f64>, &Array1<f64>, &Context) -> Array1<f64>>>,
    step_change: StepChange,
    error_tolerance: ErrorTolerance,
    normalize_to_photons: bool,
}

impl NumberDensitySolver {
    /// Normalize the number densities to the photon number density during the
    /// calculations.  As a consequence, the effects from the expansion of the
    /// Universe are automatically taken into account.
    pub fn normalize_to_photons(mut self, v: bool) -> Self {
        self.normalize_to_photons = v;
        self
    }
}

impl Solver for NumberDensitySolver {
    /// The solution is a one-dimensional array of number densities for each
    /// particle species (or aggregated number density in the case of
    /// \\(n_{\mathsc{b-l}}\\)), in the same order as [`Solver::add_particle`]
    /// is invoked.
    type Solution = Array1<f64>;

    type Context = Context;

    /// Create a new instance of the number density solver.
    ///
    /// The default range of temperatures span 1 GeV through to 10^{20} GeV.
    fn new() -> Self {
        Self {
            initialized: false,
            beta_range: (1e-20, 1e0),
            particles: Vec::with_capacity(20),
            interactions: Vec::with_capacity(100),
            step_change: StepChange {
                increase: 1.1,
                decrease: 0.5,
            },
            error_tolerance: ErrorTolerance {
                upper: 1e-2,
                lower: 1e-5,
            },
            normalize_to_photons: true,
        }
    }

    /// Set the range of inverse temperature values over which the phase space
    /// is evolved.
    ///
    /// Inverse temperature must be provided in units of GeV^{-1}.
    ///
    /// This function has a convenience alternative called
    /// [`Solver::temperature_range`] allowing for the limits to be specified as
    /// temperature in the units of GeV.
    ///
    /// # Panics
    ///
    /// Panics if the starting value is larger than the final value.
    fn beta_range(mut self, start: f64, end: f64) -> Self {
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
    /// [`NumberDensitySolver::beta_range`].
    ///
    /// # Panics
    ///
    /// Panics if the starting value is smaller than the final value.
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
        self.initialized = true;

        self
    }

    fn add_particle(&mut self, s: Particle) {
        self.particles.push(s);
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

    fn solve<U>(&self, universe: &U) -> Self::Solution
    where
        U: Universe,
    {
        assert!(
            self.initialized,
            "The phase space solver has to be initialized first with the `initialize()` method."
        );

        let mut y = self.equilibrium_number_densities(self.beta_range.0);
        let mut beta = self.beta_range.0;
        let mut h = beta / 10.0;

        // Allocate variables which will be re-used each for loop
        let mut k1: Self::Solution;
        let mut k2: Self::Solution;
        let mut k3: Self::Solution;
        let mut k4: Self::Solution;
        let mut tmp: Self::Solution;

        let mut n_eval = 0;
        while beta < self.beta_range.1 {
            n_eval += 1;

            // Standard Runge-Kutta integration.
            let c = self.context(beta, universe);
            k1 = self.interactions
                .iter()
                .fold(Self::Solution::zeros(y.dim()), |s, f| f(s, &y, &c));
            if !self.normalize_to_photons {
                Zip::from(&mut k1).and(&y).apply(|k, &t| {
                    *k -= -t * 3.0 * c.hubble_rate;
                    *k *= h
                });
            } else {
                k1 *= h;
            };

            let c = self.context(beta + 0.5 * h, universe);
            tmp = &y + &(&k1 * 0.5);
            k2 = self.interactions
                .iter()
                .fold(Self::Solution::zeros(y.dim()), |s, f| f(s, &tmp, &c));
            if !self.normalize_to_photons {
                Zip::from(&mut k2).and(&tmp).apply(|k, &t| {
                    *k -= t * 3.0 * c.hubble_rate;
                    *k *= h
                })
            } else {
                k2 *= h;
            }

            let c = self.context(beta + 0.5 * h, universe);
            tmp = &y + &(&k2 * 0.5);
            k3 = self.interactions
                .iter()
                .fold(Self::Solution::zeros(y.dim()), |s, f| f(s, &tmp, &c));
            if !self.normalize_to_photons {
                Zip::from(&mut k3).and(&tmp).apply(|k, &t| {
                    *k -= t * 3.0 * c.hubble_rate;
                    *k *= h
                })
            } else {
                k3 *= h;
            }

            let c = self.context(beta + h, universe);
            let tmp = &y + &k3;
            k4 = self.interactions
                .iter()
                .fold(Self::Solution::zeros(y.dim()), |s, f| f(s, &tmp, &c));
            if !self.normalize_to_photons {
                Zip::from(&mut k4).and(&tmp).apply(|k, &t| {
                    *k -= t * 3.0 * c.hubble_rate;
                    *k *= h
                })
            } else {
                k4 *= h;
            }

            // Calculate dy.  Note that we consume k2, k3 and k4 here.  We use
            // k1 by reference since we need it later to get the error estimate.
            let dy = (k2 * 2.0 + k3 * 2.0 + k4 + &k1) / 6.0;

            // Check the error on the RK method vs the Euler method.  If it is
            // small enough, increase the step size.  We use the maximum error
            // for any given element of `dy`.
            let err = Zip::from(&k1)
                .and(&dy)
                .fold_while(0.0, |e, k, d| {
                    let v = (d / k - 1.0).abs();
                    if v.is_finite() && v > e {
                        FoldWhile::Continue(v)
                    } else {
                        FoldWhile::Continue(e)
                    }
                })
                .into_inner();

            // Adjust the step size as needed based on the step size.
            if err < self.error_tolerance.lower {
                h *= self.step_change.increase;
                debug!(
                    "Step {:>7}, β = {:>9.2e} -> Increased h to {:.3e} (error was {:.3e})",
                    n_eval, beta, h, err
                );
            } else if err > self.error_tolerance.upper {
                h *= self.step_change.decrease;
                debug!(
                    "Step {:>7}, β = {:>9.2e} -> Decreased h to {:.3e} (error was {:.3e})",
                    n_eval, beta, h, err
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

impl Default for NumberDensitySolver {
    fn default() -> Self {
        Self::new()
    }
}

impl NumberDensitySolver {
    /// Create an array containing the initial conditions for all the particle
    /// species.
    ///
    /// All particles species are assumed to be in thermal equilibrium at this
    /// energy, with the distribution following either the Bose–Einstein or
    /// Fermi–Dirac distribution as determined by their spin.
    fn equilibrium_number_densities(&self, beta: f64) -> Array1<f64> {
        let a = Array1::from_iter(self.particles.iter().map(|p| p.number_density(0.0, beta)));

        if self.normalize_to_photons {
            a / BoseEinstein.massless_number_density(0.0, beta)
        } else {
            a
        }
    }

    fn context<U: Universe>(&self, beta: f64, universe: &U) -> Context {
        Context {
            beta,
            hubble_rate: universe.hubble_rate(beta),
            eq_n: self.equilibrium_number_densities(beta),
            eq_boson: BoseEinstein.massless_number_density(0.0, beta),
            eq_fermion: FermiDirac.massless_number_density(0.0, beta),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use special_functions::bessel;
    use std::f64::consts;
    use universe::StandardModel;
    use utilities::test::*;

    #[test]
    fn no_interaction() {
        let phi = Particle::new(0, 1e3);
        let mut solver = NumberDensitySolver::new()
            .temperature_range(1e20, 1e-10)
            .normalize_to_photons(true)
            .initialize();

        solver.add_particle(phi);

        let sol = solver.solve(&StandardModel::new());
        approx_eq(sol[0], 1.0, 8.0, 0.0);
    }

    #[test]
    fn minimal_leptogenesis() {
        let n1 = Particle::new(1, 1e10);
        let b_minus_l = Particle::new(0, 0.0).set_dof(0.0);

        let yukawa: f64 = 1e-4;
        let epsilon = 1e-7;
        let decay_zero_temp = yukawa.powi(2) / (16.0 * consts::PI.powi(2)) * n1.mass;

        let universe = StandardModel::new();
        let mut solver = NumberDensitySolver::new()
            .beta_range(1e-14, 1e0)
            .error_tolerance(1e-1, 1e-2)
            .normalize_to_photons(true)
            .initialize();

        solver.add_particle(n1);
        solver.add_particle(b_minus_l);

        // Interaction N ↔ e± H∓ (decay of N, asymmetry in B-L and washout)
        solver.add_interaction(move |mut s, n, ref c| {
            let m_beta = n1.mass * c.beta;
            let dilation_factor = if m_beta > 600.0 {
                1.0
            } else if m_beta < 1e-15 {
                0.0
            } else {
                bessel::k_1(m_beta) / bessel::k_2(m_beta)
            };

            let decay = decay_zero_temp * dilation_factor;

            let full_decay = (n[0] - c.eq_n[0]) / (c.hubble_rate * c.beta) * decay;
            let inverse_decay = decay * c.eq_n[0] / (2.0 * c.eq_fermion);
            let washout = inverse_decay / (2.0 * c.hubble_rate * c.beta);

            s[0] -= full_decay;
            s[1] += epsilon * full_decay;
            s[1] -= washout * n[1] / (c.hubble_rate * c.beta);
            s
        });

        let sol = solver.solve(&universe);

        assert!(sol[0] < 1e-20);
        assert!(1e-10 < sol[1] && sol[1] < 1e-5);
    }
}
