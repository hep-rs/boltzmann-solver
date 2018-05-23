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
//!   Fermi–Dirac and Bose–Einstein approach the Maxwell–Boltzmann distribution:
//!
//!   \\begin{equation}
//!     f_{\textsc{MB}} = \exp[-(E - \mu) \beta]
//!   \\end{equation}
//!
//!   Furthermore, the assumption that \\(\beta \gg E - \mu\\) implies that
//!   \\(f_{\textsc{BE}}, f_{\textsc{FD}} \ll 1\\).  Consequently, the Pauli
//!   suppression and Bose enhancement factors in the collision term can all be
//!   neglected resulting in:
//!
//!   \\begin{equation}
//!     \begin{aligned}
//!       g_{a_1} \int \vt C[f_{a_{1}}] \dd \Pi_{a_1}
//!         &= - \int \left( \prod_{\vt a, \vt b} \dd \Pi_i \right)
//!                   (2 \pi)^4 \delta(p_{\vt a} - p_{\vt b}) \\\\
//!         &\quad \times \Biggl[ \abs{\mathcal M(\vt a | \vt b)}^2 \left( \prod_{\vt a} f_i \right)
//!                             - \abs{\mathcal M(\vt b | \vt a)}^2 \left( \prod_{\vt b} f_i \right) \Biggr],
//!     \end{aligned}
//!   \\end{equation}
//!
//!   Combined with the assumption that all species are in kinetic equilibrium,
//!   then the collision term can be further simplified to:
//!
//!   \\begin{equation}
//!     \begin{aligned}
//!       g_{a_1} \int \vt C[f_{a_{1}}] \dd \Pi_{a_1}
//!         &= - \int \left( \prod_{\vt a, \vt b} \dd \Pi_i \right)
//!                   (2 \pi)^4 \delta(p_{\vt a} - p_{\vt b}) \\\\
//!         &\quad \times \Biggl[ \abs{\mathcal M(\vt a | \vt b)}^2 e^{ \beta \sum_{\vt a} \mu_i }
//!                             - \abs{\mathcal M(\vt b | \vt a)}^2 e^{ \beta \sum_{\vt b} \mu_i } \Biggr]
//!                       e^{ - \beta \sum_{\vt a} E_i },
//!     \end{aligned}
//!   \\end{equation}
//!
//!   where energy conservation (as imposed by the Dirac delta) is used to
//!   equate \\(\sum_{\vt a} E_i = \sum_{\vt b} E_i\\).
//!
//!   It should be noted that this assumption also simplifies greatly the
//!   expression for the number density.  In particular, we obtain:
//!
//!  \\begin{equation}
//!    n = e^{\mu \beta} \frac{m^2 K_2(m \beta)}{2 \pi^2 \beta} = e^{\mu \beta} n^{(0)}
//!  \\end{equation}
//!
//! - *Assume \\(\mathcal{CP}\\) symmetry.* If \\(\mathcal{CP}\\) symmetry is
//!   assumed, then \\(\abs{\mathcal M(\vt a | \vt b)}^2 \equiv \abs{\mathcal
//!   M(\vt b | \vt a)}^2\\).  This simplification allows for the exponentials
//!   of \\(\mu\\) to be taken out of the integral entirely:
//!
//!   \\begin{equation}
//!     g_{a_1} \int \vt C[f_{a_{1}}] \dd \Pi_{a_1}
//!       = - \left[ e^{ \beta \sum_{\vt a} \mu_i } - e^{ \beta \sum_{\vt b} \mu_i } \right]
//!            \int \left( \prod_{\vt a, \vt b} \dd \Pi_i \right) (2 \pi)^4 \delta(p_{\vt a} - p_{\vt b})
//!            \abs{\mathcal M(\vt a | \vt b)}^2 e^{ - \beta \sum_{\vt a} E_i }.
//!   \\end{equation}
//!
//!   The remaining integrand is then independent of time and can be
//!   pre-calculated.  In the case of a \\(2 \leftrightarrow 2\\) interaction,
//!   this integral is related to the thermally averaged cross-section
//!   \\(\angles{\sigma v_\text{rel}}\\).
//!
//!   Solving the Boltzmann equation is generally required within the context of
//!   baryogenesis and leptogenesis where the assumption of \\(\mathcal{CP}\\)
//!   symmetry is evidently not correct.  In such cases, it is convenient to
//!   define the parameter \\(\epsilon\\) to account for all of the
//!   \\(\mathcal{CP}\\) asymmetry as.  That is:
//!
//!   \\begin{equation}
//!     \abs{\mathcal M(\vt a | \vt b)}^2 = (1 + \epsilon) \abs{\mathcal{M^{(0)}(\vt a | \vt b)}}^2, \qquad
//!     \abs{\mathcal M(\vt b | \vt a)}^2 = (1 - \epsilon) \abs{\mathcal{M^{(0)}(\vt a | \vt b)}}^2,
//!   \\end{equation}
//!
//!   where \\(\abs{\mathcal{M^{(0)}}(\vt a | \vt b)}^2\\) is the
//!   \\(\mathcal{CP}\\)-symmetric squared amplitude.  With \\(\epsilon\\)
//!   defined as above, the collision term becomes:
//!
//!   \\begin{equation}
//!     \begin{aligned}
//!       g_{a_1} \int \vt C[f_{a_{1}}] \dd \Pi_{a_1}
//!         &= - \left[ e^{ \beta \sum_{\vt a} \mu_i } - e^{ \beta \sum_{\vt b} \mu_i } \right] \times
//!              \int \left( \prod_{\vt a, \vt b} \dd \Pi_i \right) (2 \pi)^4 \delta(p_{\vt a} - p_{\vt b})
//!              \abs{\mathcal M^{(0)}(\vt a | \vt b)}^2 e^{ - \beta \sum_{\vt a} E_i } \\\\
//!         &\quad - \left[ e^{ \beta \sum_{\vt a} \mu_i } + e^{ \beta \sum_{\vt b} \mu_i } \right] \times
//!              \int \left( \prod_{\vt a, \vt b} \dd \Pi_i \right) (2 \pi)^4 \delta(p_{\vt a} - p_{\vt b})
//!              \epsilon \abs{\mathcal M^{(0)}(\vt a | \vt b)}^2 e^{ - \beta \sum_{\vt a} E_i }.
//!     \\end{aligned}
//!   \\end{equation}
