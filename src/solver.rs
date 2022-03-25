#![doc = include_str!("solver.md")]

mod builder;
mod context;
pub(crate) mod options;
pub(crate) mod tableau;

#[allow(clippy::useless_attribute)] // Waiting on rust-lang/rust-clippy#7511
#[allow(clippy::module_name_repetitions)]
pub use builder::SolverBuilder;
pub use context::Context;

use crate::{
    model::{
        interaction::{self, fix_equilibrium, FastInteractionResult, Interaction},
        Model, ModelInteractions, ParticleData,
    },
    solver::options::{ErrorTolerance, StepPrecision},
    statistic::{Statistic, Statistics},
};
use ndarray::prelude::*;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "serde")]
use serde::Serialize;
#[cfg(feature = "debug")]
use std::collections::HashMap;
use std::{collections::HashSet, convert::TryFrom, error, fmt, io::Write, iter, ops, sync::RwLock};

/// The minimum multiplicative step size between two consecutive calls of the logger.
///
/// For debug builds, this is set to `1.0` so that the logger is called every time.  Otherwise
#[cfg(feature = "debug")]
const BETA_LOG_STEP: f64 = 1.0;
#[cfg(not(feature = "debug"))]
const BETA_LOG_STEP: f64 = 1.1;

/// Error type returned by the solver in case there is an error during
/// integration.
///
/// The underlying number density and number density asymmetry arrays can be
/// obtained with [`Error::into_inner`].
#[derive(Debug)]
pub enum Error {
    /// Result is inaccurate.
    ///
    /// If [`SolverBuilder::abort_when_inaccurate`] was set to true, the number
    /// densities returned
    Inaccurate((Array1<f64>, Array1<f64>)),
    /// One of more number density was NAN.
    NAN((Array1<f64>, Array1<f64>)),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Inaccurate(_) => write!(f, "Result my be inaccurate"),
            Error::NAN(_) => write!(f, "NAN number density encountered"),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}

impl Error {
    /// Unwrap the error and return the underlying number density and number
    /// density asymmetry arrays respectively.
    #[must_use]
    pub fn into_inner(self) -> (Array1<f64>, Array1<f64>) {
        match self {
            Error::Inaccurate(v) | Error::NAN(v) => v,
        }
    }
}

#[cfg(feature = "debug")]
#[derive(Serialize)]
struct InteractionDebug {
    incoming: Vec<isize>,
    outgoing: Vec<isize>,
    incoming_names: Vec<String>,
    outgoing_names: Vec<String>,
    gamma: Option<f64>,
    delta_gamma: Option<f64>,
    symmetric_prefactor: f64,
    asymmetric_prefactor: f64,
    dn: Vec<f64>,
    dna: Vec<f64>,
    fast: bool,
}

#[cfg(feature = "debug")]
#[derive(Serialize)]
struct Debug {
    step: usize,
    h: f64,
    beta: HashMap<usize, f64>,
    n: HashMap<usize, Vec<f64>>,
    na: HashMap<usize, Vec<f64>>,
    eq: HashMap<usize, Vec<f64>>,
    eqn: HashMap<usize, Vec<f64>>,
    k: HashMap<usize, Vec<f64>>,
    ka: HashMap<usize, Vec<f64>>,
    dn: Vec<f64>,
    dn_error: Vec<f64>,
    dna: Vec<f64>,
    dna_error: Vec<f64>,
    interactions: HashMap<usize, Vec<InteractionDebug>>,
    normalizations: HashMap<usize, f64>,
}

/// Workspace of functions to reuse during the integration
#[cfg_attr(feature = "serde", derive(Serialize))]
struct Workspace {
    /// Number density
    n: Array1<f64>,
    /// Number density at the current sub-quadrature step
    ni: Array1<f64>,
    /// Number density change
    dn: Array1<f64>,
    /// Number density local error estimate
    dn_error: Array1<f64>,
    /// Number density asymmetry
    na: Array1<f64>,
    /// Number density asymmetry at the current sub-quadrature step
    nai: Array1<f64>,
    /// Number density asymmetry change
    dna: Array1<f64>,
    /// Number density asymmetry local error estimate
    dna_error: Array1<f64>,

    /// k array for the number density of shape `RK_S × p` where `p` is the
    /// number of particles (including the 0-index particle).
    k: Vec<Array1<f64>>,
    /// k array for the number density asymmetry of shape `RK_S × p` where `p`
    /// is the number of particles (including the 0-index particle).
    ka: Vec<Array1<f64>>,
}

/// Workspace of variables allocated once and then reused during the integration.
impl Workspace {
    fn new(initial_densities: &Array1<f64>, initial_asymmetries: &Array1<f64>) -> Self {
        let dim = initial_densities.dim();

        let k: Vec<_> = iter::repeat(Array::zeros(dim))
            .take(tableau::RK_S)
            .collect();
        let ka: Vec<_> = iter::repeat(Array::zeros(dim))
            .take(tableau::RK_S)
            .collect();

        Self {
            n: initial_densities.clone(),
            ni: initial_asymmetries.clone(),
            dn: Array1::zeros(dim),
            dn_error: Array1::zeros(dim),

            na: initial_asymmetries.clone(),
            nai: initial_asymmetries.clone(),
            dna: Array1::zeros(dim),
            dna_error: Array1::zeros(dim),

            k,
            ka,
        }
    }

    /// Clear the workspace for the next step, filling the changes to 0 and the
    /// error estimates to 0.
    ///
    /// This does not alter the sub-quadrature densities.
    fn clear_step(&mut self) {
        self.dn.fill(0.0);
        self.dna.fill(0.0);
        self.dn_error.fill(0.0);
        self.dna_error.fill(0.0);
    }

    /// Compute and update the changes to number density from k and a given step
    /// of the Runge-Kutta integration.
    fn compute_dn(&mut self, i: usize) {
        let bi = tableau::RK_B[i];
        let ei = tableau::RK_E[i];

        ndarray::azip!(
            (
                dn in &mut self.dn,
                dn_err in &mut self.dn_error,
                &ki in &self.k[i],
                dna in &mut self.dna,
                dna_err in &mut self.dna_error,
                &kai in &self.ka[i]
            ) {
                *dn += bi * ki;
                *dn_err += ei * ki;
                *dna += bi * kai;
                *dna_err += ei * kai;
            }
        );
    }

    /// Advance the integration by apply the computed changes to far to the number densities.
    fn advance(&mut self) {
        self.n += &self.dn;
        self.na += &self.dna;
    }

    /// Unwrap the workspace and return the number density and number density
    /// asymmetry.
    fn result(self) -> (Array1<f64>, Array1<f64>) {
        (self.n, self.na)
    }
}

impl ops::AddAssign<&FastInteractionResult> for Workspace {
    fn add_assign(&mut self, rhs: &FastInteractionResult) {
        self.dn += &rhs.dn;
        self.dna += &rhs.dna;
        self.dn_error += &rhs.dn_error;
        self.dna_error += &rhs.dna_error;
    }
}

impl ops::AddAssign<FastInteractionResult> for Workspace {
    fn add_assign(&mut self, rhs: FastInteractionResult) {
        self.dn += &rhs.dn;
        self.dna += &rhs.dna;
        self.dn_error += &rhs.dn_error;
        self.dna_error += &rhs.dna_error;
    }
}

type LoggerFn<M> = Box<dyn Fn(&Context<M>, &Array1<f64>, &Array1<f64>)>;

/// Boltzmann solver
#[allow(clippy::struct_excessive_bools)]
pub struct Solver<M> {
    model: M,
    particles_next: Vec<ParticleData>,
    initial_densities: Array1<f64>,
    initial_asymmetries: Array1<f64>,
    beta_range: (f64, f64),
    /// The indices of particles forced in equilibrium (irrespective of
    /// interactions) **must** be ordered.
    in_equilibrium: Vec<usize>,
    /// The indices of particles forced to have no asymmetry (irrespective of
    /// interactions) **must** be ordered.
    no_asymmetry: Vec<usize>,
    logger: LoggerFn<M>,
    step_precision: StepPrecision,
    /// The absolute and relative error tolerances for the integration.
    error_tolerance: ErrorTolerance,
    use_fast_interactions: bool,
    is_inaccurate: bool,
    abort_when_inaccurate: bool,
    is_precomputed: bool,
}

impl<M> Solver<M>
where
    M: ModelInteractions,
{
    fn compute_ki(&self, ki: &mut Array1<f64>, kai: &mut Array1<f64>, ci: &Context<M>) {
        #[cfg(feature = "parallel")]
        if self.is_precomputed && self.model.interactions().len() < 50 {
            self.compute_ki_serial(ki, kai, ci);
        } else {
            self.compute_ki_parallel(ki, kai, ci);
        }

        #[cfg(not(feature = "parallel"))]
        self.compute_ki_serial(ki, kai, ci);
    }

    fn compute_ki_serial(&self, ki: &mut Array1<f64>, kai: &mut Array1<f64>, ci: &Context<M>) {
        ki.fill(0.0);
        kai.fill(0.0);
        for interaction in self.model.interactions() {
            interaction.apply_change(ki, kai, ci);
        }
    }

    #[cfg(feature = "parallel")]
    fn compute_ki_parallel(&self, ki: &mut Array1<f64>, kai: &mut Array1<f64>, ci: &Context<M>) {
        let dim = ki.dim();

        let (new_ki, new_kai) = self
            .model
            .interactions()
            .par_iter()
            .fold(
                || (Array1::zeros(dim), Array1::zeros(dim)),
                |(mut dn, mut dna), interaction| {
                    interaction.apply_change(&mut dn, &mut dna, ci);
                    (dn, dna)
                },
            )
            .reduce(
                || (Array1::zeros(dim), Array1::zeros(dim)),
                |(dn, dna), (dni, dnai)| (dn + dni, dna + dnai),
            );

        new_ki.move_into(ki);
        new_kai.move_into(kai);
    }

    /// Adjust `dn` and/or `dna` for fast interaction (if applicable).
    #[cfg(not(feature = "parallel"))]
    fn fix_fast_interactions(
        c: &mut Context<M>,
        ws: &mut Workspace,
        eq: &[Array1<f64>],
    ) -> Option<()> {
        let interactions = c.fast_interactions.take()?.into_inner().ok()?;

        let mut n = &c.n + &ws.dn;
        let mut na = &c.na + &ws.dna;

        let result =
            interactions
                .iter()
                .fold(FastInteractionResult::zero(c.n.dim()), |mut acc, fi| {
                    let result = fi.fast_interaction_de(c, &n, &na, eq);
                    n += &result.dn;
                    na += &result.dna;
                    acc += result;
                    acc
                });

        *ws += result;

        Ok(())
    }

    /// Adjust `dn` and/or `dna` for fast interaction (if applicable).
    #[cfg(feature = "parallel")]
    fn fix_fast_interactions(
        context: &mut Context<M>,
        ws: &mut Workspace,
        eq: &[Array1<f64>],
    ) -> Option<()> {
        let mut interactions = context.fast_interactions.take()?.into_inner().ok()?;

        if interactions.is_empty() {
            return Some(());
        }

        // TODO: Is the bug in full::custom due to overlapping interactions for
        // particles in equilibrium?

        // As results from one fast interaction might affect the computation of
        // the next, we have to be careful as to how we parallelize the tasks.
        // Specifically, we can only run interactions in parallel if they share
        // no common particles.  We ignore particles whose number density is not
        // changed by the interaction, and particles who are forced to remain in
        // equilibrium.
        let mut anticliques = Vec::new();
        #[allow(clippy::unnecessary_filter_map)]
        while !interactions.is_empty() {
            let mut anticlique = Vec::new();
            let mut particle_set = HashSet::<usize>::new();

            interactions = interactions
                .drain()
                .filter_map(|interaction| {
                    let relevant_particles: Vec<_> = interaction
                        .particle_counts
                        .iter()
                        .filter_map(|(&idx, &(c, ca))| {
                            if c == 0.0 && ca == 0.0 {
                                None
                            } else {
                                Some(idx)
                            }
                        })
                        .filter(|idx| {
                            context
                                .in_equilibrium
                                .binary_search(idx)
                                .and_then(|_| context.no_asymmetry.binary_search(idx))
                                .is_err()
                        })
                        .collect();

                    if particle_set.is_empty()
                        || relevant_particles
                            .iter()
                            .all(|idx| !particle_set.contains(idx))
                    {
                        particle_set.extend(relevant_particles);
                        anticlique.push(interaction);
                        None
                    } else {
                        Some(interaction)
                    }
                })
                .collect();

            anticliques.push(anticlique);
        }

        log::trace!(
            "Divided {} fast into interactions into {} groups of sizes {:?}.",
            anticliques.iter().map(Vec::len).sum::<usize>(),
            anticliques.len(),
            anticliques.iter().map(Vec::len).collect::<Vec<_>>()
        );

        let mut n = &context.n + &ws.dn;
        let mut na = &context.na + &ws.dna;

        let mut result = FastInteractionResult::zero(context.n.dim());
        for anticlique in anticliques {
            let intermediate_result = match anticlique.len() {
                1 => anticlique[0].fast_interaction_de(context, &n, &na, eq),
                _ => anticlique
                    .into_par_iter()
                    .map(|fi| fi.fast_interaction_de(context, &n, &na, eq))
                    .reduce(
                        || FastInteractionResult::zero(n.dim()),
                        |acc, result| acc + result,
                    ),
            };

            log::trace!(
                "Anticlique result:\n         dn: {:e}\n        dna: {:e}\n   dn error: {:e}\n  dna error: {:e}",
                intermediate_result.dn,
                intermediate_result.dna,
                intermediate_result.dn_error,
                intermediate_result.dna_error
            );

            if intermediate_result
                .dn_error
                .iter()
                .chain(&intermediate_result.dna_error)
                .any(|x| x.is_nan())
            {
                log::warn!("Fast interaction produced NaN");
                *ws += result;
                return None;
            }

            n += &intermediate_result.dn;
            na += &intermediate_result.dna;
            result += intermediate_result;
        }

        *ws += result;
        Some(())
    }

    /// Check if the local error is within the tolerance and compute the
    /// adjustment to the step size based on the local error.
    ///
    /// Given a local error estimate of `$\tilde\varepsilon$` and target local
    /// error `$\varepsilon_\text{abs}$` and `$\varepsilon_\text{rel}$`, the new
    /// step size should be:
    ///
    /// ```math
    /// h_{\text{new}} = h \times \underbrace{S \sqrt[p + 1]{\frac{\max[\varepsilon_\text{abs}, \varepsilon_\text{rel} n]}{\tilde \varepsilon}}}_{\Delta}
    /// ```
    ///
    /// where `$S \in [0, 1]$` is a safety factor to purposefully underestimate
    /// the result.  The values of `\Delta` are bounded so as to avoid too large
    /// step size adjustments.
    ///
    /// This function returns three values:
    /// - Whether the local error was within the tolerance;
    /// - The ratio of the target error to the local error; and,
    /// - The value of delta.
    fn delta(&self, ws: &Workspace, n: &Array1<f64>, na: &Array1<f64>) -> (bool, f64, f64) {
        const MIN_DELTA: f64 = 0.1;
        const MAX_DELTA: f64 = 2.0;

        let ratio_symmetric =
            ws.dn_error
                .iter()
                .map(|err| err.abs())
                .zip(n)
                .fold(f64::INFINITY, |min, (err, &n)| {
                    let max_tolerance = self.error_tolerance.max_tolerance(n);
                    min.min(if err == 0.0 {
                        f64::MAX
                    } else if err.is_nan() {
                        0.0
                    } else {
                        max_tolerance / err
                    })
                });

        let ratio_asymmetric = ws.dna_error.iter().map(|err| err.abs()).zip(na).fold(
            f64::INFINITY,
            |min, (err, &na)| {
                let max_tolerance = self.error_tolerance.max_asymmetric_tolerance(na);
                min.min(if err == 0.0 {
                    f64::MAX
                } else if err.is_nan() {
                    0.0
                } else {
                    max_tolerance / err
                })
            },
        );

        let ratio = f64::min(ratio_symmetric, ratio_asymmetric);

        (ratio.is_finite() && ratio > 1.0, ratio, {
            let delta = 0.8 * ratio.powf(1.0 / f64::from(tableau::RK_ORDER + 1));
            if delta.is_nan() {
                MIN_DELTA
            } else {
                if MIN_DELTA > MAX_DELTA {
                    log::error!("Clamp Error: {} ≮ {}", MIN_DELTA, MAX_DELTA);
                }
                delta.clamp(MIN_DELTA, MAX_DELTA)
            }
        })
    }

    /// Ensure that h is within the desired range of step sizes.
    ///
    // This function also adjusts for the final step so as to ensure the
    // integration finishes at the target.
    ///
    /// This function returns a [`Result`] with:
    ///
    /// - `Ok(h)` indicating that the step size was either unchanged or made
    ///   smaller and integration can proceed with no worry about the error
    ///   estimate.
    /// - `Err(h)` indicating that the step size was increased.  The local error
    ///   estimate requires a smaller step size than the minimal value allowed
    ///   and thus the result of the integration may be inaccurate.
    ///
    /// The result should always be in the range `[step_precision.min * beta,
    /// step_precision.max * beta]`, with the ony exception being for the last
    /// integration step when the step size might be smaller than the lower
    /// bound.
    fn adjust_h(&mut self, step: usize, mut h: f64, beta: f64) -> Result<f64, f64> {
        let h_on_beta = h / beta;

        if h_on_beta > self.step_precision.max {
            h = beta * self.step_precision.max;
            log::trace!(
                "[{}|{:>10.4e}] Step size too large, decreased h to {:.3e}",
                step,
                beta,
                h
            );

            Ok(h)
        } else if h_on_beta < self.step_precision.min {
            // It is possible that there is a 'too small' step size for the last step, which is not an inaccuracy.
            if beta + h >= self.beta_range.1 {
                Ok(h)
            } else {
                h = beta * self.step_precision.min;
                log::debug!(
                    "[{}|{:>10.4e}] Step size too small, increased h to {:.3e}",
                    step,
                    beta,
                    h
                );

                // Give a warning (if not done already).
                if !self.is_inaccurate {
                    log::warn!(
                        "[{step}|{beta:.3e}] Step size too small at β = {beta:e}.  Result may not be accurate",
                        step = step,
                        beta = beta,
                    );
                    self.is_inaccurate = true;
                }

                Err(h)
            }
        } else {
            Ok(h)
        }
    }

    /// Evolve the initial conditions by solving the PDEs.
    ///
    /// Note that this is not a true PDE solver, but instead converts the PDE
    /// into an ODE in time (or inverse temperature in this case), with the
    /// derivative in energy being calculated solely from the previous
    /// time-step.
    ///
    /// # Errors
    ///
    /// If the time evolution encounters an error, the function returns an
    /// error.  The final number density (despite the error) can still be
    /// obtained by using [`Error::into_inner`], though this may not be at the
    /// expected final temperature as some errors abort the integration.
    #[allow(clippy::missing_panics_doc)]
    #[allow(clippy::too_many_lines)]
    pub fn solve(&mut self) -> Result<(Array1<f64>, Array1<f64>), Error> {
        use tableau::{RK_A, RK_C, RK_S};

        // Debug information to show the full step information.
        #[cfg(feature = "debug")]
        let debug_dir = {
            let mut temp_dir = std::env::temp_dir();
            temp_dir.push("boltzmann-solver");
            temp_dir.push("debug");
            std::fs::remove_dir_all(&temp_dir).unwrap_or_default();
            std::fs::create_dir_all(&temp_dir).unwrap_or_default();
            temp_dir
        };

        // Initialize all the variables that will be used in the integration
        let mut workspace = Workspace::new(&self.initial_densities, &self.initial_asymmetries);

        let mut step = 0_usize;
        let mut steps_discarded = 0_usize;
        let mut evals = 0_usize;
        let mut beta = self.beta_range.0;
        let mut h = beta * self.step_precision.max;
        let mut beta_logging = beta * BETA_LOG_STEP;

        // Run logger for 0th step
        {
            self.model.set_beta(beta);
            let context = self.context(
                step,
                None,
                h,
                (beta, beta + h),
                &workspace.n,
                &workspace.na,
                None,
            );
            (*self.logger)(&context, &workspace.dn, &workspace.dna);
        }

        while beta <= self.beta_range.1 {
            workspace.clear_step();

            // Log the progress of the integration
            if step % 100 == 0 {
                log::info!(
                    "Step {} (including {} discarded), β = {:>10.4e}, Δβ = {:.4e}",
                    step,
                    steps_discarded,
                    beta,
                    h
                );
            } else {
                log::debug!(
                    "Step {} (including {} discarded), β = {:>10.4e}, Δβ = {:.4e}",
                    step,
                    steps_discarded,
                    beta,
                    h
                );
            }

            // Compute next step particles
            self.model.set_beta(beta + h);
            self.particles_next = self.model.particles().to_vec();

            #[cfg(feature = "debug")]
            let mut debug = Debug {
                step,
                h,
                beta: HashMap::new(),
                n: HashMap::new(),
                na: HashMap::new(),
                eq: HashMap::new(),
                eqn: HashMap::new(),
                k: HashMap::new(),
                ka: HashMap::new(),
                dn: Vec::new(),
                dn_error: Vec::new(),
                dna: Vec::new(),
                dna_error: Vec::new(),
                interactions: HashMap::new(),
                normalizations: HashMap::new(),
            };

            // Initialize the fast interactions if they are being used
            let mut fast_interactions = if self.use_fast_interactions {
                Some(RwLock::new(HashSet::new()))
            } else {
                None
            };

            // For fast interactions, we need to know the equilibrium number
            // densities over the step interval.  Instead of recomputing them,
            // we collect them during the Runge-Kutta integration.
            let mut eq: Vec<Array1<f64>> = iter::repeat(Array1::zeros(workspace.n.dim()))
                .take(RK_S)
                .collect();

            for i in 0..RK_S {
                evals += 1;

                // Compute the sub-step values
                let beta_i = beta + RK_C[i] * h;
                let ai = RK_A[i];
                workspace.ni.assign(&workspace.n);
                workspace.nai.assign(&workspace.na);
                #[allow(clippy::needless_range_loop)]
                for j in 0..i {
                    workspace.ni.scaled_add(ai[j], &workspace.k[j]);
                    workspace.nai.scaled_add(ai[j], &workspace.ka[j]);
                }
                self.model.set_beta(beta_i);
                let context_i = self.context(
                    step,
                    Some(i),
                    h,
                    (beta_i, beta + h),
                    &workspace.ni,
                    &workspace.nai,
                    fast_interactions,
                );

                // Collect the equilibrium number densities for this substep.
                eq[i].assign(&context_i.eq);

                let ki = &mut workspace.k[i];
                let kai = &mut workspace.ka[i];

                // Compute k[i] and ka[i] from each interaction
                self.compute_ki(ki, kai, &context_i);

                #[cfg(feature = "debug")]
                {
                    debug.beta.insert(i, beta_i);
                    debug.n.insert(i, context_i.n.to_vec());
                    debug.na.insert(i, context_i.na.to_vec());
                    debug.eq.insert(i, context_i.eq.to_vec());
                    debug.eqn.insert(i, context_i.eqn.to_vec());
                    debug.k.insert(i, ki.to_vec());
                    debug.ka.insert(i, kai.to_vec());
                    debug.normalizations.insert(i, context_i.normalization);

                    #[cfg(feature = "parallel")]
                    let iter = self.model.interactions().par_iter();
                    #[cfg(not(feature = "parallel"))]
                    let iter = self.model.interactions().iter();

                    debug.interactions.insert(
                        i,
                        iter.map(|interaction| {
                            let mut dn = Array1::zeros(workspace.n.dim());
                            let mut dna = Array1::zeros(workspace.na.dim());
                            interaction.apply_change(&mut dn, &mut dna, &context_i);

                            InteractionDebug {
                                incoming: interaction.particles().incoming_signed.clone(),
                                outgoing: interaction.particles().outgoing_signed.clone(),
                                incoming_names: interaction
                                    .particles()
                                    .incoming_signed
                                    .iter()
                                    .map(|&i| self.model.particle_name(i).unwrap())
                                    .collect(),
                                outgoing_names: interaction
                                    .particles()
                                    .outgoing_signed
                                    .iter()
                                    .map(|&i| self.model.particle_name(i).unwrap())
                                    .collect(),
                                gamma: interaction.gamma(&context_i, false),
                                delta_gamma: interaction.delta_gamma(&context_i, false),
                                symmetric_prefactor: interaction.symmetric_prefactor(&context_i),
                                asymmetric_prefactor: interaction.asymmetric_prefactor(&context_i),
                                dn: dn.to_vec(),
                                dna: dna.to_vec(),
                                fast: context_i
                                    .fast_interactions
                                    .as_ref()
                                    .and_then(|lock| lock.read().ok())
                                    .map_or(false, |s| s.contains(interaction.particles())),
                            }
                        })
                        .collect(),
                    );
                }

                fast_interactions = context_i.into_fast_interactions();
                workspace.compute_dn(i);
            }

            self.model.set_beta(beta);
            let mut context = self.context(
                step,
                None,
                h,
                (beta, beta + h),
                &workspace.n,
                &workspace.na,
                fast_interactions,
            );

            // If the error is already too large and the step will be discarded,
            // there's no need to evaluate the fast interactions.  Otherwise, we
            // compute the fast interactions and recompute the error
            let (mut within_tolerance, mut error_ratio, mut delta) =
                self.delta(&workspace, &context.n, &context.na);

            if within_tolerance {
                Self::fix_fast_interactions(&mut context, &mut workspace, &eq);
                fix_equilibrium(&context, &mut workspace.dn, &mut workspace.dna);

                let tmp = self.delta(&workspace, &context.n, &context.na);

                within_tolerance = tmp.0;
                error_ratio = tmp.1;
                delta = tmp.2;
            }

            #[cfg(feature = "debug")]
            {
                debug.dn = workspace.dn.to_vec();
                debug.dn_error = workspace.dn_error.to_vec();
                debug.dna = workspace.dna.to_vec();
                debug.dna_error = workspace.dna_error.to_vec();

                serde_json::to_writer(
                    std::fs::File::create(debug_dir.join(format!("{}.json", step))).unwrap(),
                    &debug,
                )
                .unwrap();
            }

            // If the error is within the tolerance, we'll be advancing the
            // iteration step
            if within_tolerance {
                if beta > beta_logging || beta + h >= self.beta_range.1 {
                    (*self.logger)(&context, &workspace.dn, &workspace.dna);
                    beta_logging = beta * BETA_LOG_STEP;
                }

                workspace.advance();
                beta += h;
            } else {
                if log::log_enabled!(log::Level::Debug) {
                    log::debug!(
                        "[{}|{:>10.4e}] Error is not within tolerance (error ratio: {:e}).",
                        step,
                        beta,
                        error_ratio
                    );
                }
                steps_discarded += 1;
            }

            // Adjust the step size based on the error
            h *= delta;
            h = match self.adjust_h(step, h, beta) {
                Ok(h) => h,
                Err(h) => {
                    if self.abort_when_inaccurate {
                        return Err(Error::Inaccurate(workspace.result()));
                    }

                    workspace.advance();
                    beta += h;

                    h
                }
            };

            if beta + h >= self.beta_range.1 {
                log::trace!(
                    "[{}|{:>10.4e}] Fixing overshoot of last integration step.",
                    step,
                    beta
                );
                h = (self.beta_range.1 - beta).max(beta * f64::EPSILON);

                // Also adjust `beta_logging` so that the last step is always logged.
                beta_logging = beta;
            }

            step += 1;
        }

        log::info!("Number of integration steps: {}", step - steps_discarded);
        log::info!("Number of integration steps discarded: {}", steps_discarded);
        log::info!("Number of evaluations: {}", evals);

        if self.is_inaccurate {
            Err(Error::Inaccurate(workspace.result()))
        } else {
            Ok(workspace.result())
        }
    }

    /// Compute the interaction rates.
    ///
    /// The number of logarithmic steps in beta is specified by `n`, with the
    /// range of `$\beta$` values being taken from the solver.  If `normalize`
    /// is true, the interaction rate is divided by `$n_1 H \beta$`, where
    /// `$n_1$` is the equilibrium number density of a single bosonic degree of
    /// freedom, `$H$` is the Hubble rate and `$\beta$` is the inverse
    /// temperature.
    ///
    /// The interactions rates are returned as two dimensional array with
    /// `data[[i, j]]` giving the interaction rate for the `$j$`th interaction
    /// at the `$i$`th `$\beta$` step.  The interaction index corresponds to the
    /// index of the interaction as returned by
    /// [`ModelInteractions::interactions`], offset by 1 such that the 0th value
    /// is for `$\beta$` itself.
    ///
    /// The entries of the returned array as `Option<f64>` in order to
    /// distinguish cases where the rate is not computed due to being unphysical
    /// from cases where it is 0.
    ///
    pub fn gammas(&mut self, size: usize, normalize: bool) -> (Vec<String>, Array2<Option<f64>>) {
        let mut gammas = Array2::from_elem((size, self.model.interactions().len() + 1), None);
        let n = Array1::zeros(self.model.particles().len());
        let na = Array1::zeros(n.dim());

        for (i, &beta) in Array1::geomspace(self.beta_range.0, self.beta_range.1, size)
            .unwrap_or_else(|| ndarray::array![])
            .iter()
            .enumerate()
        {
            if log::log_enabled!(log::Level::Info) {
                print!("Step {:>5} / {}\r", i, size);
                std::io::stdout().flush().unwrap_or(());
            }

            self.model.set_beta(beta);
            gammas[[i, 0]] = Some(beta);
            let mut c = self.context(0, None, 1.0, (beta, beta), &n, &na, None);
            c.n = c.eq.clone();
            let normalization = if normalize { c.normalization } else { 1.0 };

            #[cfg(not(feature = "parallel"))]
            let values: Vec<_> = self
                .model
                .interactions()
                .iter()
                .enumerate()
                .map(|(j, interaction)| (j, interaction.gamma(&c, true).map(|v| v * normalization)))
                .collect();

            #[cfg(feature = "parallel")]
            let values: Vec<_> = self
                .model
                .interactions()
                .par_iter()
                .enumerate()
                .map(|(j, interaction)| (j, interaction.gamma(&c, true).map(|v| v * normalization)))
                .collect();

            for (j, v) in values {
                gammas[[i, j + 1]] = v;
            }
        }

        let mut names = vec!["beta".to_string()];
        names.extend(self.model.interactions().iter().map(|interaction| {
            let ptcls = interaction.particles();
            ptcls
                .display(&self.model)
                .unwrap_or_else(|_| format!("{}", ptcls))
        }));

        (names, gammas)
    }

    /// Compute the asymmetric interaction rates.
    ///
    /// The arguments and returned values are identical to [`Solver::gammas`].
    pub fn asymmetries(
        &mut self,
        size: usize,
        normalize: bool,
    ) -> (Vec<String>, Array2<Option<f64>>) {
        let mut gammas = Array2::from_elem((size, self.model.interactions().len() + 1), None);
        let n = Array1::zeros(self.model.particles().len());
        let na = Array1::zeros(n.dim());

        for (i, &beta) in Array1::geomspace(self.beta_range.0, self.beta_range.1, size)
            .unwrap_or_else(|| Array1::zeros(10))
            .iter()
            .enumerate()
        {
            self.model.set_beta(beta);
            gammas[[i, 0]] = Some(beta);
            let mut c = self.context(0, None, 1.0, (beta, beta), &n, &na, None);
            c.n = c.eq.clone();
            let normalization = if normalize { c.normalization } else { 1.0 };

            #[cfg(not(feature = "parallel"))]
            let values: Vec<_> = self
                .model
                .interactions()
                .iter()
                .enumerate()
                .map(|(j, interaction)| {
                    (
                        j,
                        interaction.delta_gamma(&c, true).map(|v| v * normalization),
                    )
                })
                .collect();

            #[cfg(feature = "parallel")]
            let values: Vec<_> = self
                .model
                .interactions()
                .par_iter()
                .enumerate()
                .map(|(j, interaction)| {
                    (
                        j,
                        interaction.delta_gamma(&c, true).map(|v| v * normalization),
                    )
                })
                .collect();

            for (j, v) in values {
                gammas[[i, j + 1]] = v;
            }
        }

        let mut names = vec!["beta".to_string()];
        names.extend(self.model.interactions().iter().map(|interaction| {
            let ptcls = interaction.particles();
            ptcls
                .display(&self.model)
                .unwrap_or_else(|_| format!("{}", ptcls))
        }));

        (names, gammas)
    }
}

impl<M> Solver<M>
where
    M: Model,
{
    /// Generate the context at a given beta to pass to the logger/interaction
    /// functions.
    #[allow(clippy::too_many_arguments)]
    fn context(
        &self,
        step: usize,
        substep: Option<usize>,
        step_size: f64,
        (beta, next_beta): (f64, f64),
        n: &Array1<f64>,
        na: &Array1<f64>,
        fast_interactions: Option<RwLock<HashSet<interaction::Particles>>>,
    ) -> Context<M> {
        let hubble_rate = self.model.hubble_rate(beta);
        let normalization =
            (hubble_rate * beta * Statistic::BoseEinstein.number_density(beta, 0.0, 0.0)).recip();
        Context {
            step,
            substep: substep.map_or(-1, |v| TryFrom::try_from(v).unwrap()),
            step_size,
            beta,
            hubble_rate,
            normalization,
            eq: equilibrium_number_densities(self.model.particles(), beta),
            eqn: equilibrium_number_densities(&self.particles_next, next_beta),
            n: n.clone(),
            na: na.clone(),
            model: &self.model,
            fast_interactions,
            in_equilibrium: &self.in_equilibrium,
            no_asymmetry: &self.no_asymmetry,
        }
    }
}

/// Create an array containing the equilibrium number densities of the model's
/// particles at the specified temperature.
///
/// All particles species are assumed to be in thermal equilibrium at this
/// energy, with the distribution following either the Bose–Einstein or
/// Fermi–Dirac distribution as determined by their spin.
///
/// The `normalized` flag
fn equilibrium_number_densities<'a, I>(particles: I, beta: f64) -> Array1<f64>
where
    I: IntoIterator<Item = &'a ParticleData>,
{
    particles
        .into_iter()
        .map(|p| p.normalized_number_density(beta, 0.0))
        .collect()
}
