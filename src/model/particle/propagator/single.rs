// TODO: Find out how best to replace particle widths with particle self-energies.

use super::{Multi, Propagator};
use crate::prelude::{Model, ParticleData};
use num::{Complex, One};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops,
};

#[cfg(feature = "min-width")]
const MIN_ABSORPTIVE: f64 = 1e-3;

/// A propagator for a single particle with the real intermediate state removed.
///
/// The propagator is defined as
#[derive(Debug, Copy, Clone)]
pub struct Single {
    particle: u64,
    momentum: f64,
    mass2: f64,
    absorptive: f64,
    numerator: Complex<f64>,
    pub(crate) conjugated: bool,
}

impl Single {
    /// Create a new propagator.
    ///
    /// The momentum transfer is specified by the Lorentz-invariant quantity
    /// `$p^2$`.
    pub(crate) fn new<M: Model>(p: &ParticleData, model: &M, momentum: f64) -> Self {
        #[cfg(feature = "min-width")]
        let absorptive = {
            let absorptive = model.self_energy_absorptive(p, momentum);
            if f64::abs(momentum / p.mass2 - 1.0) < MIN_ABSORPTIVE
                && absorptive < p.mass2 * MIN_ABSORPTIVE
            {
                // log::warn!(
                //     "[{}, β={:.3e}, q²={:.3e}, m²={:.3e}] Absorptive self-energy is {:.3e}, setting it to {:.3e}.",
                //     p.name,
                //     model.get_beta(),
                //     momentum,
                //     p.mass2,
                //     absorptive,
                //     MIN_ABSORPTIVE * p.mass2,
                // );
                p.mass2 * MIN_ABSORPTIVE
            } else {
                absorptive
            }
        };
        #[cfg(not(feature = "min-width"))]
        let absorptive = model.self_energy_absorptive(p, momentum);

        Single {
            particle: {
                let mut s = DefaultHasher::new();
                p.hash(&mut s);
                s.finish()
            },
            momentum,
            mass2: p.mass2,
            absorptive,
            numerator: One::one(),
            conjugated: false,
        }
    }

    /// Evaluate the propagator denominator.
    fn denominator(&self) -> Complex<f64> {
        if self.conjugated {
            Complex::new(self.momentum - self.mass2, self.absorptive)
        } else {
            Complex::new(self.momentum - self.mass2, -self.absorptive)
        }
    }

    /// Equivalent to `self * rhs.conj()` without the need to mutate the
    /// other.
    pub(crate) fn mul_conj(&self, rhs: &Self) -> Complex<f64> {
        debug_assert!(
                self.conjugated == rhs.conjugated,
                "Multiplying propagators with the same complex conjugation (after the automatic conjugation)."
            );

        #[allow(clippy::float_cmp)]
        if self.particle == rhs.particle && self.momentum == rhs.momentum {
            let numerator = (self.momentum - self.mass2).powi(2) - self.absorptive.powi(2);
            let denominator = (self.momentum - self.mass2).powi(2) + self.absorptive.powi(2);

            self.numerator * rhs.numerator.conj() * numerator / denominator.powi(2)
        } else {
            self.numerator
                * rhs.numerator.conj()
                * (self.denominator() * rhs.denominator().conj()).finv()
        }
    }
}

impl Propagator for Single {
    fn conj(mut self) -> Self {
        self.numerator = self.numerator.conj();
        self.conjugated = !self.conjugated;
        self
    }

    fn ref_conj(&mut self) -> &mut Self {
        self.numerator = self.numerator.conj();
        self.conjugated = !self.conjugated;
        self
    }

    fn eval(&self) -> Complex<f64> {
        self.numerator * self.denominator().finv()
    }

    fn norm_sqr(&self) -> f64 {
        let propagator_squared = if self.momentum > 0.0 {
            let numerator = (self.momentum - self.mass2).powi(2) - self.absorptive.powi(2);
            let denominator = (self.momentum - self.mass2).powi(2) + self.absorptive.powi(2);

            numerator / denominator.powi(2)
        } else {
            (self.momentum - self.mass2).powi(-2)
        };

        self.numerator.norm_sqr() * propagator_squared
    }
}

impl ops::Add<Self> for Single {
    type Output = Multi;

    /// Add two propagators, combining them into a [`Multi`].
    ///
    /// ## Panic
    ///
    /// The two propagators should have the same conjugation or this will panic.
    /// The check is only done on debug builds.
    fn add(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.conjugated == rhs.conjugated,
            "When adding two propagators, they must have the same conjugation."
        );

        Multi {
            conjugated: self.conjugated,
            propagators: vec![self, rhs],
        }
    }
}

impl ops::Sub<Self> for Single {
    type Output = Multi;

    /// Take the difference of two propagators, combining them into a
    /// [`Multi`].
    ///
    /// ## Panic
    ///
    /// The two propagators should have the same conjugation or this will panic.
    /// The check is only done on debug builds.
    fn sub(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.conjugated == rhs.conjugated,
            "When adding two propagators, they must have the same conjugation."
        );

        Multi {
            conjugated: self.conjugated,
            propagators: vec![self, -rhs],
        }
    }
}

impl<'a, 'x> ops::Mul<Self> for Single {
    type Output = Complex<f64>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.conjugated != rhs.conjugated,
            "When multiplying two propagators, they must have the opposite conjugation."
        );

        // if self.particle.width == 0.0 && (self.momentum - self.mass2).abs() < 1e-1 {
        //     log::warn!("Evaluating propagator near pole of stable particle.");
        // }

        #[allow(clippy::float_cmp)]
        if self.particle == rhs.particle && self.momentum == rhs.momentum {
            let numerator = (self.momentum - self.mass2).powi(2) - self.absorptive.powi(2);
            let denominator = (self.momentum - self.mass2).powi(2) + self.absorptive.powi(2);

            self.numerator * rhs.numerator * numerator / denominator.powi(2)
        } else {
            self.numerator * rhs.numerator * (self.denominator() * rhs.denominator()).finv()
        }
    }
}

impl<'a, 'x> ops::Mul<Self> for &'x Single {
    type Output = Complex<f64>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.conjugated != rhs.conjugated,
            "When multiplying two propagators, they must have the opposite conjugation."
        );

        // if self.particle.width == 0.0 && (self.momentum - self.mass2).abs() < 1e-1 {
        //     log::warn!("Evaluating propagator near pole of stable particle.");
        // }

        #[allow(clippy::float_cmp)]
        if self.particle == rhs.particle && self.momentum == rhs.momentum {
            let numerator = (self.momentum - self.mass2).powi(2) - self.absorptive.powi(2);
            let denominator = (self.momentum - self.mass2).powi(2) + self.absorptive.powi(2);

            self.numerator * rhs.numerator * numerator / denominator.powi(2)
        } else {
            self.numerator * rhs.numerator * (self.denominator() * rhs.denominator()).finv()
        }
    }
}

impl ops::Mul<f64> for Single {
    type Output = Self;

    fn mul(mut self, rhs: f64) -> Self::Output {
        self.numerator *= rhs;
        self
    }
}

impl ops::Mul<Single> for f64 {
    type Output = Single;

    fn mul(self, rhs: Single) -> Self::Output {
        rhs * self
    }
}

impl ops::MulAssign<f64> for Single {
    fn mul_assign(&mut self, rhs: f64) {
        self.numerator *= rhs;
    }
}

impl ops::Mul<Complex<f64>> for Single {
    type Output = Self;

    fn mul(mut self, rhs: Complex<f64>) -> Self::Output {
        self.numerator *= rhs;
        self
    }
}

impl ops::Mul<Single> for Complex<f64> {
    type Output = Single;

    fn mul(self, rhs: Single) -> Self::Output {
        rhs * self
    }
}

impl ops::MulAssign<Complex<f64>> for Single {
    fn mul_assign(&mut self, rhs: Complex<f64>) {
        self.numerator *= rhs;
    }
}

impl ops::Div<f64> for Single {
    type Output = Self;

    fn div(mut self, rhs: f64) -> Self {
        self.numerator /= rhs;
        self
    }
}

impl ops::DivAssign<f64> for Single {
    fn div_assign(&mut self, rhs: f64) {
        self.numerator /= rhs;
    }
}

impl ops::Div<Complex<f64>> for Single {
    type Output = Self;

    fn div(mut self, rhs: Complex<f64>) -> Self {
        self.numerator /= rhs;
        self
    }
}

impl ops::DivAssign<Complex<f64>> for Single {
    fn div_assign(&mut self, rhs: Complex<f64>) {
        self.numerator /= rhs;
    }
}

impl ops::Neg for Single {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.numerator = -self.numerator;
        self
    }
}
