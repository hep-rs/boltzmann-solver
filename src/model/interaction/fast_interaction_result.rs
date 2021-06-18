use ndarray::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::{fmt, ops};

/// Result from a fast interaction.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct FastInteractionResult {
    /// Array of changes to be added to the number densities.
    pub dn: Array1<f64>,
    /// Value of the change in number density.  This is equivalent to
    /// `self.dn.abs().max()` provided that no particle is repeated in the
    /// interaction.
    pub symmetric_delta: f64,
    /// Array of changes to be added to the number density asymmetries.
    pub dna: Array1<f64>,
    /// Value of teh change in number density asymmetry.  This is equivalent to
    /// `self.dna.abs().max()` provided that no particle is repeated.
    pub asymmetric_delta: f64,
}

impl FastInteractionResult {
    /// Create a new interaction result filled with 0 values and the specified
    /// size for the [`dn`](FastInteractionResult::dn) and
    /// [`dna`](FastInteractionResult::dna) arrays.
    #[must_use]
    pub fn zero(n: usize) -> Self {
        Self {
            dn: Array1::zeros(n),
            symmetric_delta: 0.0,
            dna: Array1::zeros(n),
            asymmetric_delta: 0.0,
        }
    }
}

impl ops::AddAssign<&Self> for FastInteractionResult {
    fn add_assign(&mut self, rhs: &Self) {
        self.dn += &rhs.dn;
        self.symmetric_delta += rhs.symmetric_delta;
        self.dna += &rhs.dna;
        self.asymmetric_delta += rhs.asymmetric_delta;
    }
}

impl fmt::Display for FastInteractionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Fast Interaction Result: δ = {:e}, δ' = {:e}",
            self.symmetric_delta, self.asymmetric_delta
        )
    }
}
