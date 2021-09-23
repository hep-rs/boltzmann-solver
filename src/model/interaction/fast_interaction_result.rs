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
    /// Array of changes to be added to the number density asymmetries.
    pub dna: Array1<f64>,
    /// Array of error estimates of changes to be added to the number densities.
    pub dn_error: Array1<f64>,
    /// Array of error estimates of changes to be added to the number density asymmetries.
    pub dna_error: Array1<f64>,
}

impl FastInteractionResult {
    /// Create a new interaction result filled with 0 values and the specified
    /// size for the [`dn`](FastInteractionResult::dn) and
    /// [`dna`](FastInteractionResult::dna) arrays.
    #[must_use]
    pub fn zero(n: usize) -> Self {
        Self {
            dn: Array1::zeros(n),
            dna: Array1::zeros(n),
            dn_error: Array1::zeros(n),
            dna_error: Array1::zeros(n),
        }
    }
}

impl ops::Add<Self> for FastInteractionResult {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            dn: self.dn + rhs.dn,
            dna: self.dna + rhs.dna,
            dn_error: self.dn_error + rhs.dn_error,
            dna_error: self.dna_error + rhs.dna_error,
        }
    }
}

impl ops::AddAssign<&Self> for FastInteractionResult {
    fn add_assign(&mut self, rhs: &Self) {
        self.dn += &rhs.dn;
        self.dna += &rhs.dna;
        self.dn_error += &rhs.dn_error;
        self.dna_error += &rhs.dna_error;
    }
}

impl ops::AddAssign<Self> for FastInteractionResult {
    fn add_assign(&mut self, rhs: Self) {
        self.dn += &rhs.dn;
        self.dna += &rhs.dna;
        self.dn_error += &rhs.dn_error;
        self.dna_error += &rhs.dna_error;
    }
}

impl fmt::Display for FastInteractionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Fast Interaction Result: max |δn| = {:e}, max |δΔ| = {:e}",
            self.dn
                .iter()
                .map(|v| v.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0),
            self.dna
                .iter()
                .map(|v| v.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0)
        )
    }
}
