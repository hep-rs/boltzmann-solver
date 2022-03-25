use crate::statistic::Statistic;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::cmp;

/// Representation under the Lorentz group.  Although the physics convention is
/// to use half-integer multiples, we use integer weights instead to make it
/// easier to store and manipulate.  Thus a Weyl spinor will be `$(1, 0)$`
/// instead of `$(\sfrac{1}{2}, 0)$`.
#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub enum LorentzRepresentation {
    /// A simple representation under the Lorentz group.
    Simple(u8, u8),
    /// A representation which is made from the sum of two simple
    /// representation.
    ///
    /// The most common use case is for a Dirac spinor which is composed from
    /// the sum of a left- and right-handed Weyl spinor.
    Sum((u8, u8), (u8, u8)),
}

impl LorentzRepresentation {
    /// Create a new Lorentz representation.
    ///
    /// The first argument is the left-handed representation and the second is
    /// for the right-handed representation.
    ///
    /// # Panics
    ///
    /// If the left and right handed representations are identical, this panics.
    pub fn new(left: (u8, u8), right: impl Into<Option<(u8, u8)>>) -> Self {
        let (l1, r1) = left;
        match right.into() {
            None => LorentzRepresentation::Simple(l1, r1),
            Some((l2, r2)) => match l1.cmp(&l2) {
                cmp::Ordering::Less => LorentzRepresentation::Sum((l2, r2), (l1, r1)),
                cmp::Ordering::Greater => LorentzRepresentation::Sum((l1, r1), (l2, r2)),
                cmp::Ordering::Equal => match r1.cmp(&r2) {
                    cmp::Ordering::Less => LorentzRepresentation::Sum((l2, r2), (l1, r1)),
                    cmp::Ordering::Greater => LorentzRepresentation::Sum((l1, r1), (l2, r2)),
                    cmp::Ordering::Equal => panic!("Cannot add two equal representations"),
                },
            },
        }
    }

    /// Returns true of the representation is fermionic
    #[must_use]
    pub fn is_fermionic(&self) -> bool {
        match self {
            LorentzRepresentation::Simple(l, r) => l % 2 == 1 || r % 2 == 1,
            LorentzRepresentation::Sum((a, b), (c, d)) => {
                a % 2 == 1 || b % 2 == 1 || c % 2 == 1 || d % 2 == 1
            }
        }
    }

    /// Returns true of the representation is bosonic
    #[must_use]
    pub fn is_bosonic(&self) -> bool {
        match self {
            LorentzRepresentation::Simple(l, r) => l % 2 == 0 && r % 2 == 0,
            LorentzRepresentation::Sum((a, b), (c, d)) => {
                a % 2 == 0 && b % 2 == 0 && c % 2 == 0 && d % 2 == 0
            }
        }
    }

    /// Return the corresponding quantum statistic distribution.
    #[must_use]
    pub fn statistic(&self) -> Statistic {
        if self.is_fermionic() {
            Statistic::FermiDirac
        } else {
            Statistic::BoseEinstein
        }
    }

    /// Return the spin of the representation.
    ///
    /// # Panics
    ///
    /// If the representation is a direct sum, both must have the same spin.
    #[must_use]
    pub fn spin(&self) -> u8 {
        match self {
            LorentzRepresentation::Simple(l, r) => l + r,
            LorentzRepresentation::Sum((a, b), (c, d)) => {
                if a + b == c + d {
                    a + b
                } else {
                    panic!("representation is a direct sum, but the spins are different")
                }
            }
        }
    }
}

/// Build a [`LorentzRepresentation`] with `rep!(a, b)` for simple
/// representations, and either `rep!((a, b) + (c, d))` or `rep!((a, b), (c,
/// d))` for representations which are direct sums.
///
/// The sum representations are automatically sorted so that `a >= c` (and
/// if equal `b > d`).  Thus the Dirac spinor is always `$(1, 0) \otimes (0,
/// 1)$` and not the other way around.
macro_rules! rep {
    ($l:expr, $r:expr) => {
        LorentzRepresentation::Simple($l, $r)
    };
    (($l1:expr, $r1:expr), ($l2:expr,$r2:expr)) => {
        rep!(($l1, $r1) + ($l2, $r2))
    };
    (($l1:expr, $r1:expr) + ($l2:expr,$r2:expr)) => {
        match $l1.cmp(&$l2) {
            cmp::Ordering::Less => LorentzRepresentation::Sum(($l2, $r2), ($l1, $r1)),
            cmp::Ordering::Greater => LorentzRepresentation::Sum(($l1, $r1), ($l2, $r2)),
            cmp::Ordering::Equal => match $r1.cmp(&$r2) {
                cmp::Ordering::Less => LorentzRepresentation::Sum(($l2, $r2), ($l1, $r1)),
                cmp::Ordering::Greater => LorentzRepresentation::Sum(($l1, $r1), ($l2, $r2)),
                cmp::Ordering::Equal => panic!("Cannot add two equal representations"),
            },
        }
    };
}

/// Scalar (trivial) representation
pub const SCALAR: LorentzRepresentation = rep!(0, 0);
/// Lorentz 4-vector
pub const VECTOR: LorentzRepresentation = rep!(1, 1);
/// Traceless symmetric tensor (e.g. `$g_{\mu\nu}$`)
pub const TENSOR: LorentzRepresentation = rep!(2, 2);

/// Left-handed Weyl spinor
pub const LEFT_WEYL_SPINOR: LorentzRepresentation = rep!(1, 0);
/// Right-handed Weyl spinor
pub const RIGHT_WEYL_SPINOR: LorentzRepresentation = rep!(0, 1);

/// Dirac spinor
pub const DIRAC_SPINOR: LorentzRepresentation = LorentzRepresentation::Sum((1, 0), (0, 1));
