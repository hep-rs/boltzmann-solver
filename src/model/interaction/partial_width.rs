#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::{convert::TryFrom, fmt};

/// Partial width associated with a single specific interaction.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct PartialWidth {
    /// Partial width of this process in GeV
    pub width: f64,
    /// Parent particle decaying.  This is signed to distinguish between
    /// particles (> 0) an antiparticles (< 0).
    pub parent: isize,
    /// Daughter particles of the decay, signed just as the parent particle.
    pub daughters: Vec<isize>,
}

impl PartialWidth {
    /// Return the parent particle index
    #[must_use]
    pub fn parent_idx(&self) -> usize {
        usize::try_from(self.parent.abs()).expect("parent particle index")
    }

    /// Return the daughter particle indices
    #[must_use]
    pub fn daughters_idx(&self) -> Vec<usize> {
        self.daughters
            .iter()
            .map(|p| usize::try_from(p.abs()).expect("daughter particle index"))
            .collect()
    }
}

impl fmt::Display for PartialWidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PartialWidth {{ width: {:e}, parent: {}, daughters: {:?} }}",
            self.width, self.parent, self.daughters
        )
    }
}
