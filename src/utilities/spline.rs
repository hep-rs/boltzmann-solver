//! Spline implementation
//!
//! Two implementations are provided: linear and cubic.

mod cubic;
mod linear;

pub use cubic::{ConstCubicHermite, CubicHermite};
pub use linear::{ConstLinear, Linear};

pub use linear::Linear as Spline;
// pub use cubic::CubicHermiteSpline as Spline;
