//! Spline implementation
//!
//! Two implementations are provided: linear and cubic.

mod cubic;
mod linear;

pub use cubic::{ConstCubicHermiteSpline, CubicHermiteSpline};
pub use linear::{ConstLinearSpline, LinearSpline};

pub use linear::LinearSpline as Spline;
// pub use cubic::CubicHermiteSpline as Spline;
