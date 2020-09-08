//! Linear interpolation

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::f64;

/// Linear spline interpolator using a constant data array
#[allow(clippy::module_name_repetitions)]
pub struct ConstLinearSpline {
    /// Data array arranged in tuples of `(xi, yi)` where `xi`, `yi` are x
    /// and y values of a particular point.
    ///
    /// The data array *must* be sorted in increasing `x` values.
    pub data: &'static [(f64, f64)],
}

impl ConstLinearSpline {
    /// Sample the spline at the specific `x` value.
    ///
    /// For values of `x` outside of the domain of the underlying data, the
    /// boundary value is returned.
    #[must_use]
    pub fn sample(&self, x: f64) -> f64 {
        match self
            .data
            .binary_search_by(|&(xi, _)| xi.partial_cmp(&x).unwrap())
        {
            Ok(i) => self.data[i].1,
            Err(0) => self.data[0].1,
            Err(i) if i == self.data.len() => self.data[i - 1].1,
            Err(i) => {
                let (x0, y0) = self.data[i - 1];
                let (x1, y1) = self.data[i];

                let t = (x - x0) / (x1 - x0);
                y0 + t * (y1 - y0)
            }
        }
    }
}

/// Single point within a linear spline going through the coordinate `(x, y)`.
///
/// The `accurate` flag indicates whether the interval between the current point
/// and the next point is deemed accurate.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
struct LinearSplinePoint {
    x: f64,
    y: f64,
    accurate: bool,
}

impl LinearSplinePoint {
    /// Create a new spline point going through coordinate `(x, y)`.
    ///
    /// The gradient will be NaN initially, and `accurate` is set to `false`.
    fn new(x: f64, y: f64) -> Self {
        LinearSplinePoint {
            x,
            y,
            accurate: false,
        }
    }
}

/// Linear spline interpolator
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[allow(clippy::module_name_repetitions)]
pub struct LinearSpline {
    // `(x, y, m, accurate)` tuples through which the spline goes through, with
    // gradient `m`.  The accurate flag determines whether the interval between
    // it and the next value is accurate or not.
    data: Vec<LinearSplinePoint>,
    // Number of points required before it begins to consider whether an
    // interval is accurate.
    min_points: usize,
}

impl LinearSpline {
    /// Create a new empty linear Spline.
    #[must_use]
    pub fn empty() -> Self {
        LinearSpline {
            data: Vec::new(),
            min_points: 64,
        }
    }

    /// Return the number of data points in the underlying data.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check whether the spline is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Adjust the minimum number of points before accuracy is considered.
    pub fn min_points(&mut self, min_points: usize) {
        self.min_points = min_points;
    }

    /// Adds a data point to the spline.
    ///
    /// If the `x` value is already present, the input is ignored and the spline
    /// remains unchanged.
    ///
    /// If the data point is within the range that is being interpolated, the
    /// given value is compared to the interpolated value and if they are in
    /// good agreement, both sides adjacent to the new point are marked as
    /// accurate (provided the spline has reached the minimum number of points).
    ///
    /// Any addition to the spline which extends the interval will result in the
    /// new interval being marked as inaccurate.
    pub fn add(&mut self, x: f64, y: f64) {
        match self.data.binary_search_by(|p| p.x.partial_cmp(&x).unwrap()) {
            Ok(_) => (),
            Err(0) => {
                self.data.insert(0, LinearSplinePoint::new(x, y));
            }
            Err(i) if i == self.data.len() => {
                self.data.insert(i, LinearSplinePoint::new(x, y));
            }
            Err(i) => {
                let ny = self.sample(x);

                self.data.insert(i, LinearSplinePoint::new(x, y));

                let delta = (ny - y).abs();
                if self.data.len() > self.min_points
                    && (delta / (ny.abs() + y.abs()) < 0.05 || delta < 1e-3)
                {
                    self.data[i - 1].accurate = true;
                    self.data[i].accurate = true;
                }
            }
        }
    }

    /// Check whether a given interval is accurate or not.
    ///
    /// If the point checked falls outside of the boundary, this is
    /// automatically determined to be false.  If the `x` value is a known
    /// control point, then `true` is returned even if the interval on either
    /// side might not be accurate.
    #[must_use]
    pub fn accurate(&self, x: f64) -> bool {
        match self.data.binary_search_by(|p| p.x.partial_cmp(&x).unwrap()) {
            Ok(_) => true,
            Err(0) => false,
            Err(i) if i == self.data.len() => false,
            Err(i) => self.data[i].accurate,
        }
    }

    /// Sample the spline at the specific `x` value.
    ///
    /// For values of `x` outside of the domain of the underlying data, the
    /// boundary value is returned.
    #[must_use]
    pub fn sample(&self, x: f64) -> f64 {
        match self.data.binary_search_by(|p| p.x.partial_cmp(&x).unwrap()) {
            Ok(i) => self.data[i].y,
            Err(0) => self.data[0].y,
            Err(i) if i == self.data.len() => self.data[i - 1].y,
            Err(i) => {
                let p0 = &self.data[i - 1];
                let p1 = &self.data[i];

                let t = (x - p0.x) / (p1.x - p0.x);
                p0.y + t * (p1.y - p0.y)
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::utilities::rec_linspace;
    use ndarray::prelude::*;
    use std::{env, error, f64, fs, io::prelude::*, io::BufWriter};

    pub(crate) const RECURSIONS: u32 = 12;

    pub(crate) fn f(x: f64) -> f64 {
        x * (x - 1.0) * (x - 0.2) * (x - 0.5) * f64::exp(-3.0 * f64::exp(x))
    }

    #[test]
    fn spline() -> Result<(), Box<dyn error::Error>> {
        let mut spline = super::LinearSpline::empty();

        let mut path = env::temp_dir();
        path.push("sampled.csv");
        let mut sampled = BufWriter::new(fs::File::create(path)?);
        for x in rec_linspace(0.0, 1.0, RECURSIONS) {
            if !spline.accurate(x) {
                spline.add(x, f(x));
            }
        }

        for p in &spline.data {
            writeln!(sampled, "{:e},{:e}", p.x, p.y)?;
        }

        let mut path = env::temp_dir();
        path.push("spline.csv");
        let mut output = BufWriter::new(fs::File::create(path)?);
        for &x in Array1::linspace(0.0, 1.0, 2_usize.pow(RECURSIONS)).iter() {
            writeln!(output, "{:e},{:e}", x, spline.sample(x))?;
        }

        Ok(())
    }
}
