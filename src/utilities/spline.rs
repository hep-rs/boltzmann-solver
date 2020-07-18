//! Cubic Hermite interpolation

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::f64;

/// Cubic Hermite spline interpolator using a constant data array
#[allow(clippy::module_name_repetitions)]
pub struct ConstCubicHermiteSpline {
    /// Data array arranged in triples of `(xi, yi, mi)` where `xi`, `yi` are x
    /// and y values of a particular point, and `mi` is the gradient for that
    /// point.
    ///
    /// The data array *must* be sorted in increasing `x` values.
    pub data: &'static [(f64, f64, f64)],
}

impl ConstCubicHermiteSpline {
    /// Sample the spline at the specific `x` value.
    ///
    /// For values of `x` outside of the domain of the underlying data, the
    /// boundary value is returned.
    #[must_use]
    pub fn sample(&self, x: f64) -> f64 {
        match self
            .data
            .binary_search_by(|&(xi, _, _)| xi.partial_cmp(&x).unwrap())
        {
            Ok(i) => self.data[i].1,
            Err(0) => self.data[0].1,
            Err(i) if i == self.data.len() => self.data[i - 1].1,
            Err(i) => {
                let (x0, y0, m0) = self.data[i - 1];
                let (x1, y1, m1) = self.data[i];

                let t = (x - x0) / (x1 - x0);
                let t2 = t.powi(2);
                let t3 = t.powi(3);
                y0 * (2.0 * t3 - 3.0 * t2 + 1.0)
                    + m0 * (t3 - 2.0 * t2 + t)
                    + y1 * (-2.0 * t3 + 3.0 * t2)
                    + m1 * (t3 - t2)
            }
        }
    }
}

/// Single point within a cubic Hermite spline going through the coordinate `(x, y)`.
///
/// The gradient at the point is `m` and is normalized to the size of the
/// interval between `x` and the next value (and thus is not meaningful without
/// being part of a spline).
///
/// The `accurate` flag indicates whether the interval between the current point
/// and the next point is deemed accurate.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
struct CubicHermiteSplinePoint {
    x: f64,
    y: f64,
    m: f64,
    accurate: bool,
}

impl CubicHermiteSplinePoint {
    /// Create a new spline point going through coordinate `(x, y)`.
    ///
    /// The gradient will be NaN initially, and `accurate` is set to `false`.
    fn new(x: f64, y: f64) -> Self {
        CubicHermiteSplinePoint {
            x,
            y,
            m: f64::NAN,
            accurate: false,
        }
    }
}

/// Cubic Hermite spline interpolator
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[allow(clippy::module_name_repetitions)]
pub struct CubicHermiteSpline {
    // `(x, y, m, accurate)` tuples through which the spline goes through, with
    // gradient `m`.  The accurate flag determines whether the interval between
    // it and the next value is accurate or not.
    data: Vec<CubicHermiteSplinePoint>,
    // Number of points required before it begins to consider whether an
    // interval is accurate.
    min_points: usize,
}

impl CubicHermiteSpline {
    /// Create a new empty cubic Hermite Spline.
    #[must_use]
    pub fn empty() -> Self {
        CubicHermiteSpline {
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
                self.data.insert(0, CubicHermiteSplinePoint::new(x, y));

                self.compute_gradient(0);
                self.compute_gradient(1);
            }
            Err(i) if i == self.data.len() => {
                self.data.insert(i, CubicHermiteSplinePoint::new(x, y));

                self.compute_gradient(i - 1);
                self.compute_gradient(i);
            }
            Err(i) => {
                let ny = self.sample(x);

                self.data.insert(i, CubicHermiteSplinePoint::new(x, y));

                self.compute_gradient(i - 1);
                self.compute_gradient(i);
                self.compute_gradient(i + 1);

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

    /// Compute the gradient of point `i`.
    ///
    /// If `i` is bigger than the number of points within the spline, this
    /// function does nothing.  Similarly, this function will not do anything
    /// until there are at least 2 points within the spline.
    fn compute_gradient(&mut self, i: usize) {
        if self.data.len() < 2 || i > self.data.len() {
            return;
        }

        if i == 0 {
            self.data[0].m = self.data[1].y - self.data[0].y;
        } else if i == self.data.len() - 1 {
            self.data[i].m = self.data[i].y - self.data[i - 1].y;
        } else {
            let p0 = &self.data[i - 1];
            let p1 = &self.data[i];
            let p2 = &self.data[i + 1];
            self.data[i].m = 0.5
                * ((p2.y - p1.y) / (p2.x - p1.x) + (p1.y - p0.y) / (p1.x - p0.x))
                * (p2.x - p1.x);
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
                let t2 = t.powi(2);
                let t3 = t.powi(3);
                p0.y * (2.0 * t3 - 3.0 * t2 + 1.0)
                    + p0.m * (t3 - 2.0 * t2 + t)
                    + p1.y * (-2.0 * t3 + 3.0 * t2)
                    + p1.m * (t3 - t2)
            }
        }
    }
}

/// Create a recursively generated geometrically spaced interval between `start`
/// and `end`.
///
/// This is the analogous version of the recursively generated linearly spaced
/// interval [`rec_linspace`].
#[must_use]
pub fn rec_geomspace(start: f64, end: f64, recursions: u32) -> Vec<f64> {
    let mut v = Vec::with_capacity(2_usize.pow(recursions));

    v.push(start);
    v.push(end);

    let start = start.ln();
    let end = end.ln();

    let mut base = 2.0;
    for i in 2..2_u64.pow(recursions) {
        let i = i as f64;
        if i > base {
            base *= 2.0;
        }

        let b = 2.0 * i - base - 1.0;
        let a = base - b;

        v.push(((a * start + b * end) / base).exp())
    }
    v
}

/// Create a recursively generated linearly spaced interval spanning `start` and
/// `end` inclusively.
///
/// The resulting vector will have values in the following order:
///
/// ```math
/// \left[
///   a,
///   b,
///   \frac{a + b}{2},
///   \frac{3a + b}{4}, \frac{a + 3b}{4},
///   \frac{7a + b}{8}, \frac{5a + 3b}{8}, \frac{3a + 5b}{8}, \frac{a + 8b}{8},
///   \dots
/// \right]
/// ```
///
/// The number of recursions is determined by `recursions`.
#[must_use]
pub fn rec_linspace(start: f64, end: f64, recursions: u32) -> Vec<f64> {
    let mut v = Vec::with_capacity(2_usize.pow(recursions));

    v.push(start);
    v.push(end);

    let mut base = 2.0;
    for i in 2..2_u64.pow(recursions) {
        let i = i as f64;
        if i > base {
            base *= 2.0;
        }

        let b = 2.0 * i - base - 1.0;
        let a = base - b;

        v.push((a * start + b * end) / base)
    }
    v
}

#[cfg(test)]
mod test {
    use ndarray::prelude::*;
    use std::{env, error, f64, fs, io::prelude::*, io::BufWriter};

    pub(crate) const RECURSIONS: u32 = 12;

    pub(crate) fn f(x: f64) -> f64 {
        x * (x - 1.0) * (x - 0.2) * (x - 0.5) * f64::exp(-3.0 * f64::exp(x))
    }

    #[test]
    fn spline() -> Result<(), Box<dyn error::Error>> {
        let mut spline = super::CubicHermiteSpline::empty();

        let mut path = env::temp_dir();
        path.push("sampled.csv");
        let mut sampled = BufWriter::new(fs::File::create(path)?);
        for x in super::rec_linspace(0.0, 1.0, RECURSIONS) {
            if !spline.accurate(x) {
                spline.add(x, f(x));
            }
        }

        for p in &spline.data {
            writeln!(sampled, "{:e},{:e},{:e}", p.x, p.y, p.m)?;
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

#[cfg(all(test, feature = "nightly"))]
mod benches {
    use super::test::{f, RECURSIONS};
    use ndarray::prelude::*;
    use test::Bencher;

    #[bench]
    fn spline_add_linear(b: &mut Bencher) {
        let data: Vec<_> = Array1::linspace(0.0, 1.0, 2_usize.pow(RECURSIONS))
            .iter()
            .map(|&x| (x, f(x)))
            .collect();

        b.iter(|| {
            let mut spline = super::CubicHermiteSpline::empty();
            for &(x, fx) in &data {
                if !spline.accurate(x) {
                    spline.add(x, fx);
                }
            }
        })
    }

    #[bench]
    fn spline_add_recursive(b: &mut Bencher) {
        let data: Vec<_> = super::rec_linspace(0.0, 1.0, RECURSIONS)
            .iter()
            .map(|&x| (x, f(x)))
            .collect();

        b.iter(|| {
            let mut spline = super::CubicHermiteSpline::empty();
            for &(x, fx) in &data {
                if !spline.accurate(x) {
                    spline.add(x, fx);
                }
            }
        })
    }

    #[bench]
    fn spline_sample(b: &mut Bencher) {
        let mut spline = super::CubicHermiteSpline::empty();
        for x in super::rec_linspace(0.0, 1.0, RECURSIONS) {
            if !spline.accurate(x) {
                spline.add(x, f(x));
            }
        }

        let xs = Array1::linspace(0.0, 1.0, 2_usize.pow(RECURSIONS));

        b.iter(|| {
            for &x in &xs {
                test::black_box(spline.sample(x));
            }
        })
    }
}
