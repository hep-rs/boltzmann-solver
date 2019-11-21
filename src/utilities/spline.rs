//! Cubic Hermite interpolation

/// Cubic Hermite spline interpolator using a constant data array
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

/// Cubic Hermite spline interpolator
pub struct CubicHermiteSpline {
    // `(x, y)` pairs through which the spline goes through
    data: Vec<(f64, f64)>,
    // Gradient at each point.
    gradients: Vec<f64>,
    // Segments where the spline is deemed to be accurate.  The value of
    // `accurate[i]` determines the accuracy for the interval `x[i]` to
    // `x[i+1]`.
    accurate: Vec<bool>,
    // Number of points required before it begins to consider whether an
    // interval is accurate.
    min_points: usize,
}

impl CubicHermiteSpline {
    /// Create a new empty Cubic Hermite Spline
    pub fn empty() -> Self {
        CubicHermiteSpline {
            data: Vec::new(),
            gradients: Vec::new(),
            accurate: Vec::new(),
            min_points: 16,
        }
    }

    /// Return the number of data points in the underlying data
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Adjust the minimum number of points before accuracy is considered.
    pub fn min_points(&mut self, min_points: usize) {
        self.min_points = min_points;
    }

    /// Adds a data point to the spline.
    ///
    /// If the `x` value is already present, the input is ignored.
    ///
    /// If the data point is within the range that is being interpolated, the
    /// given value is compared to the interpolated value and if they are in
    /// good agreement, both sides adjacent to the new point are marked as
    /// accurate (provided the spline has reached the minimum number of points).
    pub fn add(&mut self, x: f64, y: f64) {
        match self
            .data
            .binary_search_by(|&(xi, _)| xi.partial_cmp(&x).unwrap())
        {
            Ok(_) => (),
            Err(0) => {
                self.data.insert(0, (x, y));
                self.gradients.insert(0, 0.0);
                self.accurate.insert(0, false);

                self.compute_gradient(0);
                self.compute_gradient(1);
            }
            Err(i) if i == self.data.len() => {
                self.data.insert(i, (x, y));
                self.gradients.insert(i, 0.0);
                self.accurate.insert(i, false);

                self.compute_gradient(i - 1);
                self.compute_gradient(i);
            }
            Err(i) => {
                let ny = self.sample(x);

                self.data.insert(i, (x, y));
                self.gradients.insert(i, 0.0);
                self.accurate.insert(i, false);

                self.compute_gradient(i - 1);
                self.compute_gradient(i);
                self.compute_gradient(i + 1);

                let delta = (ny - y).abs();
                if self.data.len() > self.min_points
                    && (delta / (ny.abs() + y.abs()) < 0.05 || delta < 1e-3)
                {
                    self.accurate[i - 1] = true;
                    self.accurate[i] = true;
                }
            }
        }
    }

    fn compute_gradient(&mut self, i: usize) {
        if self.data.len() < 2 || i > self.data.len() {
            return;
        }

        if i == 0 {
            self.gradients[0] = self.data[1].1 - self.data[0].1;
        } else if i == self.data.len() - 1 {
            self.gradients[i] = self.data[i].1 - self.data[i - 1].1;
        } else {
            let (x0, y0) = self.data[i - 1];
            let (x1, y1) = self.data[i];
            let (x2, y2) = self.data[i + 1];
            self.gradients[i] = 0.5 * ((y2 - y1) / (x2 - x1) + (y1 - y0) / (x1 - x0)) * (x2 - x1)
        }
    }

    /// Check whether a given interval is accurate or not.
    ///
    /// If the point checked falls outside of the boundary, this is
    /// automatically determined to be false.  If the `x` value is a known
    /// control point, then `true` is returned even if the interval on either
    /// side might not be accurate.
    pub fn accurate(&self, x: f64) -> bool {
        match self
            .data
            .binary_search_by(|&(xi, _)| xi.partial_cmp(&x).unwrap())
        {
            Ok(_) => true,
            Err(0) => false,
            Err(i) if i == self.data.len() => false,
            Err(i) => self.accurate[i],
        }
    }

    /// Sample the spline at the specific `x` value.
    ///
    /// For values of `x` outside of the domain of the underlying data, the
    /// boundary value is returned.
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
                let m0 = self.gradients[i - 1];
                let m1 = self.gradients[i];

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

/// Create a recursively generated geometrically spaced interval between `start`
/// and `end`.
///
/// This is the analogous version of the recursively generated linearly spaced
/// interval [`rec_linspace`].
pub fn rec_geomspace(start: f64, end: f64, recursions: u32) -> Vec<f64> {
    let mut v = Vec::with_capacity(2usize.pow(recursions));

    v.push(start);
    v.push(end);

    let start = start.ln();
    let end = end.ln();

    let mut base = 2.0;
    for i in 2..2u64.pow(recursions) {
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
/// \\begin{equation}
///   \left[
///     a,
///     b,
///     \frac{a + b}{2},
///     \frac{3a + b}{4}, \frac{a + 3b}{4},
///     \frac{7a + b}{8}, \frac{5a + 3b}{8}, \frac{3a + 5b}{8}, \frac{a + 8b}{8},
///     \dots
///   \right]
/// \\end{equation}
///
/// The number of recursions is determined by `recursions`.
pub fn rec_linspace(start: f64, end: f64, recursions: u32) -> Vec<f64> {
    let mut v = Vec::with_capacity(2usize.pow(recursions));

    v.push(start);
    v.push(end);

    let mut base = 2.0;
    for i in 2..2u64.pow(recursions) {
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
    use std::{f64, fs, io::prelude::*, io::BufWriter};

    pub(crate) const RECURSIONS: u32 = 12;

    pub(crate) fn f(x: f64) -> f64 {
        x * (x - 1.0) * (x - 0.2) * (x - 0.5) * f64::exp(-3.0 * f64::exp(x))
    }

    #[test]
    fn spline() {
        let mut spline = super::CubicHermiteSpline::empty();

        let mut path = std::env::temp_dir();
        path.push("sampled.csv");
        let mut sampled = BufWriter::new(fs::File::create(path).unwrap());
        for x in super::rec_linspace(0.0, 1.0, RECURSIONS) {
            if !spline.accurate(x) {
                writeln!(sampled, "{:e},{:e}", x, f(x)).unwrap();
                spline.add(x, f(x))
            }
        }

        let mut path = std::env::temp_dir();
        path.push("spline.csv");
        let mut output = BufWriter::new(fs::File::create(path).unwrap());
        for &x in Array1::linspace(0.0, 1.0, 2usize.pow(RECURSIONS)).iter() {
            writeln!(output, "{:e},{:e}", x, spline.sample(x)).unwrap();
        }
    }
}

#[cfg(all(test, feature = "nightly"))]
mod benches {
    use super::test::{f, RECURSIONS};
    use ndarray::prelude::*;
    use test::Bencher;

    #[bench]
    fn spline_add_linear(b: &mut Bencher) {
        let data: Vec<_> = Array1::linspace(0.0, 1.0, 2usize.pow(RECURSIONS))
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

        let xs = Array1::linspace(0.0, 1.0, 2usize.pow(RECURSIONS));

        b.iter(|| {
            for &x in &xs {
                test::black_box(spline.sample(x));
            }
        })
    }
}
