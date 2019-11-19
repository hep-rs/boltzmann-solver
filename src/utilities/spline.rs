//! Cubic Hermite interpolation

/// Cubic Hermite spline interpolator using constant data
pub(crate) struct ConstCubicHermiteSpline {
    data: &'static [(f64, f64, f64)],
}

impl ConstCubicHermiteSpline {
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
pub(crate) struct CubicHermiteSpline {
    data: Vec<(f64, f64)>,
    gradients: Vec<f64>,
    accurate: Vec<bool>,
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
    /// accurate.
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
                if self.data.len() > self.min_points / 2
                    && (delta / (ny.abs() + y.abs()) < 0.05 || delta < 1e-6)
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

    // fn compute_gradients(&mut self) {
    //     let n = self.data.len();
    //     if n < 2 {
    //         return;
    //     }

    //     self.gradients[0] = self.data[1].1 - self.data[0].1;
    //     for i in 1..(n - 1) {
    //         let (x0, y0) = self.data[i - 1];
    //         let (x1, y1) = self.data[i];
    //         let (x2, y2) = self.data[i + 1];
    //         self.gradients[i] = 0.5 * ((y2 - y1) / (x2 - x1) + (y1 - y0) / (x1 - x0)) * (x2 - x1)
    //     }
    //     self.gradients[n - 1] = self.data[n - 1].1 - self.data[n - 2].1;
    // }

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

pub(crate) fn rec_geomspace(start: f64, end: f64, pow: u32) -> Vec<f64> {
    let mut v = Vec::with_capacity(2usize.pow(pow));

    v.push(start);
    v.push(end);

    let start = start.ln();
    let end = end.ln();

    let mut base = 2.0;
    for i in 2..2u64.pow(pow) {
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

#[allow(unused)]
pub(crate) fn rec_linspace(start: f64, end: f64, pow: u32) -> Vec<f64> {
    let mut v = Vec::with_capacity(2usize.pow(pow));

    v.push(start);
    v.push(end);

    let mut base = 2.0;
    for i in 2..2u64.pow(pow) {
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

    fn f(x: f64) -> f64 {
        x * (x - 1.0) * (x - 0.2) * (x - 0.5) * f64::exp(-3.0 * f64::exp(x))
    }

    #[test]
    fn spline() {
        let mut spline = super::CubicHermiteSpline::empty();

        let mut path = std::env::temp_dir();
        path.push("sampled.csv");
        let mut sampled = BufWriter::new(fs::File::create(path).unwrap());
        for x in super::rec_linspace(0.0, 1.0, 14) {
            if !spline.accurate(x) {
                writeln!(sampled, "{:e},{:e}", x, f(x)).unwrap();
                spline.add(x, f(x))
            }
        }

        println!("Points in spline: {}", spline.len());

        let mut path = std::env::temp_dir();
        path.push("spline.csv");
        let mut output = BufWriter::new(fs::File::create(path).unwrap());
        for &x in Array1::linspace(0.0, 1.0, 2048).iter() {
            writeln!(output, "{:e},{:e}", x, spline.sample(x)).unwrap();
        }
    }
}
