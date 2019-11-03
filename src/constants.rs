//! Collection of physical and mathematical constants which appear frequently.
//!
//! When implementing code that is numerically intensive, it advisable to
//! collect all the constants into a single multiplicative constants; however,
//! make sure to document in the code what the resulting 'magic number'
//! corresponds to.

/// Planck mass, \\(M_{\text{Pl}} = \sqrt{\hbar c / G}\\), in units of GeV /
/// \\(c^2\\).
///
/// In this library, the Planck mass will be denoted by \\(M_{\text{Pl}}\\), in
/// contrast to the reduced Planck mass \\(m_{\text{Pl}}\\).
pub const PLANCK_MASS: f64 = 1.220_910e19;

/// Reduced Planck mass, \\(m_{\text{Pl}} = \sqrt{\hbar c / 8 \pi G}\\), in
/// units of GeV / \\(c^2\\).
///
/// In this library, the reduced Planck mass will be denoted by
/// \\(m_{\text{Pl}}\\), in contrast to the Planck mass \\(M_{\text{Pl}}\\).
pub const REDUCED_PLANCK_MASS: f64 = 2.435_363e18;

/// Riemann zeta function evaluated at 3: \\(\zeta(3) \approx 1.202\dots\\)
pub const ZETA_3: f64 = 1.202_056_903_159_594_2;

/// Euler gamma constant: \\(\gamma_\textsc{E} \approx 0.577\dots\\)
pub const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// \\(\pi\\)
pub const PI: f64 = ::std::f64::consts::PI;
/// \\(\pi\\) [named to follow the convention `PI_n`]
pub const PI_1: f64 = PI;
/// \\(\pi^2\\)
pub const PI_2: f64 = 9.869_604_401_089_358;
/// \\(\pi^3\\)
pub const PI_3: f64 = 3.100_627_668_029_982e1;
/// \\(\pi^4\\)
pub const PI_4: f64 = 9.740_909_103_400_244e1;
/// \\(\pi^5\\)
pub const PI_5: f64 = 3.060_196_847_852_814_7e2;
/// \\(\pi^{-1}\\)
pub const PI_N1: f64 = 3.183_098_861_837_907e-1;
/// \\(\pi^{-2}\\)
pub const PI_N2: f64 = 1.013_211_836_423_377_8e-1;
/// \\(\pi^{-3}\\)
pub const PI_N3: f64 = 3.225_153_443_319_949e-2;
/// \\(\pi^{-4}\\)
pub const PI_N4: f64 = 1.026_598_225_468_433_6e-2;
/// \\(\pi^{-5}\\)
pub const PI_N5: f64 = 3.267_763_643_053_385_6e-3;
