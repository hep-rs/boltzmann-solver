//! Predefined constants.
//!
//! Although computing some of these values is generally very fast (e.g. `PI_2
//! == PI.powi(2)`), it will be slightly faster to use pre-defined constant
//! which, in certain performance-sensitive functions, can result in an
//! appreciable performance improvement.

/// Planck mass, \\(M_{\text{Pl}} = \sqrt{\hbar c / G}\\), in units of GeV /
/// \\(c\^2\\).
///
/// In this library, the Planck mass will be denoted by \\(M_{\text{Pl}}\\), in
/// contrast to the reduced Planck mass \\(m_{\text{Pl}}\\).
pub const PLANCK_MASS: f64 = 1.220_910e19;

/// Reduced Planck mass, \\(m_{\text{Pl}} = \sqrt{\hbar c / 8 \pi G}\\), in
/// units of GeV / \\(c\^2\\).
///
/// In this library, the reduced Planck mass will be denoted by
/// \\(m_{\text{Pl}}\\), in contrast to the Planck mass \\(M_{\text{Pl}}\\).
pub const REDUCED_PLANCK_MASS: f64 = 2.435_363e18;

/// The Riemann zeta function evaluated at 3.
pub const ZETA_3: f64 = 1.2020569031595942854;

/// \\(\pi\\)
pub const PI: f64 = ::std::f64::consts::PI;
/// \\(\pi\\), with a different name to follow the convention for `PI_n`.
pub const PI_1: f64 = PI;
/// \\(\pi\^2\\)
pub const PI_2: f64 = 9.8696044010893586188;
/// \\(\pi^{-1}\\)
pub const PI_M1: f64 = 0.31830988618379067154;
/// \\(\pi^{-2}\\)
pub const PI_M2: f64 = 0.10132118364233777144;

////////////////////////////////////////////////////////////////////////////////
// Crate Constants
////////////////////////////////////////////////////////////////////////////////

/// Value of \\(g_{*}\\) for the Standard Model.
///
/// The first column is the log-transformed inverse temperature,
/// \\(\ln(\beta)\\), where \\(\beta\\) is in units of inverse
/// gigaelectronvolts.
///
/// The second column is the value of \\(g_{*}\\).
///
/// Data is sourced from [`arXiv:1609.04979`](https://arxiv.org/abs/1609.04979).
#[cfg_attr(feature = "cargo-clippy", allow(approx_constant))]
pub(crate) const STANDARD_MODEL_GSTAR: [(f64, f64); 37] = [
    (-9.2103403719761840, 106.75),
    (-8.5171931914162380, 106.75),
    (-7.6009024595420820, 106.74),
    (-6.9077552789821370, 106.7),
    (-6.2146080984221910, 106.56),
    (-5.2983173665480360, 105.61),
    (-4.6051701859880920, 102.85),
    (-3.9120230054281460, 96.53),
    (-2.9957322735539910, 88.14),
    (-2.3025850929940460, 86.13),
    (-1.6094379124341003, 85.37),
    (-0.6931471805599453, 81.8),
    (0.0000000000000000, 75.5),
    (0.6931471805599453, 68.55),
    (1.5394455406140655, 62.25),
    (1.5441184463134578, 54.8),
    (1.6094379124341003, 45.47),
    (1.6607312068216509, 39.77),
    (1.7147984280919266, 34.91),
    (1.7719568419318754, 30.84),
    (1.8325814637483102, 27.49),
    (1.8971199848858813, 24.77),
    (1.9661128563728327, 22.59),
    (2.0402208285265546, 20.86),
    (2.3025850929940460, 17.55),
    (2.9957322735539910, 14.32),
    (3.9120230054281460, 11.25),
    (4.6051701859880920, 10.76),
    (5.2983173665480360, 10.74),
    (6.2146080984221910, 10.7),
    (6.9077552789821370, 10.56),
    (7.6009024595420820, 10.03),
    (8.5171931914162380, 7.55),
    (9.2103403719761840, 4.78),
    (9.9034875525361270, 3.93),
    (10.819778284410283, 3.91),
    (11.512925464970229, 3.91),
];

/// Value of \\(g_{*}\\) for a fermion.
///
/// The first column is the log-transformed inverse temperature scaled to the
/// fermion's mass: \\(\ln(m\beta)\\).
///
/// The second column is is log-transformed value of \\(g_{*}\\),
/// \\(\log(g_{*})\\).
///
/// Data is sourced from [`arXiv:1609.04979`](https://arxiv.org/abs/1609.04979).
#[cfg_attr(feature = "cargo-clippy", allow(approx_constant))]
pub(crate) const FERMION_GSTAR: [(f64, f64); 21] = [
    (-2.3025850929940460, -0.13353139262452263),
    (-0.6931471805599453, -0.13467490332660160),
    (0.0000000000000000, -0.16016875215282134),
    (0.6931471805599453, -0.2395270305647338),
    (1.0986122886681098, -0.5361434317502807),
    (1.3862943611198906, -0.9755100915341263),
    (1.6094379124341003, -1.5050778971098575),
    (1.7917594692280550, -2.1202635362000910),
    (1.9459101490553132, -3.4737680744969910),
    (2.0794415416798357, -4.1997050778799270),
    (2.1972245773362196, -4.9805911727478790),
    (2.3025850929940460, -5.7603528261445955),
    (2.4849066497880004, -7.3713793012638340),
    (2.6390573296152584, -9.0363870648527450),
    (2.7725887222397810, -10.738198297417860),
    (2.8903717578961645, -12.470038191364639),
    (2.9957322735539910, -14.229511997094725),
    (3.4011973816621555, -23.285917835359264),
    (3.6888794541139363, -32.602916581708875),
    (3.9120230054281460, -42.066428392713180),
    (4.6051701859880920, -90.377072055856230),
];

/// Value of \\(g_{*}\\) for a boson.
///
/// The first column is the log-transformed inverse temperature scaled to the
/// fermion's mass: \\(\ln(m\beta)\\).
///
/// The second column is is log-transformed value of \\(g_{*}\\),
/// \\(\log(g_{*})\\).
///
/// Data is sourced from [`arXiv:1609.04979`](https://arxiv.org/abs/1609.04979).
#[cfg_attr(feature = "cargo-clippy", allow(approx_constant))]
pub(crate) const BOSON_GSTAR: [(f64, f64); 21] = [
    (-2.302585092994046, 0.00000000000000000),
    (-0.693147180559945, -0.0020020026706730793),
    (0.0000000000000000, -0.040821994520255166),
    (0.6931471805599453, -0.14734058789870913),
    (1.0986122886681098, -0.48939034304592566),
    (1.3862943611198906, -0.9545119446943529),
    (1.6094379124341003, -1.5050778971098575),
    (1.7917594692280550, -2.1202635362000910),
    (1.9459101490553132, -3.4737680744969910),
    (2.0794415416798357, -4.1997050778799270),
    (2.1972245773362196, -4.9805911727478790),
    (2.3025850929940460, -5.7603528261445955),
    (2.4849066497880004, -7.3713793012638340),
    (2.6390573296152584, -9.0363870648527450),
    (2.7725887222397810, -10.738198297417860),
    (2.8903717578961645, -12.470038191364639),
    (2.9957322735539910, -14.229511997094725),
    (3.4011973816621555, -23.285917835359264),
    (3.6888794541139363, -32.602916581708875),
    (3.9120230054281460, -42.066428392713180),
    (4.6051701859880920, -90.377072055856230),
];
