#![allow(clippy::approx_constant)]
#![allow(clippy::unreadable_literal)]

use crate::utilities::spline::ConstCubicHermiteSpline;

/// Value of \\(g_{*}\\) for the Standard Model.
///
/// The first column is the log-transformed inverse temperature,
/// \\(\ln(\beta)\\), where \\(\beta\\) is in units of inverse
/// gigaelectronvolts.
///
/// The second column is the value of \\(g_{*}\\).
///
/// Data is sourced from [`arXiv:1609.04979`](https://arxiv.org/abs/1609.04979).
pub(crate) const STANDARD_MODEL_GSTAR: ConstCubicHermiteSpline = ConstCubicHermiteSpline {
    data: &[
        (-9.210340371976184, 106.75, 0.),
        (-8.517193191416238, 106.75, -0.005000000000002558),
        (-7.600902459542082, 106.74, -0.0237823539868281),
        (-6.907755278982137, 106.7, -0.0899999999999963),
        (-6.214608098422191, 106.56, -0.5675349666421171),
        (-5.298317366548036, 105.61, -1.7393236287488676),
        (-4.605170185988092, 102.85, -4.540000000000002),
        (-3.912023005428146, 96.53, -8.37229277984406),
        (-2.995732273553991, 88.14, -4.178394994950497),
        (-2.302585092994046, 86.13, -1.384999999999999),
        (-1.6094379124341003, 85.37, -2.287332676057195),
        (-0.6931471805599453, 81.8, -4.500300373298365),
        (0., 75.5, -6.624999999999999),
        (0.6931471805599453, 68.55, -7.392802803889832),
        (1.5394455406140655, 62.25, -3.7423929829571523),
        (1.5441184463134578, 54.8, -56.73431766909642),
        (1.6094379124341003, 45.47, -6.513275781770426),
        (1.6607312068216509, 39.77, -5.434127195575209),
        (1.7147984280919266, 34.91, -4.6039307193494565),
        (1.7719568419318754, 30.84, -3.8334067350416747),
        (1.8325814637483102, 27.49, -3.1431372743693733),
        (1.8971199848858813, 24.77, -2.5438651268789396),
        (1.9661128563728327, 22.59, -2.035812112999736),
        (2.0402208285265546, 20.86, -4.717357289896269),
        (2.302585092994046, 17.55, -5.98738884706743),
        (2.995732273553991, 14.32, -3.669913873243092),
        (3.912023005428146, 11.25, -1.4061826739568573),
        (4.605170185988092, 10.76, -0.25499999999999945),
        (5.298317366548036, 10.74, -0.033219280948873824),
        (6.214608098422191, 10.7, -0.08512941594732035),
        (6.907755278982137, 10.56, -0.33499999999999996),
        (7.600902459542082, 10.03, -1.590310945145152),
        (8.517193191416238, 7.55, -2.3230237887338756),
        (9.210340371976184, 4.78, -1.809999999999996),
        (9.903487552536127, 3.93, -0.571819440327131),
        (10.819778284410283, 3.91, -0.0075647079736603),
        (11.512925464970229, 3.91, 0.),
    ],
};

/// Value of \\(g_{*}\\) for a fermion.
///
/// The first column is the log-transformed inverse temperature scaled to the
/// fermion's mass: \\(\ln(m\beta)\\).
///
/// The second column is is log-transformed value of \\(g_{*}\\),
/// \\(\log(g_{*})\\).
///
/// Data is sourced from [`arXiv:1609.04979`](https://arxiv.org/abs/1609.04979).
pub(crate) const FERMION_GSTAR: ConstCubicHermiteSpline = ConstCubicHermiteSpline {
    #[rustfmt::skip]
    data: &[
        (-2.302585092994046, -0.13353139262452263, -0.0011435107020789648),
        (-0.6931471805599453, -0.1346749033266016, -0.0129931660397556),
        (0., -0.16016875215282134, -0.0524260636190661),
        (0.6931471805599453, -0.2395270305647338, -0.17151900908915246),
        (1.0986122886681098, -0.5361434317502807, -0.324909672812512),
        (1.3862943611198906, -0.9755100915341263, -0.43518353172140395),
        (1.6094379124341003, -1.5050778971098575, -0.5239370207821284),
        (1.791759469228055, -2.120263536200091, -0.9368182924791323),
        (1.9459101490553132, -3.473768074496991, -0.9491979822391018),
        (2.0794415416798357, -4.199705077879927, -0.7106039559409633),
        (2.1972245773362196, -4.980591172747879, -0.7391440341153358),
        (2.302585092994046, -5.7603528261445955, -1.480184188474602),
        (2.4849066497880004, -7.371379301263834, -1.5135556200702458),
        (2.6390573296152584, -9.036387064852745, -1.572053280541367),
        (2.772588722239781, -10.73819829741786, -1.6164718926468724),
        (2.8903717578961645, -12.470038191364639, -1.6543286904096994),
        (2.995732273553991, -14.229511997094725, -7.913746509155695),
        (3.4011973816621555, -23.285917835359264, -7.871310473829766),
        (3.6888794541139363, -32.602916581708875, -8.345168743729955),
        (3.912023005428146, -42.06642839271318, -38.85349816550636),
        (4.605170185988092, -90.37707205585623, -48.310643663143054),
    ],
};

/// Value of \\(g_{*}\\) for a boson.
///
/// The first column is the log-transformed inverse temperature scaled to the
/// fermion's mass: \\(\ln(m\beta)\\).
///
/// The second column is is log-transformed value of \\(g_{*}\\),
/// \\(\log(g_{*})\\).
///
/// Data is sourced from [`arXiv:1609.04979`](https://arxiv.org/abs/1609.04979).
pub(crate) const BOSON_GSTAR: ConstCubicHermiteSpline = ConstCubicHermiteSpline {
    #[rustfmt::skip]
    data: &[
        (-2.302585092994046, 0., -0.0020020026706730793),
        (-0.693147180559945, -0.0020020026706730793, -0.01984110373452065),
        (0., -0.040821994520255166, -0.07266929261401803),
        (0.6931471805599453, -0.14734058789870913, -0.20217956895158848),
        (1.0986122886681098, -0.48939034304592566, -0.35390488256468877),
        (1.3862943611198906, -0.9545119446943529, -0.45567114760167815),
        (1.6094379124341003, -1.5050778971098575, -0.5325153883824197),
        (1.791759469228055, -2.120263536200091, -0.9368182924791323),
        (1.9459101490553132, -3.473768074496991, -0.9491979822391018),
        (2.0794415416798357, -4.199705077879927, -0.7106039559409633),
        (2.1972245773362196, -4.980591172747879, -0.7391440341153358),
        (2.302585092994046, -5.7603528261445955, -1.480184188474602),
        (2.4849066497880004, -7.371379301263834, -1.5135556200702458),
        (2.6390573296152584, -9.036387064852745, -1.572053280541367),
        (2.772588722239781, -10.73819829741786, -1.6164718926468724),
        (2.8903717578961645, -12.470038191364639, -1.6543286904096994),
        (2.995732273553991, -14.229511997094725, -7.913746509155695),
        (3.4011973816621555, -23.285917835359264, -7.871310473829766),
        (3.6888794541139363, -32.602916581708875, -8.345168743729955),
        (3.912023005428146, -42.06642839271318, -38.85349816550636),
        (4.605170185988092, -90.37707205585623, -48.310643663143054),
    ],
};

pub(crate) const G1_RUNNING: ConstCubicHermiteSpline = ConstCubicHermiteSpline {
    data: &[
        (1e-17, 0.588709, -0.003659000000000079),
        (1.99894855306109e-17, 0.58505, -0.005452578400413114),
        (3.9957964221638836e-17, 0.581459, -0.0053520720780386765),
        (7.987348040703526e-17, 0.577933, -0.005255179791777989),
        (1.5966330202868192e-16, 0.574471, -0.005161166766141278),
        (3.1915844301745156e-16, 0.571069, -0.005070732058987444),
        (6.379829530954933e-16, 0.567728, -0.004981211390402123),
        (1.2752906068470353e-15, 0.564444, -0.004896280540555485),
        (2.549238542447371e-15, 0.561216, -0.004812810737160987),
        (5.0958010599266205e-15, 0.558043, -0.00473181048692762),
        (1.0186214181451108e-14, 0.554922, -0.004653364492382703),
        (2.0361705333545095e-14, 0.551854, -0.004575893101585903),
        (4.070202858910488e-14, 0.548835, -0.00450189195710564),
        (8.136100692382169e-14, 0.545866, -0.004428450830066243),
        (1.626365333697639e-13, 0.542944, -0.004358457139854289),
        (3.25101756849894e-13, 0.540068, -0.004289986907369462),
        (6.498612546221382e-13, 0.537237, -0.004223511365508486),
        (1.2990387113535983e-12, 0.534449, -0.004158527507199373),
        (2.5967078937323262e-12, 0.531705, -0.00409405507554802),
        (5.190679615682081e-12, 0.529002, -0.0040325802188680765),
        (1.037589803397484e-11, 0.52634, -0.00397209558565116),
        (2.0740863649562367e-11, 0.523717, -0.003913619612512733),
        (4.145988548779628e-11, 0.521133, -0.003856140852032615),
        (8.287613333112331e-11, 0.518586, -0.0038006567983132965),
        (1.6566494596009463e-10, 0.516076, -0.003746179088753531),
        (3.311554344262567e-10, 0.513601, -0.003693198192922159),
        (6.619623211046827e-10, 0.511162, -0.0036402145753571726),
        (1.3232274045688397e-9, 0.508757, -0.0035902347540362266),
        (2.6450618018690007e-9, 0.506384, -0.003541252245340059),
        (5.287340520591548e-9, 0.504045, -0.0034917682972154143),
        (1.0569115143111104e-8, 0.501737, -0.003445281605479519),
        (2.1127087884460182e-8, 0.49946, -0.0033992962030987197),
        (4.2231869858269844e-8, 0.497213, -0.0033543340338083832),
        (8.44195313027622e-8, 0.494996, -0.0033103183873418062),
        (1.6874988398445475e-7, 0.492807, -0.0032678597877374015),
        (3.3732273690175814e-7, 0.490647, -0.003225360906649058),
        (6.742906462401553e-7, 0.488514, -0.0031843678357189906),
        (1.3478687499326066e-6, 0.486409, -0.0031439003110233727),
        (2.694321179250494e-6, 0.484329, -0.00310590833164595),
        (5.385812692206191e-6, 0.482275, -0.003067402641102525),
        (0.000010765919834807726, 0.480246, -0.003029939510746484),
        (0.000021520510122171935, 0.478242, -0.002993442341571449),
        (0.000043018338717795396, 0.476261, -0.0029584486460459574),
        (0.00008599117730520848, 0.474304, -0.0029229829632056842),
        (0.00017189218921892188, 0.47237, -0.002889480477405368),
        (0.00034360364632189476, 0.470457, -0.002857495367334073),
        (0.0006868462082655072, 0.468566, -0.0028249996532714807),
        (0.0013729681787165219, 0.466696, -0.0027940202707888956),
        (0.002744493174445475, 0.464846, -0.0027640236420699463),
        (0.005486095490978116, 0.463016, -0.002735033204235737),
        (0.010966403326768112, 0.461204, -0.0018119999999999803),
    ],
};

pub(crate) const G2_RUNNING: ConstCubicHermiteSpline = ConstCubicHermiteSpline {
    data: &[
        (1e-17, 0.518173, 0.0019240000000000368),
        (1.99894855306109e-17, 0.520097, 0.0028954895715755007),
        (3.9957964221638836e-17, 0.522042, 0.0029274557760470866),
        (7.987348040703526e-17, 0.524009, 0.0029604845860542305),
        (1.5966330202868192e-16, 0.525998, 0.0029939467642562194),
        (3.1915844301745156e-16, 0.52801, 0.0030284544099597158),
        (6.379829530954933e-16, 0.530045, 0.0030629105595534915),
        (1.2752906068470353e-15, 0.532103, 0.00309792245811913),
        (2.549238542447371e-15, 0.534185, 0.003133910456867809),
        (5.0958010599266205e-15, 0.536291, 0.0031703786276297673),
        (1.0186214181451108e-14, 0.538422, 0.003207883285250775),
        (2.0361705333545095e-14, 0.540578, 0.003245870771518654),
        (4.070202858910488e-14, 0.54276, 0.0032853377775437552),
        (8.136100692382169e-14, 0.544969, 0.0033253473841752725),
        (1.626365333697639e-13, 0.547204, 0.003364819886233516),
        (3.25101756849894e-13, 0.549466, 0.00340580993896721),
        (6.498612546221382e-13, 0.551756, 0.003448295841403864),
        (1.2990387113535983e-12, 0.554075, 0.0034912752113325776),
        (2.5967078937323262e-12, 0.556422, 0.003534264126206754),
        (5.190679615682081e-12, 0.558799, 0.0035797514540324123),
        (1.037589803397484e-11, 0.561207, 0.003625729590626563),
        (2.0740863649562367e-11, 0.563645, 0.0036712169711421343),
        (4.145988548779628e-11, 0.566114, 0.003718201340429026),
        (8.287613333112331e-11, 0.568615, 0.0037671810571580856),
        (1.6566494596009463e-10, 0.57115, 0.003817165932267055),
        (3.311554344262567e-10, 0.573717, 0.0038666498025176433),
        (6.619623211046827e-10, 0.576319, 0.003918628669569252),
        (1.3232274045688397e-9, 0.578955, 0.003970613227292877),
        (2.6450618018690007e-9, 0.581627, 0.004024595027201243),
        (5.287340520591548e-9, 0.584335, 0.004079573984121177),
        (1.0569115143111104e-8, 0.587081, 0.00413605038502893),
        (2.1127087884460182e-8, 0.589864, 0.004192528692676333),
        (4.2231869858269844e-8, 0.592686, 0.00425103566684789),
        (8.44195313027622e-8, 0.595547, 0.0043104751493842144),
        (1.6874988398445475e-7, 0.598449, 0.00437148839836175),
        (3.3732273690175814e-7, 0.601391, 0.004432948512667391),
        (6.742906462401553e-7, 0.604376, 0.004497415606948538),
        (1.3478687499326066e-6, 0.607404, 0.004561918119609854),
        (2.694321179250494e-6, 0.610475, 0.004627388214656094),
        (5.385812692206191e-6, 0.613591, 0.004695335262743659),
        (0.000010765919834807726, 0.616753, 0.004763847330202228),
        (0.000021520510122171935, 0.61996, 0.004832807429850063),
        (0.000043018338717795396, 0.623215, 0.004904772510287558),
        (0.00008599117730520848, 0.626518, 0.004977283458083069),
        (0.00017189218921892188, 0.62987, 0.005050732968078003),
        (0.00034360364632189476, 0.633271, 0.005124713928020437),
        (0.0006868462082655072, 0.636722, 0.005199674406895686),
        (0.0013729681787165219, 0.640223, 0.005275165758305832),
        (0.002744493174445475, 0.643775, 0.00535112539277426),
        (0.005486095490978116, 0.647377, 0.0054260970500858725),
        (0.010966403326768112, 0.651029, 0.0036519999999999886),
    ],
};

pub(crate) const G3_RUNNING: ConstCubicHermiteSpline = ConstCubicHermiteSpline {
    data: &[
        (1e-17, 0.513992, 0.004241999999999968),
        (1.99894855306109e-17, 0.518234, 0.006414772225895626),
        (3.9957964221638836e-17, 0.522584, 0.006578664589102921),
        (7.987348040703526e-17, 0.527046, 0.006748696605477427),
        (1.5966330202868192e-16, 0.531624, 0.006925575810339285),
        (3.1915844301745156e-16, 0.536324, 0.0071115575182955255),
        (6.379829530954933e-16, 0.541152, 0.007305915322616256),
        (1.2752906068470353e-15, 0.546113, 0.0075089024852910205),
        (2.549238542447371e-15, 0.551214, 0.0077223305669946105),
        (5.0958010599266205e-15, 0.556462, 0.007945205620988303),
        (1.0186214181451108e-14, 0.561862, 0.008178170220719887),
        (2.0361705333545095e-14, 0.567424, 0.00842508684192326),
        (4.070202858910488e-14, 0.573156, 0.008683446902328428),
        (8.136100692382169e-14, 0.579065, 0.008954916791802478),
        (1.626365333697639e-13, 0.585163, 0.00924228016387115),
        (3.25101756849894e-13, 0.591458, 0.009543688136957867),
        (6.498612546221382e-13, 0.597962, 0.009863079979253555),
        (1.2990387113535983e-12, 0.604687, 0.01020044816567998),
        (2.5967078937323262e-12, 0.611645, 0.010556836084425432),
        (5.190679615682081e-12, 0.61885, 0.010935215492765335),
        (1.037589803397484e-11, 0.626318, 0.01133756004269089),
        (2.0740863649562367e-11, 0.634065, 0.011764423041607165),
        (4.145988548779628e-11, 0.642108, 0.012218769494155695),
        (8.287613333112331e-11, 0.650468, 0.012704091218649164),
        (1.6566494596009463e-10, 0.659165, 0.013221923121469826),
        (3.311554344262567e-10, 0.668224, 0.013776735123103814),
        (6.619623211046827e-10, 0.677669, 0.01437002220756401),
        (1.3232274045688397e-9, 0.687529, 0.015008312754593337),
        (2.6450618018690007e-9, 0.697836, 0.015696080443623875),
        (5.287340520591548e-9, 0.708625, 0.01643831858001597),
        (1.0569115143111104e-8, 0.719935, 0.017241029444529436),
        (2.1127087884460182e-8, 0.731809, 0.018111222492575556),
        (4.2231869858269844e-8, 0.744296, 0.019058020507416548),
        (8.44195313027622e-8, 0.757451, 0.020090488671845406),
        (1.6874988398445475e-7, 0.771336, 0.021220767543505693),
        (3.3732273690175814e-7, 0.786022, 0.02246175521992968),
        (6.742906462401553e-7, 0.801589, 0.023828737270809846),
        (1.3478687499326066e-6, 0.818129, 0.02534185921345674),
        (2.694321179250494e-6, 0.83575, 0.027023751784583532),
        (5.385812692206191e-6, 0.854574, 0.0289004431918763),
        (0.000010765919834807726, 0.874747, 0.03100895625938338),
        (0.000021520510122171935, 0.89644, 0.03338855100584284),
        (0.000043018338717795396, 0.919854, 0.03609107374988384),
        (0.00008599117730520848, 0.945233, 0.039184310742563934),
        (0.00017189218921892188, 0.97287, 0.04274743094832082),
        (0.00034360364632189476, 1.00312, 0.04689411388492176),
        (0.0006868462082655072, 1.03644, 0.05177737358382074),
        (0.0013729681787165219, 1.07339, 0.057570641179491996),
        (0.002744493174445475, 1.11467, 0.06454321402413331),
        (0.005486095490978116, 1.16124, 0.07309039689686267),
        (0.010966403326768112, 1.21433, 0.05308999999999986),
    ],
};