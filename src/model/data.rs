#![allow(clippy::approx_constant)]

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
        (1.99894855306109e-17, 0.58505, 0.5829472488826573),
        (3.9957964221638836e-17, 0.581459, 0.5793838285781978),
        (7.987348040703526e-17, 0.577933, 0.5759036567219664),
        (1.5966330202868192e-16, 0.574471, 0.572465800205641),
        (3.1915844301745156e-16, 0.571069, 0.5691017285990428),
        (6.379829530954933e-16, 0.567728, 0.5657820659234288),
        (1.2752906068470353e-15, 0.564444, 0.5625344645046562),
        (2.549238542447371e-15, 0.561216, 0.5593358068979456),
        (5.0958010599266205e-15, 0.558043, 0.5561853613477813),
        (1.0186214181451108e-14, 0.554922, 0.5530972024485735),
        (2.0361705333545095e-14, 0.551854, 0.5500554604571442),
        (4.070202858910488e-14, 0.548835, 0.5470581680284916),
        (8.136100692382169e-14, 0.545866, 0.5441201770982905),
        (1.626365333697639e-13, 0.542944, 0.5412193173651003),
        (3.25101756849894e-13, 0.540068, 0.538368364773554),
        (6.498612546221382e-13, 0.537237, 0.5355605028155226),
        (1.2990387113535983e-12, 0.534449, 0.5327947287279013),
        (2.5967078937323262e-12, 0.531705, 0.5300735169244574),
        (5.190679615682081e-12, 0.529002, 0.5273931357508858),
        (1.037589803397484e-11, 0.52634, 0.5247508142516307),
        (2.0740863649562367e-11, 0.523717, 0.522149387182272),
        (4.145988548779628e-11, 0.521133, 0.5195853912882119),
        (8.287613333112331e-11, 0.518586, 0.5170575152354029),
        (1.6566494596009463e-10, 0.516076, 0.514566910039701),
        (3.311554344262567e-10, 0.513601, 0.51211135461544),
        (6.619623211046827e-10, 0.511162, 0.5096901022802292),
        (1.3232274045688397e-9, 0.508757, 0.5073028474711077),
        (2.6450618018690007e-9, 0.506384, 0.5049482352995713),
        (5.287340520591548e-9, 0.504045, 0.5026255704089003),
        (1.0569115143111104e-8, 0.501737, 0.5003336274660988),
        (2.1127087884460182e-8, 0.49946, 0.49807243655974875),
        (4.2231869858269844e-8, 0.497213, 0.49584647552471256),
        (8.44195313027622e-8, 0.494996, 0.4936376357160486),
        (1.6874988398445475e-7, 0.492807, 0.4914702209764325),
        (3.3732273690175814e-7, 0.490647, 0.4893215847023686),
        (6.742906462401553e-7, 0.488514, 0.4872018669812143),
        (1.3478687499326066e-6, 0.486409, 0.485114217455707),
        (2.694321179250494e-6, 0.484329, 0.48304645733759194),
        (5.385812692206191e-6, 0.482275, 0.48100014949301584),
        (0.000010765919834807726, 0.480246, 0.47898760876076746),
        (0.000021520510122171935, 0.478242, 0.47698833622810594),
        (0.000043018338717795396, 0.476261, 0.47500823024210675),
        (0.00008599117730520848, 0.474304, 0.4730475121321818),
        (0.00017189218921892188, 0.47237, 0.4710785407678481),
        (0.00034360364632189476, 0.470457, 0.46909263249368127),
        (0.0006868462082655072, 0.468566, 0.46703970479237567),
        (0.0013729681787165219, 0.466696, 0.4648400025721579),
        (0.002744493174445475, 0.464846, 0.46231342881047643),
        (0.005486095490978116, 0.463016, 0.4591223436871914),
        (0.010966403326768112, 0.461204, -0.0018119999999999803),
    ],
};

pub(crate) const G2_RUNNING: ConstCubicHermiteSpline = ConstCubicHermiteSpline {
    data: &[
        (1e-17, 0.518173, 0.0019240000000000368),
        (1.99894855306109e-17, 0.520097, 0.5207963602950575),
        (3.9957964221638836e-17, 0.522042, 0.5227452281435485),
        (7.987348040703526e-17, 0.524009, 0.5247329936216151),
        (1.5966330202868192e-16, 0.525998, 0.526725468127315),
        (3.1915844301745156e-16, 0.52801, 0.5287531053691946),
        (6.379829530954933e-16, 0.530045, 0.5307902395766703),
        (1.2752906068470353e-15, 0.532103, 0.5328653978292284),
        (2.549238542447371e-15, 0.534185, 0.5349584526426171),
        (5.0958010599266205e-15, 0.536291, 0.537070943537618),
        (1.0186214181451108e-14, 0.538422, 0.5392178489963739),
        (2.0361705333545095e-14, 0.540578, 0.5413858663849532),
        (4.070202858910488e-14, 0.54276, 0.5435754038219939),
        (8.136100692382169e-14, 0.544969, 0.5458021451364956),
        (1.626365333697639e-13, 0.547204, 0.5480460680207402),
        (3.25101756849894e-13, 0.549466, 0.5503219203908899),
        (6.498612546221382e-13, 0.551756, 0.5526253682396898),
        (1.2990387113535983e-12, 0.554075, 0.5549558631813787),
        (2.5967078937323262e-12, 0.556422, 0.5573175015462946),
        (5.190679615682081e-12, 0.558799, 0.5597094845417154),
        (1.037589803397484e-11, 0.561207, 0.5621299191670703),
        (2.0740863649562367e-11, 0.563645, 0.5645828745587134),
        (4.145988548779628e-11, 0.566114, 0.5670667319094497),
        (8.287613333112331e-11, 0.568615, 0.5695826316322686),
        (1.6566494596009463e-10, 0.57115, 0.5721329268261598),
        (3.311554344262567e-10, 0.573717, 0.5747162346419893),
        (6.619623211046827e-10, 0.576319, 0.5773332626273179),
        (1.3232274045688397e-9, 0.578955, 0.5799864170194791),
        (2.6450618018690007e-9, 0.581627, 0.5826751715389804),
        (5.287340520591548e-9, 0.584335, 0.5854002901965994),
        (1.0569115143111104e-8, 0.587081, 0.5881625743196721),
        (2.1127087884460182e-8, 0.589864, 0.5909631420691643),
        (4.2231869858269844e-8, 0.592686, 0.5938089346745954),
        (8.44195313027622e-8, 0.595547, 0.5966805442255676),
        (1.6874988398445475e-7, 0.598449, 0.5996081938846807),
        (3.3732273690175814e-7, 0.601391, 0.6025661829643797),
        (6.742906462401553e-7, 0.604376, 0.605568869176279),
        (1.3478687499326066e-6, 0.607404, 0.6086215075422972),
        (2.694321179250494e-6, 0.610475, 0.6117112508009639),
        (5.385812692206191e-6, 0.613591, 0.6148414933192178),
        (0.000010765919834807726, 0.616753, 0.6180287611971743),
        (0.000021520510122171935, 0.61996, 0.6212495411995733),
        (0.000043018338717795396, 0.623215, 0.6245142389907372),
        (0.00008599117730520848, 0.626518, 0.6278254077690714),
        (0.00017189218921892188, 0.62987, 0.6311525134624434),
        (0.00034360364632189476, 0.633271, 0.6344921289584632),
        (0.0006868462082655072, 0.636722, 0.6377922495811199),
        (0.0013729681787165219, 0.640223, 0.6409770884166425),
        (0.002744493174445475, 0.643775, 0.643863997054763),
        (0.005486095490978116, 0.647377, 0.6461179450894355),
        (0.010966403326768112, 0.651029, 0.0036519999999999886),
    ],
};

pub(crate) const G3_RUNNING: ConstCubicHermiteSpline = ConstCubicHermiteSpline {
    data: &[
        (1e-17, 0.513992, 0.004241999999999968),
        (1.99894855306109e-17, 0.518234, 0.5201368386880695),
        (3.9957964221638836e-17, 0.522584, 0.5245344371567195),
        (7.987348040703526e-17, 0.527046, 0.5290629258472618),
        (1.5966330202868192e-16, 0.531624, 0.533692488989912),
        (3.1915844301745156e-16, 0.536324, 0.5384592847749624),
        (6.379829530954933e-16, 0.541152, 0.5433427934267736),
        (1.2752906068470353e-15, 0.546113, 0.5483775623764825),
        (2.549238542447371e-15, 0.551214, 0.5535495411008313),
        (5.0958010599266205e-15, 0.556462, 0.5588657031757571),
        (1.0186214181451108e-14, 0.561862, 0.5643485656581654),
        (2.0361705333545095e-14, 0.567424, 0.5699928055000685),
        (4.070202858910488e-14, 0.573156, 0.5758052136690238),
        (8.136100692382169e-14, 0.579065, 0.5818118544668892),
        (1.626365333697639e-13, 0.585163, 0.5880015250952542),
        (3.25101756849894e-13, 0.591458, 0.5943988279758198),
        (6.498612546221382e-13, 0.597962, 0.6010100715793872),
        (1.2990387113535983e-12, 0.604687, 0.6078466322611401),
        (2.5967078937323262e-12, 0.611645, 0.614925422441699),
        (5.190679615682081e-12, 0.61885, 0.6222589420789881),
        (1.037589803397484e-11, 0.626318, 0.6298610679940181),
        (2.0740863649562367e-11, 0.634065, 0.6377528151279634),
        (4.145988548779628e-11, 0.642108, 0.6459502601251486),
        (8.287613333112331e-11, 0.650468, 0.6544734651274946),
        (1.6566494596009463e-10, 0.659165, 0.6633476080997218),
        (3.311554344262567e-10, 0.668224, 0.6725950255989571),
        (6.619623211046827e-10, 0.677669, 0.6822418482002793),
        (1.3232274045688397e-9, 0.687529, 0.6923207973457686),
        (2.6450618018690007e-9, 0.697836, 0.7028635673159361),
        (5.287340520591548e-9, 0.708625, 0.7139068398592364),
        (1.0569115143111104e-8, 0.719935, 0.7254919406170641),
        (2.1127087884460182e-8, 0.731809, 0.7376655990515322),
        (4.2231869858269844e-8, 0.744296, 0.7504872643878517),
        (8.44195313027622e-8, 0.757451, 0.7639897529238915),
        (1.6874988398445475e-7, 0.771336, 0.7782771400347753),
        (3.3732273690175814e-7, 0.786022, 0.7933908163234183),
        (6.742906462401553e-7, 0.801589, 0.8094331914855372),
        (1.3478687499326066e-6, 0.818129, 0.8265114211006811),
        (2.694321179250494e-6, 0.83575, 0.8447220173449772),
        (5.385812692206191e-6, 0.854574, 0.8642012470489207),
        (0.000010765919834807726, 0.874747, 0.8851309165128605),
        (0.000021520510122171935, 0.89644, 0.9076631223361359),
        (0.000043018338717795396, 0.919854, 0.93203380709494),
        (0.00008599117730520848, 0.945233, 0.9585172742065293),
        (0.00017189218921892188, 0.97287, 0.9873961984417838),
        (0.00034360364632189476, 1.00312, 1.0190813987510061),
        (0.0006868462082655072, 1.03644, 1.054023297125467),
        (0.0013729681787165219, 1.07339, 1.0927811438374002),
        (0.002744493174445475, 1.11467, 1.135994477018287),
        (0.005486095490978116, 1.16124, 1.1844284693671374),
        (0.010966403326768112, 1.21433, 0.05308999999999986),
    ],
};
