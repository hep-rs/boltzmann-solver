(* BeginPackage["DegreesOfFreedom`", {"Extra`", "Particle`"}]; *)

GStar::usage = "Degrees of freedom contributing to entrpopy.

This function remains undefined but can be substituted with
a more applicable function (such as GStarSM).

GStar[Particle[_]] converts to either GStarFermion or GStarBoson based on the
particle statistic and is scaled.";
GStarSM::usage = "Degrees of freedom contribution to entropy for the SM.";
GStarFermion::usage = "Degrees of freedom contribution to entropy for a
massive fermionic particle of mass m at inverse temperature \[Beta]";
GStarBoson::usage = "Degrees of freedom contribution to entropy for a
massive bosonic particle of mass m at inverse temperature \[Beta]";

Begin["`Private`"];

MakeBoxes[GStar, TraditionalForm] = SubscriptBox["g", "*"];
MakeBoxes[GStar[p_?ParticleQ], TraditionalForm] = SubsuperscriptBox["g", "*", MakeBoxes[p, TraditionalForm]];
MakeBoxes[GStarSM, TraditionalForm] = SubsuperscriptBox["g", "*", "SM"];
MakeBoxes[GStarFermion, TraditionalForm] = SubsuperscriptBox["g", "*", "f"];
MakeBoxes[GStarBoson, TraditionalForm] = SubsuperscriptBox["g", "*", "b"];

Attributes[GStarSM] = {Listable, NumericFunction};
Attributes[GStarFermion] = {Listable, NumericFunction};
Attributes[GStarBoson] = {Listable, NumericFunction};

GStarSM[beta_?InexactNumberQ] /; 1 / beta >= 10^4 = 106.75;
GStarSM[beta_?InexactNumberQ] /; 1 / beta <= 10^-6 = 3.91;
GStarSM[beta_?InexactNumberQ] := Interpolation[
  MapAt[
    Log,
    {
      {10000, 106.75},
      {5000, 106.75},
      {2000, 106.74},
      {1000, 106.70},
      {500, 106.56},
      {200, 105.61},
      {100, 102.85},
      {50, 96.53},
      {20, 88.14},
      {10, 86.13},
      {5, 85.37},
      {2, 81.80},
      {1, 75.50},
      {500*^-3, 68.55},
      {214.5*^-3, 62.25},
      {213.5*^-3, 54.80},
      {200*^-3, 45.47},
      {190*^-3, 39.77},
      {180*^-3, 34.91},
      {170*^-3, 30.84},
      {160*^-3, 27.49},
      {150*^-3, 24.77},
      {140*^-3, 22.59},
      {130*^-3, 20.86},
      {100*^-3, 17.55},
      {50*^-3, 14.32},
      {20*^-3, 11.25},
      {10*^-3, 10.76},
      {5*^-3, 10.74},
      {2*^-3, 10.70},
      {1*^-3, 10.56},
      {500*^-6, 10.03},
      {200*^-6, 7.55},
      {100*^-6, 4.78},
      {50*^-6, 3.93},
      {20*^-6, 3.91},
      {10*^-6, 3.91}
    },
    {All, 1}
  ],
  InterpolationOrder -> 1,
  Method -> "Hermite"
][Log[beta]];

GStarFermion[m_?NumericQ, beta_?NumericQ] /; AnyInexactNumberQ[m, beta] && 1 / (m beta) >= 10 = 0.875;
GStarFermion[m_?NumericQ, beta_?NumericQ] /; AnyInexactNumberQ[m, beta] && 1 / (m beta) <= 1 / 100 = 0;
GStarFermion[m_?NumericQ, beta_?NumericQ] /; AnyInexactNumberQ[m, beta] := Exp@Interpolation[
  Log@{
    {10, 0.875},
    {2, 0.874},
    {1, 0.852},
    {1/2, 0.787},
    {1/3, 0.585},
    {1/4, 0.377},
    {1/5, 0.222},
    {1/6, 0.120},
    {1/7, 0.031},
    {1/8, 0.015},
    {1/9, 6.87*^-3},
    {1/10, 3.15*^-3},
    {1/12, 6.29*^-4},
    {1/14, 1.19*^-4},
    {1/16, 2.17*^-5},
    {1/18, 3.84*^-6},
    {1/20, 6.61*^-7},
    {1/30, 7.71*^-11},
    {1/40, 6.93*^-15},
    {1/50, 5.38*^-19},
    {1/100, 5.62*^-40}
  },
  InterpolationOrder -> 1,
  Method -> "Hermite"
][Log[1 / (m beta)]];

GStarBoson[m_?NumericQ, beta_?NumericQ] /; AnyInexactNumberQ[m, beta] && 1 / (m beta) >= 10 = 1;
GStarBoson[m_?NumericQ, beta_?NumericQ] /; AnyInexactNumberQ[m, beta] && 1 / (m beta) <= 1 / 100 = 0;
GStarBoson[m_?NumericQ, beta_?NumericQ] /; AnyInexactNumberQ[m, beta] := Exp@Interpolation[
  Log@{
    {10, 1},
    {2, 0.998},
    {1, 0.960},
    {1/2, 0.863},
    {1/3, 0.613},
    {1/4, 0.385},
    {1/5, 0.222},
    {1/6, 0.120},
    {1/7, 0.031},
    {1/8, 0.015},
    {1/9, 6.87*^-3},
    {1/10, 3.15*^-3},
    {1/12, 6.29*^-4},
    {1/14, 1.19*^-4},
    {1/16, 2.17*^-5},
    {1/18, 3.84*^-6},
    {1/20, 6.61*^-7},
    {1/30, 7.71*^-11},
    {1/40, 6.93*^-15},
    {1/50, 5.38*^-19},
    {1/100, 5.62*^-40}
  },
  InterpolationOrder -> 1,
  Method -> "Hermite"
][Log[1 / (m beta)]];

GStar::unknownStatistic = "Unknown statistic ``."
GStar[p_?ParticleQ][beta_?InexactNumberQ] := DoF[p] * Switch[BoltzmannSolver`Statistic[p],
  FermiDirac, GStarFermion[BoltzmannSolver`Mass[p], beta],
  BoseEinstein, GStarBoson[BoltzmannSolver`Mass[p], beta],
  _, Message[GStar::unknownStatistic, Statistic[p]]
];

Protect[GStar, GStarSM, GStarFermion, GStarBoson];

End[];

(* EndPackage[]; *)