MandelstamTRange::usage = "Calculate the range of Mandelstam variables.  The
first argument is the Mandelstam s variable.  The remaining four arguments are
the masses of the four particles.";
IntegrateT::usage  = "Integrate a squared amplitude over the Mandelstam t variable.";
IntegrateST::usage = "Integrate a squared amplitude over the Mandelstam s and t variables.";

BesselK12::usage = "Ratio of Bessel function K1 / K2.";

SquaredAmplitude::usage = "Squared Amplitude.";
InteractionGamma::usage = "Interaction rate density.";

Begin["`Private`"];

(* Mandelstam T Range *)
(* ****************** *)

MandelstamTRange::invalids12 = "The Mandelstam s value must be larger than (m1 + m2)^2.";
MandelstamTRange::invalids34 = "The Mandelstam s value must be larger than (m1 + m2)^2.";

Attributes[MandelstamTRange] = {Listable, NumericFunction};

(* Allow particles to be given instead of masses *)
MandelstamTRange[s_, p1_?ParticleQ, m2_, m3_, m4_] := MandelstamTRange[s, Mass[p1], m2, m3, m4 ];
MandelstamTRange[s_, m1_, p2_?ParticleQ, m3_, m4_] := MandelstamTRange[s, m1, Mass[p2], m3, m4 ];
MandelstamTRange[s_, m1_, m2_, p3_?ParticleQ, m4_] := MandelstamTRange[s, m1, m2, Mass[p3], m4 ];
MandelstamTRange[s_, m1_, m2_, m3_, p4_?ParticleQ] := MandelstamTRange[s, m1, m2, m3, Mass[p4] ];

(* Numerical expansion expands the arguments *)
MandelstamTRange /: N[MandelstamTRange[s_, m1_, m2_, m3_, m4_], ___precision] := MandelstamTRange@@N[{s, m1, m2, m3, m4}, precision];

MandelstamTRange[s_?NumericQ, m1_?NumericQ, m2_?NumericQ, m3_?NumericQ, m4_?NumericQ] := Module[
  {
    baseline = 1/2 (m1^2 + m2^2 + m3^2 + m4^2 - s - (m1^2 - m2^2) (m3^2 - m4^2) / s),
    cosine = Sqrt[X`Kallen\[Lambda][s, m1^2, m2^2]] Sqrt[X`Kallen\[Lambda][s, m3^2, m4^2]] / (2 s)
  },
  If[s < (m1 + m2)^2, Message[MandelstamTRange::invalids12] ];
  If[s < (m3 + m4)^2, Message[MandelstamTRange::invalids12] ];
  {baseline - cosine, baseline + cosine}
];


Protect[MandelstamTRange];


(* Mandelstam T Integration *)
(* ************************ *)

Attributes[IntegrateT] = {HoldFirst, Listable};

(* Allow particles to be given instead of masses *)
Attributes[IntegrateT] = {Listable, NumericFunction};
IntegrateT[f_, s_, beta_, p1_?ParticleQ, m2_, m3_, m4_] = IntegrateT[f, s, beta, Mass[p1], m2, m3, m4 ]
IntegrateT[f_, s_, beta_, m1_, p2_?ParticleQ, m3_, m4_] = IntegrateT[f, s, beta, m1, Mass[p2], m3, m4 ]
IntegrateT[f_, s_, beta_, m1_, m2_, p3_?ParticleQ, m4_] = IntegrateT[f, s, beta, m1, m2, Mass[p3], m4 ]
IntegrateT[f_, s_, beta_, m1_, m2_, m3_, p4_?ParticleQ] = IntegrateT[f, s, beta, m1, m2, m3, Mass[p4] ]

(* Numerical expansion expands the arguments *)
IntegrateT /: N[IntegrateT[f_, s_, beta_, m1_, m2_, m3_, m4_], ___precision] := CurryApplied[IntegrateT, 7][f]@@N[{s, beta, m1, m2, m3, m4}, precision];

IntegrateT /: MakeBoxes[IntegrateT[f_, s_, beta_, m1_, m2_, m3_, m4_], TraditionalForm] := With[
  {
    integrand = f[\[FormalS], \[FormalT]]
  },
  RowBox[{
    SubsuperscriptBox["\[Integral]", SubscriptBox[\[FormalT], "min"], SubscriptBox[\[FormalT], "max"]],
    MakeBoxes[integrand, TraditionalForm],
    "\[DifferentialD]", \[FormalT]
  }]
];

IntegrateT[f_?NumberQ, s_, beta_, m1_, m2_, m3_, m4_] := IntegrateT[Function[{t}, f], s, beta, m1, m2, m3, m4];
IntegrateT[
  f:(_?ValueQ | _Function), s_?NumericQ, beta_?NumericQ, m1_?NumericQ, m2_?NumericQ, m3_?NumericQ, m4_?NumericQ
] /; AnyInexactNumberQ[s, beta, m1, m2, m3, m4] := Block[
  {
    tMin, tMax
  },
  {tMin, tMax} = MandelstamTRange[s, m1, m2, m3, m4];
  NIntegrate[
    f[s, t],
    {t, tMin, tMax},
    (* Method -> {"ClenshawCurtisRule", "SymbolicProcessing" -> 0}, *)
    Method -> {Automatic, "SymbolicProcessing" -> 0},
    WorkingPrecision -> Precision[{s, beta, m1, m2, m3, m4}]
  ]
];

Protect[IntegrateT];


(* Mandelstam S and T Integration *)
(* ****************************** *)

IntegrateST[f_, beta_, p1_?ParticleQ, m2_, m3_, m4_] = IntegrateST[f, beta, Mass[p1], m2, m3, m4 ]
IntegrateST[f_, beta_, m1_, p2_?ParticleQ, m3_, m4_] = IntegrateST[f, beta, m1, Mass[p2], m3, m4 ]
IntegrateST[f_, beta_, m1_, m2_, p3_?ParticleQ, m4_] = IntegrateST[f, beta, m1, m2, Mass[p3], m4 ]
IntegrateST[f_, beta_, m1_, m2_, m3_, p4_?ParticleQ] = IntegrateST[f, beta, m1, m2, m3, Mass[p4] ]

(* Numerical expansion expands the arguments *)
IntegrateST /: N[IntegrateST[f_, beta_, m1_, m2_, m3_, m4_], ___precision] := CurryApplied[IntegrateST, 6][f]@@N[{beta, m1, m2, m3, m4}, precision];

IntegrateST /: MakeBoxes[IntegrateST[f_, beta_, m1_, m2_, m3_, m4_], TraditionalForm] := With[
  {
    sMin = Max[m1^2 + m2^2, m3^2 + m4^2],
    integrand = f[\[FormalS], \[FormalT]]
  },
  RowBox[{
    MakeBoxes[1 / (512 \[Pi]^5 beta), TraditionalForm],
    SubsuperscriptBox["\[Integral]", MakeBoxes[sMin, TraditionalForm], "\[Infinity]"],
    SubsuperscriptBox["\[Integral]", SubscriptBox[\[FormalT], "min"], SubscriptBox[\[FormalT], "max"]],
    FractionBox[MakeBoxes[BesselK[1, Sqrt[\[FormalS]] beta], TraditionalForm], MakeBoxes[Sqrt[\[FormalS]], TraditionalForm]],
    MakeBoxes[integrand, TraditionalForm],
    "\[DifferentialD]", \[FormalT],
    "\[DifferentialD]", \[FormalS]
  }]
];

IntegrateST[f_?NumericQ, beta_, m1_, m2_, m3_, m4_] := IntegrateST[Function[{s, t}, f], beta, m1, m2, m3, m4];
IntegrateST[
  f:(_?ValueQ | _Function), beta_?NumericQ, m1_?NumericQ, m2_?NumericQ, m3_?NumericQ, m4_?NumericQ
] /; AnyInexactNumberQ[beta, m1, m2, m3, m4] := Block[
  {
    sMin = Max[(m1 + m2)^2, (m3 + m4)^2]
  },

  1/(512 \[Pi]^5 beta) NIntegrate[
    BesselK[1, Sqrt[s] beta]/Sqrt[s] IntegrateT[f, s, beta, m1, m2, m3, m4],
    {s, sMin, \[Infinity]},
    (* Method -> {"DoubleExponential", "SymbolicProcessing" -> 0}, *)
    Method -> {Automatic, "SymbolicProcessing" -> 0},
    WorkingPrecision -> Precision[{beta, m1, m2, m3, m4}]
  ]
];

Protect[IntegrateST];

(* Bessel K Ratio *)
(* ************** *)

Attributes[BesselK12] = {Listable, NumericFunction};
BesselK12 /: MakeBoxes[BesselK12[x_], TraditionalForm] := FractionBox[MakeBoxes[BesselK[1, x], TraditionalForm], MakeBoxes[BesselK[2, x], TraditionalForm]];
BesselK12[x_?InexactNumberQ] /; PossibleZeroQ[BesselK[1, x]] := With[
  {
    targetPrecision = Precision[x]
  },
  BesselK12[SetPrecision[x, 2 targetPrecision]]
];
BesselK12[x_?InexactNumberQ] := BesselK[1, x] / BesselK[2, x];

Protect[BesselK12];

(* SquaredAmplitude *)
(* **************** *)

(* For consistency, we wrap all inputs within Particle if the symbol is not
explicitly a particle. *)
SquaredAmplitude[pIn:{___} -> pOut:{___}] /; Or @@ Not @* ParticleQ /@ Join[pIn, pOut] := With[{
    in = Replace[pIn, x_ ? (Not@*ParticleQ) :> Particle[x], {1}],
    out = Replace[pOut, x_ ? (Not@*ParticleQ) :> Particle[x], {1}]
  }
  ,
  SquaredAmplitude[in -> out]
];
SquaredAmplitude[pIn:{___} -> pOut:{___}] /; Or @@ Not @* MatchQ[Particle[_]] /@ Join[pIn, pOut] := With[{
    in = Replace[pIn, x : Except[Particle[_]] :> Particle[x], {1}],
    out = Replace[pOut, x : Except[Particle[_]] :> Particle[x], {1}]
  }
  ,
  SquaredAmplitude[in -> out]
];

(* Squared Amplitude is agnostic of the order of particles *)
SquaredAmplitude[pIn:{___?ParticleQ} -> pOut:{___?ParticleQ}] /; !OrderedQ[pIn] || !OrderedQ[pOut] := SquaredAmplitude[Sort[pIn] -> Sort[pOut]];

(* Display a nice output *)
SquaredAmplitude /: MakeBoxes[
  SquaredAmplitude[pIn:{___?ParticleQ} -> pOut:{___?ParticleQ}],
  TraditionalForm
] := SuperscriptBox[
  TemplateBox[
    {
      RowBox[{
        "\[ScriptCapitalM](",
        RowBox[Riffle[MakeBoxes[#, TraditionalForm] & /@ pIn, ","]],
        "\[LeftRightArrow]",
        RowBox[Riffle[MakeBoxes[#, TraditionalForm] & /@ pOut, ","]],
        ")"
      }]
    },
    "Abs"
  ],
  "2"
];

(* When defining a squared amplitude, make sure we define its canonical form *)
SquaredAmplitude /: Set[SquaredAmplitude[args___], lhs_] /; {args} =!= List @@ SquaredAmplitude[args] := With[
  {
    rhs = SquaredAmplitude[args]
  },
  rhs = lhs
];


(* InteractionGamma *)

(* **************** *)
(* For consistency, we wrap all inputs within Particle *)
InteractionGamma[pIn:{___} -> pOut:{___}] /; Or @@ Not @* ParticleQ /@ Join[pIn, pOut] := With[{
    in = Replace[pIn, x_ ? (Not@*ParticleQ) :> Particle[x], {1}],
    out = Replace[pOut, x_ ? (Not@*ParticleQ) :> Particle[x], {1}]
  }
  ,
  InteractionGamma[in -> out]
];
InteractionGamma[pIn:{___} -> pOut:{___}] /; Or @@ Not @* MatchQ[Particle[_]] /@ Join[pIn, pOut] := With[{
    in = Replace[pIn, x : Except[Particle[_]] :> Particle[x], {1}],
    out = Replace[pOut, x : Except[Particle[_]] :> Particle[x], {1}]
  }
  ,
  InteractionGamma[in -> out]
];
(* Interaction rate is agnostic of the order of particles *)
InteractionGamma[pIn:{__?ParticleQ} -> pOut:{__?ParticleQ}] /; ! OrderedQ[pIn] || ! OrderedQ[pOut] := InteractionGamma[Sort[pIn] -> Sort[pOut]];

InteractionGamma /: MakeBoxes[
  InteractionGamma[pIn:{__?ParticleQ} -> pOut:{__?ParticleQ}],
  TraditionalForm
] := RowBox[{
  "\[Gamma](",
  RowBox[Riffle[MakeBoxes[#, TraditionalForm] & /@ pIn, ","]],
  "\[RightArrow]",
  RowBox[Riffle[MakeBoxes[#, TraditionalForm] & /@ pOut, ","]],
  ")"
}];

InteractionGamma /: MakeBoxes[
  InteractionGamma[pIn:{___?ParticleQ} -> pOut:{___?ParticleQ}][beta_],
  TraditionalForm
] := RowBox[{
  MakeBoxes[InteractionGamma[pIn -> pOut], TraditionalForm],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ")"
}];

(* 1 <-> 2 Interaction *)
InteractionGamma[
  {p1_?ParticleQ} -> {p2_?ParticleQ, p3_?ParticleQ}
][beta_?InexactNumberQ] /; Mass[p1] < Mass[p2] + Mass[p3] = 0;
InteractionGamma[
  {p1_?ParticleQ} -> {p2_?ParticleQ, p3_?ParticleQ}
][beta_?InexactNumberQ] := Block[
  {
    z = Mass[p1] beta,
    ampSq = SquaredAmplitude[{p1} -> {p2, p3}]
  },
  Times[
    1 / (32 Pi^3),
    ampSq,
    Sqrt[X`Kallen\[Lambda][Mass2[p1], Mass2[p2], Mass2[p3]]],
    BesselK[1, z] / z
  ]
];

(* 2 <-> 2 Interaction *)
InteractionGamma[
  {p1_?ParticleQ, p2_?ParticleQ} -> {p3_?ParticleQ, p4_?ParticleQ}
][beta_?InexactNumberQ] := IntegrateST[
  SquaredAmplitude[{p1, p2} -> {p3, p4}],
  beta,
  p1, p2,
  p3, p4
];

Protect[InteractionGamma];

End[];