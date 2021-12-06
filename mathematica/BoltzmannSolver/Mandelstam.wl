MandelstamTRange::usage = "Calculate the range of the Mandelstam t variables.  The
first argument is the Mandelstam s variable.  The remaining four arguments are
the masses of the four particles.";
MandelstamTMin::usage = "Calculate the lower bound of the Mandelstam t variables.  The
first argument is the Mandelstam s variable.  The remaining four arguments are
the masses of the four particles.";
MandelstamTMax::usage = "Calculate the upper bound of the Mandelstam t variables.  The
first argument is the Mandelstam s variable.  The remaining four arguments are
the masses of the four particles.";
IntegrateT::usage  = "Integrate a squared amplitude over the Mandelstam t variable.";
IntegrateST::usage = "Integrate a squared amplitude over the Mandelstam s and t variables.";


Begin["`Private`"];


(* Mandelstam T Range *)
(* ****************** *)

MandelstamTRange::invalids12 = "The Mandelstam s value must be larger than (m1 + m2)^2.";
MandelstamTRange::invalids34 = "The Mandelstam s value must be larger than (m3 + m4)^2.";

Attributes[MandelstamTRange] = {Listable, NumericFunction};
Attributes[MandelstamTMin] = {Listable, NumericFunction};
Attributes[MandelstamTMax] = {Listable, NumericFunction};

(* Allow particles to be given instead of masses *)
MandelstamTRange[s_, p1_?ParticleQ, m2_, m3_, m4_] := MandelstamTRange[s, Mass[p1], m2, m3, m4 ];
MandelstamTRange[s_, m1_, p2_?ParticleQ, m3_, m4_] := MandelstamTRange[s, m1, Mass[p2], m3, m4 ];
MandelstamTRange[s_, m1_, m2_, p3_?ParticleQ, m4_] := MandelstamTRange[s, m1, m2, Mass[p3], m4 ];
MandelstamTRange[s_, m1_, m2_, m3_, p4_?ParticleQ] := MandelstamTRange[s, m1, m2, m3, Mass[p4] ];

MandelstamTMin[s_, p1_?ParticleQ, m2_, m3_, m4_] := MandelstamTMin[s, Mass[p1], m2, m3, m4 ];
MandelstamTMin[s_, m1_, p2_?ParticleQ, m3_, m4_] := MandelstamTMin[s, m1, Mass[p2], m3, m4 ];
MandelstamTMin[s_, m1_, m2_, p3_?ParticleQ, m4_] := MandelstamTMin[s, m1, m2, Mass[p3], m4 ];
MandelstamTMin[s_, m1_, m2_, m3_, p4_?ParticleQ] := MandelstamTMin[s, m1, m2, m3, Mass[p4] ];

MandelstamTMax[s_, p1_?ParticleQ, m2_, m3_, m4_] := MandelstamTMax[s, Mass[p1], m2, m3, m4 ];
MandelstamTMax[s_, m1_, p2_?ParticleQ, m3_, m4_] := MandelstamTMax[s, m1, Mass[p2], m3, m4 ];
MandelstamTMax[s_, m1_, m2_, p3_?ParticleQ, m4_] := MandelstamTMax[s, m1, m2, Mass[p3], m4 ];
MandelstamTMax[s_, m1_, m2_, m3_, p4_?ParticleQ] := MandelstamTMax[s, m1, m2, m3, Mass[p4] ];

(* Numerical expansion expands the arguments *)
MandelstamTRange /: N[MandelstamTRange[s_, m1_, m2_, m3_, m4_], ___precision] := MandelstamTRange@@N[{s, m1, m2, m3, m4}, precision];
MandelstamTMin /: N[MandelstamTMin[s_, m1_, m2_, m3_, m4_], ___precision] := MandelstamTMin@@N[{s, m1, m2, m3, m4}, precision];
MandelstamTMax /: N[MandelstamTMax[s_, m1_, m2_, m3_, m4_], ___precision] := MandelstamTMax@@N[{s, m1, m2, m3, m4}, precision];

MandelstamTRange[s_, m1_, m2_, m3_, m4_] := Module[
  {
    x1 = m1^2 / s,
    x2 = m2^2 / s,
    x3 = m3^2 / s,
    x4 = m4^2 / s
  },
  If[s < (m1 + m2)^2, Message[MandelstamTRange::invalids12] ];
  If[s < (m3 + m4)^2, Message[MandelstamTRange::invalids34] ];
  If[PossibleZeroQ[s], Return[{0, 0}] ];
  s/4 * {
    (x1 - x2 - x3 + x4)^2 - (Sqrt[X`Kallen\[Lambda][1, x1, x2] ] - Sqrt[X`Kallen\[Lambda][1, x3, x4] ])^2,
    (x1 - x2 - x3 + x4)^2 - (Sqrt[X`Kallen\[Lambda][1, x1, x2] ] + Sqrt[X`Kallen\[Lambda][1, x3, x4] ])^2
  }
];
MandelstamTRange[Infinity, _, _, _, _] := {-Infinity, 0};

MandelstamTMin[s_, m1_, m2_, m3_, m4_] := Module[
  {
    baseline = 1/2 (m1^2 + m2^2 + m3^2 + m4^2 - s - (m1^2 - m2^2) (m3^2 - m4^2) / s),
    cosine = Sqrt[X`Kallen\[Lambda][s, m1^2, m2^2]] Sqrt[X`Kallen\[Lambda][s, m3^2, m4^2]] / (2 s)
  },
  If[s < (m1 + m2)^2, Message[MandelstamTRange::invalids12] ];
  If[s < (m3 + m4)^2, Message[MandelstamTRange::invalids34] ];
  If[PossibleZeroQ[s], Return[0] ];
  baseline - cosine
];
MandelstamTMin[Infinity, _, _, _, _] := -Infinity;

MandelstamTMax[s_, m1_, m2_, m3_, m4_] := Module[
  {
    baseline = 1/2 (m1^2 + m2^2 + m3^2 + m4^2 - s - (m1^2 - m2^2) (m3^2 - m4^2) / s),
    cosine = Sqrt[X`Kallen\[Lambda][s, m1^2, m2^2]] Sqrt[X`Kallen\[Lambda][s, m3^2, m4^2]] / (2 s)
  },
  If[s < (m1 + m2)^2, Message[MandelstamTRange::invalids12] ];
  If[s < (m3 + m4)^2, Message[MandelstamTRange::invalids34] ];
  If[PossibleZeroQ[s], Return[0] ];
  baseline + cosine
];
MandelstanTMax[Infinity, _, _, _, _] := 0;

(*
 * The following functions are used to calculate the Mandelstam variables
 * for the decay of a particle.
 *
 * The first argument is the mass of the decaying particle.
 * The second argument is the mass of the particle that decays.
*)

Protect[MandelstamTRange];
Protect[MandelstamTMin];
Protect[MandelstamTMax];


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
  f:(_?ValueQ | _Function), s_, beta_, m1_, m2_, m3_, m4_
] := Block[
  {
    tMin, tMax
  },
  {tMin, tMax} = MandelstamTRange[s, m1, m2, m3, m4];
  NIntegrate[
    f[s, t],
    {t, tMin, tMax},
    (* Method -> {"ClenshawCurtisRule", "SymbolicProcessing" -> 0}, *)
    Method -> {Automatic, "SymbolicProcessing" -> 0},
    WorkingPrecision -> (Precision[{s, beta, m1, m2, m3, m4}] /. Infinity -> MachinePrecision)
  ]
];

Protect[IntegrateT];


(* Mandelstam S and T Integration *)
(* ****************************** *)

Attributes[IntegrateST] = {HoldFirst, Listable};

IntegrateST[f_, {s_, t_}, beta_, p1_?ParticleQ, m2_, m3_, m4_] = IntegrateST[f, {s, t}, beta, Mass[p1], m2, m3, m4 ]
IntegrateST[f_, {s_, t_}, beta_, m1_, p2_?ParticleQ, m3_, m4_] = IntegrateST[f, {s, t}, beta, m1, Mass[p2], m3, m4 ]
IntegrateST[f_, {s_, t_}, beta_, m1_, m2_, p3_?ParticleQ, m4_] = IntegrateST[f, {s, t}, beta, m1, m2, Mass[p3], m4 ]
IntegrateST[f_, {s_, t_}, beta_, m1_, m2_, m3_, p4_?ParticleQ] = IntegrateST[f, {s, t}, beta, m1, m2, m3, Mass[p4] ]

IntegrateST[f_, {s_, t_}, beta_, m1_, m2_, m3_, m4_, opt:OptionsPattern[]] := Block[
  {
    sMin = Max[m1 + m2, m3 + m4]^2
  },

  1/(512 \[Pi]^5) NIntegrate[
    f BesselK[1, Sqrt[s] beta]/(Sqrt[s] beta),
    {s, Max[m1 + m2, m3 + m4]^2, \[Infinity]},
    {t, MandelstamTMin[s, m1, m2, m3, m4], MandelstamTMax[s, m1, m2, m3, m4]},
    opt
  ]
];

Protect[IntegrateST];


(* InteractionGamma *)
(* **************** *)

(* Use the same standardized arguments for the squared amplitude *)

InteractionGamma[pIn:{___} -> pOut:{___}] /; Not@NormalArgs[pIn -> pOut] := With[
  {
    in = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pIn ],
    out = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pOut ]
  },
  InteractionGamma[in -> out]
];

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

ExpandValues[InteractionGamma] := {
  InteractionGamma[
    {p1_} -> {p2_, p3_}
  ][beta_] 
  :> 
  Block[
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
  ]
};

ExpandValues[X`Kallen\[Lambda]] := {X`Kallen\[Lambda][x___]:> X`KallenExpand[X`Kallen\[Lambda][x]]};

(* 2 <-> 2 Interaction *)
InteractionGamma[
  {p1_?ParticleQ, p2_?ParticleQ} -> {p3_?ParticleQ, p4_?ParticleQ}
][beta_?InexactNumberQ] := IntegrateST[
  SquaredAmplitude[{p1, p2} -> {p3, p4}],
  beta,
  p1, p2,
  p3, p4
];

(* When defining a interactions, make sure we define its canonical form *)
InteractionGamma /: Set[InteractionGamma[pIn:{___} -> pOut:{___}], lhs_] /; Not @ NormalArgs[pIn -> pOut] := With[
  {
    in = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pIn ],
    out = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pOut ]
  },
  InteractionGamma[in -> out] = lhs
];


InteractionGamma /: Unset[InteractionGamma[pIn:{___} -> pOut:{___}]] /; Not @ NormalArgs[pIn -> pOut] := With[
  {
    in = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pIn ],
    out = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pOut ]
  },
  Unset[InteractionGamma[in -> out]]
];

Protect[InteractionGamma];

(* ScaledInteractionGamma *)

End[];