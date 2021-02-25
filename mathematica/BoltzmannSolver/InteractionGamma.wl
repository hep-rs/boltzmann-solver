InteractionGamma::usage = "Interaction rate density.";
ScaledInteractionGamma::usage = "Interaction rate density scaled by number densities.";

Begin["`Private`"];


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

(* When defining a interactions, make sure we define its canonical form *)
InteractionGamma /: Set[InteractionGamma[pIn:{___} -> pOut:{___}], lhs_] /; Not @ NormalArgs[pIn -> pOut] := With[
  {
    in = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pIn ],
    out = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pOut ]
  },
  InteractionGamma[in -> out] = lhs
];

InteractionGamma /: SetDelayed[InteractionGamma[pIn:{___} -> pOut:{___}], lhs_] /; Not @ NormalArgs[pIn -> pOut] := With[
  {
    in = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pIn ],
    out = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pOut ]
  },
  InteractionGamma[in -> out] := lhs
];

InteractionGamma /: Unset[InteractionGamma[pIn:{___} -> pOut:{___}]] /; Not @ NormalArgs[pIn -> pOut] := With[
  {
    in = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pIn ],
    out = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pOut ]
  },
  Unset[InteractionGamma[in -> out]]
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


(* ScaledInteractionGamma *)

(* ********************** *)
(* Use the same standardized arguments for the squared amplitude *)
ScaledInteractionGamma[pIn:{___} -> pOut:{___}] /; Not@NormalArgs[pIn -> pOut] := With[
  {
    in = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pIn ],
    out = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pOut ]
  },
  ScaledInteractionGamma[in -> out]
];


(* When defining a interactions, make sure we define its canonical form *)
ScaledInteractionGamma /: Set[ScaledInteractionGamma[pIn:{___} -> pOut:{___}], lhs_] /; Not @ NormalArgs[pIn -> pOut] := With[
  {
    in = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pIn ],
    out = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pOut ]
  },
  ScaledInteractionGamma[in -> out] := lhs
];

ScaledInteractionGamma /: SetDelayed[ScaledInteractionGamma[pIn:{___} -> pOut:{___}], lhs_] /; Not @ NormalArgs[pIn -> pOut] := With[
  {
    in = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pIn ],
    out = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pOut ]
  },
  ScaledInteractionGamma[in -> out] = lhs
];

ScaledInteractionGamma /: Unset[ScaledInteractionGamma[pIn:{___} -> pOut:{___}]] /; Not @ NormalArgs[pIn -> pOut] := With[
  {
    in = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pIn ],
    out = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pOut ]
  },
  Unset[ScaledInteractionGamma[in -> out]]
];


ScaledInteractionGamma[pIn:{___} -> pOut:{___}][beta_?InexactNumberQ] := Block[
  {forward, backward}
  ,
  forward = Product[\[FormalN][i][beta] / NormalizedNumberDensity[i, beta], {i, pIn}];
  backward = Product[\[FormalN][i][beta] / NormalizedNumberDensity[i, beta], {i, pOut}];

  (forward - backward) InteractionGamma[pIn -> pOut][beta]
];

ScaledInteractionGamma[{p1_?ParticleQ} -> {p2_?ParticleQ, p3_?ParticleQ}][beta_?InexactNumberQ] := Block[
  {}
  ,
  If[
    Mass[p1] beta < 10
    ,
    (* m β < threshold *)
    1 / (32 Pi^2)
    * SquaredAmplitude[{p1} -> {p2, p3}]
    * Sqrt[X`Kallen\[Lambda][Mass2[p1], Mass2[p2], Mass2[p3]]]
    * BesselK[1, Mass[p1] beta] / (Mass[p1] beta)
    * (
        \[FormalN][p1][beta] / NormalizedNumberDensity[p1, beta]
        -
        \[FormalN][p2][beta] / NormalizedNumberDensity[p2, beta]
        * \[FormalN][p3][beta] / NormalizedNumberDensity[p3, beta]
      )

    ,
    (* m β > threshold *)
    Zeta[3] / (16 Pi^2)
    * SquaredAmplitude[{p1} -> {p2, p3}]
    * Sqrt[X`Kallen\[Lambda][Mass2[p1], Mass2[p2], Mass2[p3]]]
    * BesselK12[Mass[p1] beta] / (Mass[p1] beta)^3 / DoF[p1]
    *  (
        \[FormalN][p1][beta]
        -
        NormalizedNumberDensity[p1, beta]
        * \[FormalN][p2][beta] / NormalizedNumberDensity[p2, beta]
        * \[FormalN][p3][beta] / NormalizedNumberDensity[p3, beta]
      )
  ]
]

End[];