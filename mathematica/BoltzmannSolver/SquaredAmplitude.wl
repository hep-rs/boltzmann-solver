SquaredAmplitude::usage = "Squared Amplitude.";


Begin["`Private`"];


(* SquaredAmplitude *)
(* **************** *)

SquaredAmplitude[pIn:{___} -> pOut:{___}] /; Not@NormalArgs[pIn -> pOut] := With[
  {
    in = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pIn ],
    out = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pOut ]
  },
  SquaredAmplitude[in -> out]
];

(* When defining a squared amplitude, make sure we define its canonical form *)
SquaredAmplitude /: Set[SquaredAmplitude[pIn:{___} -> pOut:{___}], lhs_] /; Not @ NormalArgs[pIn -> pOut] := With[
  {
    in = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pIn ],
    out = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pOut ]
  },
  SquaredAmplitude[in -> out] = lhs
];

SquaredAmplitude /: SetDelayed[SquaredAmplitude[pIn:{___} -> pOut:{___}], lhs_] /; Not @ NormalArgs[pIn -> pOut] := With[
  {
    in = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pIn ],
    out = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pOut ]
  },
  SquaredAmplitude[in -> out] := lhs
];

SquaredAmplitude /: Unset[SquaredAmplitude[pIn:{___} -> pOut:{___}]] /; Not @ NormalArgs[pIn -> pOut] := With[
  {
    in = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pIn ],
    out = Sort[ Replace[x : Except[Particle[_]] :> Particle[x]] /@ pOut ]
  },
  Unset[SquaredAmplitude[in -> out]]
];


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

End[];