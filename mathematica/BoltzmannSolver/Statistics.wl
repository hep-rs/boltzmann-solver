(* BeginPackage["Statistics`", {"Particle`", "Extra`"}]; *)

FermiDirac::usage = "Fermi\[LongDash]Dirac statistics";
BoseEinstein::usage = "Bose\[LongDash]Einstein statistics";
MaxwellBoltzmann::usage = "Maxwell\[LongDash]Boltzmann statistics";

Protect[FermiDirac, BoseEinstein, MaxwellBoltzmann];

PhaseSpace::usage = "Phase space statistic.

If a statistic is given, this is a function of inverse temperature
\[Beta], energy, mass and chemical potential \[Mu].  The latter two are optional
and if unspecified are assume to be 0.

If a particle is given, the mass is automatically inferred and the result is
automatically multiplied by the appropriate degrees of freedom.";

NumberDensity::usage = "Equilibrium number density of a particle species.

If a statistic is specified, the arguments are inverse temperaure \[Beta], mass
and chemical potential \[Mu].  If a paricle is given, the mass is automatically infered.";

NormalizedNumberDensity::usage = "Equilibrium number density of a particle species normalized to
a massless bosonic degree of freedom.

If a statistic is specified, the arguments are inverse temperaure \[Beta], mass
and chemical potential \[Mu].  If a paricle is given, the mass is automatically infered.";


Begin["`Private`"] (* Begin Private Context *)

StatisticPattern = FermiDirac | BoseEinstein | MaxwellBoltzmann;

(* Phase Space *)
(* *********** *)

Attributes[PhaseSpace] = {NumericFunction, Listable};

PhaseSpace::chemicalPotentialFermiDirac = "Chemical potential for fermions must be non-negative."
PhaseSpace::chemicalPotentialBoseEinstein = "Chemical potential for bosons must be non-positive."

(* Formats *)
PhaseSpace /: MakeBoxes[PhaseSpace[s_ : StatisticPattern, beta_, e_], TraditionalForm] := Block[
  {
    stat = Switch[
      s,
      FermiDirac, "FD",
      BoseEinstein, "BE",
      MaxwellBoltzmann, "MB"
    ]
  }
  ,
  RowBox[{
    SubscriptBox["f", stat],
    "(",
    MakeBoxes[beta, TraditionalForm],
    ";",
    MakeBoxes[e, TraditionalForm],
    ")"
  }]
];

PhaseSpace /: MakeBoxes[PhaseSpace[s_ : StatisticPattern, beta_, e_, m_], TraditionalForm] := Block[
  {
    stat = Switch[
      s,
      FermiDirac, "FD",
      BoseEinstein, "BE",
      MaxwellBoltzmann, "MB"
    ]
  }
  ,
  RowBox[{
    SubscriptBox["f", stat],
    "(",
    MakeBoxes[beta, TraditionalForm],
    ";",
    MakeBoxes[e, TraditionalForm],
    ",", MakeBoxes[m, TraditionalForm], ")"
  }]
];

PhaseSpace /: MakeBoxes[PhaseSpace[s_ : StatisticPattern, beta_, e_, m_, mu_], TraditionalForm] := Block[
  {
    stat = Switch[
      s,
      FermiDirac, "FD",
      BoseEinstein, "BE",
      MaxwellBoltzmann, "MB"
    ]
  }
  ,
  RowBox[{
    SubscriptBox["f", stat],
    "(",
    MakeBoxes[beta, TraditionalForm],
    ";",
    MakeBoxes[e, TraditionalForm],
    ",",
    MakeBoxes[m, TraditionalForm],
    ",",
    MakeBoxes[mu, TraditionalForm],
    ")"
  }]
];

PhaseSpace /: MakeBoxes[PhaseSpace[p_?ParticleQ, beta_, e_], TraditionalForm] := RowBox[{
  SubscriptBox["f", MakeBoxes[p, TraditionalForm]],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ";",
  MakeBoxes[e, TraditionalForm],
  ")"
}];

PhaseSpace /: MakeBoxes[PhaseSpace[p_?ParticleQ, beta_, e_, mu_], TraditionalForm] := RowBox[{
  SubscriptBox["f", MakeBoxes[p, TraditionalForm]],
  "(",
  MakeBoxes[beta, TraditionalForm], ";",
  MakeBoxes[e, TraditionalForm],
  ",",
  MakeBoxes[mu, TraditionalForm],")"
  }];


(* Definitions *)
PhaseSpace[
  FermiDirac,
  beta_,
  e_,
  m_:0,
  mu_:0
] /; AnyInexactNumberQ[beta, e, m, mu] := Block[{},
  If[mu < 0, Message[PhaseSpace::chemicalPotentialFermiDirac]];

  1/(Exp[(e - mu) beta] + 1)
];


PhaseSpace[
  BoseEinstein,
  beta_,
  e_,
  m_:0,
  mu_:0
] /; AnyInexactNumberQ[beta, e, m, mu] = Block[{},
  If[mu > 0, Message[PhaseSpace::chemicalPotentialBoseEinstein]];

  1/(Exp[(e - mu) beta] - 1)
];


PhaseSpace[
  MaxwellBoltzmann,
  beta_,
  e_,
  m_:0,
  mu_:0
] /; AnyInexactNumberQ[beta, e, m, mu] = Exp[-(e - mu) beta];


PhaseSpace[
  p_?ParticleQ,
  e_,
  mu_:0
] /; AnyInexactNumberQ[beta, e, mu] := PhaseSpace[Statistic[p]][beta, e, Mass[p], mu];


(* ExpandValues *)
PhaseSpace /: ExpandValues[PhaseSpace] := {
  PhaseSpace[FermiDirac, beta_, e_, m_:0, mu_:0] :> 1/(Exp[(e - mu) beta] + 1),
  PhaseSpace[BoseEinstein, beta_, e_, m_:0, mu_:0] :> 1/(Exp[(e - mu) beta] - 1),
  PhaseSpace[MaxwellBoltzmann, beta_, e_, m_:0, mu_:0] :> Exp[-(e - mu) beta]
};

Protect[PhaseSpace];


(* Number Density *)
(* ************** *)

NumberDensity::chemicalPotentialFermiDirac = "Chemical potential for fermions must be non-negative."
NumberDensity::chemicalPotentialBoseEinstein = "Chemical potential for bosons must be non-positive."

Attributes[NumberDensity] = {NumericFunction, Listable};

(* Formats *)
NumberDensity /: MakeBoxes[NumberDensity[s_ : StatisticPattern, beta_], TraditionalForm] := Block[
  {
    stat = Switch[
      s,
      FermiDirac, "FD",
      BoseEinstein, "BE",
      MaxwellBoltzmann, "MB"
    ]
  }
  ,
  RowBox[{
    SubscriptBox["N", stat],
    "(",
    MakeBoxes[beta, TraditionalForm],
    ")"
  }]
];

NumberDensity /: MakeBoxes[NumberDensity[s_ : StatisticPattern, beta_, m_], TraditionalForm] := Block[
  {
    stat = Switch[
      s,
      FermiDirac, "FD",
      BoseEinstein, "BE",
      MaxwellBoltzmann, "MB"
    ]
  }
  ,
  RowBox[{
    SubscriptBox["N", stat],
    "(",
    MakeBoxes[beta, TraditionalForm],
    ";",
    MakeBoxes[m, TraditionalForm],
    ")"
  }]
];

NumberDensity /: MakeBoxes[NumberDensity[s_ : StatisticPattern, beta_, m_, mu_], TraditionalForm] := Block[
  {
    stat = Switch[
      s,
      FermiDirac, "FD",
      BoseEinstein, "BE",
      MaxwellBoltzmann, "MB"
    ]
  }
  ,
  RowBox[{
    SubscriptBox["N", stat], "(",
    MakeBoxes[beta, TraditionalForm],
    ";",
    MakeBoxes[m, TraditionalForm],
    ",",
    MakeBoxes[mu, TraditionalForm],
    ")"
  }]
];

NumberDensity /: MakeBoxes[NumberDensity[p_?ParticleQ, beta_], TraditionalForm] := RowBox[{
  SubscriptBox["N", MakeBoxes[p, TraditionalForm]],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ")"
  }];

NumberDensity /: MakeBoxes[NumberDensity[p_?ParticleQ, beta_, mu_], TraditionalForm] := RowBox[{
  SubscriptBox["N", MakeBoxes[p, TraditionalForm]],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ";",
  MakeBoxes[mu, TraditionalForm],
  ")"
  }];


(* Definitions *)

(* Massive Fermi-Dirac case *)
NumberDensity[
  FermiDirac,
  beta_,
  m_?(Not@*PossibleZeroQ),
  mu_:0
] /; AnyInexactNumberQ[beta, m, mu] := Block[{},
  If[mu < 0, Message[NumberDensity::chemicalPotentialFermiDirac]];

  1 / (2 Pi^2) NIntegrate[
    1/(Exp[(u - mu) beta] + 1) u Sqrt[u^2 - m^2],
    {u, m, Infinity},
    Method -> {
      "GlobalAdaptive",
      "SymbolicProcessing" -> False
    },
    WorkingPrecision -> Precision[{beta, m, mu}]
  ]
];

(* Massless Fermi-Dirac case *)
NumberDensity[
  FermiDirac,
  beta_,
  m:(_?PossibleZeroQ):0,
  mu_:0
] /; AnyInexactNumberQ[beta, m, mu] = Block[{},
  If[mu < 0, Message[NumberDensity::chemicalPotentialFermiDirac]];

  1 / (2 Pi^2) Integrate[
    1/(Exp[(u - mu) beta] + 1) u^2,
    {u, 0, Infinity},
    Assumptions -> m > 0 && beta > 0 && Element[mu, Reals]
  ]
];


(* Massive Bose-Einstein case *)
NumberDensity[
  BoseEinstein,
  beta_,
  m_?(Not@*PossibleZeroQ),
  mu_:0
] /; AnyInexactNumberQ[beta, m, mu] := Block[{},
  If[mu > 0, Message[NumberDensity::chemicalPotentialBoseEinstein]];

  1 / (2 Pi^2) NIntegrate[
    1/(Exp[(u - mu) beta] - 1) u Sqrt[u^2 - m^2],
    {u, m, Infinity},
    Method -> {
      "GlobalAdaptive",
      "SymbolicProcessing" -> False
    },
    WorkingPrecision -> Precision[{beta, m, mu}]
  ]
];

(* Massless Bose-Einstein case *)
NumberDensity[
  BoseEinstein,
  beta_,
  m:(_?PossibleZeroQ):0,
  mu_:0
] /; AnyInexactNumberQ[beta, m, mu] = Block[{},
  If[mu > 0, Message[NumberDensity::chemicalPotentialBoseEinstein]];

  1 / (2 Pi^2) Integrate[
    1/(Exp[(u - mu) beta] - 1) u^2,
    {u, 0, Infinity},
    Assumptions -> m > 0 && beta > 0 && Element[mu, Reals]
  ]
];


(* Massive Maxwell-Boltzmann case *)
NumberDensity[
  MaxwellBoltzmann,
  beta_,
  m_?(Not@*PossibleZeroQ),
  mu_:0
] /; AnyInexactNumberQ[beta, m, mu] = 1 / (2 Pi^2) Integrate[
  Exp[-(u - mu) beta] u Sqrt[u^2 - m^2],
  {u, m, Infinity},
  Assumptions -> m >  0 && beta > 0 && Element[mu, Reals]
];

(* Massless Maxwell-Boltzmann case *)
NumberDensity[
  MaxwellBoltzmann,
  beta_?NumberQ,
  m:(_?PossibleZeroQ):0,
  mu_:0
] /; AnyInexactNumberQ[beta, m, mu] = 1 / (2 Pi^2) Integrate[
  Exp[-(u - mu) beta] u^2,
  {u, 0, Infinity},
  Assumptions -> m > 0 && beta > 0 && Element[mu, Reals]
];


(* Particle *)
NumberDensity[p_?ParticleQ, beta_, mu_:0] /; AnyInexactNumberQ[beta, mu] := BoltzmannSolver`DoF[p] NumberDensity[BoltzmannSolver`Statistic[p], beta, Mass[p], mu];

NumberDensity[Particle[p_?ParticleQ]] := \[FormalN][p];
NumberDensity[p_?ParticleQ] := \[FormalN][p];


(* ExpandValues *)
NumberDensity /: ExpandValues[NumberDensity] = {
  NumberDensity[FermiDirac, beta_, m:(_?PossibleZeroQ):0, mu_:0] :> 1 / (2 Pi^2) Integrate[
    u^2 / (Exp[(u - mu) beta] + 1),
    {u, 0, Infinity},
    Assumptions -> m >= 0 && beta > 0 && Element[mu, Reals]
  ]
  ,
  NumberDensity[BoseEinstein, beta_, m:(_?PossibleZeroQ):0, mu_:0] :> 1 / (2 Pi^2) Integrate[
    u^2 / (Exp[(u - mu) beta] - 1),
    {u, 0, Infinity},
    Assumptions -> m >= 0 && beta > 0 && Element[mu, Reals]
  ]
  ,
  NumberDensity[MaxwellBoltzmann, beta_, m_:0, mu_:0] :> 1 / (2 Pi^2) Integrate[
    (u Sqrt[u^2 - m^2])/Exp[(u - mu) beta],
    {u, m, Infinity},
    Assumptions -> m >= 0 && beta > 0 && Element[mu, Reals]
  ]
  ,
  NumberDensity[stat:StatisticPattern, beta_, m_, mu_] :> 1 / (2 Pi^2) Inactive[NIntegrate][
    PhaseSpace[stat, beta, u, m, mu] u Sqrt[u^2 - m^2] /. ExpandValues[PhaseSpace],
    {u, m, Infinity}
  ]
};

Protect[NumberDensity];


(* Normalized Number Density *)
(* ************************* *)

Attributes[NormalizedNumberDensity] = {Listable, NumericFunction};

(* Formats *)
NormalizedNumberDensity /: MakeBoxes[NormalizedNumberDensity[s_ : StatisticPattern, beta_], TraditionalForm] := Block[
  {
    stat = Switch[
      s,
      FermiDirac, "FD",
      BoseEinstein, "BE",
      MaxwellBoltzmann, "MB"
    ]
  }
  ,
  RowBox[{
    SubscriptBox["n", stat],
    "(",
    MakeBoxes[beta, TraditionalForm],
    ")"
  }]
];

NormalizedNumberDensity /: MakeBoxes[NormalizedNumberDensity[s_ : StatisticPattern, beta_, m_], TraditionalForm] := Block[
  {
    stat = Switch[
      s,
      FermiDirac, "FD",
      BoseEinstein, "BE",
      MaxwellBoltzmann, "MB"
    ]
  }
  ,
  RowBox[{
    SubscriptBox["n", stat],
    "(",
    MakeBoxes[beta, TraditionalForm],
    ";",
    MakeBoxes[m, TraditionalForm],
    ")"
  }]
];

NormalizedNumberDensity /: MakeBoxes[NormalizedNumberDensity[s_ : StatisticPattern, beta_, m_, mu_], TraditionalForm] := Block[
  {
    stat = Switch[
      s,
      FermiDirac, "FD",
      BoseEinstein, "BE",
      MaxwellBoltzmann, "MB"
    ]
  }
  ,
  RowBox[{
    SubscriptBox["n", stat],
    "(",
    MakeBoxes[beta, TraditionalForm],
    ";",
    MakeBoxes[m, TraditionalForm],
    ",",
    MakeBoxes[mu, TraditionalForm],
    ")"
  }]
];

NormalizedNumberDensity /: MakeBoxes[NormalizedNumberDensity[p_?ParticleQ, beta_], TraditionalForm] := RowBox[{
  SubscriptBox["n", MakeBoxes[p, TraditionalForm]],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ")"
  }];

NormalizedNumberDensity /: MakeBoxes[NormalizedNumberDensity[p_?ParticleQ, beta_, m_], TraditionalForm] := RowBox[{
  SubscriptBox["n", MakeBoxes[p, TraditionalForm]],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ";",
  MakeBoxes[m, TraditionalForm],
  ")"
  }];

NormalizedNumberDensity /: MakeBoxes[NormalizedNumberDensity[p_?ParticleQ, beta_, m_, mu_], TraditionalForm] := RowBox[{
  SubscriptBox["n", MakeBoxes[p, TraditionalForm]],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ";",
  MakeBoxes[m, TraditionalForm],
  ",",
  MakeBoxes[mu, TraditionalForm],
  ")"
  }];


(* Definitions *)
NormalizedNumberDensity[
  stat : StatisticPattern,
  beta_,
  m_:0,
  mu_:0
] /; AnyInexactNumberQ[beta, m, mu] := NumberDensity[stat, beta, m, mu] / NumberDensity[BoseEinstein, beta];


NormalizedNumberDensity[
  p_?ParticleQ,
  beta_,
  mu_:0
] /; AnyInexactNumberQ[beta, mu] := NumberDensity[p, beta, mu] / NumberDensity[BoseEinstein, beta];


NormalizedNumberDensity[Particle[p_?ParticleQ]] := \[FormalN][p];
NormalizedNumberDensity[p_?ParticleQ] := \[FormalN][p];

(* ExpandValues *)
NormalizedNumberDensity /: ExpandValues[NormalizedNumberDensity] := {
  NormalizedNumberDensity[stat:StatisticPattern, beta_, m_:0, mu_:0] :> NumberDensity[stat, beta, m, mu] / NumberDensity[BoseEinstein, beta]
};

Protect[NormalizedNumberDensity];


End[] (* End Private Context *)

(* EndPackage[]; *)