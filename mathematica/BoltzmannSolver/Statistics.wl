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


PhaseSpace /: MakeBoxes[PhaseSpace[FermiDirac, beta_, e_], TraditionalForm] := RowBox[{
  SubscriptBox["f", "FD"],
  "(", MakeBoxes[beta, TraditionalForm],
  ";", MakeBoxes[e, TraditionalForm],
  ")"
  }];
PhaseSpace /: MakeBoxes[PhaseSpace[FermiDirac, beta_, e_, m_], TraditionalForm] := RowBox[{
  SubscriptBox["f", "FD"],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ";",
  MakeBoxes[e, TraditionalForm],
  ",", MakeBoxes[m, TraditionalForm], ")"
  }];
PhaseSpace /: MakeBoxes[PhaseSpace[FermiDirac, beta_, e_, m_, mu_], TraditionalForm] := RowBox[{
  SubscriptBox["f", "FD"],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ";",
  MakeBoxes[e, TraditionalForm],
  ",",
  MakeBoxes[m, TraditionalForm],
  ",",
  MakeBoxes[mu, TraditionalForm],
  ")"
  }];

PhaseSpace[
  FermiDirac,
  beta_?NumericQ,
  e_?NumericQ,
  m:_?NumericQ:0,
  mu:_?NumericQ:0
] /; AnyInexactNumberQ[beta, e, m, mu] := Block[{},
  If[mu < 0, Message[PhaseSpace::chemicalPotentialFermiDirac]];

  1/(Exp[(e - mu) beta] + 1)
];


PhaseSpace /: MakeBoxes[PhaseSpace[BoseEinstein, beta_, e_], TraditionalForm] := RowBox[{
  SubscriptBox["f", "BE"],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ";",
  MakeBoxes[e, TraditionalForm],
  ")"
  }];
PhaseSpace /: MakeBoxes[PhaseSpace[BoseEinstein, beta_, e_, m_], TraditionalForm] := RowBox[{
  SubscriptBox["f", "BE"],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ";",
  MakeBoxes[e, TraditionalForm],
  ",",
  MakeBoxes[m, TraditionalForm],
  ")"}];
PhaseSpace /: MakeBoxes[PhaseSpace[BoseEinstein, beta_, e_, m_, mu_], TraditionalForm] := RowBox[{
  SubscriptBox["f", "BE"],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ";",
  MakeBoxes[e, TraditionalForm],
  ",",
  MakeBoxes[m, TraditionalForm],
  ",",
  MakeBoxes[mu, TraditionalForm],
  ")"
  }];

PhaseSpace[
  BoseEinstein,
  beta_?NumericQ,
  e_?NumericQ,
  m:_?NumericQ:0,
  mu:_?NumericQ:0
] /; AnyInexactNumberQ[beta, e, m, mu] = Block[{},
  If[mu > 0, Message[PhaseSpace::chemicalPotentialBoseEinstein]];

  1/(Exp[(e - mu) beta] - 1)
];


PhaseSpace /: MakeBoxes[PhaseSpace[MaxwellBoltzmann, beta_, e_], TraditionalForm] := RowBox[{
  SubscriptBox["f", "MB"],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ";",
  MakeBoxes[e, TraditionalForm],
  ")"
  }];
PhaseSpace /: MakeBoxes[PhaseSpace[MaxwellBoltzmann, beta_, e_, m_], TraditionalForm] := RowBox[{
  SubscriptBox["f", "MB"],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ";",
  MakeBoxes[e, TraditionalForm],
  ",",
  m,
  ")"
  }];
PhaseSpace /: MakeBoxes[PhaseSpace[MaxwellBoltzmann, beta_, e_, m_, mu_], TraditionalForm] := RowBox[{
  SubscriptBox["f", "MB"],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ";",
  MakeBoxes[e, TraditionalForm],
  ",",
  MakeBoxes[m, TraditionalForm],
  ",",
  MakeBoxes[mu, TraditionalForm],
  ")"
  }];

PhaseSpace[
  MaxwellBoltzmann,
  beta_?NumericQ,
  e_?NumericQ,
  m:_?NumericQ:0,
  mu:_?NumericQ:0
] /; AnyInexactNumberQ[beta, e, MakeBoxes[m, TraditionalForm], mu] = Exp[-(e - mu) beta];


PhaseSpace /: MakeBoxes[PhaseSpace[p_?ParticleQ, beta_, e_], TraditionalForm] := RowBox[{SubscriptBox["f", MakeBoxes[p, TraditionalForm]], "(", MakeBoxes[beta, TraditionalForm], ";", MakeBoxes[e, TraditionalForm],
")"}];
PhaseSpace /: MakeBoxes[PhaseSpace[p_?ParticleQ, beta_, e_, mu_], TraditionalForm] := RowBox[{
  SubscriptBox["f", MakeBoxes[p, TraditionalForm]],
  "(",
  MakeBoxes[beta, TraditionalForm], ";",
  MakeBoxes[e, TraditionalForm],
  ",",
  MakeBoxes[mu, TraditionalForm],")"
  }];

PhaseSpace[
  p_?ParticleQ,
  e_?NumericQ,
  mu:_?NumericQ:0
] /; AnyInexactNumberQ[beta, e, mu] := PhaseSpace[Statistic[p]][beta, e, Mass[p], mu];


Protect[PhaseSpace];


(* Number Density *)
(* ************** *)

NumberDensity::chemicalPotentialFermiDirac = "Chemical potential for fermions must be non-negative."
NumberDensity::chemicalPotentialBoseEinstein = "Chemical potential for bosons must be non-positive."

Attributes[NumberDensity] = {NumericFunction, Listable};
NumberDensity[stat : StatisticPattern, beta_] := NumberDensity[stat, beta, 0, 0];
NumberDensity[stat : StatisticPattern, beta_, m_] := NumberDensity[stat, beta, m, 0];


NumberDensity /: MakeBoxes[NumberDensity[FermiDirac, beta_], TraditionalForm] := RowBox[{
  SubscriptBox["N", "FD"],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ")"
  }];
NumberDensity /: MakeBoxes[NumberDensity[FermiDirac, beta_, m_], TraditionalForm] := RowBox[{
  SubscriptBox["N", "FD"],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ";",
  MakeBoxes[m, TraditionalForm],
  ")"}];
NumberDensity /: MakeBoxes[NumberDensity[FermiDirac, beta_, m_, mu_], TraditionalForm] := RowBox[{
  SubscriptBox["N", "FD"], "(",
  MakeBoxes[beta, TraditionalForm],
  ";",
  MakeBoxes[m, TraditionalForm],
  ",",
  MakeBoxes[mu, TraditionalForm],
  ")"
  }];

(* Massless case *)
NumberDensity[
  FermiDirac,
  beta_?NumericQ,
  m_?PossibleZeroQ,
  mu_?NumericQ
] /; AnyInexactNumberQ[beta, m, mu] = Block[{},
  If[mu < 0, Message[NumberDensity::chemicalPotentialFermiDirac]];

  1 / (2 Pi^2) Integrate[
    1/(Exp[(u - mu) beta] + 1) u^2,
    {u, 0, Infinity},
    Assumptions -> m > 0 && beta > 0 && Element[mu, Reals]
  ]
];

(* Massive case *)
NumberDensity[
  FermiDirac,
  beta_?NumericQ,
  m_?NumericQ,
  mu_?NumericQ
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


NumberDensity /: MakeBoxes[NumberDensity[BoseEinstein, beta_], TraditionalForm] := RowBox[{
  SubscriptBox["N", "BE"],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ")"
  }];
NumberDensity /: MakeBoxes[NumberDensity[BoseEinstein, beta_, m_], TraditionalForm] := RowBox[{
  SubscriptBox["N", "BE"],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ";",
  MakeBoxes[m, TraditionalForm],
  ")"
  }];
NumberDensity /: MakeBoxes[NumberDensity[BoseEinstein, beta_, m_, mu_], TraditionalForm] := RowBox[{
  SubscriptBox["N", "BE"],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ";",
  MakeBoxes[m, TraditionalForm],
  ",",
  MakeBoxes[mu, TraditionalForm],
  ")"
  }];

(* Massless case *)
NumberDensity[
  BoseEinstein,
  beta_?NumericQ,
  m_?PossibleZeroQ,
  mu_?NumericQ
] /; AnyInexactNumberQ[beta, m, mu] = Block[{},
  If[mu > 0, Message[NumberDensity::chemicalPotentialBoseEinstein]];

  1 / (2 Pi^2) Integrate[
    1/(Exp[(u - mu) beta] - 1) u^2,
    {u, 0, Infinity},
    Assumptions -> m > 0 && beta > 0 && Element[mu, Reals]
  ]
];

(* Massive case *)
NumberDensity[
  BoseEinstein,
  beta_?NumericQ,
  m_?NumericQ,
  mu:_?NumericQ
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


NumberDensity /: MakeBoxes[NumberDensity[MaxwellBoltzmann, beta_], TraditionalForm] := RowBox[{
  SubscriptBox["N", "MB"],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ")"
  }];
NumberDensity /: MakeBoxes[NumberDensity[MaxwellBoltzmann, beta_, m_], TraditionalForm] := RowBox[{
  SubscriptBox["N", "MB"],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ";",
  MakeBoxes[m, TraditionalForm],")"
  }];
NumberDensity /: MakeBoxes[NumberDensity[MaxwellBoltzmann, beta_, m_, mu_], TraditionalForm] := RowBox[{
  SubscriptBox["N", "MB"],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ";",
  MakeBoxes[m, TraditionalForm],
  ",",
  MakeBoxes[mu, TraditionalForm],
  ")"
  }];

(* Massless case *)
NumberDensity[
  MaxwellBoltzmann,
  beta_?NumberQ,
  m_?PossibleZeroQ,
  mu_?NumericQ
] /; AnyInexactNumberQ[beta, m, mu] = 1 / (2 Pi^2) Integrate[
  Exp[-(u - mu) beta] u^2,
  {u, 0, Infinity},
  Assumptions -> m > 0 && beta > 0 && Element[mu, Reals]
];

(* Massive case *)
NumberDensity[
  MaxwellBoltzmann,
  beta_?NumericQ,
  m_?NumericQ,
  mu_?NumericQ
] /; AnyInexactNumberQ[beta, m, mu] = 1 / (2 Pi^2) Integrate[
  Exp[-(u - mu) beta] u Sqrt[u^2 - m^2],
  {u, m, Infinity},
  Assumptions -> m > 0 && beta > 0 && Element[mu, Reals]
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

NumberDensity[p_?ParticleQ, beta_, mu_:0] /; AnyInexactNumberQ[beta, mu] := BoltzmannSolver`DoF[p] NumberDensity[BoltzmannSolver`Statistic[p], beta, Mass[p], mu];

Protect[NumberDensity];


(* Normalized Number Density *)
(* ************************* *)

Attributes[NormalizedNumberDensity] = {Listable, NumericFunction};

NormalizedNumberDensity /: MakeBoxes[NormalizedNumberDensity[FermiDirac, beta_], TraditionalForm] := RowBox[{
  SubscriptBox["n", "FD"],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ")"
  }];
NormalizedNumberDensity /: MakeBoxes[NormalizedNumberDensity[FermiDirac, beta_, m_], TraditionalForm] := RowBox[{
  SubscriptBox["n", "FD"],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ";",
  MakeBoxes[m, TraditionalForm],
  ")"
  }];
NormalizedNumberDensity /: MakeBoxes[NormalizedNumberDensity[FermiDirac, beta_, m_, mu_], TraditionalForm] := RowBox[{
  SubscriptBox["n", "FD"],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ";",
  MakeBoxes[m, TraditionalForm],
  ",",
  MakeBoxes[mu, TraditionalForm],
  ")"
  }];

NormalizedNumberDensity[
  FermiDirac,
  beta_?NumericQ,
  m_?NumericQ,
  mu_?NumericQ
] /; AnyInexactNumberQ[beta, m, mu] := NumberDensity[FermiDirac, beta, m, mu] / NumberDensity[BoseEinstein, beta];


NormalizedNumberDensity /: MakeBoxes[NormalizedNumberDensity[BoseEinstein, beta_], TraditionalForm] := RowBox[{
  SubscriptBox["n", "BE"],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ")"
  }];
NormalizedNumberDensity /: MakeBoxes[NormalizedNumberDensity[BoseEinstein, beta_, m_], TraditionalForm] := RowBox[{
  SubscriptBox["n", "BE"],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ";",
  MakeBoxes[m, TraditionalForm],
  ")"
  }];
NormalizedNumberDensity /: MakeBoxes[NormalizedNumberDensity[BoseEinstein, beta_, m_, mu_], TraditionalForm] := RowBox[{
  SubscriptBox["n", "BE"],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ";",
  MakeBoxes[m, TraditionalForm],
  ",",
  MakeBoxes[mu, TraditionalForm],
  ")"
  }];


NormalizedNumberDensity[
  BoseEinstein,
  beta_?NumericQ,
  m_?NumericQ,
  mu_?NumericQ
] /; AnyInexactNumberQ[beta, m, mu] := NumberDensity[BoseEinstein,beta, m, mu] / NumberDensity[BoseEinstein,beta];


NormalizedNumberDensity /: MakeBoxes[NormalizedNumberDensity[MaxwellBoltzmann, beta_], TraditionalForm] := RowBox[{
  SubscriptBox["n", "MB"],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ")"
  }];
NormalizedNumberDensity /: MakeBoxes[NormalizedNumberDensity[MaxwellBoltzmann, beta_, m_], TraditionalForm] := RowBox[{
  SubscriptBox["n", "MB"],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ";",
  MakeBoxes[m, TraditionalForm],
  ")"
  }];
NormalizedNumberDensity /: MakeBoxes[NormalizedNumberDensity[MaxwellBoltzmann, beta_, m_, mu_], TraditionalForm] := RowBox[{
  SubscriptBox["n", "MB"],
  "(",
  MakeBoxes[beta, TraditionalForm],
  ";",
  MakeBoxes[m, TraditionalForm],
  ",",
  MakeBoxes[mu, TraditionalForm],
  ")"
  }];

NormalizedNumberDensity[
  MaxwellBoltzmann,
  beta_?NumericQ,
  m_?NumericQ,
  mu_?NumericQ
] /; AnyInexactNumberQ[beta, m, mu] := NumberDensity[MaxwellBoltzmann,beta, m, mu] / NumberDensity[BoseEinstein,beta];


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

NormalizedNumberDensity[
  p_?ParticleQ,
  beta_?NumericQ,
  mu_?NumericQ
] /; AnyInexactNumberQ[beta, mu] := NumberDensity[p, beta, mu] / NumberDensity[BoseEinstein][beta];

Protect[NormalizedNumberDensity];


End[] (* End Private Context *)

(* EndPackage[]; *)