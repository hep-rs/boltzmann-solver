#!/usr/bin/env wolframscript

SetOptions[$Output, FormatType -> OutputForm]

$Sections = {
  "all",
  "number_density",
  "phase_space",
  "st_integral"
};

sections = ToLowerCase @ Rest @ $ScriptCommandLine;
If[sections === {},
  Echo["At least one of the followiong must be specified: " <> StringRiffle[$Sections, ", "]];
  Exit[];
];
Block[{unknown = Complement[sections, $Sections]},
  If[unknown =!= {},
    Echo["Unknown sections: " <> StringRiffle[unknown, ", "]];
    Echo["Valid sections: " <> StringRiffle[$Sections, ", "]];
    Exit[];
  ];
];

If[MemberQ[sections, "all"],
  sections = $Sections
];

LaunchKernels[2 $ProcessorCount];
$KernelIDs = ParallelEvaluate[$KernelID];

(*****************************************************************************)
(* Definitions *)
(*****************************************************************************)

$MaxExtraPrecision = 4 $MachinePrecision;
ParallelEvaluate[$MaxExtraPrecision = 4 $MachinePrecision];
$DataDir = ExpandFileName@FileNameJoin[{DirectoryName[$InputFileName], "..", "tests", "data"}];
Echo[$DataDir, "Base output directory: "];

ExactMachinePrecision::usenormal = "Convert the input into an exact machine precision number.";
Attributes[ExactMachinePrecision] = {Listable};
ExactMachinePrecision[x_] := SetPrecision[N[x], Infinity];

CreateDataDir::usage = "Create the directory structure in the output and return the resulting path.";
CreateDataDir[subdirs___String] := Block[{dir},
  dir = FileNameJoin[{$DataDir} ~Join~ {subdirs}];

  Quiet@CreateDirectory[
    dir,
    CreateIntermediateDirectories -> True
  ];

  Echo[dir, "Created sub-directory: "];

  dir
];

GenerateCSV[
  csv_String,
  {headings__String},
  f_Function,
  xInput_List
] := Block[{
    shortName = StringReplace[csv, $DataDir <> "/" -> ""],
    xValues = xInput
  },
  Echo["Generating " <> shortName <> "."];

  (* Makes sure the arguments are a list of lists *)
  If[Head[First[xValues]] =!= List,
    xValues = {#} & /@ xValues;
  ];

  (* Open a CSV file for each kernel *)
  ParallelEvaluate[$csv = OpenWrite[csv <> "." <> ToString[$KernelID]]];

  ParallelDo[
    WriteString[$csv,
      StringRiffle[
        If[Head[#] === String, #, ToString[#, CForm]] & /@ (
          N[f @@ ExactMachinePrecision[x]] //. {
            _Complex -> "NaN",
            Indeterminate -> "NaN",
            ComplexInfinity -> "NaN",
            -Infinity -> "-inf",
            Infinity -> "inf",
            HoldPattern[-Overflow[]] -> "-inf",
            Overflow[] -> "inf",
            Underflow[] -> 0,
            x_ :> 0         /; Abs[x] < $MinMachineNumber,
            x_ :>  Infinity /; x > $MaxMachineNumber,
            x_ :> -Infinity /; x < -$MaxMachineNumber
          }
        ),
        ","
      ] <> "\n"
    ];
    ,
    {x, xValues}
  ];

  ParallelEvaluate[Close@$csv];

  (* Write the CSV headers *)
  $csv = OpenWrite[csv];
  WriteString[$csv, StringRiffle[{headings}, ","] <> "\n"];
  Close@$csv;


  (* Concatenate each sub-file *)
  Run[StringJoin[
    "cat ",
    StringRiffle[csv <> "." <> ToString[#] & /@ $KernelIDs, " "],
    " | sort -n -t, >> ",
    csv
  ]];
  DeleteFile[csv <> "." <> ToString[#] & /@ $KernelIDs];

  (* Compress using zstd *)
  Run["zstd --ultra -22 --rm -f " <> csv];

  Echo["Generated " <> shortName <> "."];
];

Attributes[ReservoirSample] = {HoldRest};
ReservoirSample[n_, arg_, iter__] := Block[{
    sample = {}, p = 0
  },

  Do[
    p += 1;
    If[Length[sample] < n,

      (* Fill the reservoir to begin with *)
      AppendTo[sample, arg]
      ,

      (* Otherwise we add the next argument with decreasing probability *)
      If[RandomInteger[{1, p}] <= n,
        sample[[RandomInteger[{1, n}]]] = arg;
      ];
    ];
    ,
    iter
  ];

  sample
];

Attributes[RandomAccessSample] = {HoldRest};
RandomAccessSample[n_, arg_, iter__] := Block[{
    iterArgs, iterLists
  },
  (* Separate the variables we're iterating over from the corresponding lists,
     and convert range specifications into lists for later. *)

  iterArgs = First /@ {iter};
  iterLists = Table[
    If[Head[i[[2]]] === List, i[[2]], Range @@ i[[2 ;;]]],
    {i, {iter}}
  ];

  Table[
    arg /. Thread[iterArgs -> RandomChoice /@ iterLists],
    n
  ]
];

(* Cover a broad range of positive / negative numbers *)
RealSample[n_] := Join[-10^Subdivide[-10, 10, n], 10^Subdivide[-10, 10, n], Subdivide[-10, 10, n]];
PositiveSample[n_] := Join[10^Subdivide[-10, 10, n], Subdivide[0, 10, n]];

(*****************************************************************************)
(* Test *)
(*****************************************************************************)

If[MemberQ[sections, "test"],
  dir = CreateDataDir["tmp-DELETE-ME"];

  GenerateCSV[
    FileNameJoin[{dir, "poly.csv"}],
    {"x", "x^3 - 2x^2 + x - 1"},
    {#, #^3 - 2 #^2 + # - 1} &,
    Join[
      -10^Subdivide[-10, 10, 10],
      10^Subdivide[-10, 10, 10],
      Subdivide[-10, 10, 10]
    ]
  ];
];

(*****************************************************************************)
(* Particle Statistics *)
(*****************************************************************************)

If[MemberQ[sections, "number_density"],
  dir = CreateDataDir["."];

  Normalization[beta_] = 1/(2 Pi^2) Integrate[
    u^2 / (Exp[u beta] - 1),
    {u, 0, Infinity},
    Assumptions -> beta > 0
  ];

  (* Maxwell-Boltzmann distribution is always analytical *)
  MaxwellBoltzmann[beta_, m_?PossibleZeroQ, mu_] = 1/(2 Pi^2) Integrate[
    u^2 / Exp[beta * (u - mu)],
    {u, 0, Infinity},
    Assumptions -> beta > 0 && mu \[Element] Reals
  ];
  MaxwellBoltzmann[beta_, m_, mu_] = 1/(2 Pi^2) Integrate[
    u Sqrt[u^2 - m^2] / Exp[beta * (u - mu)],
    {u, m, Infinity},
    Assumptions -> beta > 0 && mu \[Element] Reals && m > 0
  ];
  MaxwellBoltzmannAsymmetry[beta_, m_?PossibleZeroQ, mu_] = 1/(2 Pi^2) Integrate[
    u^2 / Exp[beta * (u - mu)] - u^2 / Exp[beta * (u + mu)],
    {u, 0, Infinity},
    Assumptions -> beta > 0 && mu \[Element] Reals
  ];
  MaxwellBoltzmannAsymmetry[beta_, m_, mu_] = 1/(2 Pi^2) Integrate[
    u Sqrt[u^2 - m^2] / Exp[beta * (u - mu)] - u Sqrt[u^2 - m^2] / Exp[beta * (u + mu)],
    {u, m, Infinity},
    Assumptions -> beta > 0 && mu \[Element] Reals && m > 0
  ];
  (* For large values, Mathematica computes K2 to go to zero faster than Exp, leading to 0. *)
  MaxwellBoltzmann[beta_, m_, mu_] /; N@BesselK[2, m beta] == 0 = Simplify@Normal@Series[
    MaxwellBoltzmann[beta, m, mu],
    {mu, Infinity, 0},
    {m, Infinity, 0}
  ];
  MaxwellBoltzmannAsymmetry[beta_, m_, mu_] /; N@BesselK[2, m beta] == 0 = Simplify@Normal@Series[
    MaxwellBoltzmannAsymmetry[beta, m, mu],
    {mu, Infinity, 0},
    {m, Infinity, 0}
  ];

  (* Massless case can be done analytically *)
  FermiDirac[beta_, m_?PossibleZeroQ, mu_] = 1/(2 Pi^2) Integrate[
    u^2 / (Exp[beta * (u - mu)] + 1),
    {u, 0, Infinity},
    Assumptions -> beta > 0
  ];
  FermiDiracAsymmetry[beta_, m_?PossibleZeroQ, mu_] = 1/(2 beta^3 Pi^2) ((mu beta)^3 / 3 + Pi^2 / 3 mu beta);
  (* For large values, PolyLog[3, ...] fails to allocate enough memory, so we
  use the series expansion *)
  FermiDirac[beta_, m_?PossibleZeroQ, mu_] /; beta mu > 10^5 = Normal@Series[
      FermiDirac[beta, 0, mu] 
        /. {beta -> x / mu},
      {x, Infinity, 10}
    ] /. x -> beta mu;
  
  BoseEinstein[beta_, m_?PossibleZeroQ, mu_] /; mu > 0 = "NaN";
  BoseEinstein[beta_, m_?PossibleZeroQ, mu_] = 1/(2 Pi^2) Integrate[
    u^2 / (Exp[beta * (u - mu)] - 1),
    {u, 0, Infinity},
    Assumptions -> beta > 0 && mu < 0
  ];
  BoseEinsteinAsymmetry[beta_, m_?PossibleZeroQ, mu_] /; Abs[mu] > 0 = "NaN";
  BoseEinsteinAsymmetry[beta_, m_?PossibleZeroQ, mu_?PossibleZeroQ] = 0;

  (* Massive case must be done numerically *)
  (* We define the dimensionless Fermi-Dirac and Bose-Einstein integrals *)
  FermiDiracIntegral[m_, mu_] := NIntegrate[
    u Sqrt[u^2 - m^2] / (Exp[u - mu] + 1),
    {u, m, Infinity},
    Exclusions -> u == m,
    PrecisionGoal -> 10,
    AccuracyGoal -> Abs@Log10@$MinMachineNumber,
    WorkingPrecision -> 4 $MachinePrecision,
    MaxRecursion -> Infinity
  ];
  FermiDiracAsymmetryIntegral[m_, mu_] := NIntegrate[
    u Sqrt[u^2 - m^2] / (Exp[u - mu] + 1) - u Sqrt[u^2 - m^2] / (Exp[u + mu] + 1),
    {u, m, Infinity},
    Exclusions -> u == m,
    PrecisionGoal -> 10,
    AccuracyGoal -> Abs@Log10@$MinMachineNumber,
    WorkingPrecision -> 4 $MachinePrecision,
    MaxRecursion -> Infinity
  ];
  BoseEinsteinIntegral[m_, mu_] := NIntegrate[
    u Sqrt[u^2 - m^2] / (Exp[u - mu] - 1),
    {u, m, Infinity},
    Exclusions -> u == m,
    PrecisionGoal -> 10,
    AccuracyGoal -> Abs@Log10@$MinMachineNumber,
    WorkingPrecision -> 4 $MachinePrecision,
    MaxRecursion -> Infinity
  ];
  BoseEinsteinAsymmetryIntegral[m_, mu_] := NIntegrate[
    u Sqrt[u^2 - m^2] / (Exp[u - mu] - 1) - u Sqrt[u^2 - m^2] / (Exp[u + mu] - 1),
    {u, m, Infinity},
    Exclusions -> u == m,
    PrecisionGoal -> 10,
    AccuracyGoal -> Abs@Log10@$MinMachineNumber,
    WorkingPrecision -> 4 $MachinePrecision,
    MaxRecursion -> Infinity
  ];

  FermiDirac[beta_?NumericQ, m_, mu_] := 1/(2 Pi^2 beta^3) FermiDiracIntegral[m beta, mu beta];
  FermiDirac[beta_?NumericQ, m_, mu_] /; m < 10^-2 mu := FermiDirac[beta, 0, mu];
  FermiDiracAsymmetry[beta_?NumericQ, m_, mu_] := 1/(2 Pi^2 beta^3) FermiDiracAsymmetryIntegral[m beta, mu beta];
  FermiDiracAsymmetry[beta_?NumericQ, m_, mu_] /; m < 10^-2 mu := FermiDiracAsymmetry[beta, 0, mu];
  BoseEinstein[beta_?NumericQ, m_, mu_] /; mu > m = "NaN";
  BoseEinstein[beta_?NumericQ, m_, mu_] := 1/(2 Pi^2 beta^3) BoseEinsteinIntegral[m beta, mu beta];
  BoseEinsteinAsymmetry[beta_?NumericQ, m_, mu_] /; Abs[mu] > m = "NaN";
  BoseEinsteinAsymmetry[beta_?NumericQ, m_, mu_] := 1/(2 Pi^2 beta^3) BoseEinsteinAsymmetryIntegral[m beta, mu beta];


  (* Turn off warning *)
  ParallelEvaluate[
    Off[
      General::munfl,
      General::ovfl,
      General::unfl,
      N::meprec,
      NIntegrate::inumri,
      NIntegrate::slwcon,
      Power::indet,
      Power::infy
    ];
  ];

  (* Prevent infinite memory being allocation *)
  ParallelEvaluate[
    $MinPrecision = 2 $MachinePrecision;
    $MaxPrecision = 20 $MachinePrecision;
  ];

  GenerateCSV[
    FileNameJoin[{dir, "number_density.csv"}],
    {"beta", "m", "mu", "bose-einstein", "normalized bose-einstein", "fermi-dirac", "normalized fermi-dirac", "maxwell-boltzmann", "maxwell-boltzmann normalized"},
    Function[{beta, m, mu}, Block[{
        be = BoseEinstein[beta, m, mu],
        fd = FermiDirac[beta, m, mu],
        mb = MaxwellBoltzmann[beta, m, mu]
      },
      {
        beta, m, mu,
        be, If[be =!= "NaN", be / Normalization[beta], "NaN"],
        fd, If[fd =!= "NaN", fd / Normalization[beta], "NaN"],
        mb, If[mb =!= "NaN", mb / Normalization[beta], "NaN"]
      }]],
    RandomAccessSample[10^5,
      {beta, m, mu},
      {m, PositiveSample[200]},
      {mu, RealSample[200]},
      {beta, Select[# > 0&] @ PositiveSample[200]}
    ]
  ];

  GenerateCSV[
    FileNameJoin[{dir, "number_density_asymmetry.csv"}],
    {"beta", "m", "mu", "bose-einstein", "normalized bose-einstein", "fermi-dirac", "normalized fermi-dirac", "maxwell-boltzmann", "maxwell-boltzmann normalized"},
    Function[{beta, m, mu}, Block[{
        be = BoseEinsteinAsymmetry[beta, m, mu],
        fd = FermiDiracAsymmetry[beta, m, mu],
        mb = MaxwellBoltzmannAsymmetry[beta, m, mu]
      },
      {
        beta, m, mu,
        be, If[be =!= "NaN", be / Normalization[beta], "NaN"],
        fd, If[fd =!= "NaN", fd / Normalization[beta], "NaN"],
        mb, If[mb =!= "NaN", mb / Normalization[beta], "NaN"]
      }]],
    RandomAccessSample[10^5,
      {beta, m, mu},
      {m, PositiveSample[200]},
      {mu, RealSample[200]},
      {beta, Select[# > 0&] @ PositiveSample[200]}
    ]
  ];

  (* Turn on warning *)
  ParallelEvaluate[
    On[
      General::munfl,
      General::ovfl,
      General::unfl,
      N::meprec,
      NIntegrate::inumri,
      NIntegrate::slwcon,
      Power::indet,
      Power::infy
    ];
  ];

  (* Reset precision *)
  ParallelEvaluate[
    $MinPrecision = 0;
    $MaxPrecision = Infinity;
  ];

];

(*****************************************************************************)
(* Phase Space *)
(*****************************************************************************)


If[MemberQ[sections, "phase_space"],
  dir = CreateDataDir["."];

  FermiDirac[beta_, e_, m_, mu_] = 1 / (Exp[(e - mu) beta] + 1);
  BoseEinsteinExact[x_] = 1 / (Exp[x] - 1);
  BoseEinsteinApprox[x_] = Normal@Series[BoseEinsteinExact[x], {x, 0, 10}];
  BoseEinstein[beta_, e_, m_, mu_] /; mu > e = "NaN";
  BoseEinstein[beta_, e_, m_, mu_] = With[{x = (e - mu) beta},
    If[Abs[x] < 1/3,
      BoseEinsteinApprox[x],
      BoseEinsteinExact[x]
    ]
  ];
  MaxwellBoltzmann[beta_, e_, m_, mu_] = 1 / (Exp[(e - mu) beta]);

  (* Turn off warnings *)
  ParallelEvaluate[
    Off[
      General::munfl,
      General::ovfl,
      Power::infy
    ];
  ];

  (* Prevent infinite memory being allocation *)
  ParallelEvaluate[
    $MinPrecision = 2 $MachinePrecision;
    $MaxPrecision = 20 $MachinePrecision;
  ];

  GenerateCSV[
    FileNameJoin[{dir, "phase_space.csv"}],
    {"beta", "e", "m", "mu", "fermi-dirac", "bose-einstein", "maxwell-boltzmann"},
    Function[{beta, e, m, mu}, 
      {
        beta, ExactMachinePrecision[m + e], m, mu,
        FermiDirac[beta, ExactMachinePrecision[m + e], m, mu],
        BoseEinstein[beta, ExactMachinePrecision[m + e], m, mu],
        MaxwellBoltzmann[beta, ExactMachinePrecision[m + e], m, mu]
      }
    ],
    SeedRandom[123454321];
    RandomAccessSample[10^6,
      {beta, e, m, mu},
      {beta, Select[# > 0&] @ PositiveSample[200]},
      {e, PositiveSample[200]},
      {m, PositiveSample[200]},
      {mu, RealSample[200]}
    ]
  ];

  (* Reinstate warnings *)
  ParallelEvaluate[
    On[General::munfl];
    On[General::ovfl];
    On[Power::infy];
  ];

  (* Reset precision *)
  ParallelEvaluate[
    $MinPrecision = 0;
    $MaxPrecision = Infinity;
  ];
];


(*****************************************************************************)
(* S T Integrals *)
(*****************************************************************************)


If[MemberQ[sections, "st_integral"],
  dir = CreateDataDir["."];

  Lambda[a_, b_, c_] := a^2 + b^2 + c^2 - 2(a b + b c + a c);

  MandelstamTMin[s_, m1_, m2_, m3_, m4_] := Module[
    {
      baseline = 1/2 (m1^2 + m2^2 + m3^2 + m4^2 - s - (m1^2 - m2^2) (m3^2 - m4^2) / s),
      cosine = Sqrt[Lambda[s, m1^2, m2^2]] Sqrt[Lambda[s, m3^2, m4^2]] / (2 s)
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
      cosine = Sqrt[Lambda[s, m1^2, m2^2]] Sqrt[Lambda[s, m3^2, m4^2]] / (2 s)
    },
    If[s < (m1 + m2)^2, Message[MandelstamTRange::invalids12] ];
    If[s < (m3 + m4)^2, Message[MandelstamTRange::invalids34] ];
    If[PossibleZeroQ[s], Return[0] ];
    baseline + cosine
  ];
  MandelstanTMax[Infinity, _, _, _, _] := 0;

  f1[s_, t_] := s^2;
  f2[s_, t_] := s t;
  f3[s_, t_] := t^2;
  f4[s_, t_] := s^2 / (s + 1);
  f5[s_, t_] := s t / (s + 1);
  f6[s_, t_] := s^2 t^2 / ((s + 1) (t^2 + 1));

  stIntegral[f_, beta_, m1_, m2_, m3_, m4_] := Block[{
      sMin = Max[m1 + m2, m3 + m4]^2
    },
    NIntegrate[
      f[s, t] BesselK[1, Sqrt[s] beta] / (Sqrt[s] beta),
      {s, sMin, Infinity},
      {t, MandelstamTMin[s, m1, m2, m3, m4], MandelstamTMax[s, m1, m2, m3, m4]},
      Method -> {
        "GlobalAdaptive",
        "SingularityDepth" -> 2
      },
      MaxRecursion -> 5,
      WorkingPrecision -> 60,
      PrecisionGoal -> 6,
      AccuracyGoal -> 100
    ] / (512 \[Pi]^5)
  ];

  (* Turn off warnings *)
  ParallelEvaluate[
    Off[
      General::munfl,
      General::ovfl,
      Power::infy
    ];
  ];

  GenerateCSV[
    FileNameJoin[{dir, "st_integral_massless.csv"}],
    {"beta", "m1", "m2", "m3", "m4", "f1", "f2", "f3", "f4", "f5", "f6"},
    Function[{beta, m1, m2, m3, m4}, 
      {
        beta, m1, m2, m3, m4,
        stIntegral[f1, beta, m1, m2, m3, m4],
        stIntegral[f2, beta, m1, m2, m3, m4],
        stIntegral[f3, beta, m1, m2, m3, m4],
        stIntegral[f4, beta, m1, m2, m3, m4],
        stIntegral[f5, beta, m1, m2, m3, m4],
        stIntegral[f6, beta, m1, m2, m3, m4]
      }
    ],
    SeedRandom[123454321];
    RandomAccessSample[10^3,
      {beta, m1, m2, m3, m4},
      {beta, Select[# > 0&] @ PositiveSample[200]},
      {m1, {0}},
      {m2, {0}},
      {m3, {0}},
      {m4, {0}}
    ]
  ];

  GenerateCSV[
    FileNameJoin[{dir, "st_integral_massive.csv"}],
    {"beta", "m1", "m2", "m3", "m4", "f1", "f2", "f3", "f4", "f5", "f6"},
    Function[{beta, m1, m2, m3, m4}, 
      {
        beta, m1, m2, m3, m4,
        stIntegral[f1, beta, m1, m2, m3, m4],
        stIntegral[f2, beta, m1, m2, m3, m4],
        stIntegral[f3, beta, m1, m2, m3, m4],
        stIntegral[f4, beta, m1, m2, m3, m4],
        stIntegral[f5, beta, m1, m2, m3, m4],
        stIntegral[f6, beta, m1, m2, m3, m4]
      }
    ],
    SeedRandom[123454321];
    RandomAccessSample[10^3,
      {beta, m1, m2, m3, m4},
      {beta, Select[# > 0&] @ PositiveSample[200]},
      {m1, PositiveSample[200]},
      {m2, PositiveSample[200]},
      {m3, PositiveSample[200]},
      {m4, PositiveSample[200]}
    ]
  ];

  (* Reinstate warnings *)
  ParallelEvaluate[
    On[General::munfl];
    On[General::ovfl];
    On[Power::infy];
  ];

  ClearAll[MandelstamTMax, MandelstamTMin, f1, f2, f3, f4, f5, f6, Lambda];
];