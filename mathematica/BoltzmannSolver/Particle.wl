Particle::usage = "Indicates that the symbol within is a particle, irrespective of the content.";
ParticleQ::usage = "Test whether a symbol is defined as a particle.";
Mass::usage = "Mass of a particle.";
Mass2::usage = "Sqared mass of a particle.";
Width::usage = "Width of a particle.";
Width2::usage = "Squared width of a particle.";
Momentum::usage = "Momentum of a particle.";
DoF::usage = "Internal degrees of freedom for a particle.";
Spin::usage = "Get the spin of a particle."
Statistic::usage = "Quantum statistic of a particle.";

DefineParticle::usage = "Define a new particle.

Note that the Spin should be specified as:
- 0 for a scalar
- 1 for a fermion
- 2 for a vector
- 3 for a spin-3/2 particle
- 4 for a spin-2 particle";

Begin["`Private`"];

Attributes[Particle] = {HoldAll, Listable};
Attributes[ParticleQ] = {HoldAll, Listable};

Attributes[Mass] = {HoldAll, Listable};
Attributes[Mass2] = {HoldAll, Listable};
Attributes[Width] = {HoldAll, Listable};
Attributes[Width2] = {HoldAll, Listable};
Attributes[Momentum] = {HoldAll, Listable};
Attributes[DoF] = {HoldAll, Listable};
Attributes[Spin] = {HoldAll, Listable};
Attributes[Statistic] = {HoldAll, Listable};

Particle  /: MakeBoxes[Particle[p_?ParticleQ],     TraditionalForm] := MakeBoxes[p, TraditionalForm];

DoF       /: MakeBoxes[DoF[p_?ParticleQ],         TraditionalForm] := SubscriptBox["g", MakeBoxes[p, TraditionalForm]];
Spin      /: MakeBoxes[Spin[p_?ParticleQ],        TraditionalForm] := RowBox[{"Spin(", MakeBoxes[p, TraditionalForm], ")"}];
Statistic /: MakeBoxes[Statistic[p_?ParticleQ],   TraditionalForm] := RowBox[{"Statistic(", MakeBoxes[p, TraditionalForm], ")"}];

Mass      /: MakeBoxes[Mass[p_?ParticleQ],        TraditionalForm] := SubscriptBox["m", MakeBoxes[p, TraditionalForm]];
             MakeBoxes[Mass[p_?ParticleQ]^i_,     TraditionalForm] := SubsuperscriptBox["m", MakeBoxes[p, TraditionalForm], MakeBoxes[i, TraditionalForm]];
Width     /: MakeBoxes[Width[p_?ParticleQ],       TraditionalForm] := SubscriptBox["\[CapitalGamma]", MakeBoxes[p, TraditionalForm]];
             MakeBoxes[Width[p_?ParticleQ]^i_,    TraditionalForm] := SubsuperscriptBox["\[CapitalGamma]", MakeBoxes[p, TraditionalForm], MakeBoxes[i, TraditionalForm]];
Momentum  /: MakeBoxes[Momentum[p_?ParticleQ],    TraditionalForm] := SubscriptBox["p", MakeBoxes[p, TraditionalForm]];
             MakeBoxes[Momentum[p_?ParticleQ]^i_, TraditionalForm] := SubsuperscriptBox["p", MakeBoxes[p, TraditionalForm], MakeBoxes[i, TraditionalForm]];

(* Nested applications of Particle are flattened.  Note we can't use the `Flat`
attribute for some reason. *)
Particle[Particle[p_]] := Particle[p];

ParticleQ[_] := False;
ParticleQ[Particle[_]] := True;

(* Wrap the argument inside Particle *)
DoF[p : Except[Particle[_]]]       /; ParticleQ[p] := DoF[Particle[p]];
Spin[p : Except[Particle[_]]]      /; ParticleQ[p] := Spin[Particle[p]];
Statistic[p : Except[Particle[_]]] /; ParticleQ[p] := Statistic[Particle[p]];
Mass[p : Except[Particle[_]]]      /; ParticleQ[p] := Mass[Particle[p]];
Width[p : Except[Particle[_]]]     /; ParticleQ[p] := Width[Particle[p]];
Momentum[p : Except[Particle[_]]]  /; ParticleQ[p] := Momentum[Particle[p]];


Mass2[p_]  := Mass[p]^2;
Width2[p_] := Width[p]^2;

Options[DefineParticle] = {
  Display -> Automatic,
  Spin -> 0,
  DoF -> Automatic,
  Mass -> Automatic,
  Width -> Automatic,
  Momentum -> Automatic,
  Complex -> False
};

Attributes[DefineParticle] = {HoldFirst};

DefineParticle[symbol_, opt : OptionsPattern[]] := With[{
    p = Particle[symbol]
  }
  ,
  ParticleQ[symbol] = True;

  Quiet[
    If[OptionValue[Display] =!= Automatic,
      Particle /: MakeBoxes[Particle[symbol], TraditionalForm] = OptionValue[Display];
      ,
      Particle /: MakeBoxes[Particle[symbol], TraditionalForm] =.;
    ];

    If[OptionValue[Mass] =!= Automatic,
      Mass /: N@Mass[p] = OptionValue[Mass];
      Mass /: InexactNumberQ@Mass[p] = InexactNumberQ@OptionValue[Mass];
      ,
      Mass /: N@Mass[p] =.;
      Mass /: InexactNumberQ@Mass[p] = False;
    ];

    If[OptionValue[Momentum] =!= Automatic,
      Momentum /: N@Momentum[p] = OptionValue[Momentum];
      Momentum /: InexactNumberQ@Momentum[p] = InexactNumberQ@OptionValue[Momentum];
      ,
      Momentum /: N@Momentum[p] =.;
      Momentum /: InexactNumberQ@Momentum[p] = False;
    ];

    If[OptionValue[Width] =!= Automatic,
      Width /: N@Width[p] = OptionValue[Width];
      Width /: InexactNumberQ@Width[p] = InexactNumberQ@OptionValue[Width];
      ,
      Width /: N@Width[p] =.;
      Width /: InexactNumberQ@Width[p] = False;
    ];
    ,
    {Unset::norep, TagUnset::norep}
  ];

  Spin /: N@Spin[p] = OptionValue[Spin];

  (* Due to circulate dependencies, we have to explicitly specify the  *)
  Statistic[p] = If[
    Mod[N@Spin[p], 2] == 0,
    BoltzmannSolver`BoseEinstein,
    BoltzmannSolver`FermiDirac
  ];

  DoF /: N[DoF[p]] = OptionValue[DoF] /. {
    Automatic ->
      If[OptionValue[Complex], 2, 1]
      * Switch[
        Statistic[p]
        ,
        BoseEinstein,
        If[PossibleZeroQ[ Mass[p] ],
          2,
          2 * Spin[p] + 1
        ]
        ,
        FermiDirac,
        2 * Spin[p] + 1
      ]
  };
];

Protect[DefineParticle];

End[]; (* End Private Context *)

(* EndPackage[]; *)