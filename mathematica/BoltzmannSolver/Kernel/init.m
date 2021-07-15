(* Wolfram Language Init File *)

(* Import Package-X under it's own namespace. *)
Check[
  Needs["X`"];
  ,
  Abort[]
];
(* $ContextPath = Cases[Except["X`"]][$ContextPath]; *)
X`Utilities`DisableFancyIO[X`LDot, X`LTensor, X`LoopIntegrate, X`FermionLine, X`FermionLineProduct];

BeginPackage["BoltzmannSolver`"];

Get["BoltzmannSolver`Extra`"];
Get["BoltzmannSolver`Particle`"];

Get["BoltzmannSolver`Statistics`"];
Get["BoltzmannSolver`GStar`"];
Get["BoltzmannSolver`Hubble`"];

Get["BoltzmannSolver`Mandelstam`"];
Get["BoltzmannSolver`NormalArgs`"];
Get["BoltzmannSolver`SquaredAmplitude`"];
Get["BoltzmannSolver`InteractionGamma`"];

Protect[ExpandValues];

EndPackage[];