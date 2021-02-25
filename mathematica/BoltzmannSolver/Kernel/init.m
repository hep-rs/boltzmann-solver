(* Wolfram Language Init File *)

(* Import Package-X under it's own namespace. *)
Check[
  Needs["X`"],
  Abort[]
];

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

EndPackage[];