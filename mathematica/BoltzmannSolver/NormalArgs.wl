Begin["`Private`"];

(* Test whether the arguments are "normal"

Specifically, all particles should be wrapped inside the "Particle" head and the
incoming / outgoing particles are ordered. *)
NormalArgs[pIn:{___} -> pOut:{___}] := Block[{},
  And[
    AllTrue[pIn, MatchQ[Particle[_]]],
    AllTrue[pOut, MatchQ[Particle[_]]],
    OrderedQ[pIn],
    OrderedQ[pOut]
  ]
];

End[];