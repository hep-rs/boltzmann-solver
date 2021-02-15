Needs["BoltzmannSolver`"];

ApproxEq[n_] := Function[{x, y}, Abs[x - y] < 10^(-n) || Abs[x - y] / (Abs[x] + Abs[y]) < 10^(-n)];

ApproxEqList[n_] := Function[
  {x, y},
  And[
    Head[x] === List,
    Head[x] === Head[y],
    Dimensions[x] === Dimensions[y],
    And @@ ApproxEq[n] @@@ Transpose[{x, y}]
  ]
]