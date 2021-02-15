Get[FileNameJoin[{DirectoryName[$TestFileName], "Common.wl"}]];

BeginTestSection["ExpandValues"]

f1[] := 1;
f2[] = 2;
f3[] := f1[] + f2[];

VerificationTest[
  ExpandValues[f1]
  ,
  {HoldPattern[f1[]] :> 1}
]

VerificationTest[
  ExpandValues[f2]
  ,
  {HoldPattern[f2[]] :> 2}
]

VerificationTest[
  ExpandValues[f3]
  ,
  {HoldPattern[f3[]] :> f1[] + f2[]}
]

VerificationTest[
  ExpandValues[f1, f2, f3]
  ,
  {
    HoldPattern[f1[]] :> 1,
    HoldPattern[f2[]] :> 2,
    HoldPattern[f3[]] :> f1[] + f2[]
  }
]


N[one] = 1.0;
N[f[x_]] := x / 2;

VerificationTest[
  ExpandValues[one]
  ,
  {HoldPattern[one] :> 1.0}
]

VerificationTest[
  ExpandValues[f]
  ,
  {HoldPattern[f[x_]] :> x / 2}
]


g /: f1[g[x_]] := 2 * x;

VerificationTest[
  ExpandValues[g]
  ,
  {HoldPattern[f1[g[x_]]] :> 2 * x}
]

VerificationTest[
  ExpandValues[f1]
  ,
  {HoldPattern[f1[]] :> 1}
]

EndTestSection[];