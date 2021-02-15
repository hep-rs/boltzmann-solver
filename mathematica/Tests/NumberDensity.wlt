Get[FileNameJoin[{DirectoryName[$TestFileName], "TestCommon.wl"}]];

BeginTestSection["NumberDensity"]

(* FermiDirac *)
VerificationTest[
  NumberDensity[FermiDirac, 1.0]
  ,
  NumberDensity[FermiDirac, 1.0, 0.0, 0.0]
  ,
  SameTest -> ApproxEq[6],
  TestID -> "ed46a58e-7e4e-58b3-9340-0b32552b5a79"
]
VerificationTest[
  NumberDensity[FermiDirac, 1.0]
  ,
  0.0913454
  ,
  SameTest -> ApproxEq[6],
  TestID -> "5619eed1-9f64-5c46-80df-3030d27ce4e7"
]

VerificationTest[
  NumberDensity[FermiDirac, 1.0, 2.0]
  ,
  NumberDensity[FermiDirac, 1.0, 2.0, 0.0]
  ,
  SameTest -> ApproxEq[6],
  TestID -> "96596b59-70f6-596d-be3f-8318536be6ce"
]
VerificationTest[
  NumberDensity[FermiDirac, 1.0, 2.0]
  ,
  0.049765
  ,
  SameTest -> ApproxEq[6],
  TestID -> "8d9af9c7-16c0-51cd-9d8c-09a41a2a4225"
]

VerificationTest[
  NumberDensity[FermiDirac, 1.0, 2.0, 3.0]
  ,
  0.692817
  ,
  SameTest -> ApproxEq[6],
  TestID -> "7ad6235d-a78d-5088-9889-066a211b8079"
]


(* BoseEinstein *)
VerificationTest[
  NumberDensity[BoseEinstein, 1.0]
  ,
  NumberDensity[BoseEinstein, 1.0, 0.0, 0.0]
  ,
  TestID -> "88132e57-14e5-51f6-9801-4aee6dd32ab3"
]
VerificationTest[
  NumberDensity[BoseEinstein, 1.0]
  ,
  0.121794
  ,
  SameTest -> ApproxEq[6]
  ,
  TestID -> "ea8ba9c9-905b-54fd-82a9-981e6ad9be39"
]

VerificationTest[
  NumberDensity[BoseEinstein, 1.0, 2.0]
  ,
  NumberDensity[BoseEinstein, 1.0, 2.0, 0.0]
  ,
  TestID -> "74924929-755c-522e-af76-ad0019bd2618"
]
VerificationTest[
  NumberDensity[BoseEinstein, 1.0, 2.0]
  ,
  0.0533103
  ,
  SameTest -> ApproxEq[6],
  TestID -> "60755342-75bf-5fae-8fc1-2d4c5782a174"
]

VerificationTest[
  NumberDensity[BoseEinstein, 1.0, 2.0, -3.0]
  ,
  0.00256456
  ,
  SameTest -> ApproxEq[6],
  TestID -> "679daf26-8d0f-5f2c-b3fa-5f0694f6e7be"
]


(* MaxwellBoltzmann *)
VerificationTest[
  NumberDensity[MaxwellBoltzmann, 1.0]
  ,
  NumberDensity[MaxwellBoltzmann, 1.0, 0.0, 0.0]
  ,
  TestID -> "1ce0b2fd-8cbc-5423-b816-b5ef8ddcc815"
]
VerificationTest[
  NumberDensity[MaxwellBoltzmann, 1.0]
  ,
  0.101321
  ,
  SameTest -> ApproxEq[6]
  ,
  TestID -> "0427b5da-fb49-5b66-a6c6-99cb6b030230"
]

VerificationTest[
  NumberDensity[MaxwellBoltzmann, 1.0, 2.0]
  ,
  NumberDensity[MaxwellBoltzmann, 1.0, 2.0, 0.0]
  ,
  TestID -> "90b4a6ba-364a-5636-8e89-21dfe1345532"
]
VerificationTest[
  NumberDensity[MaxwellBoltzmann, 1.0, 2.0]
  ,
  0.0514225
  ,
  SameTest -> ApproxEq[6],
  TestID -> "7e784eee-fe55-5358-b5ad-2493b3882a0f"
]

VerificationTest[
  NumberDensity[MaxwellBoltzmann, 1.0, 2.0, 3.0]
  ,
  1.03285
  ,
  SameTest -> ApproxEq[6],
  TestID -> "3f2b47ac-8267-584f-bffc-ba248db8b270"
]



DefineParticle["F", Spin -> 1, DoF -> 2, Mass -> 3];
DefineParticle["B", Spin -> 0, DoF -> 3, Mass -> 4];

VerificationTest[
  NumberDensity["F", 1.0, 2.0] // N
  ,
  2 * NumberDensity[FermiDirac, 1.0, 3.0, 2.0]
  ,
  SameTest -> ApproxEq[6],
  TestID -> "619b5745-36c0-5934-8d06-2e7d29f09ef9"
]


VerificationTest[
  NumberDensity["B", 1.0, -2.0] // N
  ,
  3 * NumberDensity[BoseEinstein, 1.0, 4.0, -2.0]
  ,
  SameTest -> ApproxEq[6],
  TestID -> "41d9603a-3d74-5dff-9abb-9c2f2dadfcd3"
]

EndTestSection[]
