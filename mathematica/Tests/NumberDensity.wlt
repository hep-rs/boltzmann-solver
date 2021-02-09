Get[FileNameJoin[{DirectoryName[$TestFileName], "TestCommon.wl"}]];

BeginTestSection["NumberDensity"]

(* FermiDirac *)
VerificationTest[
  NumberDensity[FermiDirac, 1.0]
  ,
  NumberDensity[FermiDirac, 1.0, 0.0, 0.0]
  ,
  TestID -> "f7e5dab8-386d-5a89-97b9-facd87e70fe4"
]
VerificationTest[
  NumberDensity[FermiDirac, 1.0]
  ,
  0.0913454
  ,
  SameTest -> ApproxEq[6],
  TestID -> "5142f304-c30e-5020-b79e-9e2966bfada6"
]

VerificationTest[
  NumberDensity[FermiDirac, 1.0, 2.0]
  ,
  NumberDensity[FermiDirac, 1.0, 2.0, 0.0]
  ,
  SameTest -> ApproxEq[6],
  TestID -> "2c8ec1c5-3842-5f6d-a672-2c4342d4ec4b"
]
VerificationTest[
  NumberDensity[FermiDirac, 1.0, 2.0]
  ,
  0.049765
  ,
  SameTest -> ApproxEq[6],
  TestID -> "a94f6041-8a11-54c3-9227-fe4c7d86c950"
]

VerificationTest[
  NumberDensity[FermiDirac, 1.0, 2.0, 3.0]
  ,
  0.692817
  ,
  SameTest -> ApproxEq[6],
  TestID -> "0b4fb38a-8fd8-505d-9217-60f241996c5a"
]


(* BoseEinstein *)
VerificationTest[
  NumberDensity[BoseEinstein, 1.0]
  ,
  NumberDensity[BoseEinstein, 1.0, 0.0, 0.0]
  ,
  TestID -> "4993f46f-49e4-56cb-800a-5c678d675016"
]
VerificationTest[
  NumberDensity[BoseEinstein, 1.0]
  ,
  0.121794
  ,
  SameTest -> ApproxEq[6]
  ,
  TestID -> "aebe65c8-dc67-50a5-8573-db32dcce387a"
]

VerificationTest[
  NumberDensity[BoseEinstein, 1.0, 2.0]
  ,
  NumberDensity[BoseEinstein, 1.0, 2.0, 0.0]
  ,
  TestID -> "34dba41d-ece1-5580-bd2d-c0a4ef6f159e"
]
VerificationTest[
  NumberDensity[BoseEinstein, 1.0, 2.0]
  ,
  0.0533103
  ,
  SameTest -> ApproxEq[6],
  TestID -> "cec1847a-3183-5adc-bcb3-2303757222f0"
]

VerificationTest[
  NumberDensity[BoseEinstein, 1.0, 2.0, -3.0]
  ,
  0.00256456
  ,
  SameTest -> ApproxEq[6],
  TestID -> "fb0c9910-f07e-5f00-9af5-44b67aecbc7a"
]


(* MaxwellBoltzmann *)
VerificationTest[
  NumberDensity[MaxwellBoltzmann, 1.0]
  ,
  NumberDensity[MaxwellBoltzmann, 1.0, 0.0, 0.0]
  ,
  TestID -> "4993f46f-49e4-56cb-800a-5c678d675016"
]
VerificationTest[
  NumberDensity[MaxwellBoltzmann, 1.0]
  ,
  0.101321
  ,
  SameTest -> ApproxEq[6]
  ,
  TestID -> "aebe65c8-dc67-50a5-8573-db32dcce387a"
]

VerificationTest[
  NumberDensity[MaxwellBoltzmann, 1.0, 2.0]
  ,
  NumberDensity[MaxwellBoltzmann, 1.0, 2.0, 0.0]
  ,
  TestID -> "6257deae-8875-52e6-bb9b-6f6effa5af53"
]
VerificationTest[
  NumberDensity[MaxwellBoltzmann, 1.0, 2.0]
  ,
  0.0514225
  ,
  SameTest -> ApproxEq[6],
  TestID -> "99838762-c859-51ab-92ec-9eef01073501"
]

VerificationTest[
  NumberDensity[MaxwellBoltzmann, 1.0, 2.0, 3.0]
  ,
  1.03285
  ,
  SameTest -> ApproxEq[6],
  TestID -> "db877f5c-723f-5f86-97a8-8abf3215f7c5"
]



DefineParticle["F", Spin -> 1, DoF -> 2, Mass -> 3];
DefineParticle["B", Spin -> 0, DoF -> 3, Mass -> 4];

VerificationTest[
  NumberDensity["F", 1.0, 2.0] // N
  ,
  2 * NumberDensity[FermiDirac, 1.0, 3.0, 2.0]
  ,
  SameTest -> ApproxEq[6],
  TestID -> "4d8b4b09-40ec-563a-b74a-b4f00f57d1aa"
]


VerificationTest[
  NumberDensity["B", 1.0, -2.0] // N
  ,
  3 * NumberDensity[BoseEinstein, 1.0, 4.0, -2.0]
  ,
  SameTest -> ApproxEq[6],
  TestID -> "8a1637e0-194d-5c06-8dd4-d919ba9729dc"
]

EndTestSection[]
