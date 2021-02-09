Get[FileNameJoin[{DirectoryName[$TestFileName], "TestCommon.wl"}]];

BeginTestSection["PhaseSpace"]

(* FermiDirac *)
VerificationTest[
  PhaseSpace[FermiDirac, 1.0, 2.0]
  ,
  PhaseSpace[FermiDirac, 1.0, 2.0, 0.0, 0.0]
]
VerificationTest[
  PhaseSpace[FermiDirac, 1.0, 2.0]
  ,
  0.119203
  ,
  SameTest -> ApproxEq[6]
]

VerificationTest[
  PhaseSpace[FermiDirac, 1.0, 2.0, 3.0]
  ,
  PhaseSpace[FermiDirac, 1.0, 2.0, 3.0, 0.0]
]
VerificationTest[
  PhaseSpace[FermiDirac, 1.0, 2.0, 3.0]
  ,
  0.119203
  ,
  SameTest -> ApproxEq[6]
]

VerificationTest[
  PhaseSpace[FermiDirac, 1.0, 2.0, 3.0, 4.0]
  ,
  0.880797
  ,
  SameTest -> ApproxEq[6]
]


(* BoseEinstein *)
VerificationTest[
  PhaseSpace[BoseEinstein, 1.0, 2.0]
  ,
  PhaseSpace[BoseEinstein, 1.0, 2.0, 0.0, 0.0]
]
VerificationTest[
  PhaseSpace[BoseEinstein, 1.0, 2.0]
  ,
  0.156518
  ,
  SameTest -> ApproxEq[6]
]

VerificationTest[
  PhaseSpace[BoseEinstein, 1.0, 2.0, 3.0]
  ,
  PhaseSpace[BoseEinstein, 1.0, 2.0, 3.0, 0.0]
]
VerificationTest[
  PhaseSpace[BoseEinstein, 1.0, 2.0, 3.0]
  ,
  0.156518
  ,
  SameTest -> ApproxEq[6]
]

VerificationTest[
  PhaseSpace[BoseEinstein, 1.0, 2.0, 3.0, 4.0]
  ,
  -1.156517
  ,
  SameTest -> ApproxEq[6]
]


(* MaxwellBoltzmann *)
VerificationTest[
  PhaseSpace[MaxwellBoltzmann, 1.0, 2.0]
  ,
  PhaseSpace[MaxwellBoltzmann, 1.0, 2.0, 0.0, 0.0]
]
VerificationTest[
  PhaseSpace[MaxwellBoltzmann, 1.0, 2.0]
  ,
  0.135335
  ,
  SameTest -> ApproxEq[6]
]

VerificationTest[
  PhaseSpace[MaxwellBoltzmann, 1.0, 2.0, 3.0]
  ,
  PhaseSpace[MaxwellBoltzmann, 1.0, 2.0, 3.0, 0.0]
]
VerificationTest[
  PhaseSpace[MaxwellBoltzmann, 1.0, 2.0, 3.0]
  ,
  0.135335
  ,
  SameTest -> ApproxEq[6]
]

VerificationTest[
  PhaseSpace[MaxwellBoltzmann, 1.0, 2.0, 3.0, 4.0]
  ,
  7.38906
  ,
  SameTest -> ApproxEq[6]
]

EndTestSection[]
