Get[FileNameJoin[{DirectoryName[$TestFileName], "TestCommon.wl"}]];

BeginTestSection["GStar"]

VerificationTest[
  N@GStarSM[10]
  ,
  GStarSM[10.0]
  ,
  TestID -> "9df894ae-9acd-5c5d-a8f2-2fa5786ce9bf"
]
VerificationTest[
  GStarSM[10.0]
  ,
  86.13
  ,
  SameTest -> ApproxEq[6],
  TestID -> "5a9042ff-de2f-5395-a1cf-a77863c74e0b"
]

VerificationTest[
  N@GStarBoson[2, 10]
  ,
  GStarBoson[2, 10.0]
  ,
  TestID -> "79ce3962-8e1e-58b2-ae0b-df3cb7bd30c6"
]
VerificationTest[
  GStarBoson[2, 10.0]
  ,
  6.61*^-7
  ,
  SameTest -> ApproxEq[6],
  TestID -> "bed36162-3f87-591a-808d-5fe7f6167893"
]

VerificationTest[
  N@GStarFermion[3, 10]
  ,
  GStarFermion[3, 10.0]
  ,
  TestID -> "956537c7-0652-532c-a613-af334a97094f"
]
VerificationTest[
  GStarFermion[3, 10.0]
  ,
  7.7*^-11
  ,
  SameTest -> ApproxEq[6],
  TestID -> "92f4c83f-86b9-58a2-9c9d-5212c00bc03c"
]

DefineParticle["F", Spin -> 1, DoF -> 2, Mass -> 3];
DefineParticle["B", Spin -> 0, DoF -> 3, Mass -> 4];

VerificatioNTest[
  N@GStar["F"][1.0]
  ,
  N@GStar[Particle["F"]][1.0]
  ,
  TestID -> "b528cc05-0921-5ec2-aafe-664dadde5945"
]

VerificationTest[
  N@GStar[Particle["F"]][1.0]
  ,
  2 GStarFermion[3, 1.0]
  ,
  (* SameTest -> ApproxEq[6], *)
  TestID -> "efda1ed9-093a-5112-8e91-9a587665c774"
]

VerificationTest[
  N@GStar[Particle["B"]][1.0]
  ,
  3 GStarBoson[4, 1.0]
  ,
  (* SameTest -> ApproxEq[6], *)
  TestID -> "55e4c42f-a757-5024-8e63-f2d38a191d16"
]

EndTestSection[];