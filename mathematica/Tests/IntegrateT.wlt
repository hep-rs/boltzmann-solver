Get[FileNameJoin[{DirectoryName[$TestFileName], "TestCommon.wl"}]];

BeginTestSection["IntegrateT"];

DefineParticle["3", Mass -> 3.0];
DefineParticle["4", Mass -> 4.0];
DefineParticle["5", Mass -> 5.0];
DefineParticle["6", Mass -> 6.0];
DefineParticle["7", Mass -> 7.0];

VerificationTest[
  IntegrateT[1, 200, 3, 4, 5, 6, 7] // N
  ,
  IntegrateT[1, 200, 3, 4, 5, 6, 7.0]
  ,
  SameTest -> ApproxEq[6],
  TestID -> "accf07dd-f9f9-5b5d-96f4-c8b9d84cfe99"
]

VerificationTest[
  IntegrateT[1, 200, 3, 4, 5, 6, 7] // N
  ,
  IntegrateT[1, 200, 3, Particle["4"], 5, 6, 7] // N
  ,
  SameTest -> ApproxEq[6],
  TestID -> "07643a6f-fb0f-545c-8455-b7326747118f"
]

VerificationTest[
  IntegrateT[1, 200, 3, 4, 5, 6, 7] // N
  ,
  IntegrateT[1, 200, 3, 4, Particle["5"], 6, 7] // N
  ,
  SameTest -> ApproxEq[6],
  TestID -> "73eedd6f-8476-5728-a328-48741c2d2c59"
]

VerificationTest[
  IntegrateT[1, 200, 3, 4, 5, 6, 7] // N
  ,
  IntegrateT[1, 200, 3, 4, 5, Particle["6"], 7] // N
  ,
  SameTest -> ApproxEq[6],
  TestID -> "3d476d8a-ad0d-51bc-b11e-b66086f753ce"
]

VerificationTest[
  IntegrateT[1, 200, 3, 4, 5, 6, 7] // N
  ,
  IntegrateT[1, 200, 3, 4, 5, 6, Particle["7"]] // N
  ,
  SameTest -> ApproxEq[6],
  TestID -> "5527f5da-0824-59be-a57e-02cc393c078b"
]

VerificationTest[
  IntegrateT[1, 200, 3, 4, 5, 6, 7.0]
  ,
  IntegrateT[1 &, 200, 3, 4, 5, 6, 7.0]
  ,
  SameTest -> ApproxEq[6],
  TestID -> "3cad6925-cc3a-56b5-aba7-78e88560435a"
]

VerificationTest[
  IntegrateT[1, 200, 3, 4, 5, 6, 7.0]
  ,
  60.4335
  ,
  SameTest -> ApproxEq[6],
  TestID -> "688a5eb0-fc2a-5526-a610-ff1a6c99e9cb"
]

VerificationTest[
  IntegrateT[1 / (# + 1) &, 200, 3, 4, 5, 6, 7.0]
  ,
  0.300664
  ,
  SameTest -> ApproxEq[6],
  TestID -> "9a6409ac-f22d-5d90-9d8b-51f6b8acf54c"
]

VerificationTest[
  IntegrateT[# &, 200, 3, 4, 5, 6, 7.0]
  ,
  12086.7
  ,
  SameTest -> ApproxEq[6],
  TestID -> "7230054b-2a53-5788-8ac2-e63a4deceb16"
]


EndTestSection[];