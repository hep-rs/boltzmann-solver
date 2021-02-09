Get[FileNameJoin[{DirectoryName[$TestFileName], "TestCommon.wl"}]];

BeginTestSection["InteractionGamma"]

DefineParticle["1", Mass -> 10];
DefineParticle["2", Mass -> 2];
DefineParticle["3", Mass -> 3];

VerificationTest[
  InteractionGamma[{"1", "2"} -> {"3"}]
  ,
  InteractionGamma[{Particle["1"], Particle["2"]} -> {Particle["3"]}]
  ,
  TestID -> "14dcfcc8-40fb-5539-9518-92a8d11dc3d3"
]

VerificationTest[
  InteractionGamma[{"2", "1"} -> {"3"}]
  ,
  InteractionGamma[{Particle["1"], Particle["2"]} -> {Particle["3"]}]
  ,
  TestID -> "22971c45-e0b4-560c-8e26-40bf74a29b02"
]

VerificatioNTest[
  InteractionGamma[{"1"} -> {"2", "3"}][1.0]
  / SquaredAmplitude[{"1"} -> {"2", "3"}] // N
  ,
  1.61957*^-7
  ,
  SameTest -> ApproxEq[6],
  TestID -> "34344e60-61f4-5465-8289-98737d3ed772"
]

SquaredAmplitude[{"1", "4"} -> {"2", "3"}] = 1;

VerificationTest[
  InteractionGamma[{"1", "4"} -> {"2", "3"}][1.0] // N
  ,
  2.24457*^-10
  ,
  SameTest -> ApproxEq[6],
  TestID -> "ea08bfff-29d7-5779-b327-01f6db0022ee"
]

EndTestSection[];