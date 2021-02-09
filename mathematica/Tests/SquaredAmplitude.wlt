Get[FileNameJoin[{DirectoryName[$TestFileName], "TestCommon.wl"}]];

BeginTestSection["SquaredAmplitude"]

DefineParticle["1"];
DefineParticle["2"];
DefineParticle["3"];

VerificationTest[
  SquaredAmplitude[{"1", "2"} -> {"3"}]
  ,
  SquaredAmplitude[{Particle["1"], Particle["2"]} -> {Particle["3"]}]
  ,
  TestID -> "c3892a30-5c60-5f10-81e6-97a3535386c9"
]

VerificationTest[
  SquaredAmplitude[{"2", "1"} -> {"3"}]
  ,
  SquaredAmplitude[{Particle["1"], Particle["2"]} -> {Particle["3"]}]
  ,
  TestID -> "e87fdf27-14d8-536d-8a96-5e1547e7b6b6"
]

SquaredAmplitude[{"1", "2"} -> {"3", "2"}] = 1;

VerificationTest[
  SquaredAmplitude[{"1", "2"} -> {"3", "2"}]
  ,
  SquaredAmplitude[{"1", "2"} -> {"2", "3"}]
]

VerificationTest[
  SquaredAmplitude[{"1", "2"} -> {"3", "2"}]
  ,
  1
]

VerificationTest[
  SquaredAmplitude[{"1", "2"} -> {"2", "3"}]
  ,
  1
]

EndTestSection[];