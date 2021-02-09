Get[FileNameJoin[{DirectoryName[$TestFileName], "TestCommon.wl"}]];

BeginTestSection["Planck"];

VerificationTest[
  N@Mass[Planck]
  ,
  1.22089*^19
  ,
  SameTest -> ApproxEq[6],
  TestID -> "563a794a-0c5a-5356-8b11-585c84905fe5"
]

VerificationTest[
  N@Mass[ReducedPlanck]
  ,
  N@Mass[Planck] / Sqrt[8 Pi]
  ,
  SameTest -> ApproxEq[6],
  TestID -> "3db0f464-a8d3-50df-b5f6-3c43f41ffa47"
]

VerificationTest[
  N@Mass[ReducedPlanck]
  ,
  2.43532*10^18
  ,
  SameTest -> ApproxEq[6],
  TestID -> "3db0f464-a8d3-50df-b5f6-3c43f41ffa47"
]

EndTestSection[];