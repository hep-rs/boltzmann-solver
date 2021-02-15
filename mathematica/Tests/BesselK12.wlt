Get[FileNameJoin[{DirectoryName[$TestFileName], "Common.wl"}]];

BeginTestSection["BesselK12"];

VerificationTest[
  BesselK12[10^-5]
  ,
  BesselK[1, 10^-5] / BesselK[2, 10^-5]
  ,
  SameTest -> ApproxEq[6],
  TestID -> "2e2b19b2-c57f-59cf-9bae-4f7199d83451"
]

VerificationTest[
  BesselK12[10^-4]
  ,
  BesselK[1, 10^-4] / BesselK[2, 10^-4]
  ,
  SameTest -> ApproxEq[6],
  TestID -> "900e4a01-5190-523e-b757-9d1e325c5ff2"
]

VerificationTest[
  BesselK12[10^-3]
  ,
  BesselK[1, 10^-3] / BesselK[2, 10^-3]
  ,
  SameTest -> ApproxEq[6],
  TestID -> "d5f5bf9b-71ce-5c66-aaae-589d4934ab05"
]

VerificationTest[
  BesselK12[10^-2]
  ,
  BesselK[1, 10^-2] / BesselK[2, 10^-2]
  ,
  SameTest -> ApproxEq[6],
  TestID -> "8a1921e6-83ce-5c05-a8c0-03f0725057b9"
]

VerificationTest[
  BesselK12[10^-1]
  ,
  BesselK[1, 10^-1] / BesselK[2, 10^-1]
  ,
  SameTest -> ApproxEq[6],
  TestID -> "168e3e77-367f-5e63-ae63-3709d2daaa6c"
]

VerificationTest[
  BesselK12[10^0]
  ,
  BesselK[1, 10^0] / BesselK[2, 10^0]
  ,
  SameTest -> ApproxEq[6],
  TestID -> "f2fdd063-e33f-5922-8f88-04b7fe8c4dbb"
]

VerificationTest[
  BesselK12[10^1]
  ,
  BesselK[1, 10^1] / BesselK[2, 10^1]
  ,
  SameTest -> ApproxEq[6],
  TestID -> "0073b52d-e3ad-5609-b741-d76205d7c3b2"
]

VerificationTest[
  BesselK12[10^2]
  ,
  BesselK[1, 10^2] / BesselK[2, 10^2]
  ,
  SameTest -> ApproxEq[6],
  TestID -> "e9c6bd60-bcdc-59c7-b890-d2b6514fd400"
]

VerificationTest[
  BesselK12[10^3]
  ,
  BesselK[1, 10^3] / BesselK[2, 10^3]
  ,
  SameTest -> ApproxEq[6],
  TestID -> "9abb7ce4-53d5-5945-9b1e-b1d5e2a51da6"
]

VerificationTest[
  BesselK12[10^4]
  ,
  BesselK[1, 10^4] / BesselK[2, 10^4]
  ,
  SameTest -> ApproxEq[6],
  TestID -> "d92d04dd-c24a-52ee-b382-8e40c704883a"
]

VerificationTest[
  BesselK12[10^5]
  ,
  BesselK[1, 10^5] / BesselK[2, 10^5]
  ,
  SameTest -> ApproxEq[6],
  TestID -> "f2161530-4163-5a84-9bfc-42ba9175933a"
]


EndTestSection[];