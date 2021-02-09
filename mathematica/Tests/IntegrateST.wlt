Get[FileNameJoin[{DirectoryName[$TestFileName], "TestCommon.wl"}]];

BeginTestSection["IntegrateST"];

DefineParticle["0", Mass -> 0];
DefineParticle["1", Mass -> 1];
DefineParticle["2", Mass -> 2];
DefineParticle["3", Mass -> 3];
DefineParticle["4", Mass -> 4];
DefineParticle["5", Mass -> 5];
DefineParticle["6", Mass -> 6];
DefineParticle["7", Mass -> 7];

VerificationTest[
  IntegrateST[1, 3, 4, 5, 6, 7] // N
  ,
  IntegrateST[1, 3, 4, 5, 6, 7.0]
  ,
  SameTest -> ApproxEq[6],
  TestID -> "fe2785d1-faad-5a42-9886-0cbfca17a692"
]

VerificationTest[
  IntegrateST[1, 3, 4, 5, 6, 7] // N
  ,
  IntegrateST[1, 3, Particle["4"], 5, 6, 7] // N
  ,
  SameTest -> ApproxEq[6],
  TestID -> "312f6f45-4901-57e8-b7dc-14fad96d2be9"
]

VerificationTest[
  IntegrateST[1, 3, 4, 5, 6, 7] // N
  ,
  IntegrateST[1, 3, 4, Particle["5"], 6, 7] // N
  ,
  SameTest -> ApproxEq[6],
  TestID -> "e83a9d9e-6edf-5f51-a767-4802de2b7520"
]

VerificationTest[
  IntegrateST[1, 3, 4, 5, 6, 7] // N
  ,
  IntegrateST[1, 3, 4, 5, Particle["6"], 7] // N
  ,
  SameTest -> ApproxEq[6],
  TestID -> "8d09a1a0-98f3-5ada-a04a-196c3c9ba381"
]

VerificationTest[
  IntegrateST[1, 3, 4, 5, 6, 7] // N
  ,
  IntegrateST[1, 3, 4, 5, 6, Particle["7"]] // N
  ,
  SameTest -> ApproxEq[6],
  TestID -> "a3f6cf56-3565-51e3-8853-84fd9f9a503d"
]

VerificationTest[
  IntegrateST[1, 3, 4, 5, 6, 7.0]
  ,
  IntegrateST[1 &, 3, 4, 5, 6, 7.0]
  ,
  SameTest -> ApproxEq[6],
  TestID -> "bbd48ea3-6003-5463-ae36-22353835bb4d"
]

VerificationTest[
  IntegrateST[1, 3, 4, 5, 6, 7.0]
  ,
  8.57338*^-23
  ,
  SameTest -> ApproxEq[6],
  TestID -> "8d8328ee-7cca-524e-a368-9af02c6ac925"
]

VerificationTest[
  IntegrateST[1 / ((#1 + 1) (#2 + 2)) &, 3, 4, 5, 6, 7.0]
  ,
  -2.18325*^-26
  ,
  SameTest -> ApproxEq[6],
  TestID -> "301b3e9e-dac0-5528-9114-18b69849404f"
]

VerificationTest[
  IntegrateST[#1 / (#2 + 1) &, 3, 4, 5, 6, 7.0]
  ,
  -2.18325*^-26
  ,
  SameTest -> ApproxEq[6],
  TestID -> "9a0cf8d9-edf4-59d7-b0c1-d5dbe0a022f7"
]

VerificationTest[
  IntegrateST[#2 / (#1 + 1) &, 3, 4, 5, 6, 7.0]
  ,
  -1.33102*^-23
  ,
  SameTest -> ApproxEq[6],
  TestID -> "143bee98-5f15-595b-8f93-af01ef42851a"
]

VerificationTest[
  IntegrateST[#1 #2 &, 3, 4, 5, 6, 7.0]
  ,
  -4.57378*^-19
  ,
  SameTest -> ApproxEq[6],
  TestID -> "780b6b26-fcc2-585d-a2e1-7db6a540d730"
]

EndTestSection[];