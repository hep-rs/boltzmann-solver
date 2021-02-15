Get[FileNameJoin[{DirectoryName[$TestFileName], "Common.wl"}]];

BeginTestSection["MandelstamTRange"];

DefineParticle["0", Mass -> 0];
DefineParticle["1", Mass -> 1];
DefineParticle["2", Mass -> 2];
DefineParticle["3", Mass -> 3];
DefineParticle["4", Mass -> 4];

VerificationTest[
  MandelstamTRange[1, 0, 0, 0, 0]
  ,
  {-1, 0}
  ,
  SameTest -> ApproxEqList[6],
  TestID -> "7fd0067e-be35-5114-b3eb-ff3f94462c89"
]

VerificationTest[
  MandelstamTRange[4, 0, 0, 0, 0]
  ,
  {-4, 0}
  ,
  SameTest -> ApproxEqList[6],
  TestID -> "329d77c1-eb85-527c-b78a-c958bb0cd835"
]

VerificationTest[
  MandelstamTRange[100, 1, 2, 3, 4]
  ,
  MandelstamTRange[100, 2, 1, 4, 3]
  ,
  SameTest -> ApproxEqList[6],
  TestID -> "00b4a6cd-7d35-5597-9bcb-83a6bebc997d"
]

VerificationTest[
  MandelstamTRange[100, 1, 2, 3, 4]
  ,
  MandelstamTRange[100, 2, 1, 4, 3]
  ,
  SameTest -> ApproxEqList[6],
  TestID -> "06bcf778-a9f5-5262-ba2a-2793a926dfc5"
]

VerificationTest[
  MandelstamTRange[100, "1", 2, 3, 4] // N
  ,
  MandelstamTRange[100, 1, 2, 3, 4]
  ,
  SameTest -> ApproxEqList[6],
  TestID -> "59c143d4-7b1e-58a0-bece-3798f61b9305"
]

VerificationTest[
  MandelstamTRange[100, 1, "2", 3, 4] // N
  ,
  MandelstamTRange[100, 1, 2, 3, 4]
  ,
  SameTest -> ApproxEqList[6],
  TestID -> "d1eb3a09-1ca9-5e2e-8974-e36bdadc8a6c"
]

VerificationTest[
  MandelstamTRange[100, 1, 2, "3", 4] // N
  ,
  MandelstamTRange[100, 1, 2, 3, 4]
  ,
  SameTest -> ApproxEqList[6],
  TestID -> "796e874c-6f45-5706-a503-8e9c29a74651"
]

VerificationTest[
  MandelstamTRange[100, 1, 2, 3, "4"] // N
  ,
  MandelstamTRange[100, 1, 2, 3, 4]
  ,
  SameTest -> ApproxEqList[6],
  TestID -> "f3d1b6c0-a9b0-58f0-b1f9-ad9fa13ad8b2"
]

VerificationTest[
  MandelstamTRange[100, 1, 2, 3, 4]
  ,
  {-68.8268, -1.38318}
  ,
  SameTest -> ApproxEqList[6],
  TestID -> "06bcf778-a9f5-5262-ba2a-2793a926dfc5"
]

EndTestSection[];