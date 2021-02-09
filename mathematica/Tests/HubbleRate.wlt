Get[FileNameJoin[{DirectoryName[$TestFileName], "TestCommon.wl"}]];

BeginTestSection["HubbleRate"];

VerificationTest[
  HubbleRate[1]
  ,
  HubbleRate[1]
  ,
  TestID -> "7e12ef63-6b20-59af-aa0d-e30770147663"
]

VerificationTest[
  HubbleRate[beta]
  ,
  HubbleRate[beta]
  ,
  TestID -> "03d675e7-38f5-5c38-8575-64dbdcff2bbb"
]

VerificationTest[
  N@HubbleRate[1] / Sqrt[GStar[1.0]] // N
  ,
  1.35979*^-19
  ,
  SameTest -> ApproxEq[6],
  TestID -> "c8891750-f692-557c-afa6-1e175671e770"
]

VerificationTest[
  HubbleRate[1.0] / Sqrt[GStar[1.0]] // N
  ,
  1.36*^-19
  ,
  SameTest -> ApproxEq[6],
  TestID -> "76215bab-30b0-5f93-90c1-d82e89630fb2"
]

VerificationTest[
  HubbleRate[1.0] /. GStar -> GStarSM // N
  ,
  1.23879*^-18
  ,
  SameTest -> ApproxEq[6],
  TestID -> "98b29175-3d69-5052-82dc-9e6e551de97d"
]

EndTestSection[];
