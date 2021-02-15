Get[FileNameJoin[{DirectoryName[$TestFileName], "Common.wl"}]];

BeginTestSection["DefineParticle"]


(* Particle 1 *)
DefineParticle["1"]

VerificationTest[
  N@Mass["1"]
  ,
  N@Mass["1"]
  ,
  TestID -> "7d72f8b5-e191-5d4b-b778-45bc90804ef4"
]
VerificationTest[
  N@Mass2["1"]
  ,
  Mass["1"]^2
  ,
  TestID -> "db1260b3-b235-5b38-859f-8a1a37a597ce"
]

VerificationTest[
  N@Width["1"]
  ,
  N@Width["1"]
  ,
  TestID -> "d8b8309b-a1e9-53c1-8faf-ae211f2d83fe"
]
VerificationTest[
  N@Width2["1"]
  ,
  Width["1"]^2
  ,
  TestID -> "fb14af4b-176a-59d3-9a62-27526e482739"
]

VerificationTest[
  N@Momentum["1"]
  ,
  N@Momentum["1"]
  ,
  TestID -> "6816d928-d963-5cec-9ecf-7eff3a9b824a"
]

VerificationTest[
  N@DoF["1"]
  ,
  1.0
  ,
  TestID -> "018c3b78-7822-5943-8850-6d1fb4bdfbae"
]

VerificationTest[
  N@Spin["1"]
  ,
  0.0
  ,
  TestID -> "d22abf97-40d7-5c1b-90de-ab7ec1aeb52c"
]

VerificatioNTest[
  N@Statistic["1"]
  ,
  BoseEinstein
  ,
  TestID -> "6e69430d-a45f-56e3-949f-5e31c5ae3522"
]


(* Particle 2 *)
DefineParticle[
  "2",
  Mass -> 20,
  Width -> 2,
  Momentum -> 3,
  Spin -> 1
]

VerificationTest[
  N@Mass["2"]
  ,
  20.0
  ,
  TestID -> "bb30e92c-8f1e-53f2-a364-42224a157e15"
]
VerificationTest[
  N@Mass2["2"]
  ,
  400.0
  ,
  TestID -> "12c66e51-d7bc-5c7c-a524-82daee832e97"
]

VerificationTest[
  N@Width["2"]
  ,
  2.0
  ,
  TestID -> "4788211d-680a-5590-bf87-193b22750dee"
]
VerificationTest[
  N@Width2["2"]
  ,
  4.0
  ,
  TestID -> "1a4fbac0-4257-5064-9c97-ca02646d4f39"
]

VerificationTest[
  N@Momentum["2"]
  ,
  3.0
  ,
  TestID -> "b79e5394-7cbf-5972-ad6a-1aabf02d83c1"
]

VerificationTest[
  N@DoF["2"]
  ,
  3.0
  ,
  TestID -> "39e58c86-0da0-58e7-91aa-33f70eb64b71"
]

VerificationTest[
  N@Spin["2"]
  ,
  1.0
  ,
  TestID -> "d07234cd-f31f-504d-afb3-2453b6d5a284"
]

VerificatioNTest[
  N@Statistic["2"]
  ,
  BoseEinstein
  ,
  TestID -> "ca01956f-72fa-5424-9d9e-ee04aff9ed54"
]

EndTestSection[];