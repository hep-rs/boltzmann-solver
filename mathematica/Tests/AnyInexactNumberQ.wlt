Get[FileNameJoin[{DirectoryName[$TestFileName], "TestCommon.wl"}]];

BeginTestSection["AnyInexactNumberQ"]

VerificationTest[
	AnyInexactNumberQ[1]
	,
	False
	,
	TestID -> "b603c554-f3bb-554a-b06b-aeb46d713e72"
]

VerificationTest[
	AnyInexactNumberQ[1, 3 / 4, Pi, E, Sin[2]]
	,
	False
	,
	TestID -> "710dcc47-8401-5020-9fed-14057f76594e"
]

VerificationTest[
	AnyInexactNumberQ[1.`]
	,
	True
	,
	TestID -> "9aacc12d-a3df-518f-98b9-124f42a68eb6"
]

VerificationTest[
	AnyInexactNumberQ[1.`, 3 / 4, Pi, E, Sin[2]]
	,
	True
	,
	TestID -> "0f5dfb08-12de-55dd-9144-68ca25d13722"
]

VerificationTest[
	AnyInexactNumberQ[1, 3 / 4, Pi, E, Sin[2.`]]
	,
	True
	,
	TestID -> "2effdacd-f57f-5263-a51c-0d5473d14f2c"
]

EndTestSection[]