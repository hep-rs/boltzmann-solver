Planck::usage = "In Mass[Planck] refers to the Planck mass.";
ReducedPlanck::usage = "In Mass[ReducedPlanck] refers to the reduced Planck mass.";

HubbleRate::usage = "Hubble rate.";

Begin["`Private`"];

Mass /: MakeBoxes[Mass[Planck], TraditionalForm] = SubscriptBox["M", "Pl"];
Mass /: MakeBoxes[Mass[ReducedPlanck], TraditionalForm] = SubscriptBox["m", "Pl"];

N[Mass[Planck], precision_:MachinePrecision] := N[
  SetPrecision[
    UnitConvert[Quantity["PlanckMass"], "Gigaelectronvolts" / "SpeedOfLight"^2] // QuantityMagnitude,
    Infinity
  ]
  ,
  precision
]
N[Mass[ReducedPlanck], precision_:MachinePrecision] := N[Mass[Planck]/Sqrt[8 Pi], precision];

Mass /: NumberQ[Mass[m:(Planck|ReducedPlanck)]] = True;
Mass /: NumericQ[Mass[m:(Planck|ReducedPlanck)]] = True;


HubbleRate /: MakeBoxes[HubbleRate, TraditionalForm] = "H";

Attributes[HubbleRate] = {Listable, NumericFunction};
HubbleRate[beta_?InexactNumberQ] := Sqrt[Pi^2/90] Sqrt[GStar[beta]] / (Mass[ReducedPlanck] beta^2);

End[]; (* End Private *)