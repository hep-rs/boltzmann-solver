AnyInexactNumberQ::usage = "AnyInexactNumberQ[a, b, ...] returns True if any of the arguments are inexact and False otherwise.";

ExpandValues::usage = "ExpandValues[symbol] expands the definition associated with the symbol as replacement rules.  Multiple symbols can be given and the replacement rules will be joined.";

BesselK12::usage = "Ratio of Bessel function K1 / K2.";

Begin["`Private`"] (* Begin Private Context *)

(* AnyInexactNumberQ *)
(* ***************** *)
AnyInexactNumberQ[args__] := Or @@ InexactNumberQ /@ {args};

Protect[AnyInexactNumberQ];


(* ExpandValues *)
(* ************ *)

Attributes[ExpandValues] = {HoldAll};

ExpandValues[symbol_] := Join @@ Through[
	{OwnValues, DownValues, UpValues, SubValues, DefaultValues, NValues}[symbol]
] //. {
	Verbatim[Condition][p_, q_] :> p,
	Verbatim[PatternTest][p_, q_] :> p,
	HoldPattern[N[f_, ___]] :> f
};

ExpandValues[symbol_, symbols__] := Join[ExpandValues[symbol], ExpandValues[symbols]];

Protect[ExpandValues];


(* Bessel K Ratio *)
(* ************** *)

Attributes[BesselK12] = {Listable, NumericFunction};
BesselK12 /: MakeBoxes[BesselK12[x_], TraditionalForm] := FractionBox[MakeBoxes[BesselK[1, x], TraditionalForm], MakeBoxes[BesselK[2, x], TraditionalForm]];
BesselK12[x_?InexactNumberQ] /; PossibleZeroQ[BesselK[1, x]] := With[
  {
    targetPrecision = Precision[x]
  },
  BesselK12[SetPrecision[x, 2 targetPrecision]]
];
BesselK12[x_?InexactNumberQ] := BesselK[1, x] / BesselK[2, x];

Protect[BesselK12];

End[] (* End Private Context *)