(* BeginPackage["Extra`"]; *)

AnyInexactNumberQ::usage = "AnyInexactNumberQ[a, b, ...] returns True if any of the arguments are inexact and False otherwise.";

ExpandValues::usage = "ExpandValues[symbol] expands the definition associated with the symbol as replacement rules.  Multiple symbols can be given and the replacement rules will be joined.";

Begin["`Private`"] (* Begin Private Context *)

AnyInexactNumberQ[args__] := Or @@ InexactNumberQ /@ {args};

Protect[AnyInexactNumberQ];


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

End[] (* End Private Context *)

(* EndPackage[]; *)