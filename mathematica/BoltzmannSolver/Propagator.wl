Propagator::usage = "Propagator with RIS removed when appropriate.";

Begin["`Private`"];

Propagator /: MakeBoxes[Propagator[p_, q_], TraditionalForm] := RowBox[{
    SubscriptBox[
        "\[ScriptCapitalP]", 
        MakeBoxes[p, TraditionalForm]
    ], 
    "(", MakeBoxes[q, TraditionalForm], ")"
}];

Propagator /: Times[
    Propagator[p_?ParticleQ, q_],
    Conjugate[Propagator[p_?ParticleQ, q_]]
] := Block[{
        mw2 = (Mass[p] Width[p])^2
    },
    ((q - Mass2[p])^2 - mw2) 
    / ((q - Mass2[p])^2 + mw2)
];

Propagator /: Times[
    Propagator[p1_?ParticleQ, q1_],
    Conjugate[Propagator[p2_?ParticleQ, q2_]]
] := (1 
    / (q1 - Mass2[p1] + I HeavisideTheta[q1] Mass[p1] Width[p1])
    / (q2 - Mass2[p2] - I HeavisideTheta[q2] Mass[p2] Width[p2])
);


End[];