(* Paclet Info File *)

Paclet[
  Name -> "BoltzmannSolver",
  Version -> "0.1.0",
  MathematicaVersion -> "12.1+",
  Description -> "Solves Boltzmann equations for particle physics",
  Creator -> "J. P. Ellis <josh@jpellis.me>",
  Extensions -> {
    {
      "Kernel",
      Root -> ".",
      Context -> "BoltzmannSolver`"
    },
    {
      "Resource",
      Root -> "Tests",
      "Resources" -> {
        "AnyInexactNumberQ.wlt",
        "BesselK12.wlt",
        "DefineParticle.wlt",
        "ExpandValues.wlt",
        "GStar.wlt",
        "HubbleRate.wlt",
        "IntegrateST.wlt",
        "IntegrateT.wlt",
        "InteractionGamma.wlt",
        "MandelstamTRange.wlt",
        "NumberDensity.wlt",
        "PhaseSpace.wlt",
        "Planck.wlt",
        "SquaredAmplitude.wlt",
        "TestCommon.wl"
      }
    }
    (* {
      "Documentation",
      Language -> "English"
    } *)
  }
]