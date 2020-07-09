var searchIndex = JSON.parse('{\
"boltzmann_solver":{"doc":"`boltzmann-solver` is a library allowing for Boltzmann…","i":[[0,"pave","boltzmann_solver","Passarino-Veltman coefficient functions.",null,null],[5,"a","boltzmann_solver::pave","Absorptive part of the Passarin-Veltman coefficient…",null,[[]]],[5,"b","","Absorptive part of the Passarin-Veltman coefficient…",null,[[]]],[5,"c","","Absorptive part of the Passarin-Veltman coefficient…",null,[[]]],[5,"d","","Absorptive part of the Passarin-Veltman coefficient…",null,[[]]],[0,"constants","boltzmann_solver","Collection of physical and mathematical constants which…",null,null],[17,"PLANCK_MASS","boltzmann_solver::constants","Planck mass, `$M_{\\\\text{Pl}} = \\\\sqrt{\\\\hbar c / G}$`, in…",null,null],[17,"REDUCED_PLANCK_MASS","","Reduced Planck mass, `$m_{\\\\text{Pl}} = \\\\sqrt{\\\\hbar c / 8…",null,null],[17,"ZETA_3","","Riemann zeta function evaluated at 3: `$\\\\zeta(3) \\\\approx…",null,null],[17,"EULER_GAMMA","","Euler gamma constant: `$\\\\gamma_\\\\textsc{E} \\\\approx…",null,null],[17,"PI","","`$\\\\pi$`",null,null],[17,"PI_1","","`$\\\\pi$` (named to follow the convention `PI_n`)",null,null],[17,"PI_2","","`$\\\\pi^2$`",null,null],[17,"PI_3","","`$\\\\pi^3$`",null,null],[17,"PI_4","","`$\\\\pi^4$`",null,null],[17,"PI_5","","`$\\\\pi^5$`",null,null],[17,"PI_N1","","`$\\\\pi^{-1}$`",null,null],[17,"PI_N2","","`$\\\\pi^{-2}$`",null,null],[17,"PI_N3","","`$\\\\pi^{-3}$`",null,null],[17,"PI_N4","","`$\\\\pi^{-4}$`",null,null],[17,"PI_N5","","`$\\\\pi^{-5}$`",null,null],[0,"model","boltzmann_solver","Model information",null,null],[3,"Particle","boltzmann_solver::model","Particle type",null,null],[12,"own_antiparticle","","Whether the particle is its own antiparticle or not.",0,null],[12,"mass","","Mass of the particle in GeV.",0,null],[12,"mass2","","Squared mass of the particle in GeV².",0,null],[12,"width","","Width of the particle in GeV.",0,null],[12,"width2","","Squared width of the particle in GeV².",0,null],[12,"decays","","Decays",0,null],[12,"name","","Name",0,null],[3,"StandardModel","","The Standard Model of particle physics.",null,null],[12,"beta","","Inverse temperature in GeV`$^{-1}$`",1,null],[12,"particles","","Particles",1,null],[12,"g1","","Hypercharge gauge coupling",1,null],[12,"g2","","Weak gauge coupling",1,null],[12,"g3","","Strong gauge coupling",1,null],[12,"yu","","Up-quark Yukawa",1,null],[12,"yd","","Down-quark Yukawa",1,null],[12,"ye","","Electron Yukawa",1,null],[12,"ckm","","CKM matrix",1,null],[12,"mh","","0-temperature mass of the Higgs",1,null],[12,"vev","","Vacuum expectation value of the Higgs",1,null],[12,"mu2","","Quadratic coupling of the Higgs",1,null],[12,"lambda","","Quartic term in scalar potential",1,null],[0,"interaction","","Common trait and implementations for interactions.",null,null],[3,"FourParticle","boltzmann_solver::model::interaction","Three particle interaction, all determined from the…",null,null],[3,"PartialWidth","","Partial width associated with a single specific interaction.",null,null],[12,"width","","Partial width of this process in GeV",2,null],[12,"parent","","Parent particle decaying. This is signed to distinguish…",2,null],[12,"daughters","","Daughter particles of the decay, signed just as the parent…",2,null],[3,"RateDensity","","Rate density associate with an interaction.",null,null],[12,"symmetric","","Net symmetric rate",3,null],[12,"asymmetric","","Net asymmetric rate",3,null],[3,"ThreeParticle","","Three particle interaction, all determined from the…",null,null],[3,"InteractionParticles","","List of particles involved in the interaction.",null,null],[12,"incoming","","Initial state particles",4,null],[12,"outgoing","","Final state particles",4,null],[3,"InteractionParticleIndices","","List of particle indices involved in the interaction.",null,null],[12,"incoming","","Initial state particles",5,null],[12,"outgoing","","Final state particles",5,null],[3,"InteractionParticleSigns","","The signed particle to its signum (as a floating point).",null,null],[12,"incoming","","Initial state particles",6,null],[12,"outgoing","","Final state particles",6,null],[5,"overshoots","","Check whether particle `i` from the model with the given…",null,[[["context",3]]]],[5,"asymmetry_overshoots","","Check whether particle asymmetry `i` from the model with…",null,[[["context",3]]]],[11,"new","","Create a new four-particle interaction.",7,[[]]],[11,"new_all","","Create a set of related four-particle interactions.",7,[[],["vec",3]]],[11,"set_asymmetry","","Specify the asymmetry between this process and its…",7,[[]]],[11,"enable_width","","Adjust whether this interaction will calculate decay widths.",7,[[]]],[11,"enable_gamma","","Adjust whether this interaction will calculate decay widths.",7,[[]]],[11,"parent_idx","","Return the parent particle index",2,[[]]],[11,"daughters_idx","","Return the daughter particle indices",2,[[],["vec",3]]],[11,"zero","","Create a new instanse with both rates being 0.",3,[[]]],[11,"new","","Create a new three-particle interaction.",8,[[]]],[11,"new_all","","Create the set of related three particle interactions.",8,[[],["vec",3]]],[11,"set_asymmetry","","Specify the asymmetry between this process and its…",8,[[]]],[11,"enable_width","","Adjust whether this interaction will calculate decay widths.",8,[[]]],[11,"enable_gamma","","Adjust whether this interaction will calculate decay widths.",8,[[]]],[8,"Interaction","","Generic interaction between particles.",null,null],[10,"particles","","Return the particles involved in this interaction",9,[[],["interactionparticles",3]]],[10,"particles_idx","","Return the particles involved in this interaction, not…",9,[[],["interactionparticleindices",3]]],[10,"particles_sign","","Return the sign of the particles, with +1 for a particle…",9,[[],["interactionparticlesigns",3]]],[10,"width_enabled","","Whether this interaction is to be used to determine decays.",9,[[]]],[11,"width","","Calculate the decay width associated with a particular…",9,[[["context",3]],[["option",4],["partialwidth",3]]]],[10,"gamma_enabled","","Whether this interaction is to be used within the…",9,[[]]],[10,"gamma","","Calculate the reaction rate density of this interaction.",9,[[["context",3]],["option",4]]],[11,"asymmetry","","Asymmetry between the interaction and its `$\\\\CP$` conjugate:",9,[[["context",3]],["option",4]]],[11,"rate","","Calculate the actual interaction rates density taking into…",9,[[["context",3]],[["option",4],["ratedensity",3]]]],[11,"adjusted_rate","","Adjust the backward and forward rates such that they do…",9,[[["context",3]],[["option",4],["ratedensity",3]]]],[11,"change","","Add this interaction to the `dn` and `dna` array.",9,[[["array1",6],["context",3]]]],[11,"as_idx","","Convert the signed particle numbers to indices which can…",4,[[],["interactionparticleindices",3]]],[11,"as_sign","","Convert the signed particle to its signum (as a floating…",4,[[],["interactionparticlesigns",3]]],[11,"display","","Output a \'pretty\' version of the interaction particles…",4,[[],[["result",4],["string",3]]]],[11,"new","boltzmann_solver::model","Create a new particle with the specified spin and mass.",0,[[]]],[11,"set_mass","","Set the mass of the particle.",0,[[]]],[11,"set_width","","Set the width of the particle.",0,[[]]],[11,"real","","Indicate that the particle is real.",0,[[]]],[11,"complex","","Indicate that the particle is complex.",0,[[]]],[11,"own_antiparticle","","Indicate that the particle is its own antiparticle,…",0,[[]]],[11,"dof","","Specify how many internal degrees of freedom this particle…",0,[[]]],[11,"name","","Specify the particle\'s name",0,[[["into",8],["string",3]]]],[11,"is_real","","Returns true if the particle is real (real scalar,…",0,[[]]],[11,"is_complex","","Returns true if the particle is complex (complex scalar,…",0,[[]]],[11,"is_bosonic","","Returns true if the particle is bosonic.",0,[[]]],[11,"is_fermionic","","Returns true if the particle is fermionic.",0,[[]]],[11,"degrees_of_freedom","","Return the number of degrees of freedom for the underlying…",0,[[]]],[11,"phase_space","","Return the equilibrium phase space occupation of the…",0,[[]]],[11,"number_density","","Return the equilibrium number density of the particle.",0,[[]]],[11,"normalized_number_density","","Return the equilibrium number density of the particle,…",0,[[]]],[11,"entropy_dof","","Return the entropy degrees of freedom associated with this…",0,[[]]],[11,"propagator","","Return the propagator denominator for the particle.",0,[[],["propagator",3]]],[8,"Model","","Contains all the information relevant to a particular…",null,null],[10,"zero","","Instantiate a new instance of the model at 0 temperature…",10,[[]]],[10,"set_beta","","Update the model to be valid at the given inverse…",10,[[]]],[10,"get_beta","","Return the current value of beta for the model",10,[[]]],[10,"entropy_dof","","Return the effective degrees of freedom contributing to…",10,[[]]],[11,"hubble_rate","","Return the Hubble rate at the specified inverse temperature.",10,[[]]],[11,"len_particles","","Return the number of particles in the model.",10,[[]]],[10,"particles","","Return a list of particles in the model.",10,[[]]],[10,"particles_mut","","Return a mutable list of particles in the model.",10,[[]]],[10,"particle_idx","","Return the index corresponding to a particle\'s name and…",10,[[["asref",8]],["result",4]]],[11,"particle_name","","Convert a signed particle number to the corresponding…",10,[[],[["result",4],["string",3]]]],[11,"particle","","Return a reference to the matching particle by name.",10,[[["asref",8]],["particle",3]]],[11,"particle_mut","","Return a mutable reference to the matching particle by name.",10,[[],["particle",3]]],[11,"as_context","","Return a instance of [`Context`] for the model.",10,[[],["context",3]]],[8,"ModelInteractions","","Supertrait for [`Model`] for the handling of interactions.",null,null],[16,"Item","","The underlying interaction type.",11,null],[10,"interactions","","Return an iterator over all interactions in the model.",11,[[]]],[11,"update_widths","","Calculate the widths of all particles.",11,[[]]],[0,"prelude","boltzmann_solver","Common imports for this crate.",null,null],[3,"Particle","boltzmann_solver::prelude","Particle type",null,null],[12,"own_antiparticle","","Whether the particle is its own antiparticle or not.",0,null],[12,"mass","","Mass of the particle in GeV.",0,null],[12,"mass2","","Squared mass of the particle in GeV².",0,null],[12,"width","","Width of the particle in GeV.",0,null],[12,"width2","","Squared width of the particle in GeV².",0,null],[12,"decays","","Decays",0,null],[12,"name","","Name",0,null],[3,"StandardModel","","The Standard Model of particle physics.",null,null],[12,"beta","","Inverse temperature in GeV`$^{-1}$`",1,null],[12,"particles","","Particles",1,null],[12,"g1","","Hypercharge gauge coupling",1,null],[12,"g2","","Weak gauge coupling",1,null],[12,"g3","","Strong gauge coupling",1,null],[12,"yu","","Up-quark Yukawa",1,null],[12,"yd","","Down-quark Yukawa",1,null],[12,"ye","","Electron Yukawa",1,null],[12,"ckm","","CKM matrix",1,null],[12,"mh","","0-temperature mass of the Higgs",1,null],[12,"vev","","Vacuum expectation value of the Higgs",1,null],[12,"mu2","","Quadratic coupling of the Higgs",1,null],[12,"lambda","","Quartic term in scalar potential",1,null],[3,"Context","","Current context at a particular step in the numerical…",null,null],[12,"step","","Evaluation step",12,null],[12,"step_size","","Step size",12,null],[12,"beta","","Inverse temperature in GeV`$^{-1}$`",12,null],[12,"hubble_rate","","Hubble rate, in GeV",12,null],[12,"normalization","","Normalization factor, which is `math \\\\frac{1}{H \\\\beta n_1}…",12,null],[12,"eq","","Equilibrium number densities for the particles",12,null],[12,"n","","Current number density",12,null],[12,"na","","Current number density asymmetries",12,null],[12,"model","","Model data",12,null],[3,"Solver","","Boltzmann solver",null,null],[3,"SolverBuilder","","Boltzmann solver builder",null,null],[0,"sm_data","","Standard model data",null,null],[17,"MASS_Z","boltzmann_solver::prelude::sm_data","Z boson mass [GeV]",null,null],[17,"MASS_W","","W boson mass [GeV]",null,null],[17,"MASS_H","","Higgs boson mass [GeV]",null,null],[17,"MASS_EL","","Electron mass [GeV]",null,null],[17,"MASS_MU","","Muon mass [GeV]",null,null],[17,"MASS_TA","","Tau mass [GeV]",null,null],[17,"MASS_UP","","Up quark mass [GeV]",null,null],[17,"MASS_DO","","Down quark mass [GeV]",null,null],[17,"MASS_ST","","Strange quark mass [GeV]",null,null],[17,"MASS_CH","","Charm quark mass [GeV]",null,null],[17,"MASS_BO","","Bottom quark mass [GeV]",null,null],[17,"MASS_TO","","Top quark mass [GeV]",null,null],[17,"VEV","","Higgs boson VEV",null,null],[17,"CKM_A","","CKM Wolfenstain `$A$` parameter",null,null],[17,"CKM_LAMBDA","","CKM Wolfenstain `$\\\\lambda$` parameter",null,null],[17,"CKM_RHO","","CKM Wolfenstain `$\\\\overline\\\\rho$` parameter",null,null],[17,"CKM_ETA","","CKM Wolfenstain `$\\\\overline\\\\eta$` parameter",null,null],[0,"solver","boltzmann_solver","Solver for the number density evolution given by…",null,null],[3,"Context","boltzmann_solver::solver","Current context at a particular step in the numerical…",null,null],[12,"step","","Evaluation step",12,null],[12,"step_size","","Step size",12,null],[12,"beta","","Inverse temperature in GeV`$^{-1}$`",12,null],[12,"hubble_rate","","Hubble rate, in GeV",12,null],[12,"normalization","","Normalization factor, which is `math \\\\frac{1}{H \\\\beta n_1}…",12,null],[12,"eq","","Equilibrium number densities for the particles",12,null],[12,"n","","Current number density",12,null],[12,"na","","Current number density asymmetries",12,null],[12,"model","","Model data",12,null],[3,"Solver","","Boltzmann solver",null,null],[3,"SolverBuilder","","Boltzmann solver builder",null,null],[11,"new","boltzmann_solver::prelude","Creates a new builder for the Boltzmann solver.",13,[[]]],[11,"model","","Set a model function.",13,[[]]],[11,"initial_densities","","Specify initial number densities explicitly.",13,[[]]],[11,"initial_asymmetries","","Specify initial number density asymmetries explicitly.",13,[[]]],[11,"beta_range","","Set the range of inverse temperature values over which the…",13,[[]]],[11,"temperature_range","","Set the range of temperature values over which the phase…",13,[[]]],[11,"in_equilibrium","","Specify the particles which must remain in equilibrium.",13,[[]]],[11,"no_asymmetry","","Specify the particles which never develop an asymmetry.",13,[[]]],[11,"logger","","Set the logger.",13,[[]]],[11,"step_precision","","Specify how large or small the step size is allowed to…",13,[[]]],[11,"error_tolerance","","Specify the local error tolerance.",13,[[]]],[11,"skip_precomputation","","Upon calling [`SolverBuilder::build`], skip the…",13,[[]]],[11,"build","","Build the Boltzmann solver.",13,[[],[["result",4],["solver",3],["error",4]]]],[11,"solve","","Evolve the initial conditions by solving the PDEs.",14,[[]]],[0,"statistic","boltzmann_solver","If the rate of collisions between particles is…",null,null],[4,"Statistic","boltzmann_solver::statistic","The statistics which describe the distribution of…",null,null],[13,"FermiDirac","","Fermi–Dirac statistic describing half-integer-spin…",15,null],[13,"BoseEinstein","","Bose–Einstein statistic describing integer-spin particles:",15,null],[13,"MaxwellBoltzmann","","Maxwell–Boltzmann statistic describing classical particles:",15,null],[13,"MaxwellJuttner","","Maxwell–Jüttner statistic describing relativistic…",15,null],[17,"BOSON_EQ_DENSITY","","Equilibrium number density for massless bosons, normalized…",null,null],[17,"FERMION_EQ_DENSITY","","Equilibrium number density for massless fermions,…",null,null],[8,"Statistics","","Equilibrium statistics.",null,null],[10,"phase_space","","Evaluate the phase space distribution, for a given energy,…",16,[[]]],[11,"number_density","","Return number density for a particle following the…",16,[[]]],[10,"normalized_number_density","","Return number density for a particle following the…",16,[[]]],[11,"massless_number_density","","Return number density for a massless particle following…",16,[[]]],[0,"utilities","boltzmann_solver","Module of various useful miscellaneous functions.",null,null],[5,"kallen_lambda","boltzmann_solver::utilities","Kallen lambda function:",null,[[]]],[5,"kallen_lambda_sqrt","","Square root of the Kallen lambda function:",null,[[]]],[5,"t_range","","Return the minimum and maximum value of the Mandelstam…",null,[[]]],[5,"integrate_st","","Integrate the amplitude with respect to the Mandelstam…",null,[[]]],[0,"spline","","Cubic Hermite interpolation",null,null],[3,"ConstCubicHermiteSpline","boltzmann_solver::utilities::spline","Cubic Hermite spline interpolator using a constant data…",null,null],[12,"data","","Data array arranged in triples of `(xi, yi, mi)` where…",17,null],[3,"CubicHermiteSpline","","Cubic Hermite spline interpolator",null,null],[5,"rec_geomspace","","Create a recursively generated geometrically spaced…",null,[[],["vec",3]]],[5,"rec_linspace","","Create a recursively generated linearly spaced interval…",null,[[],["vec",3]]],[11,"sample","","Sample the spline at the specific `x` value.",17,[[]]],[11,"empty","","Create a new empty cubic Hermite Spline.",18,[[]]],[11,"len","","Return the number of data points in the underlying data.",18,[[]]],[11,"is_empty","","Check whether the spline is empty",18,[[]]],[11,"min_points","","Adjust the minimum number of points before accuracy is…",18,[[]]],[11,"add","","Adds a data point to the spline.",18,[[]]],[11,"accurate","","Check whether a given interval is accurate or not.",18,[[]]],[11,"sample","","Sample the spline at the specific `x` value.",18,[[]]],[0,"sm_data","boltzmann_solver","Standard model data",null,null],[17,"MASS_Z","boltzmann_solver::sm_data","Z boson mass [GeV]",null,null],[17,"MASS_W","","W boson mass [GeV]",null,null],[17,"MASS_H","","Higgs boson mass [GeV]",null,null],[17,"MASS_EL","","Electron mass [GeV]",null,null],[17,"MASS_MU","","Muon mass [GeV]",null,null],[17,"MASS_TA","","Tau mass [GeV]",null,null],[17,"MASS_UP","","Up quark mass [GeV]",null,null],[17,"MASS_DO","","Down quark mass [GeV]",null,null],[17,"MASS_ST","","Strange quark mass [GeV]",null,null],[17,"MASS_CH","","Charm quark mass [GeV]",null,null],[17,"MASS_BO","","Bottom quark mass [GeV]",null,null],[17,"MASS_TO","","Top quark mass [GeV]",null,null],[17,"VEV","","Higgs boson VEV",null,null],[17,"CKM_A","","CKM Wolfenstain `$A$` parameter",null,null],[17,"CKM_LAMBDA","","CKM Wolfenstain `$\\\\lambda$` parameter",null,null],[17,"CKM_RHO","","CKM Wolfenstain `$\\\\overline\\\\rho$` parameter",null,null],[17,"CKM_ETA","","CKM Wolfenstain `$\\\\overline\\\\eta$` parameter",null,null],[11,"from","boltzmann_solver::model","",0,[[]]],[11,"into","","",0,[[]]],[11,"to_owned","","",0,[[]]],[11,"clone_into","","",0,[[]]],[11,"try_from","","",0,[[],["result",4]]],[11,"try_into","","",0,[[],["result",4]]],[11,"borrow","","",0,[[]]],[11,"borrow_mut","","",0,[[]]],[11,"type_id","","",0,[[],["typeid",3]]],[11,"from","","",1,[[]]],[11,"into","","",1,[[]]],[11,"try_from","","",1,[[],["result",4]]],[11,"try_into","","",1,[[],["result",4]]],[11,"borrow","","",1,[[]]],[11,"borrow_mut","","",1,[[]]],[11,"type_id","","",1,[[],["typeid",3]]],[11,"from","boltzmann_solver::model::interaction","",7,[[]]],[11,"into","","",7,[[]]],[11,"try_from","","",7,[[],["result",4]]],[11,"try_into","","",7,[[],["result",4]]],[11,"borrow","","",7,[[]]],[11,"borrow_mut","","",7,[[]]],[11,"type_id","","",7,[[],["typeid",3]]],[11,"from","","",2,[[]]],[11,"into","","",2,[[]]],[11,"to_string","","",2,[[],["string",3]]],[11,"try_from","","",2,[[],["result",4]]],[11,"try_into","","",2,[[],["result",4]]],[11,"borrow","","",2,[[]]],[11,"borrow_mut","","",2,[[]]],[11,"type_id","","",2,[[],["typeid",3]]],[11,"from","","",3,[[]]],[11,"into","","",3,[[]]],[11,"to_owned","","",3,[[]]],[11,"clone_into","","",3,[[]]],[11,"to_string","","",3,[[],["string",3]]],[11,"try_from","","",3,[[],["result",4]]],[11,"try_into","","",3,[[],["result",4]]],[11,"borrow","","",3,[[]]],[11,"borrow_mut","","",3,[[]]],[11,"type_id","","",3,[[],["typeid",3]]],[11,"from","","",8,[[]]],[11,"into","","",8,[[]]],[11,"try_from","","",8,[[],["result",4]]],[11,"try_into","","",8,[[],["result",4]]],[11,"borrow","","",8,[[]]],[11,"borrow_mut","","",8,[[]]],[11,"type_id","","",8,[[],["typeid",3]]],[11,"from","","",4,[[]]],[11,"into","","",4,[[]]],[11,"to_owned","","",4,[[]]],[11,"clone_into","","",4,[[]]],[11,"try_from","","",4,[[],["result",4]]],[11,"try_into","","",4,[[],["result",4]]],[11,"borrow","","",4,[[]]],[11,"borrow_mut","","",4,[[]]],[11,"type_id","","",4,[[],["typeid",3]]],[11,"from","","",5,[[]]],[11,"into","","",5,[[]]],[11,"to_owned","","",5,[[]]],[11,"clone_into","","",5,[[]]],[11,"try_from","","",5,[[],["result",4]]],[11,"try_into","","",5,[[],["result",4]]],[11,"borrow","","",5,[[]]],[11,"borrow_mut","","",5,[[]]],[11,"type_id","","",5,[[],["typeid",3]]],[11,"from","","",6,[[]]],[11,"into","","",6,[[]]],[11,"to_owned","","",6,[[]]],[11,"clone_into","","",6,[[]]],[11,"try_from","","",6,[[],["result",4]]],[11,"try_into","","",6,[[],["result",4]]],[11,"borrow","","",6,[[]]],[11,"borrow_mut","","",6,[[]]],[11,"type_id","","",6,[[],["typeid",3]]],[11,"from","boltzmann_solver::prelude","",12,[[]]],[11,"into","","",12,[[]]],[11,"to_string","","",12,[[],["string",3]]],[11,"try_from","","",12,[[],["result",4]]],[11,"try_into","","",12,[[],["result",4]]],[11,"borrow","","",12,[[]]],[11,"borrow_mut","","",12,[[]]],[11,"type_id","","",12,[[],["typeid",3]]],[11,"from","","",14,[[]]],[11,"into","","",14,[[]]],[11,"try_from","","",14,[[],["result",4]]],[11,"try_into","","",14,[[],["result",4]]],[11,"borrow","","",14,[[]]],[11,"borrow_mut","","",14,[[]]],[11,"type_id","","",14,[[],["typeid",3]]],[11,"from","","",13,[[]]],[11,"into","","",13,[[]]],[11,"try_from","","",13,[[],["result",4]]],[11,"try_into","","",13,[[],["result",4]]],[11,"borrow","","",13,[[]]],[11,"borrow_mut","","",13,[[]]],[11,"type_id","","",13,[[],["typeid",3]]],[11,"from","boltzmann_solver::statistic","",15,[[]]],[11,"into","","",15,[[]]],[11,"to_string","","",15,[[],["string",3]]],[11,"try_from","","",15,[[],["result",4]]],[11,"try_into","","",15,[[],["result",4]]],[11,"borrow","","",15,[[]]],[11,"borrow_mut","","",15,[[]]],[11,"type_id","","",15,[[],["typeid",3]]],[11,"from","boltzmann_solver::utilities::spline","",17,[[]]],[11,"into","","",17,[[]]],[11,"try_from","","",17,[[],["result",4]]],[11,"try_into","","",17,[[],["result",4]]],[11,"borrow","","",17,[[]]],[11,"borrow_mut","","",17,[[]]],[11,"type_id","","",17,[[],["typeid",3]]],[11,"from","","",18,[[]]],[11,"into","","",18,[[]]],[11,"try_from","","",18,[[],["result",4]]],[11,"try_into","","",18,[[],["result",4]]],[11,"borrow","","",18,[[]]],[11,"borrow_mut","","",18,[[]]],[11,"type_id","","",18,[[],["typeid",3]]],[11,"particles","boltzmann_solver::model::interaction","",7,[[],["interactionparticles",3]]],[11,"particles_idx","","",7,[[],["interactionparticleindices",3]]],[11,"particles_sign","","",7,[[],["interactionparticlesigns",3]]],[11,"width_enabled","","",7,[[]]],[11,"gamma_enabled","","",7,[[]]],[11,"gamma","","",7,[[["context",3]],["option",4]]],[11,"asymmetry","","",7,[[["context",3]],["option",4]]],[11,"particles","","",8,[[],["interactionparticles",3]]],[11,"particles_idx","","",8,[[],["interactionparticleindices",3]]],[11,"particles_sign","","",8,[[],["interactionparticlesigns",3]]],[11,"width_enabled","","",8,[[]]],[11,"width","","",8,[[["context",3]],[["option",4],["partialwidth",3]]]],[11,"gamma_enabled","","",8,[[]]],[11,"gamma","","Unlike other function, this is computed in one of two ways:",8,[[["context",3]],["option",4]]],[11,"asymmetry","","",8,[[["context",3]],["option",4]]],[11,"rate","","Override the default implementation of…",8,[[["context",3]],[["option",4],["ratedensity",3]]]],[11,"zero","boltzmann_solver::model","",1,[[]]],[11,"set_beta","","Update beta for the model.",1,[[]]],[11,"get_beta","","",1,[[]]],[11,"entropy_dof","","",1,[[]]],[11,"particles","","",1,[[]]],[11,"particles_mut","","",1,[[]]],[11,"particle_idx","","",1,[[["asref",8]],["result",4]]],[11,"phase_space","boltzmann_solver::statistic","Evaluate the phase space distribution, `$f$` as defined…",15,[[]]],[11,"number_density","","Return number density for a particle following the…",15,[[]]],[11,"normalized_number_density","","Return number density for a particle following the…",15,[[]]],[11,"massless_number_density","","Return number density for a massless particle following…",15,[[]]],[11,"clone","boltzmann_solver::model::interaction","",3,[[],["ratedensity",3]]],[11,"clone","","",4,[[],["interactionparticles",3]]],[11,"clone","","",5,[[],["interactionparticleindices",3]]],[11,"clone","","",6,[[],["interactionparticlesigns",3]]],[11,"clone","boltzmann_solver::model","",0,[[],["particle",3]]],[11,"default","boltzmann_solver::model::interaction","",3,[[]]],[11,"default","boltzmann_solver::prelude","",13,[[]]],[11,"eq","boltzmann_solver::model","",0,[[]]],[11,"fmt","boltzmann_solver::model::interaction","",7,[[["formatter",3]],["result",6]]],[11,"fmt","","",2,[[["formatter",3]],["result",6]]],[11,"fmt","","",3,[[["formatter",3]],["result",6]]],[11,"fmt","","",8,[[["formatter",3]],["result",6]]],[11,"fmt","","",4,[[["formatter",3]],["result",6]]],[11,"fmt","","",5,[[["formatter",3]],["result",6]]],[11,"fmt","","",6,[[["formatter",3]],["result",6]]],[11,"fmt","boltzmann_solver::model","",0,[[["formatter",3]],["result",6]]],[11,"fmt","boltzmann_solver::prelude","",12,[[["formatter",3]],["result",6]]],[11,"fmt","boltzmann_solver::statistic","",15,[[["formatter",3]],["result",6]]],[11,"fmt","boltzmann_solver::utilities::spline","",18,[[["formatter",3]],["result",6]]],[11,"fmt","boltzmann_solver::model::interaction","",2,[[["formatter",3]],["result",6]]],[11,"fmt","","",3,[[["formatter",3]],["result",6]]],[11,"fmt","boltzmann_solver::prelude","",12,[[["formatter",3]],["result",6]]],[11,"fmt","boltzmann_solver::statistic","",15,[[["formatter",3]],["result",6]]],[11,"mul","boltzmann_solver::model::interaction","",3,[[]]],[11,"mul_assign","","",3,[[]]],[11,"serialize","","",2,[[],["result",4]]],[11,"serialize","","",3,[[],["result",4]]],[11,"serialize","","",4,[[],["result",4]]],[11,"serialize","","",5,[[],["result",4]]],[11,"serialize","","",6,[[],["result",4]]],[11,"serialize","boltzmann_solver::model","",0,[[],["result",4]]],[11,"serialize","","",1,[[],["result",4]]],[11,"serialize","boltzmann_solver::statistic","",15,[[],["result",4]]],[11,"serialize","boltzmann_solver::utilities::spline","",18,[[],["result",4]]],[11,"deserialize","boltzmann_solver::model::interaction","",2,[[],["result",4]]],[11,"deserialize","","",3,[[],["result",4]]],[11,"deserialize","","",4,[[],["result",4]]],[11,"deserialize","","",5,[[],["result",4]]],[11,"deserialize","","",6,[[],["result",4]]],[11,"deserialize","boltzmann_solver::model","",0,[[],["result",4]]],[11,"deserialize","","",1,[[],["result",4]]],[11,"deserialize","boltzmann_solver::statistic","",15,[[],["result",4]]],[11,"deserialize","boltzmann_solver::utilities::spline","",18,[[],["result",4]]]],"p":[[3,"Particle"],[3,"StandardModel"],[3,"PartialWidth"],[3,"RateDensity"],[3,"InteractionParticles"],[3,"InteractionParticleIndices"],[3,"InteractionParticleSigns"],[3,"FourParticle"],[3,"ThreeParticle"],[8,"Interaction"],[8,"Model"],[8,"ModelInteractions"],[3,"Context"],[3,"SolverBuilder"],[3,"Solver"],[4,"Statistic"],[8,"Statistics"],[3,"ConstCubicHermiteSpline"],[3,"CubicHermiteSpline"]]}\
}');
addSearchOptions(searchIndex);initSearch(searchIndex);