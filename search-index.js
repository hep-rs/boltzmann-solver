var searchIndex = {};
searchIndex["boltzmann_solver"] = {"doc":"`boltzmann-solver` is a library allowing for Boltzmann equation in the context of particle physics / early cosmology.  It provides functionalities to solve Boltzmann equation in the case where a single species is out of equilibrium, as well as functionalities to solve the Boltzmann equations more general when multiple species are all out of equilibrium.","items":[[0,"constants","boltzmann_solver","Collection of physical and mathematical constants which appear frequently.",null,null],[17,"PLANCK_MASS","boltzmann_solver::constants","Planck mass, \\(M_{\\text{Pl}} = \\sqrt{\\hbar c / G}\\), in units of GeV / \\(c^2\\).",null,null],[17,"REDUCED_PLANCK_MASS","","Reduced Planck mass, \\(m_{\\text{Pl}} = \\sqrt{\\hbar c / 8 \\pi G}\\), in units of GeV / \\(c^2\\).",null,null],[17,"ZETA_3","","Riemann zeta function evaluated at 3: \\(\\zeta(3) \\approx 1.202\\dots\\)",null,null],[17,"PI","","\\(\\pi\\)",null,null],[17,"PI_1","","\\(\\pi\\) [named to follow the convention `PI_n`]",null,null],[17,"PI_2","","\\(\\pi^2\\)",null,null],[17,"PI_3","","\\(\\pi^3\\)",null,null],[17,"PI_4","","\\(\\pi^4\\)",null,null],[17,"PI_M1","","\\(\\pi^{-1}\\)",null,null],[17,"PI_M2","","\\(\\pi^{-2}\\)",null,null],[17,"PI_M3","","\\(\\pi^{-3}\\)",null,null],[17,"PI_M4","","\\(\\pi^{-4}\\)",null,null],[0,"statistic","boltzmann_solver","If the rate of collisions between particles is sufficiently high (as is usually the case), the phase space distribution of the particles will quickly converge onto either the Fermi–Dirac statistic or the Bose–Einstein statistic depending on whether the particle is a half-integer or integer spin particle.",null,null],[4,"Statistic","boltzmann_solver::statistic","The statistics which describe the distribution of particles over energy states.  Both Fermi–Dirac and Bose–Einstein quantum statistics are implemented, as well as the classical Maxwell–Boltzmann statistic.",null,null],[13,"FermiDirac","","Fermi–Dirac statistic describing half-integer-spin particles:",0,null],[13,"BoseEinstein","","Bose–Einstein statistic describing integer-spin particles:",0,null],[13,"MaxwellBoltzmann","","Maxwell–Boltzmann statistic describing classical particles:",0,null],[13,"MaxwellJuttner","","Maxwell–Jüttner statistic describing relativistic classical particles:",0,null],[11,"phase_space","","Evaluate the phase space distribution, \\(f\\) as defined above.",0,{"inputs":[{"name":"self"},{"name":"f64"},{"name":"f64"},{"name":"f64"},{"name":"f64"}],"output":{"name":"f64"}}],[11,"number_density","","Return number density for a particle following the specified statistic.",0,{"inputs":[{"name":"self"},{"name":"f64"},{"name":"f64"},{"name":"f64"}],"output":{"name":"f64"}}],[11,"massless_number_density","","Return number density for a massless particle following the specified statistic.",0,{"inputs":[{"name":"self"},{"name":"f64"},{"name":"f64"}],"output":{"name":"f64"}}],[0,"universe","boltzmann_solver","The effects of the Universe's evolution play an important role in the Boltzmann equation.  This information is provided by implementations of the [`Universe`] trait.",null,null],[3,"StandardModel","boltzmann_solver::universe","Implementation of [`Universe`] for the Standard Model.",null,null],[3,"SingleSpecies","","Contribution from a single particle in the Universe.",null,null],[11,"fmt","","",1,{"inputs":[{"name":"self"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"default","","",1,{"inputs":[],"output":{"name":"standardmodel"}}],[11,"new","","Create an instance of the Standard Model.",1,{"inputs":[],"output":{"name":"self"}}],[11,"entropy_dof","","",1,{"inputs":[{"name":"self"},{"name":"f64"}],"output":{"name":"f64"}}],[8,"Universe","","Collection of properties which determine the evolution of a Universe.",null,null],[10,"entropy_dof","","Return the effective degrees of freedom contributing to the entropy density of the Universe at the specified inverse temperature.",2,{"inputs":[{"name":"self"},{"name":"f64"}],"output":{"name":"f64"}}],[11,"hubble_rate","","Return the Hubble rate at the specified inverse temperature.",2,{"inputs":[{"name":"self"},{"name":"f64"}],"output":{"name":"f64"}}],[11,"new","","Create a new particle with the specified statistic, mass and degrees of freedom.",3,{"inputs":[{"name":"statistic"},{"name":"f64"},{"name":"f64"}],"output":{"name":"self"}}],[11,"entropy_dof","","",3,{"inputs":[{"name":"self"},{"name":"f64"}],"output":{"name":"f64"}}]],"paths":[[4,"Statistic"],[3,"StandardModel"],[8,"Universe"],[3,"SingleSpecies"]]};
searchIndex["cfg_if"] = {"doc":"A macro for defining #[cfg] if-else statements.","items":[[14,"cfg_if","cfg_if","",null,null]],"paths":[]};
searchIndex["log"] = {"doc":"A lightweight logging facade.","items":[[3,"Record","log","The \"payload\" of a log message.",null,null],[3,"RecordBuilder","","Builder for `Record`.",null,null],[3,"Metadata","","Metadata about a log message.",null,null],[3,"MetadataBuilder","","Builder for `Metadata`.",null,null],[3,"SetLoggerError","","The type returned by [`set_logger`] if [`set_logger`] has already been called.",null,null],[3,"ParseLevelError","","The type returned by [`from_str`] when the string doesn't match any of the log levels.",null,null],[4,"Level","","An enum representing the available verbosity levels of the logger.",null,null],[13,"Error","","The \"error\" level.",0,null],[13,"Warn","","The \"warn\" level.",0,null],[13,"Info","","The \"info\" level.",0,null],[13,"Debug","","The \"debug\" level.",0,null],[13,"Trace","","The \"trace\" level.",0,null],[4,"LevelFilter","","An enum representing the available verbosity level filters of the logger.",null,null],[13,"Off","","A level lower than all log levels.",1,null],[13,"Error","","Corresponds to the `Error` log level.",1,null],[13,"Warn","","Corresponds to the `Warn` log level.",1,null],[13,"Info","","Corresponds to the `Info` log level.",1,null],[13,"Debug","","Corresponds to the `Debug` log level.",1,null],[13,"Trace","","Corresponds to the `Trace` log level.",1,null],[5,"set_max_level","","Sets the global maximum log level.",null,{"inputs":[{"name":"levelfilter"}],"output":null}],[5,"max_level","","Returns the current maximum log level.",null,{"inputs":[],"output":{"name":"levelfilter"}}],[5,"set_logger","","Sets the global logger to a `&'static Log`.",null,{"inputs":[{"name":"log"}],"output":{"generics":["setloggererror"],"name":"result"}}],[5,"logger","","Returns a reference to the logger.",null,{"inputs":[],"output":{"name":"log"}}],[17,"STATIC_MAX_LEVEL","","The statically resolved maximum log level.",null,null],[8,"Log","","A trait encapsulating the operations required of a logger.",null,null],[10,"enabled","","Determines if a log message with the specified metadata would be logged.",2,{"inputs":[{"name":"self"},{"name":"metadata"}],"output":{"name":"bool"}}],[10,"log","","Logs the `Record`.",2,{"inputs":[{"name":"self"},{"name":"record"}],"output":null}],[10,"flush","","Flushes any buffered records.",2,{"inputs":[{"name":"self"}],"output":null}],[11,"fmt","","",0,{"inputs":[{"name":"self"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"hash","","",0,null],[11,"clone","","",0,{"inputs":[{"name":"self"}],"output":{"name":"level"}}],[11,"eq","","",0,{"inputs":[{"name":"self"},{"name":"level"}],"output":{"name":"bool"}}],[11,"eq","","",0,{"inputs":[{"name":"self"},{"name":"levelfilter"}],"output":{"name":"bool"}}],[11,"partial_cmp","","",0,{"inputs":[{"name":"self"},{"name":"level"}],"output":{"generics":["ordering"],"name":"option"}}],[11,"partial_cmp","","",0,{"inputs":[{"name":"self"},{"name":"levelfilter"}],"output":{"generics":["ordering"],"name":"option"}}],[11,"cmp","","",0,{"inputs":[{"name":"self"},{"name":"level"}],"output":{"name":"ordering"}}],[11,"from_str","","",0,{"inputs":[{"name":"str"}],"output":{"generics":["level"],"name":"result"}}],[11,"fmt","","",0,{"inputs":[{"name":"self"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"max","","Returns the most verbose logging level.",0,{"inputs":[],"output":{"name":"level"}}],[11,"to_level_filter","","Converts the `Level` to the equivalent `LevelFilter`.",0,{"inputs":[{"name":"self"}],"output":{"name":"levelfilter"}}],[11,"fmt","","",1,{"inputs":[{"name":"self"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"hash","","",1,null],[11,"clone","","",1,{"inputs":[{"name":"self"}],"output":{"name":"levelfilter"}}],[11,"eq","","",1,{"inputs":[{"name":"self"},{"name":"levelfilter"}],"output":{"name":"bool"}}],[11,"eq","","",1,{"inputs":[{"name":"self"},{"name":"level"}],"output":{"name":"bool"}}],[11,"partial_cmp","","",1,{"inputs":[{"name":"self"},{"name":"levelfilter"}],"output":{"generics":["ordering"],"name":"option"}}],[11,"partial_cmp","","",1,{"inputs":[{"name":"self"},{"name":"level"}],"output":{"generics":["ordering"],"name":"option"}}],[11,"cmp","","",1,{"inputs":[{"name":"self"},{"name":"levelfilter"}],"output":{"name":"ordering"}}],[11,"from_str","","",1,{"inputs":[{"name":"str"}],"output":{"generics":["levelfilter"],"name":"result"}}],[11,"fmt","","",1,{"inputs":[{"name":"self"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"max","","Returns the most verbose logging level filter.",1,{"inputs":[],"output":{"name":"levelfilter"}}],[11,"to_level","","Converts `self` to the equivalent `Level`.",1,{"inputs":[{"name":"self"}],"output":{"generics":["level"],"name":"option"}}],[11,"clone","","",3,{"inputs":[{"name":"self"}],"output":{"name":"record"}}],[11,"fmt","","",3,{"inputs":[{"name":"self"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"builder","","Returns a new builder.",3,{"inputs":[],"output":{"name":"recordbuilder"}}],[11,"args","","The message body.",3,{"inputs":[{"name":"self"}],"output":{"name":"arguments"}}],[11,"metadata","","Metadata about the log directive.",3,{"inputs":[{"name":"self"}],"output":{"name":"metadata"}}],[11,"level","","The verbosity level of the message.",3,{"inputs":[{"name":"self"}],"output":{"name":"level"}}],[11,"target","","The name of the target of the directive.",3,{"inputs":[{"name":"self"}],"output":{"name":"str"}}],[11,"module_path","","The module path of the message.",3,{"inputs":[{"name":"self"}],"output":{"generics":["str"],"name":"option"}}],[11,"file","","The source file containing the message.",3,{"inputs":[{"name":"self"}],"output":{"generics":["str"],"name":"option"}}],[11,"line","","The line containing the message.",3,{"inputs":[{"name":"self"}],"output":{"generics":["u32"],"name":"option"}}],[11,"fmt","","",4,{"inputs":[{"name":"self"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"new","","Construct new `RecordBuilder`.",4,{"inputs":[],"output":{"name":"recordbuilder"}}],[11,"args","","Set `args`.",4,{"inputs":[{"name":"self"},{"name":"arguments"}],"output":{"name":"recordbuilder"}}],[11,"metadata","","Set `metadata`. Construct a `Metadata` object with `MetadataBuilder`.",4,{"inputs":[{"name":"self"},{"name":"metadata"}],"output":{"name":"recordbuilder"}}],[11,"level","","Set `Metadata::level`.",4,{"inputs":[{"name":"self"},{"name":"level"}],"output":{"name":"recordbuilder"}}],[11,"target","","Set `Metadata::target`",4,{"inputs":[{"name":"self"},{"name":"str"}],"output":{"name":"recordbuilder"}}],[11,"module_path","","Set `module_path`",4,{"inputs":[{"name":"self"},{"generics":["str"],"name":"option"}],"output":{"name":"recordbuilder"}}],[11,"file","","Set `file`",4,{"inputs":[{"name":"self"},{"generics":["str"],"name":"option"}],"output":{"name":"recordbuilder"}}],[11,"line","","Set `line`",4,{"inputs":[{"name":"self"},{"generics":["u32"],"name":"option"}],"output":{"name":"recordbuilder"}}],[11,"build","","Invoke the builder and return a `Record`",4,{"inputs":[{"name":"self"}],"output":{"name":"record"}}],[11,"clone","","",5,{"inputs":[{"name":"self"}],"output":{"name":"metadata"}}],[11,"eq","","",5,{"inputs":[{"name":"self"},{"name":"metadata"}],"output":{"name":"bool"}}],[11,"ne","","",5,{"inputs":[{"name":"self"},{"name":"metadata"}],"output":{"name":"bool"}}],[11,"cmp","","",5,{"inputs":[{"name":"self"},{"name":"metadata"}],"output":{"name":"ordering"}}],[11,"partial_cmp","","",5,{"inputs":[{"name":"self"},{"name":"metadata"}],"output":{"generics":["ordering"],"name":"option"}}],[11,"lt","","",5,{"inputs":[{"name":"self"},{"name":"metadata"}],"output":{"name":"bool"}}],[11,"le","","",5,{"inputs":[{"name":"self"},{"name":"metadata"}],"output":{"name":"bool"}}],[11,"gt","","",5,{"inputs":[{"name":"self"},{"name":"metadata"}],"output":{"name":"bool"}}],[11,"ge","","",5,{"inputs":[{"name":"self"},{"name":"metadata"}],"output":{"name":"bool"}}],[11,"hash","","",5,null],[11,"fmt","","",5,{"inputs":[{"name":"self"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"builder","","Returns a new builder.",5,{"inputs":[],"output":{"name":"metadatabuilder"}}],[11,"level","","The verbosity level of the message.",5,{"inputs":[{"name":"self"}],"output":{"name":"level"}}],[11,"target","","The name of the target of the directive.",5,{"inputs":[{"name":"self"}],"output":{"name":"str"}}],[11,"eq","","",6,{"inputs":[{"name":"self"},{"name":"metadatabuilder"}],"output":{"name":"bool"}}],[11,"ne","","",6,{"inputs":[{"name":"self"},{"name":"metadatabuilder"}],"output":{"name":"bool"}}],[11,"cmp","","",6,{"inputs":[{"name":"self"},{"name":"metadatabuilder"}],"output":{"name":"ordering"}}],[11,"partial_cmp","","",6,{"inputs":[{"name":"self"},{"name":"metadatabuilder"}],"output":{"generics":["ordering"],"name":"option"}}],[11,"lt","","",6,{"inputs":[{"name":"self"},{"name":"metadatabuilder"}],"output":{"name":"bool"}}],[11,"le","","",6,{"inputs":[{"name":"self"},{"name":"metadatabuilder"}],"output":{"name":"bool"}}],[11,"gt","","",6,{"inputs":[{"name":"self"},{"name":"metadatabuilder"}],"output":{"name":"bool"}}],[11,"ge","","",6,{"inputs":[{"name":"self"},{"name":"metadatabuilder"}],"output":{"name":"bool"}}],[11,"hash","","",6,null],[11,"fmt","","",6,{"inputs":[{"name":"self"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"new","","Construct a new `MetadataBuilder`.",6,{"inputs":[],"output":{"name":"metadatabuilder"}}],[11,"level","","Setter for `level`.",6,{"inputs":[{"name":"self"},{"name":"level"}],"output":{"name":"metadatabuilder"}}],[11,"target","","Setter for `target`.",6,{"inputs":[{"name":"self"},{"name":"str"}],"output":{"name":"metadatabuilder"}}],[11,"build","","Returns a `Metadata` object.",6,{"inputs":[{"name":"self"}],"output":{"name":"metadata"}}],[11,"fmt","","",7,{"inputs":[{"name":"self"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"fmt","","",7,{"inputs":[{"name":"self"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"fmt","","",8,{"inputs":[{"name":"self"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"eq","","",8,{"inputs":[{"name":"self"},{"name":"parselevelerror"}],"output":{"name":"bool"}}],[11,"ne","","",8,{"inputs":[{"name":"self"},{"name":"parselevelerror"}],"output":{"name":"bool"}}],[11,"fmt","","",8,{"inputs":[{"name":"self"},{"name":"formatter"}],"output":{"name":"result"}}],[14,"log","","The standard logging macro.",null,null],[14,"error","","Logs a message at the error level.",null,null],[14,"warn","","Logs a message at the warn level.",null,null],[14,"info","","Logs a message at the info level.",null,null],[14,"debug","","Logs a message at the debug level.",null,null],[14,"trace","","Logs a message at the trace level.",null,null],[14,"log_enabled","","Determines if a message logged at the specified level in that module will be logged.",null,null]],"paths":[[4,"Level"],[4,"LevelFilter"],[8,"Log"],[3,"Record"],[3,"RecordBuilder"],[3,"Metadata"],[3,"MetadataBuilder"],[3,"SetLoggerError"],[3,"ParseLevelError"]]};
searchIndex["quadrature"] = {"doc":"The primary function of this library is `integrate`, witch uses the double exponential algorithm. It is a port of the Fast Numerical Integration from c++ to rust. The original code is by John D. Cook, and is licensed under the BSD.","items":[[3,"Output","quadrature","",null,null],[12,"num_function_evaluations","","",0,null],[12,"error_estimate","","",0,null],[12,"integral","","",0,null],[0,"double_exponential","","The double exponential algorithm is naturally adaptive, it stops calling the integrand when the error is reduced to below the desired threshold. It also does not allocate. No box, no vec, etc. It has a hard coded maximum of approximately 350 function evaluations. This guarantees that the algorithm will return. The error in the algorithm decreases exponentially in the number of function evaluations, specifically O(exp(-cN/log(N))). So if 350 function evaluations is not giving the desired accuracy than the programmer probably needs to give some guidance by splitting up the range at singularities or other preparation techniques.",null,null],[5,"integrate","quadrature::double_exponential","Integrate an analytic function over a finite interval. f is the function to be integrated. a is left limit of integration. b is right limit of integration target_absolute_error is the desired bound on error",null,{"inputs":[{"name":"f"},{"name":"f64"},{"name":"f64"},{"name":"f64"}],"output":{"name":"output"}}],[0,"clenshaw_curtis","quadrature","The `clenshaw_curtis` module provides a `integrate` function with the same signature as `quadrature::integrate`. The implemented variant of clenshaw curtis quadrature is adaptive, however the weights change for each adaptation. This unfortunately means that the sum needs to be recalculated for each layer of adaptation. It also does not allocate on the heap, however it does use a `[f64; 129]` to store the function values. It has a hard coded maximum of approximately 257 function evaluations. This guarantees that the algorithm will return. The clenshaw curtis algorithm exactly integrates polynomials of order N. This implementation starts with an N of approximately 5 and increases up to an N of approximately 257. In general the error in the algorithm decreases exponentially in the number of function evaluations. In summery clenshaw curtis will in general use more stack space and run slower than the double exponential algorithm, unless clenshaw curtis can get the exact solution.",null,null],[5,"integrate","quadrature::clenshaw_curtis","Integrate an analytic function over a finite interval. f is the function to be integrated. a is left limit of integration. b is right limit of integration target_absolute_error is the desired bound on error",null,{"inputs":[{"name":"f"},{"name":"f64"},{"name":"f64"},{"name":"f64"}],"output":{"name":"output"}}],[11,"clone","quadrature","",0,{"inputs":[{"name":"self"}],"output":{"name":"output"}}],[11,"fmt","","",0,{"inputs":[{"name":"self"},{"name":"formatter"}],"output":{"name":"result"}}]],"paths":[[3,"Output"]]};
searchIndex["special_functions"] = {"doc":"Library providing pure rust implementation of various special functions, with particular focus to high-energy particle physics.","items":[[0,"polylog","special_functions","Polylogarithms",null,null],[5,"bose_einstein","special_functions::polylog","Approximation of polylogarithm appearing in the Bose–Einstein statistics. Specifically, this approximates the function \\(\\Li_{3} e^x\\) for \\(x \\leq 0\\).",null,{"inputs":[{"name":"f64"}],"output":{"name":"f64"}}],[5,"fermi_dirac","","Approximation of polylogarithm appearing in the Fermi–Dirac statistics. Specifically, this approximates the function \\(-\\Li_{3} (-e^x)\\) for all values of \\(x\\).",null,{"inputs":[{"name":"f64"}],"output":{"name":"f64"}}],[0,"bessel","special_functions","Bessel functions",null,null],[5,"k_0","special_functions::bessel","Approximation of modified Bessel function \\(K_0(x)\\) for all \\(x \\geq 0\\).",null,{"inputs":[{"name":"f64"}],"output":{"name":"f64"}}],[5,"k_1","","Approximation of modified Bessel function \\(K_1(x)\\) for all \\(x \\geq 0\\).",null,{"inputs":[{"name":"f64"}],"output":{"name":"f64"}}],[5,"k_2","","Approximation of modified Bessel function \\(K_2(x)\\) for all \\(x \\geq 0\\).",null,{"inputs":[{"name":"f64"}],"output":{"name":"f64"}}],[5,"k_3","","Approximation of modified Bessel function \\(K_3(x)\\) for all \\(x \\geq 0\\).",null,{"inputs":[{"name":"f64"}],"output":{"name":"f64"}}],[0,"polynomial","special_functions","Utilities to handle polynomials",null,null],[5,"polynomial","special_functions::polynomial","Evaluates an arbitrary single-variable polynomial at a particular point.",null,null],[0,"interpolation","special_functions","Interpolation functions",null,null],[5,"linear","special_functions::interpolation","Perform linear interpolation on data.",null,null]],"paths":[]};
initSearch(searchIndex);
