module PpPmInterface

import JuMP
import JSON

using PowerModels
using Cbc
using Ipopt
using Juniper

try
    import Gurobi
catch e
    if isa(e, ArgumentError)
        println("Cannot import Gurobi. That's fine if you do not plan to use it")
    end
end

export run_powermodels, run_powermodels_mn_storage,
        run_powermodels_ots, run_powermodels_powerflow,
        run_powermodels_tnep
        
include("input/pp_to_pm.jl")

include("../src/pm_models/run_powermodels.jl")
include("../src/pm_models/run_powermodels_mn_storage.jl")
include("../src/pm_models/run_powermodels_ots.jl")
include("../src/pm_models/run_powermodels_powerflow.jl")
include("../src/pm_models/run_powermodels_tnep.jl")
#
# try
#     using KNITRO
# catch e
#     if isa(e, LoadError)
#         println("Cannot import KNITRO. That's fine if you do not plan to use it")
#     end
# end
#
# try
#     using SCIP
# catch e
#     if isa(e, LoadError)
#         println("Cannot import SCIP. That's fine if you do not plan to use it")
#     end
# end

end
