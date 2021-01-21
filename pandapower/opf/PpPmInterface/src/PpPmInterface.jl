module PpPmInterface

import JuMP
import JSON
import PowerModels
using Cbc
using Ipopt
using Juniper

try
    using Gurobi
catch e
    if isa(e, LoadError)
        println("Cannot import Gurobi. That's fine if you do not plan to use it")
    end
end

try
    using KNITRO
catch e
    if isa(e, LoadError)
        println("Cannot import KNITRO. That's fine if you do not plan to use it")
    end
end

try
    using SCIP
catch e
    if isa(e, LoadError)
        println("Cannot import SCIP. That's fine if you do not plan to use it")
    end
end

include("input/pp_2_pm.jl")
include("pm_models/run_powermodels.jl")
include("pm_models/run_powermodels_mn_storage.jl")
include("pm_models/run_powermodels_ots.jl")
include("pm_models/run_powermodels_powerflow.jl")
include("pm_models/run_powermodels_tnep.jl")

end
