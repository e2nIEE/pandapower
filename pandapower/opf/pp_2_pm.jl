
module PP2PM
export load_pm_from_json, get_model, get_solver

import JSON
using PowerModels

import Cbc
import Ipopt
import Juniper
import JuMP

try
    import Gurobi
catch e
    if isa(e, LoadError)
        println("Cannot import Gurobi. That's fine if you do not plan to use it")
    end
end

function get_model(model_type)
    if model_type == "DCPPowerModel"
        return DCPPowerModel
    elseif model_type == "ACPPowerModel"
        return ACPPowerModel
    elseif model_type == "SOCWRPowerModel"
        return SOCWRPowerModel
    else
        error("model_type unknown: ", model_type)
    end
end

function get_solver(optimizer::String, nl::String="ipopt", mip::String="cbc",
    log_level::Int=0)
    # import gurobi by default, if that's not possible -> get ipopt'
    try
        if optimizer == "gurobi"
            solver = JuMP.with_optimizer(Gurobi.Optimizer, TimeLimit=5*60)
        end
    catch e
        if isa(e, LoadError)
            print("could not load gurobi. using ipopt instead")
            optimizer = "ipopt"
        end
    end

    if optimizer == "ipopt"
                solver = JuMP.with_optimizer(Ipopt.Optimizer, print_level=0)
    end

    if optimizer == "juniper" && nl == "ipopt" && mip == "cbc"
        mip_solver = JuMP.with_optimizer(Cbc.Optimizer, logLevel = log_level)
        nl_solver = JuMP.with_optimizer(Ipopt.Optimizer, tol = 1e-6, print_level = log_level)
        solver = JuMP.with_optimizer(Juniper.Optimizer,
                     nl_solver = nl_solver,
                     mip_solver = mip_solver,
                     log_levels = [])
    end
    return solver

end

function load_pm_from_json(json_path)
    pm = Dict()
    open(json_path, "r") do f
        pm = JSON.parse(f)  # parse and transform data
    end

    for (idx, gen) in pm["gen"]
        if gen["model"] == 1
            pm["gen"][idx]["cost"] = convert(Array{Float64,1}, gen["cost"])
        end
    end
    if pm["correct_pm_network_data"]
        correct_network_data!(pm)
    end
    return pm
end

end
