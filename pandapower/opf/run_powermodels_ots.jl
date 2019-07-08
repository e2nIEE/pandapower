using PowerModels
using .PP2PM

import Ipopt
import Juniper
import JuMP

function run_powermodels(json_path)
    # function to run optimal transmission switching (OTS) optimization from powermodels.jl
    pm = PP2PM.load_pm_from_json(json_path)

    juniper_solver = JuMP.with_optimizer(Juniper.Optimizer, nl_solver =
                                JuMP.with_optimizer(Ipopt.Optimizer, tol = 1e-4, print_level = 0))

    result = run_ots(pm, ACPPowerModel, juniper_solver,
                        setting = Dict("output" => Dict("branch_flows" => true)))
    return result
end
