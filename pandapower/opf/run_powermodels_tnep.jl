using PowerModels
using .PP2PM

import Cbc
import Ipopt
import Juniper
import JuMP

function run_powermodels(json_path)
    # function to run transmission network expansion optimization of powermodels.jl
    pm = PP2PM.load_pm_from_json(json_path)

    cbc_solver = JuMP.with_optimizer(Cbc.Optimizer, logLevel = 1)
    juniper_solver = JuMP.with_optimizer(Juniper.Optimizer,
    nl_solver = JuMP.with_optimizer(Ipopt.Optimizer, tol = 1e-6, print_level = 1),
                                    mip_solver = cbc_solver, log_levels = [])

    result = run_tnep(pm, ACPPowerModel, juniper_solver,
                        setting = Dict("output" => Dict("branch_flows" => true)))
    return result
end
