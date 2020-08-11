using PowerModels
using Ipopt
import JSON

function run_powermodels(json_path)
    pm = PP2PM.load_pm_from_json(json_path)
    result = PowerModels.run_pf(pm, ACPPowerModel, Ipopt.Optimizer,
                                    setting = Dict("output" => Dict("branch_flows" => true)))
    return result
end
