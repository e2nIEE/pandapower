using PowerModels
using .PP2PM
#using Debugger
#include("pp_2_pm.jl")

function run_powermodels(json_path)
    pm = PP2PM.load_pm_from_json(json_path)
    model = PP2PM.get_model(pm["pm_model"])

    solver = PP2PM.get_solver(pm["pm_solver"], pm["pm_nl_solver"], pm["pm_mip_solver"],
    pm["pm_log_level"], pm["pm_time_limit"], pm["pm_nl_time_limit"], pm["pm_mip_time_limit"])

    result = PowerModels.run_opf(pm, model, solver,
                                    setting = Dict("output" => Dict("branch_flows" => true)))
    return result
end

#json_path = "C:/Users/fmeier/My_workspace/modell_one.json"
#@enter run_powermodels(json_path)
