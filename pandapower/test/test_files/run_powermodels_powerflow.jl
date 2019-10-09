using PowerModels
using .PP2PM

function run_powermodels(json_path)
    pm = PP2PM.load_pm_from_json(json_path)
    model = PP2PM.get_model(pm["pm_model"])
    solver = PP2PM.get_solver(pm["pm_solver"], "ipopt", "cbc")
    result = PowerModels.run_pf(pm, model, solver,
                                    setting = Dict("output" => Dict("branch_flows" => true)))
    return result
end
