using PowerModels
using .PP2PM


function run_powermodels(json_path)
    pm = PP2PM.load_pm_from_json(json_path)

    if pm["pm_model"] == "DCPPowerModel"
            for (i, branch) in pm["branch"]
                branch["br_r"] = 0.
            end
    end
    model = PP2PM.get_model(pm["pm_model"])
    solver = PP2PM.get_solver(pm["pm_solver"], "ipopt", "cbc")
    result = PowerModels.run_pf(pm, model, solver,
                                    setting = Dict("output" => Dict("branch_flows" => true)))
#     make_mixed_units!(result["solution"])
    return result
end
