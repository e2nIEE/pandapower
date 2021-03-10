
function run_powermodels_tnep(json_path)
    pm = load_pm_from_json(json_path)
    model = get_model(pm["pm_model"])

    solver = get_solver(pm["pm_solver"], pm["pm_nl_solver"], pm["pm_mip_solver"],
    pm["pm_log_level"], pm["pm_time_limit"], pm["pm_nl_time_limit"], pm["pm_mip_time_limit"])

    # function to run transmission network expansion optimization of powermodels.jl
    result = PowerModels.run_tnep(pm, model, solver,
                        setting = Dict("output" => Dict("branch_flows" => true)))
    return result
end
