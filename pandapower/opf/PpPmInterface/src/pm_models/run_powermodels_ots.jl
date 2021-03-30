
function run_powermodels_ots(json_path)
    # function to run optimal transmission switching (OTS) optimization from powermodels.jl
    pm = load_pm_from_json(json_path)
    model = get_model(pm["pm_model"])

    solver = get_solver(pm["pm_solver"], pm["pm_nl_solver"], pm["pm_mip_solver"],
    pm["pm_log_level"], pm["pm_time_limit"], pm["pm_nl_time_limit"], pm["pm_mip_time_limit"])

    result = PowerModels.run_ots(pm, model, solver,
                        setting = Dict("output" => Dict("branch_flows" => true)))
    return result
end
