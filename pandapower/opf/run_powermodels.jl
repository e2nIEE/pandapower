using PowerModels
using .PP2PM

function run_powermodels(json_path)
    pm = PP2PM.load_pm_from_json(json_path)
    model = PP2PM.get_model(pm["pm_model"])
    
    solver = PP2PM.get_solver(pm["pm_solver"], pm["pm_nl_solver"], pm["pm_mip_solver"], 
    pm["pm_log_level"], pm["pm_time_limit"], pm["pm_nl_time_limit"], pm["pm_mip_time_limit"])

    report_duals = get(pm, "report_duals", false)  # true or false
    branch_limits = get(pm, "branch_limits", "hard")  # "hard", "soft", "none"
    objective = get(pm, "objective", "default")  # TODO: check best options

    settings = Dict("output" => Dict("branch_flows" => true,
                                     "duals" => report_duals))

    result = PowerModels.run_opf(pm, model, solver, setting=settings) 
    return result
end
