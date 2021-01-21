# using PowerModels
# # push!(LOAD_PATH, joinpath(homedir(), "pandapower", "pandapower", "opf"))
# push!(LOAD_PATH, joinpath(@__DIR__, "pandapower", "pandapower", "opf"))
# using PP2PM

function run_powermodels(json_path)
    pm = load_pm_from_json(json_path)

    model = get_model(pm["pm_model"])
    
    solver = get_solver(pm["pm_solver"], pm["pm_nl_solver"], pm["pm_mip_solver"], 
    pm["pm_log_level"], pm["pm_time_limit"], pm["pm_nl_time_limit"], pm["pm_mip_time_limit"])
    
    result = PowerModels.run_pf(pm, model, solver)
    # add branch flows
    PowerModels.update_data!(pm, result["solution"])
    flows = PowerModels.calc_branch_flow_ac(pm)
    PowerModels.update_data!(result["solution"], flows)
    return result
end
