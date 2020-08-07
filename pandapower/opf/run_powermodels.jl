using PowerModels
using .PP2PM
#using Debugger
# include("pp_2_pm.jl")

function run_powermodels(json_path)
    pm = PP2PM.load_pm_from_json(json_path)
    model = PP2PM.get_model(pm["pm_model"])
    #
    solver = PP2PM.get_solver(pm["pm_solver"], pm["pm_nl_solver"], pm["pm_mip_solver"],
    pm["pm_log_level"], pm["pm_time_limit"], pm["pm_nl_time_limit"], pm["pm_mip_time_limit"])

    if haskey(pm["branch"]["1"],"c_rating_a")
        for (key, value) in pm["gen"]
           # value["pmin"] = 0
           value["pmax"] *= 0.01
           value["qmax"] *= 0.01
           value["qmin"] *= 0.01
           value["pg"] *= 0.01
           value["qg"] *= 0.01
           value["cost"] *= 100
        end

        for (key, value) in pm["branch"]
           value["c_rating_a"] *= 0.01
        end

        for (key, value) in pm["load"]
           value["pd"] *= 0.01
           value["qd"] *= 0.01
        end

        # pm["bus"]["4"]["vmax"] = 1.1
        # pm["bus"]["4"]["vmin"] = 0.9

        result = PowerModels._run_opf_cl(pm, model, solver,
                                        setting = Dict("output" => Dict("branch_flows" => true)))
    else
        result = PowerModels.run_opf(pm, model, solver,
                                        setting = Dict("output" => Dict("branch_flows" => true)))
    end


    return result
end


# json_path = "C:/Users/fmeier/pandapower/pandapower/test/opf/case5_clm_matfile_va.json"
# # #@enter run_powermodels(json_path)
# #
# result = run_powermodels(json_path)
# println(result["termination_status"] == LOCALLY_SOLVED)
# println(isapprox(result["objective"], 17015.5; atol = 1e0))
# mit eingeschränkter slack spannung: 17082.819507648066
