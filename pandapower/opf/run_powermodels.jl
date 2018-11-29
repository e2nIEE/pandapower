using PowerModels
using Ipopt

include("./pp_2_pm.jl")
using PP2PM

function run_powermodels(json_path)
    pm = load_pm_from_json(json_path)
    result = PowerModels.run_ac_opf(pm, Ipopt.IpoptSolver(),
                                    setting = Dict("output" => Dict("branch_flows" => true)))
    return result
end
