using PowerModels
using Ipopt
using .PP2PM

function run_powermodels(json_path)
    pm = PP2PM.load_pm_from_json(json_path)
    result = PowerModels.run_dc_opf(pm, Ipopt.IpoptSolver(),
                                    setting = Dict("output" => Dict("branch_flows" => true)))
    return result
end
