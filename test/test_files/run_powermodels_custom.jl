using PowerModels
using Ipopt
using JuMP
using .PP2PM

function run_powermodels(json_path)
    pm = PP2PM.load_pm_from_json(json_path)
    PowerModels.correct_network_data!(pm)
    ipopt_solver = JuMP.with_optimizer(Ipopt.Optimizer, print_level=0)
    result = PowerModels.run_ac_opf(pm, ipopt_solver,
                                    setting = Dict("output" => Dict("branch_flows" => true)))
    return result
end
