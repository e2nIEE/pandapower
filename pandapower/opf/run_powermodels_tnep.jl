using PowerModels
using .PP2PM
import Cbc
import Ipopt
import Juniper
import JuMP
#import Gurobi

function run_powermodels(json_path)
    pm = PP2PM.load_pm_from_json(json_path)
    model = PP2PM.get_model(pm["pm_model"])
    solver = PP2PM.get_solver(pm["pm_solver"], "ipopt", "cbc")

    # function to run transmission network expansion optimization of powermodels.jl
    result = run_tnep(pm, model, solver,
                        setting = Dict("output" => Dict("branch_flows" => true)))
    return result
end
