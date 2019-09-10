using PowerModels
using .PP2PM

import Ipopt
import Juniper
import JuMP
import Gurobi

function run_powermodels(json_path)
    # function to run optimal transmission switching (OTS) optimization from powermodels.jl
    pm = PP2PM.load_pm_from_json(json_path)
    model = PP2PM.get_model(pm["pm_model"])
    solver = get_solver(pm["pm_solver"], "ipopt", "cbc")
    result = run_ots(pm, model, solver,
                        setting = Dict("output" => Dict("branch_flows" => true)))
    return result
end
