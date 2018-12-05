using PowerModels
using Ipopt
import JSON

function run_powermodels(json_path)
    pm_net = Dict()
    open(json_path, "r") do f
        dicttxt = JSON.readstring(f)  # file information to string
        pm_net=JSON.parse(dicttxt)  # parse and transform data
    end
    result = PowerModels.run_ac_opf(pm_net, Ipopt.IpoptSolver())
    return result
end
