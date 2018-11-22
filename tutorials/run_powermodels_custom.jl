using PowerModels
using Ipopt
import JSON

function load_pm_from_json(json_path)
    pm = Dict()
    open(json_path, "r") do f
        dicttxt = JSON.readstring(f)  # file information to string
        pm=JSON.parse(dicttxt)  # parse and transform data
    end
    for (idx, gen) in pm["gen"]
        if gen["model"] == 1
            pm["gen"][idx]["cost"] = convert(Array{Float64,1}, gen["cost"])
        end
    end
    return pm
end

function run_powermodels(json_path)
    pm = load_pm_from_json(json_path)
    result = PowerModels.run_dc_opf(pm, Ipopt.IpoptSolver())
    return result
end
