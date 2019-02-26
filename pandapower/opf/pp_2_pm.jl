
module PP2PM
export load_pm_from_json

import JSON

function load_pm_from_json(json_path)
    pm = Dict()
    open(json_path, "r") do f
        pm = JSON.parse(f)  # parse and transform data
    end

    for (idx, gen) in pm["gen"]
        if gen["model"] == 1
            pm["gen"][idx]["cost"] = convert(Array{Float64,1}, gen["cost"])
        end
    end
    return pm
end

end
