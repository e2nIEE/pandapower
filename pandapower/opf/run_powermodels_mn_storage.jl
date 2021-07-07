using PowerModels
using Ipopt
using Gurobi
using JuMP
using .PP2PM

import JSON

function read_time_series(ts_path)
    time_series = Dict()
    open(ts_path, "r") do f
        time_series = JSON.parse(f)
    else
        @error "no time series data is available at $(ts_path)"
    end
    return time_series
end

function set_pq_from_timeseries!(mn, ts_data, variable)
    # This function iterates over multinetwork entries and sets p, q values
    # of loads and "sgens" (which are loads with negative P and Q values)
    for step in 1:steps
        network = mn["nw"]["$(step)"]
        for idx in keys(ts_data)
            network["load"][idx][viable] = ts_data [idx]["$(step-1)"] / network["baseMVA"]
        end
    end
    return mn
end

function run_powermodels(json_path)
    # load converted pandapower network
    pm = PP2PM.load_pm_from_json(files["net"])
    # copy network n_time_steps time step times
    n_time_steps = pm["n_time_steps"]
    mn = PowerModels.replicate(pm, pm["n_time_steps"])

    for key in keys(files)
        if key != "net"
            ts_data = read_time_series(files[key])
            if key == "load_p"
                mn = set_pq_from_timeseries!(mn, ts_data, "pd")
            elseif key == "load_q"
                mn = set_pq_from_timeseries!(mn, ts_data, "qd")
            end
        end
    end

    result = _PM.run_mn_opf_strg(mn, model, solver, setting = Dict("output" => Dict("branch_flows" => true)))
    # pm_res = get_result_for_pandapower(result)

    return result
end
