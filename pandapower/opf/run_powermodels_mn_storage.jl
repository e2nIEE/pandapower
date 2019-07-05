using PowerModels
using Ipopt
using JuMP
using .PP2PM

function set_pq_values_from_timeseries(mn)
    # Todo: The idea is that this function iterates over multinetwork entries and sets p, q values
    # of loads and "sgens"
    # The problem is that we have to get the time series as P, Q values (or even more for gens,
    # slacks...) in ppc format.
    # This requires to built the ppc from pandapower net first for each time step

    # iterate over networks (which represent the time steps)
    for (t, network) in mn["nw"]
        # iterate over all elements here for this network
        for (i, load) in network["load"]
            # update variables from time series here
            load["pd"] = load["pd"] * 1.
        end
    end

    return mn
end

function run_powermodels(json_path)
    # load converted pandapower network
    pm = PP2PM.load_pm_from_json(json_path)
    # copy network n_time_steps time step times
    n_time_steps = pm["n_time_steps"]
    mn = PowerModels.replicate(pm, pm["n_time_steps"])
    mn["time_elapsed"] = pm["time_elapsed"]
    # set P, Q values of loads and generators from time series
    mn = set_pq_values_from_timeseries(mn)

    ipopt_solver = JuMP.with_optimizer(Ipopt.Optimizer, print_level=0)

    # run multinetwork storage opf
    result = PowerModels._run_mn_strg_opf(mn, PowerModels.ACPPowerModel, ipopt_solver)
    return result
end
