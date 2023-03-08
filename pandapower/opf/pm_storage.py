import numpy as np
import pandas as pd

def add_storage_opf_settings(net, ppci, pm):
    # callback function to add storage settings. Must be called after initializing pm data structure since the
    # pm["storage"] dict is filled

    # n time steps to optimize (here 3 hours)
    pm["n_time_steps"] = net._options["n_time_steps"]
    # time step (here 1 hour)
    pm["time_elapsed"] = net._options["time_elapsed"]

    # add storage systems to pm
    # Todo: Some variables are not used and not included in pandapower as well (energy_rating, thermal_rating,
    # (efficiencies, r, x...)
    bus_lookup = net._pd2pm_lookups["bus"]

    for idx in net["storage"].index:
        energy = (net["storage"].at[idx, "soc_percent"] * 1e-2 *
                  (net["storage"].at[idx, "max_e_mwh"] -
                   net["storage"].at[idx, "min_e_mwh"]))
        qs = net["storage"].at[idx, "q_mvar"].item()
        ps = net["storage"].at[idx, "p_mw"].item()
        max_p_mw = ps
        max_q_mvar, min_q_mvar = qs, -qs
        if "max_p_mw" in net["storage"]:
            max_p_mw = net["storage"].at[idx, "max_p_mw"].item()
        if "max_q_mvar" in net["storage"]:
            max_q_mvar = net["storage"].at[idx, "max_q_mvar"].item()
        if "max_q_mvar" in net["storage"]:
            min_q_mvar = net["storage"].at[idx, "min_q_mvar"].item()

        pm_idx = int(idx) + 1
        pm["storage"][str(pm_idx)] = {
            "index": pm_idx,
            "storage_bus": bus_lookup[net["storage"].at[idx, "bus"]].item(),
			"ps": ps, #* pm["baseMVA"],
            "qs": qs, #* pm["baseMVA"],            
            "energy": energy,
			"energy_rating": net["storage"].at[idx, "max_e_mwh"],
			"charge_rating": max_p_mw,
			"discharge_rating": max_p_mw,
			"charge_efficiency": 1.,
			"discharge_efficiency": 1.0,
	        "thermal_rating": net["storage"].at[idx, "max_e_mwh"], # Todo: include in DataFrame?
            "qmax": max_q_mvar,
            "qmin": min_q_mvar,
            "r": 0.0,
			"x": 0.,       
			"p_loss":-1e-8,
			"q_loss":-1e-8,			
			"status": int(net["storage"].at[idx, "in_service"]),
			"standby_loss": 0.
        }


def read_pm_storage_results(net):
    # reads the storage results from multiple time steps from the PowerModels optimization
    storage_results = dict()
    timesteps = list(net.res_ts_opt.keys())
    for idx in net.storage.index:
        # read storage results for each storage from power models to a dataframe with rows = timesteps
        res_storage = pd.DataFrame(data=None,
                                   index=timesteps,
                                   columns=["p_mw", "q_mvar", "soc_mwh", "soc_percent"],
                                   dtype=float)
        for t in timesteps:
            pm_storage = net.res_ts_opt[str(t)].res_storage
            res_storage.at[t, "p_mw"] = pm_storage["ps"]
            res_storage.at[t, "q_mvar"] = pm_storage["qs"]
            res_storage.at[t, "soc_percent"] = pm_storage["se"] * 1e2
            res_storage.at[t, "soc_mwh"] = pm_storage["se"] * \
                                           (net["storage"].at[idx, "max_e_mwh"] - net["storage"].at[idx, "min_e_mwh"])

        storage_results[idx] = res_storage

    return storage_results
