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
    bus_lookup = net._pd2ppc_lookups["bus"]

    for idx in net["storage"].index:
        energy = (net["storage"].at[idx, "soc_percent"] * 1e-2 *
                  (net["storage"].at[idx, "max_e_mwh"] -
                   net["storage"].at[idx, "min_e_mwh"])) / pm["baseMVA"]
        qs = net["storage"].at[idx, "q_mvar"].item() / pm["baseMVA"]
        ps = net["storage"].at[idx, "p_mw"].item() / pm["baseMVA"]
        max_p_mw = ps
        max_q_mvar, min_q_mvar = qs, -qs
        if "max_p_mw" in net["storage"]:
            max_p_mw = net["storage"].at[idx, "max_p_mw"].item() / pm["baseMVA"]
        if "max_q_mvar" in net["storage"]:
            max_q_mvar = net["storage"].at[idx, "max_q_mvar"].item() / pm["baseMVA"]
        if "max_q_mvar" in net["storage"]:
            min_q_mvar = net["storage"].at[idx, "min_q_mvar"].item() / pm["baseMVA"]

        pm_idx = int(idx) + 1
        pm["storage"][str(pm_idx)] = {
            "energy_rating": net["storage"].at[idx, "max_e_mwh"],
            "standby_loss": 0.,
            "x": 0.,
            "energy": energy,
            "r": 0.0,
            "qs": qs,
            "thermal_rating": net["storage"].at[idx, "max_e_mwh"],  # Todo: include in DataFrame?
            "status": int(net["storage"].at[idx, "in_service"]),
            "discharge_rating": max_p_mw,
            "storage_bus": bus_lookup[net["storage"].at[idx, "bus"]].item(),
            "charge_efficiency": 1.,
            "index": pm_idx,
            "ps": ps,
            "qmax": max_q_mvar,
            "qmin": min_q_mvar,
            "charge_rating": max_p_mw,
            "discharge_efficiency": 1.0
        }


def read_pm_storage_results(net):
    # reads the storage results from multiple time steps from the PowerModels optimization
    pm_result = net._pm_result
    # power model networks (each network represents the result of one time step)
    networks = pm_result["solution"]["nw"]
    storage_results = dict()
    n_timesteps = len(networks)
    timesteps = np.arange(n_timesteps)
    for idx in net["storage"].index:
        # read storage results for each storage from power models to a dataframe with rows = timesteps
        pm_idx = str(int(idx) + 1)
        res_storage = pd.DataFrame(data=None,
                                   index=timesteps,
                                   columns=["p_mw", "q_mvar", "soc_mwh", "soc_percent"],
                                   dtype=float)
        for t in range(n_timesteps):
            pm_storage = networks[str(t + 1)]["storage"][pm_idx]
            res_storage.at[t, "p_mw"] = pm_storage["ps"] * pm_result["solution"]["baseMVA"]
            res_storage.at[t, "q_mvar"] = pm_storage["qs"] * pm_result["solution"]["baseMVA"]
            res_storage.at[t, "soc_percent"] = pm_storage["se"] * 1e2
            res_storage.at[t, "soc_mwh"] = pm_storage["se"] * \
                                           pm_result["solution"]["baseMVA"] * \
                                           (net["storage"].at[idx, "max_e_mwh"] - net["storage"].at[idx, "min_e_mwh"])

        storage_results[idx] = res_storage

    # DEBUG print for storage result
    # for key, val in net._pm_result.items():
    #     if key == "solution":
    #         for subkey, subval in val.items():
    #             if subkey == "nw":
    #                 for i, nw in subval.items():
    #                     print("Network {}\n".format(i))
    #                     print(nw["storage"])
    #                     print("\n")

    return storage_results
