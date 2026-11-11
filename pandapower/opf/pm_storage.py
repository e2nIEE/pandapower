import numpy as np
import pandas as pd

from pandapower.opf.make_objective import costs_from_areas


def add_storage_opf_settings(net, ppci, pm):
    """
    pp_to_pm_callback of :func:`runpm_storage_opf`: adds the controllable storages to pm["storage"]. Must be
    called after initializing the pm data structure.

    The PowerModels storage model optimizes the storage power, so the optimized storages must not be part of
    the pm network otherwise: a controllable storage is a ppc generator, which is deactivated here.
    Non-controllable storages keep their fixed power as loads and are not optimized.
    """
    # n time steps to optimize
    pm["n_time_steps"] = net._options["n_time_steps"]
    # duration of a time step in hours
    pm["time_elapsed"] = net._options["time_elapsed"]

    _deactivate_storage_gens(net, ppci, pm)

    # Todo: Some variables are not used and not included in pandapower as well (efficiencies, r, x...)
    bus_lookup = net._pd2pm_lookups["bus"]
    storage = net["storage"]
    for idx in storage.index[_controllable(storage)]:
        energy = storage.at[idx, "soc_percent"] * 1e-2 * (storage.at[idx, "max_e_mwh"] - storage.at[idx, "min_e_mwh"])
        ps = float(storage.at[idx, "p_mw"])
        qs = float(storage.at[idx, "q_mvar"])
        # charging: p_mw > 0, discharging: p_mw < 0. Without limits, |p_mw| and |q_mvar| are the limits
        max_p_mw = _value(storage, idx, "max_p_mw", abs(ps))
        min_p_mw = _value(storage, idx, "min_p_mw", -abs(ps))
        if min_p_mw > 0 or max_p_mw < 0:
            # charge and discharge ratings are upper limits only, a minimum charge / discharge cannot be modeled
            raise ValueError(
                f"storage {idx}: min_p_mw ({min_p_mw}) <= 0 <= max_p_mw ({max_p_mw}) is required by the "
                f"PowerModels storage model"
            )
        charge_rating = max_p_mw
        discharge_rating = -min_p_mw
        max_q_mvar = _value(storage, idx, "max_q_mvar", abs(qs))
        min_q_mvar = _value(storage, idx, "min_q_mvar", -abs(qs))
        # apparent power limit: the rated power or, if not given, the active and reactive power limits
        q_rating = max(abs(max_q_mvar), abs(min_q_mvar))
        thermal_rating = _value(storage, idx, "sn_mva", float(np.hypot(max(charge_rating, discharge_rating), q_rating)))

        pm_idx = int(idx) + 1
        pm["storage"][str(pm_idx)] = {
            "index": pm_idx,
            "storage_bus": bus_lookup[storage.at[idx, "bus"]].item(),
            "ps": ps,
            "qs": qs,
            "energy": energy,
            "energy_rating": storage.at[idx, "max_e_mwh"],
            "charge_rating": charge_rating,
            "discharge_rating": discharge_rating,
            "charge_efficiency": 1.0,
            "discharge_efficiency": 1.0,
            "thermal_rating": thermal_rating,
            "qmax": max_q_mvar,
            "qmin": min_q_mvar,
            "r": 0.0,
            "x": 0.0,
            "p_loss": -1e-8,
            "q_loss": -1e-8,
            "status": int(storage.at[idx, "in_service"]),
            "standby_loss": 0.0,
        }
        pm["storage"][str(pm_idx)].update(_storage_cost(net, idx))


def _storage_cost(net, idx):
    """
    Returns the active power costs of a storage in the format of the PowerModels gen costs, as a function of
    the storage power ps in MW (charging: ps > 0) in MATPOWER units like the gen costs:

        - "model": 2 (polynomial), "ncost": n, "cost": [c(n-1), ..., c1, c0] with cost = sum(c_i * ps^i)
        - "model": 1 (piecewise linear), "ncost": n, "cost": [p1, c1, ..., pn, cn] with the cost cn at ps = pn

    The ppc generator of the storage is deactivated, so the costs are part of the objective only by
    pm["storage"]. Reactive power costs are not considered, as for the gens. Returns {} if there are no costs.
    """
    if "pwl_cost" in net and len(net.pwl_cost):
        pwl = net.pwl_cost[
            (net.pwl_cost.et == "storage") & (net.pwl_cost.element == idx) & (net.pwl_cost.power_type == "p")
        ]
        if len(pwl):
            cost = [float(c) for c in costs_from_areas(pwl.points.iat[0], 1)]
            return {"model": 1, "ncost": len(cost) // 2, "cost": cost}
    if "poly_cost" in net and len(net.poly_cost):
        poly = net.poly_cost[(net.poly_cost.et == "storage") & (net.poly_cost.element == idx)]
        if len(poly):
            c2, c1, c0 = (float(poly[col].iat[0]) for col in ["cp2_eur_per_mw2", "cp1_eur_per_mw", "cp0_eur"])
            cost = [c2, c1, c0] if c2 else [c1, c0]
            return {"model": 2, "ncost": len(cost), "cost": cost}
    return {}


def _controllable(storage):
    if "controllable" not in storage.columns:
        return pd.Series(False, index=storage.index)
    return storage["controllable"].fillna(False).astype(bool)


def _value(df, idx, column, default):
    """value of an optional column, default if the column does not exist or the value is NaN"""
    if column in df.columns and pd.notna(df.at[idx, column]):
        return float(df.at[idx, column])
    return float(default)


def storage_ppci_gens(net, ppci):
    """
    Returns the pandapower indices and ppci gen indices of the in service controllable storages. In OPF mode
    a controllable storage is a ppc generator.
    """
    lookup = net._pd2ppc_lookups.get("storage_controllable")
    if lookup is None:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    pp_idx = np.flatnonzero(lookup >= 0)
    ppc_gens = lookup[pp_idx]
    # ppci contains only the gens at in service buses
    gen_is = ppci["internal"]["gen_is"]
    in_ppci = gen_is[ppc_gens]
    return pp_idx[in_ppci], (np.cumsum(gen_is) - 1)[ppc_gens[in_ppci]]


def _deactivate_storage_gens(net, ppci, pm):
    """
    The ppc generator of a controllable storage has without max_p_mw / min_p_mw the default limits of
    +-1e9 MW. Its power is given by pm["storage"] instead, so the generator is deactivated.
    """
    for ppci_gen in storage_ppci_gens(net, ppci)[1]:
        pm["gen"][str(ppci_gen + 1)]["gen_status"] = 0


def read_pm_storage_results(net):
    """
    Returns the storage results of a time series optimization as dict storage index -> DataFrame
    (index: time steps, columns: p_mw, q_mvar, soc_mwh, soc_percent). The state of charge of storages
    which are not optimized is NaN.
    """
    res = net.res_ts_opt
    time_steps = res["res_storage.p_mw"].index
    columns = ["p_mw", "q_mvar", "soc_mwh", "soc_percent"]
    return {
        idx: pd.DataFrame(
            {
                col: res[f"res_storage.{col}"][idx]
                if f"res_storage.{col}" in res and idx in res[f"res_storage.{col}"]
                else np.nan
                for col in columns
            },
            index=time_steps,
        )
        for idx in net.storage.index
    }
