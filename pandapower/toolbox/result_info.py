# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy
from collections.abc import Collection
from itertools import chain
import logging

import numpy as np
import pandas as pd

from pandapower.auxiliary import pandapowerNet
from pandapower.opf.validate_opf_input import _check_necessary_opf_parameters
from pandapower.toolbox.element_selection import pp_elements

logger = logging.getLogger(__name__)


def lf_info(net, numv=1, numi=2):  # pragma: no cover
    """
    Prints some basic information of the results in a net
    (max/min voltage, max trafo load, max line load).

    Parameters:
        numv (integer, 1): maximal number of printed maximal respectively minimal voltages
        numi (integer, 2): maximal number of printed maximal loading at trafos or lines
    """
    logger.info("Max voltage in vm_pu:")
    for _, r in net.res_bus.sort_values("vm_pu", ascending=False).iloc[:numv].iterrows():
        logger.info("  %s at busidx %s (%s)", r.vm_pu, r.name, net.bus.name.at[r.name])
    logger.info("Min voltage in vm_pu:")
    for _, r in net.res_bus.sort_values("vm_pu").iloc[:numv].iterrows():
        logger.info("  %s at busidx %s (%s)", r.vm_pu, r.name, net.bus.name.at[r.name])
    logger.info("Max loading trafo in %:")
    if net.res_trafo is not None:
        for _, r in net.res_trafo.sort_values("loading_percent", ascending=False).iloc[
                    :numi].iterrows():
            logger.info("  %s loading at trafo %s (%s)", r.loading_percent, r.name,
                        net.trafo.name.at[r.name])
    logger.info("Max loading line in %:")
    for _, r in net.res_line.sort_values("loading_percent", ascending=False).iloc[:numi].iterrows():
        logger.info("  %s loading at line %s (%s)", r.loading_percent, r.name,
                    net.line.name.at[r.name])


def opf_task(net, delta_pq=1e-3, keep=False, log=True):
    """
    Collects some basic inforamtion of the optimal powerflow task und prints them.
    """
    if keep:
        net = copy.deepcopy(net)
    _check_necessary_opf_parameters(net, logger)

    opf_task_overview = {"flexibilities": {},
                         "network_constraints": {},
                         "flexibilities_without_costs": {}}
    _determine_flexibilities_dict(net, opf_task_overview["flexibilities"], delta_pq)
    _determine_network_constraints_dict(net, opf_task_overview["network_constraints"])
    _determine_costs_dict(net, opf_task_overview)

    _check_overlapping_constraints(opf_task_overview)
    if log:
        _log_opf_task_overview(opf_task_overview)

    return opf_task_overview


def _determine_flexibilities_dict(net: pandapowerNet, data: dict, delta_pq: float, **kwargs):
    """
    Determines which flexibilities exists in the net.

    Parameters:
        net: the panpdapower net
        data: to store flexibilities information
        delta_pq: if (abs(max - min) <= delta_pq) the variable is not assumed as flexible, since the range is as small
            as delta_pq (should be small, too).

    Keyword Arguments:
        for comparing constraint columns with numpy.isclose(): rtol and atol
    """
    flex_elements = ["ext_grid", "gen", "dcline", "sgen", "load", "storage"]
    flex_tuple = tuple(zip(flex_elements, [True] * 3 + [False] * 3))

    for elm, controllable_default in flex_tuple:
        for power_type in ["P", "Q"]:
            key = power_type + elm
            if elm != "dcline":
                constraints = {"P": ["min_p_mw", "max_p_mw"],
                               "Q": ["min_q_mvar", "max_q_mvar"]}[power_type]
            else:
                constraints = {"P": ["max_p_mw"],
                               "Q": ["min_q_from_mvar", "max_q_from_mvar",
                                     "min_q_to_mvar", "max_q_to_mvar"]}[power_type]

            # determine indices of controllable elements, continue if no controllable element exists
            if elm in ["ext_grid", "dcline"]:
                controllables = net[elm].index
            elif "controllable" in net[elm].columns:
                controllables = net[elm].index[net[elm].controllable]
            elif controllable_default and net[elm].shape[0]:
                controllables = net[elm].index
            else:
                continue
            if not len(controllables):
                continue

            # consider delta_pq
            if len(constraints) >= 2 and pd.Series(constraints[:2]).isin(net[elm].columns).all():
                controllables = _find_idx_without_numerical_difference(
                    net[elm], constraints[0], constraints[1], delta_pq, idx=controllables,
                    equal_nan=False)
            if elm == "dcline" and power_type == "Q" and len(controllables) and \
                    pd.Series(constraints[2:4]).isin(net[elm].columns).all():
                controllables = _find_idx_without_numerical_difference(
                    net[elm], constraints[2], constraints[3], delta_pq, idx=controllables,
                    equal_nan=False)

            # add missing constraint columns
            for col_to_add in set(constraints) - set(net[elm].columns):
                net[elm][col_to_add] = np.nan

            data[key] = _cluster_same_floats(net[elm].loc[controllables], constraints, **kwargs)
            shorted = [col[:3] if col[:3] in ["min", "max"] else col for col in data[key].columns]
            if len(shorted) == len(set(shorted)):
                data[key].columns = shorted


def _find_idx_without_numerical_difference(df, column1, column2, delta, idx=None, equal_nan=False):
    """
    Returns indices where comlumn1 and column2 have a numerical difference bigger than delta.

    INPUT:
        **df** (DataFrame)

        **column1** (str) - name of first column within df to compare.
        The values of df[column1] must be numericals.

        **column2** (str) - name of second column within df to compare.
        The values of df[column2] must be numericals.

        **delta** (numerical) - value which defines whether indices are returned or not

    OPTIONAL:
        **idx** (iterable, None) - list of indices which should be considered only

        **equal_nan** (bool, False) - if False, indices are included where at least one value in
        df[column1] and df[column2] is NaN

    OUTPUT:
        **index** (pandas.Index) - index within idx where df[column1] and df[column2] deviates by
        at least delta or, if equal_na is True, one value is NaN
    """
    idx = idx if idx is not None else df.index
    idx_isnull = df.index[df[[column1, column2]].isnull().any(axis=1)]
    idx_without_null = idx.difference(idx_isnull)
    idx_no_delta = idx_without_null[(df.loc[idx_without_null, column1] - df.loc[
        idx_without_null, column2]).abs().values <= delta]

    if equal_nan:
        return idx_without_null.difference(idx_no_delta)
    else:
        return idx.difference(idx_no_delta)


def _determine_network_constraints_dict(net: pandapowerNet, data: dict, **kwargs):
    """
    Determines which flexibilities exists in the net.

    Parameters:
        net: the panpdapower net
        data: to store constraints information

    Keyword Arguments:
         for comparing constraint columns with numpy.isclose(): rtol and atol
    """

    const_tuple = [("VMbus", "bus", ["min_vm_pu", "max_vm_pu"]),
                   ("LOADINGline", "line", ["max_loading_percent"]),
                   ("LOADINGtrafo", "trafo", ["max_loading_percent"]),
                   ("LOADINGtrafo3w", "trafo3w", ["max_loading_percent"])
                   ]
    for key, elm, constraints in const_tuple:
        missing_columns = set(constraints) - set(net[elm].columns)
        if net[elm].shape[0] and len(missing_columns) != len(constraints):

            # add missing constraint columns
            for col_to_add in missing_columns:
                net[elm][col_to_add] = np.nan

            data[key] = _cluster_same_floats(net[elm], constraints, **kwargs)
            shorted = [col[:3] if col[:3] in ["min", "max"] else col for col in data[key].columns]
            if len(shorted) == len(set(shorted)):
                data[key].columns = shorted


def _determine_costs_dict(net: pandapowerNet, opf_task_overview: dict):
    """
    Determines which flexibilities do not have costs in the net. Each element is considered as one,
    i.e. if ext_grid 0, for instance,  is flexible in both, P and Q, and has one cost entry for P,
    it is not considered as 'flexibilities_without_costs'.

    Parameters:
        net: the panpdapower net
        opf_task_overview: both, "flexibilities_without_costs" and "flexibilities" must be in opf_task_overview.keys()
    """

    cost_dfs = [df for df in ["poly_cost", "pwl_cost"] if net[df].shape[0]]
    if not len(cost_dfs):
        opf_task_overview["flexibilities_without_costs"] = "all"
        return

    flex_elements = ["ext_grid", "gen", "sgen", "load", "dcline", "storage"]

    for flex_element in flex_elements:

        # determine keys of opf_task_overview["flexibilities"] ending with flex_element
        keys = [power_type + flex_element for power_type in ["P", "Q"] if (
                power_type + flex_element) in opf_task_overview["flexibilities"]]

        # determine indices of all flexibles
        idx_without_cost = set()
        for key in keys:
            idx_without_cost |= set(chain(*opf_task_overview["flexibilities"][key]["index"]))
            # simple alternative without itertools.chain():
        #            idx_without_cost |= {idx for idxs in opf_task_overview["flexibilities"][key][
        #                "index"] for idx in idxs}

        for cost_df in cost_dfs:
            idx_with_cost = set(net[cost_df].element[net[cost_df].et == flex_element].astype(np.int64))
            if len(idx_with_cost - idx_without_cost):
                logger.warning("These " + flex_element + "s have cost data but aren't flexible or" +
                               " have both, poly_cost and pwl_cost: " +
                               str(sorted(idx_with_cost - idx_without_cost)))
            idx_without_cost -= idx_with_cost

        if len(idx_without_cost):
            opf_task_overview["flexibilities_without_costs"][flex_element] = list(idx_without_cost)


def _cluster_same_floats(df: pd.DataFrame, subset: Collection[str] | None = None, **kwargs) -> pd.DataFrame:
    """
    Clusters indices with close values. The values of df[subset] must be numericals.

    Parameters:
        df: DataFrame on which the clustering should be done
        subset: list of columns of df which should be considered to cluster

    Keyword Arguments:
        used for numpy.isclose(): rtol and atol

    Returns:
        DataFrame of clustered values and corresponding lists of indices
    """
    if df.index.duplicated().any():
        logger.error("There are duplicated indices in df. Clusters will be determined but remain " +
                     "ambiguous.")
    if subset is None:
        subset = df.select_dtypes(include=[np.number]).columns.tolist()
    uniq: list[bool] = [not x for x in df.duplicated(subset=subset)]

    # prepare cluster_df
    cluster_df = pd.DataFrame(np.empty((sum(uniq), len(subset) + 1)), columns=["index"] + list(subset))
    cluster_df["index"] = cluster_df["index"].astype(object)
    cluster_df[subset] = df.loc[uniq, subset].values

    if sum(uniq) == df.shape[0]:  # fast return if df has no duplicates
        for i1, idx in enumerate(df.index):
            # assignment is safe because "index" column has been converted to object and the index is now added as list
            cluster_df.at[i1, "index"] = [idx]  # type: ignore[assignment]
        return cluster_df

    i2 = 0
    for i1, uni in enumerate(uniq):
        if uni:
            cluster_df.at[i2, "index"] = list(df.index[np.isclose(  # type: ignore[assignment] # see comment above
                df[subset].astype(float), df[subset].iloc[[i1]].astype(float), equal_nan=True, **kwargs
            ).all(axis=1)])  # type: ignore[call-overload] # for some reason mypy does not like the axis=1 part.
            i2 += 1
    return cluster_df


def _check_overlapping_constraints(opf_task_overview):
    """
    Logs variables where the minimum constraint is bigger than the maximum constraint.
    """
    overlap = []
    for dict_key in ["flexibilities", "network_constraints"]:
        for key, df in opf_task_overview[dict_key].items():
            min_col = [col for col in df.columns if "min" in col]
            max_col = [col for col in df.columns if "max" in col]
            n_col = min(len(min_col), len(max_col))
            for i_col in range(n_col):
                if min_col[i_col].replace("min", "") != max_col[i_col].replace("max", ""):
                    raise AssertionError("min and max are not equal.")
                if (df[min_col[i_col]] > df[max_col[i_col]]).any():
                    overlap.append(key)
    if len(overlap):
        logger.error(
            "At these variables, there is a minimum constraint exceeding the maximum constraint value: " + str(overlap)
        )


def _log_opf_task_overview(opf_task_overview):
    """
    Logs OPF task information.
    """
    s = ""
    for dict_key, data in opf_task_overview.items():
        if isinstance(data, str):
            if dict_key != "flexibilities_without_costs":
                raise AssertionError("dict_key is not 'flexibilities_without_costs'")
            s += "\n\n%s flexibilities without costs" % data
            continue
        else:
            if not isinstance(data, dict):
                raise AssertionError("data is not a dict")
        heading_logged = False
        keys, elms = _get_keys_and_elements_from_opf_task_dict(data)
        for key, elm in zip(keys, elms):
            if elm not in key:
                raise AssertionError('elm should be in key')
            df = data[key]

            if dict_key in ["flexibilities", "network_constraints"]:
                if not df.shape[0]:
                    continue
                if not heading_logged:
                    s += "\n\n%s:" % dict_key
                    heading_logged = True

                # --- logging information
                len_idx = len(list(chain(*df["index"])))
                if df.shape[0] > 1:
                    s += "\n    %ix %s" % (len_idx, key)
                else:
                    if not len(set(df.columns).symmetric_difference({"index", "min", "max"})):
                        s += "\n    %g <= %ix %s (all) <= %g" % (
                            df.loc[0, "min"], len_idx, key, df.loc[0, "max"])
                    else:
                        s += "\n    %ix %s (all) with these constraints:" % (len_idx, key)
                        for col in set(df.columns) - {"index"}:
                            s += " %s=%g" % (col, df.loc[0, col])
            elif dict_key == "flexibilities_without_costs":
                if not heading_logged:
                    s += "\n\n%s:" % dict_key
                    heading_logged = True
                s += "\n%ix %s" % (len(df), key)
            else:
                raise NotImplementedError("Key %s is unknown to this code." % dict_key)
    logger.info(s + "\n")


def _get_keys_and_elements_from_opf_task_dict(dict_):
    keys = list(dict_)
    elms = ["".join(c for c in key if not c.isupper()) for key in keys]
    keys = list(np.array(keys)[np.argsort(elms)])
    elms = sorted(elms)
    return keys, elms


def switch_info(net, sidx):  # pragma: no cover
    """
    Prints what buses and elements are connected by a certain switch.
    """
    switch_type = net.switch.at[sidx, "et"]
    bidx = net.switch.at[sidx, "bus"]
    bus_name = net.bus.at[bidx, "name"]
    eidx = net.switch.at[sidx, "element"]
    if switch_type == "b":
        bus2_name = net.bus.at[eidx, "name"]
        logger.info("Switch %u connects bus %u (%s) with bus %u (%s)" % (sidx, bidx, bus_name,
                                                                         eidx, bus2_name))
    elif switch_type == "l":
        line_name = net.line.at[eidx, "name"]
        logger.info("Switch %u connects bus %u (%s) with line %u (%s)" % (sidx, bidx, bus_name,
                                                                          eidx, line_name))
    elif switch_type == "t":
        trafo_name = net.trafo.at[eidx, "name"]
        logger.info("Switch %u connects bus %u (%s) with trafo %u (%s)" % (sidx, bidx, bus_name,
                                                                           eidx, trafo_name))


def overloaded_lines(net, max_load=100):
    """
    Returns the results for all lines with loading_percent > max_load or None, if
    there are none.
    """
    if net.converged:
        return net["res_line"].index[net["res_line"]["loading_percent"] > max_load]
    else:
        raise UserWarning("The last loadflow terminated erratically, results are invalid!")


def violated_buses(net, min_vm_pu, max_vm_pu):
    """
    Returns all bus indices where vm_pu is not within min_vm_pu and max_vm_pu or returns None, if
    there are none of those buses.
    """
    if net.converged:
        return net["bus"].index[(net["res_bus"]["vm_pu"] < min_vm_pu) |
                                (net["res_bus"]["vm_pu"] > max_vm_pu)]
    else:
        raise UserWarning("The last loadflow terminated erratically, results are invalid!")


def clear_result_tables(net):
    """
    Clears all ``res_`` DataFrames in net.
    """
    for key in net:
        if isinstance(net[key], pd.DataFrame) and key.startswith("res") and net[key].shape[0]:
            net[key] = net[key].drop(net[key].index)


def compute_switch_flows(net):
    """Compute power flow through zero-impedance bus-bus switches via nodal balance.

    After a converged load flow, switches with ``z_ohm=0`` (the default) have
    ``NaN`` values in ``res_switch`` because Pandapower fuses the adjacent buses
    into a single internal node.  This function reconstructs the individual
    switch flows by calculating the net local injection at each bus within a
    fused group and propagating the residual through the switch tree.

    The function writes ``p_from_mw``, ``q_from_mvar``, ``p_to_mw``,
    ``q_to_mvar``, and ``i_ka`` into ``net.res_switch`` for every closed
    bus-bus switch that has ``z_ohm <= 0``.  Results for open switches are set
    to zero.  Switches that already have results (``z_ohm > 0``) are not
    modified.

    **Assumption:** Within each fused-bus group the zero-impedance switches
    must form a *tree* (no parallel zero-impedance paths between the same pair
    of buses).  If a cycle is detected a ``ValueError`` is raised, since the
    flow split is physically indeterminate without impedance information.

    Parameters
    ----------
    net : pandapowerNet
        A pandapower network with valid load flow results
        (``net.converged is True``).

    Raises
    ------
    UserWarning
        If ``net.converged`` is ``False``.
    ValueError
        If zero-impedance bus-bus switches form a cycle within a fused group.

    Examples
    --------
    >>> import pandapower as pp
    >>> import pandapower.networks as pn
    >>> net = pn.example_simple()
    >>> pp.runpp(net)
    >>> from pandapower.toolbox import compute_switch_flows
    >>> compute_switch_flows(net)
    >>> net.res_switch  # p_from_mw, q_from_mvar etc. now populated
    """
    from collections import defaultdict

    if not net.converged:
        raise UserWarning("Power flow did not converge, results are invalid.")

    if len(net.switch) == 0:
        return

    bus_lookup = getattr(net, "_pd2ppc_lookups", {}).get("bus")
    if bus_lookup is None:
        return

    # Identify fused-bus groups: ppc_bus_id -> set of original bus indices
    fused_groups = defaultdict(set)
    for orig_bus, ppc_bus in enumerate(bus_lookup):
        fused_groups[int(ppc_bus)].add(orig_bus)

    multi_groups = {k: v for k, v in fused_groups.items() if len(v) > 1}
    if not multi_groups:
        return

    # Per-bus net local consumption: positive = power leaving the bus
    bus_p = defaultdict(float)
    bus_q = defaultdict(float)

    _bus_element_names = [
        ("load", 1.0), ("motor", 1.0),
        ("shunt", 1.0), ("ward", 1.0), ("xward", 1.0),
        ("ext_grid", -1.0), ("gen", -1.0), ("sgen", -1.0), ("storage", -1.0),
    ]
    for tbl_name, sign in _bus_element_names:
        tbl = net.get(tbl_name)
        res = net.get("res_%s" % tbl_name)
        if tbl is None or res is None or len(tbl) == 0 or len(res) == 0:
            continue
        in_service = tbl["in_service"].values if "in_service" in tbl.columns else np.ones(len(tbl), dtype=bool)
        buses = tbl["bus"].values
        for i, idx in enumerate(tbl.index):
            if not in_service[i]:
                continue
            b = int(buses[i])
            try:
                bus_p[b] += sign * float(res.at[idx, "p_mw"])
                bus_q[b] += sign * float(res.at[idx, "q_mvar"])
            except (KeyError, ValueError):
                pass

    # Map every bus to its fused group
    bus_to_group = {}
    for grp_id, buses in fused_groups.items():
        for b in buses:
            bus_to_group[b] = grp_id

    # Add power carried by branches that leave the fused group
    _branch_specs = [
        ("line", "res_line", [("from_bus", "p_from_mw", "q_from_mvar"),
                              ("to_bus", "p_to_mw", "q_to_mvar")]),
        ("trafo", "res_trafo", [("hv_bus", "p_hv_mw", "q_hv_mvar"),
                                ("lv_bus", "p_lv_mw", "q_lv_mvar")]),
        ("trafo3w", "res_trafo3w", [("hv_bus", "p_hv_mw", "q_hv_mvar"),
                                    ("mv_bus", "p_mv_mw", "q_mv_mvar"),
                                    ("lv_bus", "p_lv_mw", "q_lv_mvar")]),
        ("impedance", "res_impedance", [("from_bus", "p_from_mw", "q_from_mvar"),
                                        ("to_bus", "p_to_mw", "q_to_mvar")]),
        ("dcline", "res_dcline", [("from_bus", "p_from_mw", "q_from_mvar"),
                                  ("to_bus", "p_to_mw", "q_to_mvar")]),
    ]
    for tbl_name, res_name, bus_pairs in _branch_specs:
        tbl = net.get(tbl_name)
        res = net.get(res_name)
        if tbl is None or res is None or len(tbl) == 0 or len(res) == 0:
            continue
        in_service = tbl["in_service"].values if "in_service" in tbl.columns else np.ones(len(tbl), dtype=bool)
        for i, idx in enumerate(tbl.index):
            if not in_service[i]:
                continue
            ends = [(int(tbl.at[idx, bc]), pc, qc) for bc, pc, qc in bus_pairs]
            for j, (this_bus, p_col, q_col) in enumerate(ends):
                this_grp = bus_to_group.get(this_bus)
                if this_grp is None or this_grp not in multi_groups:
                    continue
                other_buses = [ends[k][0] for k in range(len(ends)) if k != j]
                if all(bus_to_group.get(ob) == this_grp for ob in other_buses):
                    continue
                try:
                    bus_p[this_bus] += float(res.at[idx, p_col])
                    bus_q[this_bus] += float(res.at[idx, q_col])
                except (KeyError, ValueError):
                    pass

    # Index closed zero-impedance bus-bus switches by fused group
    sw_by_group = defaultdict(list)
    z_ohm = net.switch["z_ohm"].values if "z_ohm" in net.switch.columns else np.zeros(len(net.switch))
    for sw_idx in net.switch.index:
        row = net.switch.loc[sw_idx]
        if row["et"] != "b" or not row["closed"] or z_ohm[sw_idx] > 0:
            continue
        a = int(row["bus"])
        b = int(row["element"])
        grp = bus_to_group.get(a)
        if grp is not None and grp in multi_groups:
            sw_by_group[grp].append((sw_idx, a, b))

    computed_switches = set()

    # For each fused group, build the coupler subgraph and compute flows
    for grp_id, grp_buses in multi_groups.items():
        couplers = sw_by_group.get(grp_id, [])
        if not couplers:
            continue

        # Skip de-energized fused groups (vm ≈ 0)
        sample_bus = next(iter(grp_buses))
        if sample_bus in net.res_bus.index:
            vm_pu = net.res_bus.at[sample_bus, "vm_pu"]
            if vm_pu == 0 or np.isnan(vm_pu):
                continue

        adj = defaultdict(list)
        coupler_buses = set()
        for sw_idx, a, b in couplers:
            adj[a].append((b, sw_idx))
            adj[b].append((a, sw_idx))
            coupler_buses.add(a)
            coupler_buses.add(b)

        # A tree with N nodes has exactly N-1 edges; more means a cycle
        if len(couplers) >= len(coupler_buses):
            raise ValueError(
                "Zero-impedance bus-bus switches form a cycle in fused "
                "group containing buses %s. The flow split is "
                "indeterminate without impedance values." % sorted(grp_buses))

        # DFS to build tree order
        root = next(iter(grp_buses))
        visited = set()
        stack = [(root, None, None)]
        order = []
        while stack:
            bus, parent, parent_sw = stack.pop()
            if bus in visited:
                continue
            visited.add(bus)
            order.append((bus, parent, parent_sw))
            for nb, sw_idx in adj.get(bus, []):
                if nb not in visited:
                    stack.append((nb, bus, sw_idx))

        # Accumulate subtree demand from leaves to root
        subtree_p = {}
        subtree_q = {}
        for bus, parent, parent_sw in reversed(order):
            sp = bus_p.get(bus, 0.0)
            sq = bus_q.get(bus, 0.0)
            for nb, sw_idx in adj.get(bus, []):
                if nb != parent and nb in subtree_p:
                    sp += subtree_p[nb]
                    sq += subtree_q[nb]
            subtree_p[bus] = sp
            subtree_q[bus] = sq
            if parent_sw is not None:
                sw_bus_col = int(net.switch.at[parent_sw, "bus"])
                sw_elem_col = int(net.switch.at[parent_sw, "element"])
                if bus == sw_elem_col:
                    p_from, q_from = sp, sq
                else:
                    p_from, q_from = -sp, -sq
                p_to, q_to = -p_from, -q_from

                net.res_switch.at[parent_sw, "p_from_mw"] = p_from
                net.res_switch.at[parent_sw, "q_from_mvar"] = q_from
                net.res_switch.at[parent_sw, "p_to_mw"] = p_to
                net.res_switch.at[parent_sw, "q_to_mvar"] = q_to

                # Derive current from apparent power and bus voltage
                s_mva = np.sqrt(p_from ** 2 + q_from ** 2)
                b_from = int(net.switch.at[parent_sw, "bus"])
                vm_pu = net.res_bus.at[b_from, "vm_pu"] if b_from in net.res_bus.index else np.nan
                vn_kv = net.bus.at[b_from, "vn_kv"] if b_from in net.bus.index else np.nan
                vm_kv = vm_pu * vn_kv
                if vm_kv > 0:
                    i_ka = s_mva / (vm_kv * np.sqrt(3))
                else:
                    i_ka = np.nan
                net.res_switch.at[parent_sw, "i_ka"] = i_ka
                computed_switches.add(parent_sw)

    if computed_switches and "in_ka" in net.switch.columns and \
            "loading_percent" in net.res_switch.columns:
        for sw_idx in computed_switches:
            in_val = net.switch.at[sw_idx, "in_ka"]
            if not np.isnan(in_val) and in_val > 0:
                i_val = net.res_switch.at[sw_idx, "i_ka"]
                net.res_switch.at[sw_idx, "loading_percent"] = i_val / in_val * 100


def res_power_columns(element_type, side=0):
    """Returns columns names of result tables for active and reactive power

    Parameters
    ----------
    element_type : str
        name of element table, e.g. "gen"
    side : typing.Union[int, str], optional
        Defines for branch elements which branch side is considered, by default 0

    Returns
    -------
    list[str]
        columns names of result tables for active and reactive power

    Examples
    --------
    >>> res_power_columns("gen")
    ["p_mw", "q_mvar"]
    >>> res_power_columns("line", "from")
    ["p_from_mw", "q_from_mvar"]
    >>> res_power_columns("line", 0)
    ["p_from_mw", "q_from_mvar"]
    >>> res_power_columns("line", "all")
    ["p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar"]
    """
    if element_type in pp_elements(branch_elements=False, other_elements=False):
        return ["p_mw", "q_mvar"]
    elif element_type in pp_elements(bus=False, bus_elements=False, other_elements=False):
        if isinstance(side, int):
            if element_type == "trafo":
                side_options = {0: "hv", 1: "lv"}
            elif element_type == "trafo3w":
                side_options = {0: "hv", 1: "mv", 2: "lv"}
            else:
                side_options = {0: "from", 1: "to"}
            side = side_options[side]
        if side != "all":
            return [f"p_{side}_mw", f"q_{side}_mvar"]
        else:
            cols = res_power_columns(element_type, side=0) + \
                res_power_columns(element_type, side=1)
            if element_type == "trafo3w":
                cols += res_power_columns(element_type, side=2)
            return cols
    else:
        raise ValueError(f'{element_type=} cannot be considered by res_power_columns().')