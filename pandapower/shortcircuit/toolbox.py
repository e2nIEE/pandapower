# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
from pandapower.auxiliary import _preserve_dtypes

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import networkx as nx
from copy import deepcopy
from scipy.sparse import csr_matrix as sparse

import pandapower as pp
import pandapower.topology as top
import pandapower.shortcircuit as sc
from pandapower.create import _get_index_with_check
from pandapower.topology import create_nxgraph
from pandapower.pypower.idx_bus import BUS_I
from pandapower.pypower.idx_brch import F_BUS, T_BUS, TAP

__all__ = ["detect_power_station_unit", "calc_sc_on_line"]

def detect_power_station_unit(net, mode="auto",
                              max_gen_voltage_kv=80, max_distance_km=0.01):
    """
    Identifies the power station units configuration (gen and trafo) according to IEC 60909.
    Updates the power_station_trafo in the gen table

    INPUT:
        **net** - panpdapower net

        **mode** (str, ("auto", trafo""))

        **max_gen_voltage_level** (float)

        **max_distance_km** (float)
    """
    logger.info("This function will overwrites the value 'power_station_trafo' in gen table")
    net.gen["power_station_trafo"] = np.nan

    required_gen = net.gen.loc[net.bus.loc[net.gen.bus.values, "vn_kv"].values < max_gen_voltage_kv,:]
    required_gen_bus = required_gen.loc[:, "bus"].values

    if mode.lower() == "auto":
        required_trafo = net.trafo.loc[net.bus.loc[net.trafo.lv_bus.values, "vn_kv"].values < max_gen_voltage_kv, :]
    elif mode.lower() == "trafo":
        if "power_station_unit" in net.trafo.columns:
            required_trafo = net.trafo.loc[net.trafo.power_station_unit, :]
        else:
            logger.warning("Using mode 'trafo' requires 'power_station_unit' defined for trafo! Using 'auto' mode instead!")
            required_trafo = net.trafo.loc[net.bus.loc[net.trafo.lv_bus.values, "vn_kv"].values < max_gen_voltage_kv, :]

    else:
        raise UserWarning(f"Unsupported modes: {mode}")

    trafo_lv_bus = net.trafo.loc[required_trafo.index, "lv_bus"].values
    trafo_hv_bus = net.trafo.loc[required_trafo.index, "hv_bus"].values

    g = create_nxgraph(net, respect_switches=True,
                       nogobuses=None, notravbuses=trafo_hv_bus)

    for t_ix in required_trafo.index:
        t_lv_bus = required_trafo.at[t_ix, "lv_bus"]
        bus_dist = pd.Series(nx.single_source_dijkstra_path_length(g, t_lv_bus, weight='weight'))

        connected_bus_at_lv_side = bus_dist[bus_dist < max_distance_km].index.values
        gen_bus_at_lv_side = np.intersect1d(connected_bus_at_lv_side, required_gen_bus)

        if len(gen_bus_at_lv_side) == 1:
            # Check parallel trafo
            if not len(np.intersect1d(connected_bus_at_lv_side, trafo_lv_bus)) == 1:
                raise UserWarning("Failure in power station units detection! Parallel trafos on generator detected!")
            if np.in1d(required_gen_bus, gen_bus_at_lv_side).sum() > 1:
                logger.info("More than 1 gen detected at the lv side of a power station trafo! Will not be considered as power station unit")
                continue
            net.gen.loc[np.in1d(net.gen.bus.values, gen_bus_at_lv_side),
                        "power_station_trafo"] = t_ix


def _create_element_from_exisiting(net, ele_type, ele_ix):
    dtypes = net[ele_type].dtypes
    ps = pd.Series(net[ele_type].loc[ele_ix, :].to_dict(), name=_get_index_with_check(net, ele_type,
                                                                                      None))

    net[ele_type] = pd.concat([net[ele_type], ps.to_frame().T])
    _preserve_dtypes(net[ele_type], dtypes)

    return net[ele_type].index.to_numpy()[-1]


def _create_aux_net(net, line_ix, distance_to_bus0):
    if distance_to_bus0 < 0 or distance_to_bus0 > 1:
        raise UserWarning("Calculating SC current on line failed! distance_to_bus0 must be between 0-1!")
    aux_net = deepcopy(net)

    # Create auxiliary bus
    aux_bus = pp.create_bus(aux_net, vn_kv=aux_net.bus.at[aux_net.line.at[line_ix, "from_bus"], "vn_kv"],
                            name="aux_bus_sc_calc")

    # Create auxiliary line, while preserve the original index
    aux_line0 = _create_element_from_exisiting(aux_net, "line", line_ix)
    aux_line1 = _create_element_from_exisiting(aux_net, "line", line_ix)

    ## Update distance and auxiliary bus
    aux_net.line.at[aux_line0, "length_km"] = distance_to_bus0 * aux_net.line.at[line_ix, "length_km"]
    aux_net.line.at[aux_line0, "to_bus"] = aux_bus
    aux_net.line.at[aux_line0, "name"] += "_aux_line0"

    aux_net.line.at[aux_line1, "length_km"] = (1 - distance_to_bus0) * aux_net.line.at[line_ix, "length_km"]
    aux_net.line.at[aux_line1, "from_bus"] = aux_bus
    aux_net.line.at[aux_line1, "name"] += "_aux_line1"

    ## Disable original line
    aux_net.line.at[line_ix, "in_service"] = False

    ## Update line switch
    for switch_ix in aux_net.switch.query(f" et == 'l' and element == {line_ix}").index:
        aux_switch_ix = _create_element_from_exisiting(aux_net, "switch", switch_ix)
        if aux_net.switch.at[aux_switch_ix, "bus"] == aux_net.line.at[line_ix, "from_bus"]:
            # The from side switch connected to aux_line0
            aux_net.switch.at[aux_switch_ix, "element"] =  aux_line0
        else:
            # The to side switch connected to aux_line1
            aux_net.switch.at[aux_switch_ix, "element"] =  aux_line1
    return aux_net, aux_bus


def calc_sc_on_line(net, line_ix, distance_to_bus0, **kwargs):
    """
    Calculate the shortcircuit in the middle of the line, returns a modified network
    with the shortcircuit calculation results and the bus added

    INPUT:
        **net** - panpdapower net

        **line_ix** (int) - The line of the shortcircuit

        **distance_to_bus0** (float) - The position of the shortcircuit should be between 0-1

    OPTIONAL:
        **kwargs**** - the parameters required for the pandapower calc_sc function
    """
    # Update network
    aux_net, aux_bus = _create_aux_net(net, line_ix, distance_to_bus0)

    pp.rundcpp(aux_net)
    sc.calc_sc(aux_net, bus=aux_bus, **kwargs)

    # Return the new net and the aux bus
    return aux_net, aux_bus


def adjust_V0_for_trafo_tap(ppci, V0, bus_idx):
    branch = ppci["branch"]
    tap = ppci["branch"][:, TAP]
    tap_branch_idx = np.flatnonzero(tap != 1)

    if len(tap_branch_idx) == 0:
        return

    Zbus = ppci["internal"]["Zbus"]
    bus = ppci["bus"]

    f = (branch[:, F_BUS]).real.astype(np.int64)  ## list of "from" buses
    t = (branch[:, T_BUS]).real.astype(np.int64)  ## list of "to" buses
    nl = len(branch)
    nb = len(bus)

    mg = nx.MultiGraph()
    mg.add_nodes_from(bus[:, BUS_I].astype(np.int64).tolist())
    mg.add_edges_from(branch[:, [F_BUS, T_BUS]].real.astype(np.int64).tolist())

    hv_buses = branch[tap_branch_idx, F_BUS].real.astype(np.int64)
    lv_buses = branch[tap_branch_idx, T_BUS].real.astype(np.int64)

    for bh, bl, t in zip(hv_buses, lv_buses, tap[tap_branch_idx]):
        c = top.connected_component(mg, bh, notravbuses={bl})
        c = [cc for cc in c if cc != bl]

        if not np.intersect1d(c, bus_idx):
            for b in c:
                V0[b] = (Zbus[:, b] / Zbus[b, b] * V0[b] * t)[b]
        else:
            c = top.connected_component(mg, bl, notravbuses={bh})
            c = [cc for cc in c if cc != bh]
            if not np.intersect1d(c, bus_idx):
                for b in c:
                    V0[b] = (Zbus[:, b] / Zbus[b, b] * V0[b] / t)[b]

        # V0[c] = (Zbus[:, c] / Zbus[c, c] * V0[c] * tap[tap_branch_idx])[c]

    ## build connection matrix Cft = Cf - Ct for line and from - to buses
    # i = np.r_[range(nl), range(nl)]  ## double set of row indices
    ## connection matrix
    # Cft = sparse((np.r_[np.ones(nl), -np.ones(nl)], (i, np.r_[f, t])), (nl, nb))


def adjacency(ppci):
    branch = ppci["branch"]
    bus = ppci["bus"]
    tap = ppci["branch"][:, TAP].real
    tap_branch_idx = np.flatnonzero(tap != 1)
    all_vertices = bus[:, BUS_I]
    num_vertices = len(bus)
    f = (branch[:, F_BUS]).real.astype(np.int64)  ## list of "from" buses
    t = (branch[:, T_BUS]).real.astype(np.int64)  ## list of "to" buses

    hv_buses = branch[tap_branch_idx, F_BUS].real.astype(np.int64)
    lv_buses = branch[tap_branch_idx, T_BUS].real.astype(np.int64)

    matrix = np.zeros((num_vertices, num_vertices))
    matrix[f, t] = -1
    matrix[t, f] = -1
    matrix[np.diag_indices(num_vertices)] = -matrix.sum(axis=1)

    # matrix[hv_buses, lv_buses] = 1 / tap[tap_branch_idx]
    # matrix[lv_buses, hv_buses] = 1 / tap[tap_branch_idx]
    # matrix[hv_buses, hv_buses] = 1 / np.square(tap[tap_branch_idx])
    #
    # init = np.ones(num_vertices)
    # init[4] = 1.1
    # fake_i = matrix.dot(init)
    # print(fake_i)
    # np.flatnonzero(fake_i)

    return matrix
