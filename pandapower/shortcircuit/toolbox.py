# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import networkx as nx
from copy import deepcopy

import pandapower as pp
from pandapower.shortcircuit import calc_sc
from pandapower.create import _get_index_with_check
from pandapower.topology import create_nxgraph


def detect_power_station_unit(net, mode="auto",
                              max_gen_voltage_kv=80, max_distance_km=0.01):
    logger.info("This function will overwrites the value 'power_station_trafo' in gen table")
    net.gen["power_station_trafo"] = np.nan

    required_gen = net.gen.loc[net.bus.loc[net.gen.bus.values, "vn_kv"].values < max_gen_voltage_kv,:]
    gen_bus = required_gen.loc[:, "bus"].values

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
        gen_bus_at_lv_side = np.intersect1d(connected_bus_at_lv_side, gen_bus)

        if len(gen_bus_at_lv_side) == 1:
            # Check parallel trafo
            assert len(np.intersect1d(connected_bus_at_lv_side, trafo_lv_bus)) == 1

            # Check parallel gen

            net.gen.loc[np.in1d(net.gen.bus.values, gen_bus_at_lv_side),
                        "power_station_trafo"] = t_ix

    # # Check parallel trafo, not allowed in the phase
    # if len(np.unique(ps_trafo_lv_bus_ppc)) != len(ps_trafo_lv_bus_ppc):
    #     raise UserWarning("Parallel trafos not allowed in power station units")

    # # Check parallel gen, not allowed in the phase
    # if len(np.unique(gen_bus_ppc[ps_gen_ppc_mask])) != len(gen_bus_ppc[ps_gen_ppc_mask]):
    #     raise UserWarning("Parallel gens not allowed in power station units")

    # # Check wrongly defined trafo (lv side no gen)
    # if np.any(~np.isin(ps_trafo_lv_bus_ppc, ps_units_bus_ppc)):
    #     raise UserWarning("Some power station units defined wrong! No gen detected at LV side!")



def _create_element_from_exisiting(net, ele_type, ele_ix):
    net[ele_type] = net[ele_type].append(pd.Series(net[ele_type].loc[ele_ix, :].to_dict(),
                                         name=_get_index_with_check(net, ele_type, None)))
    return net[ele_type].index.to_numpy()[-1]


def _create_aux_net(net, line_ix, distance_to_bus0):
    assert distance_to_bus0 > 0 and distance_to_bus0 < 1
    aux_net = deepcopy(net)

    # # Create auxiliary bus
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
    # Update network
    aux_net, aux_bus = _create_aux_net(net, line_ix, distance_to_bus0)

    calc_sc(aux_net, bus=aux_bus, **kwargs)

    # Return the new net and the aux bus
    return aux_net, aux_bus


if __name__ == "__main__":
    import pandapower.networks as nw
    net = nw.case300()
    detect_power_station_unit(net)

    import pandapower.networks as nw
    # vn_kv=10.5, xdss_pu=0.2, rdss_pu=0.001, cos_phi=0.8, p_mw=0.1, sn_mva=2.5
    # net = nw.create_cigre_network_mv()
    # net = nw.case118()
    # net = nw.case2869pegase()
    # net = nw.case1354pegase()
    net = nw.case9241pegase()
    net.ext_grid['s_sc_max_mva'] = 1000
    net.ext_grid['rx_max'] = 0.1
    net.gen["vn_kv"] = net.bus.loc[net.gen.bus.values, "vn_kv"].values
    net.gen["rdss_pu"] = 0.01
    net.gen["xdss_pu"] = 0.1
    net.gen["cos_phi"] = 0.8
    net.gen["sn_mva"] = net.gen.p_mw.values + 0.01

    net.sgen['k'] = 1.
    net.sgen["sn_mva"] = net.sgen.p_mw.values + 0.01
    net.sgen = net.sgen.iloc[0:0, :]

    net_no_inv = deepcopy(net)
    net_all = deepcopy(net)


    # for _ in range(10):
    # calc_sc(net, branch_results=True, return_all_currents=True, ip=False, ith=True, bus=[10,60,50])
    for _ in range(10):
        pass
        # calc_sc(net, branch_results=True, return_all_currents=True, ip=False, ith=True, bus=[10,60,50], inverse_y=False)
    # calc_sc(net_all, branch_results=True, return_all_currents=True, ip=True, ith=True)
    # # calc_single_sc(net,254)
    # calc_single_sc(net_no_inv, 6, with_y_inv=True)
    # n = [net]
    # n_no_inv = [net_no_inv]
    # n_all = [net_all]

# %timeit calc_sc(net, ip=True, ith=False, bus=[15, 50, 986, 5412], branch_results=True, inverse_y=False)
# %timeit calc_sc(net, ip=False, ith=False, bus=[1,3,6], inverse_y=True)
# %timeit calc_sc(net, ip=True, ith=True)

# %timeit calc_sc(net, branch_results=True, return_all_currents=True)
# %timeit calc_single_sc(net, 8, inverse_y=False)
# %timeit calc_single_sc(net_no_inv, 8, with_y_inv=False)


#     # print(net.res_bus)
#     # print(net.res_bus_sc)
#     # print(net_no_inv.res_bus_sc)

#     print((net.res_bus_sc - net_no_inv.res_bus_sc).max())


#     # print(net.res_line_sc)
#     # print(net_no_inv.res_line_sc)
#     delta = net.res_line_sc - net_no_inv.res_line_sc
    # n = [net]

    # aux_net, aux_bus = _create_aux_net(net, 4, 0.2)
    # pp.runpp(net)
    # calc_single_sc(aux_net,aux_bus)
