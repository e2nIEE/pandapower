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

if __name__ == "__main__":
    import pandapower.networks as nw
    net = nw.case300()
    detect_power_station_unit(net)
