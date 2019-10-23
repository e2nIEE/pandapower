# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from scipy.sparse import csr_matrix

import pandapower as pp
from pandapower.estimation.util import set_bb_switch_impedance, reset_bb_switch_impedance
from pandapower.estimation.ppc_conversion import pp2eppci, _initialize_voltage
from pandapower.estimation.results import eppci2pp

from pandapower.pypower.idx_bus import bus_cols, BUS_I
from pandapower.pypower.idx_brch import branch_cols, F_BUS, T_BUS
from pandapower.estimation.idx_bus import VM, P, Q
from pandapower.estimation.idx_brch import IM_FROM, IM_TO, P_FROM, P_TO, Q_FROM, Q_TO

# Get the write column definition
VM, P, Q = VM+bus_cols, P+bus_cols, Q+bus_cols
P_FROM, Q_FROM, P_TO, Q_TO = P_FROM+branch_cols, Q_FROM+branch_cols, P_TO+branch_cols, Q_TO+branch_cols

try:
    import pplog as logging
except ImportError:
    import logging
std_logger = logging.getLogger(__name__)


def find_unobserved_bus(net, zero_injection='auto', fuse_buses_with_bb_switch='all'):
    bus_to_be_fused = None
    if fuse_buses_with_bb_switch != 'all' and not net.switch.empty:
        if isinstance(fuse_buses_with_bb_switch, str):
            raise UserWarning("fuse_buses_with_bb_switch parameter is not correctly initialized")
        elif hasattr(fuse_buses_with_bb_switch, '__iter__'):
            bus_to_be_fused = fuse_buses_with_bb_switch    
        set_bb_switch_impedance(net, bus_to_be_fused)

    net, ppc, eppci = pp2eppci(net, zero_injection=zero_injection,
                               v_start=None, delta_start=None, calculate_voltage_angles=True)
    
    ppci_bus_without_p_meas = eppci.bus[np.isnan(eppci.bus[:, P])]
    ppci_bus_without_q_meas = eppci.bus[np.isnan(eppci.bus[:, Q])]

    ppci_branch_without_p_meas = eppci.branch[np.isnan(eppci.branch[:, P_FROM])&np.isnan(eppci.branch[:, P_TO])]
    ppci_branch_without_q_meas = eppci.branch[np.isnan(eppci.branch[:, Q_FROM])&np.isnan(eppci.branch[:, Q_TO])]
    
    ppci_branch_end_without_p_meas = np.unique(np.r_[ppci_branch_without_p_meas[:, F_BUS], 
                                                     ppci_branch_without_p_meas[:, T_BUS]].real)
    ppci_branch_end_without_q_meas = np.unique(np.r_[ppci_branch_without_q_meas[:, F_BUS], 
                                                     ppci_branch_without_q_meas[:, T_BUS]].real)
    
    ppci_p_unobserved_bus = np.intersect1d(ppci_bus_without_p_meas[:, BUS_I], ppci_branch_end_without_p_meas)
    ppci_q_unobserved_bus = np.intersect1d(ppci_bus_without_q_meas[:, BUS_I], ppci_branch_end_without_q_meas)
    
    bus_lookup = net._pd2ppc_lookups['bus']
    pp_p_unobserved_bus = net.bus[np.in1d(bus_lookup[net.bus.index.values], ppci_p_unobserved_bus)]
    pp_q_unobserved_bus = net.bus[np.in1d(bus_lookup[net.bus.index.values], ppci_q_unobserved_bus)]
    
    return pp_p_unobserved_bus, pp_q_unobserved_bus


def add_virtual_meas_for_unobserved_bus(net, zero_injection='auto', fuse_buses_with_bb_switch='all'):
    pp_p_unobserved_bus, pp_q_unobserved_bus = find_unobserved_bus(net, zero_injection, fuse_buses_with_bb_switch)
    pp.runpp(net)
    net.measurement = net.measurement.reset_index(drop=True)
    for bus_ix in pp_p_unobserved_bus.index:
        pp.create_measurement(net, "p", "bus", value= -net.res_bus.at[bus_ix, "p_mw"], std_dev=0.5, element=bus_ix)
    for bus_ix in pp_q_unobserved_bus.index:    
        pp.create_measurement(net, "q", "bus", value= -net.res_bus.at[bus_ix, "q_mvar"], std_dev=0.5, element=bus_ix)


if __name__ == "__main__":
    from pandapower.estimation.util import add_virtual_meas_from_loadflow
    import pandapower.networks as nw
    import pandapower as pp
    from pandapower.estimation import estimate

    net = nw.case14()
    pp.runpp(net)
    add_virtual_meas_from_loadflow(net)
    net.measurement = net.measurement.iloc[np.random.choice(net.measurement.shape[0], size=20), :]
#    observability_analysis(net)
    try:
        estimate(net)
    except:
        add_virtual_meas_for_unobserved_bus(net)
        status = estimate(net)
    assert status
#
#    bus_to_be_fused = None
#    if fuse_buses_with_bb_switch != 'all' and not net.switch.empty:
#        if isinstance(fuse_buses_with_bb_switch, str):
#            raise UserWarning("fuse_buses_with_bb_switch parameter is not correctly initialized")
#        elif hasattr(fuse_buses_with_bb_switch, '__iter__'):
#            bus_to_be_fused = fuse_buses_with_bb_switch    
#        set_bb_switch_impedance(net, bus_to_be_fused)

    
    
    


