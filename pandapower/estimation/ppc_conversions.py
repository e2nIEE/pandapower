# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pandas as pd
from pandapower.auxiliary import _select_is_elements_numba, _add_ppc_options, _add_auxiliary_elements
from pandapower.pd2ppc import _pd2ppc
from pandapower.estimation.idx_bus import *
from pandapower.estimation.idx_brch import *
from pandapower.pypower.idx_brch import branch_cols
from pandapower.pypower.idx_bus import bus_cols
from pandapower.pf.run_newton_raphson_pf import _run_dc_pf
from pandapower.run import rundcpp
from pandapower.build_branch import get_is_lines
from pandapower.create import create_buses, create_line_from_parameters

try:
    import pplog as logging
except ImportError:
    import logging
std_logger = logging.getLogger(__name__)

AUX_BUS_NAME, AUX_LINE_NAME, AUX_SWITCH_NAME =\
    "aux_bus_se", "aux_line_se", "aux_bbswitch_se"

def _add_aux_elements_for_bb_switch(net, bus_to_be_fused):
    """
    Add auxiliary elements (bus, bb switch, line) to the pandapower net to avoid
    automatic fuse of buses connected with bb switch with elements on it
    :param net: pandapower net
    :return: None
    """
    def get_bus_branch_mapping(net, bus_to_be_fused):
        bus_with_elements = set(net.load.bus).union(set(net.sgen.bus)).union(
                        set(net.shunt.bus)).union(set(net.gen.bus)).union(
                        set(net.ext_grid.bus)).union(set(net.ward.bus)).union(
                        set(net.xward.bus))
#        bus_with_pq_measurement = set(net.measurement[(net.measurement.measurement_type=='p')&(net.measurement.element_type=='bus')].element.values)
#        bus_with_elements = bus_with_elements.union(bus_with_pq_measurement)
        
        bus_ppci = pd.DataFrame(data=net._pd2ppc_lookups['bus'], columns=["bus_ppci"])
        bus_ppci['bus_with_elements'] = bus_ppci.index.isin(bus_with_elements)
        existed_bus = bus_ppci[bus_ppci.index.isin(net.bus.index)]
        bus_ppci['vn_kv'] = net.bus.loc[existed_bus.index, 'vn_kv']
        ppci_bus_with_elements = bus_ppci.groupby('bus_ppci')['bus_with_elements'].sum()
        bus_ppci.loc[:, 'elements_in_cluster'] = ppci_bus_with_elements[bus_ppci['bus_ppci'].values].values 
        bus_ppci['bus_to_be_fused'] = False
        if bus_to_be_fused is not None:
            bus_ppci.loc[bus_to_be_fused, 'bus_to_be_fused'] = True
            bus_cluster_to_be_fused_mask = bus_ppci.groupby('bus_ppci')['bus_to_be_fused'].any()
            bus_ppci.loc[bus_cluster_to_be_fused_mask[bus_ppci['bus_ppci'].values].values, 'bus_to_be_fused'] = True    
        return bus_ppci

    # find the buses which was fused together in the pp2ppc conversion with elements on them
    # the first one will be skipped
    rundcpp(net)
    bus_ppci_mapping = get_bus_branch_mapping(net, bus_to_be_fused)
    bus_to_be_handled = bus_ppci_mapping[(bus_ppci_mapping ['elements_in_cluster']>=2)&\
                                          bus_ppci_mapping ['bus_with_elements']&\
                                          (~bus_ppci_mapping ['bus_to_be_fused'])]
    bus_to_be_handled = bus_to_be_handled[bus_to_be_handled['bus_ppci'].duplicated(keep='first')]

    # create auxiliary buses for the buses need to be handled
    aux_bus_index = create_buses(net, bus_to_be_handled.shape[0], bus_to_be_handled.vn_kv.values, 
                                 name=AUX_BUS_NAME)
    bus_aux_mapping = pd.Series(aux_bus_index, index=bus_to_be_handled.index.values)

    # create auxiliary switched and disable original switches connected to the related buses
    net.switch.loc[:, 'original_closed'] = net.switch.loc[:, 'closed']
    switch_to_be_replaced_sel = ((net.switch.et == 'b') &
                                 (net.switch.element.isin(bus_to_be_handled.index) | 
                                  net.switch.bus.isin(bus_to_be_handled.index)))
    net.switch.loc[switch_to_be_replaced_sel, 'closed'] = False

    # create aux switches with selecting the existed switches
    aux_switch = net.switch.loc[switch_to_be_replaced_sel, ['bus', 'closed', 'element', 
                                                            'et', 'name', 'original_closed', 'z_ohm']]
    aux_switch.loc[:,'name'] = AUX_SWITCH_NAME
    
    # replace the original bus with the correspondent auxiliary bus
    bus_to_be_replaced = aux_switch.loc[aux_switch.bus.isin(bus_to_be_handled.index), 'bus']
    element_to_be_replaced = aux_switch.loc[aux_switch.element.isin(bus_to_be_handled.index), 'element']
    aux_switch.loc[bus_to_be_replaced.index, 'bus'] =\
        bus_aux_mapping[bus_to_be_replaced].values.astype(int)
    aux_switch.loc[element_to_be_replaced.index, 'element'] =\
        bus_aux_mapping[element_to_be_replaced].values.astype(int)
    aux_switch['closed'] = aux_switch['original_closed']

    net.switch = net.switch.append(aux_switch, ignore_index=True)
    # PY34 compatibility
#    net.switch = net.switch.append(aux_switch, ignore_index=True, sort=False)

    # create auxiliary lines as small impedance
    for bus_ori, bus_aux in bus_aux_mapping.iteritems():
        create_line_from_parameters(net, bus_ori, bus_aux, length_km=1, name=AUX_LINE_NAME,
                                    r_ohm_per_km=0.15, x_ohm_per_km=0.2, c_nf_per_km=0, max_i_ka=1)


def _drop_aux_elements_for_bb_switch(net):
    """
    Remove auxiliary elements (bus, bb switch, line) added by
    _add_aux_elements_for_bb_switch function
    :param net: pandapower net
    :return: None
    """
    # Remove auxiliary switches and restore switch status
    net.switch = net.switch[net.switch.name!=AUX_SWITCH_NAME]
    if 'original_closed' in net.switch.columns:
        net.switch.loc[:, 'closed'] = net.switch.loc[:, 'original_closed']
        net.switch.drop('original_closed', axis=1, inplace=True)
    
    # Remove auxiliary buses, lines in net and result
    for key in net.keys():
        if key.startswith('res_bus'):
            net[key] = net[key].loc[(net.bus.name != AUX_BUS_NAME).values, :]
        if key.startswith('res_line'):
            net[key] = net[key].loc[(net.line.name != AUX_LINE_NAME).values, :]
    net.bus = net.bus.loc[(net.bus.name != AUX_BUS_NAME).values, :]
    net.line = net.line.loc[(net.line.name != AUX_LINE_NAME).values, :]


def _init_ppc(net, v_start, delta_start, calculate_voltage_angles):
    # select elements in service and convert pandapower ppc to ppc
    net._options = {}
    _add_ppc_options(net, check_connectivity=False, init_vm_pu=v_start, init_va_degree=delta_start,
                     trafo_model="pi", mode="pf", enforce_q_lims=False,
                     calculate_voltage_angles=calculate_voltage_angles, switch_rx_ratio=2,
                     recycle=dict(_is_elements=False, ppc=False, Ybus=False))
    net["_is_elements"] = _select_is_elements_numba(net)
    _add_auxiliary_elements(net)
    ppc, ppci = _pd2ppc(net)

    # do dc power flow for phase shifting transformers
    if np.any(net.trafo.shift_degree):
        vm_backup = ppci["bus"][:, 7].copy()
        ppci["bus"][:, [2, 3]] = 0.
        ppci = _run_dc_pf(ppci)
        ppci["bus"][:, 7] = vm_backup

    return ppc, ppci


def _add_measurements_to_ppc(net, ppci, zero_injection):
    """
    Add pandapower measurements to the ppci structure by adding new columns
    :param net: pandapower net
    :param ppci: generated ppci
    :return: ppc with added columns
    """
    meas = net.measurement.copy(deep=False)
    meas["side"] = meas.apply(lambda row:
                              net['line']["{}_bus".format(row["side"])].loc[row["element"]] if
                              row["side"] in ("from", "to") else
                              net[row["element_type"]][row["side"]+'_bus'].loc[row["element"]] if
                              row["side"] in ("hv", "mv", "lv") else row["side"], axis=1)

    map_bus = net["_pd2ppc_lookups"]["bus"]
    meas_bus = meas[(meas['element_type'] == 'bus')]
    if (map_bus[meas_bus['element'].values.astype(int)] >= ppci["bus"].shape[0]).any():
        std_logger.warning("Measurement defined in pp-grid does not exist in ppci! Will be deleted!")
        meas_bus = meas_bus[map_bus[meas_bus['element'].values.astype(int)] < ppci["bus"].shape[0]]

    # mapping to dict instead of np array ensures good performance for large indices
    # (e.g., 999999999 requires a large np array even if there are only 2 buses)
    # downside is loop comprehension to access the map
    map_line, map_trafo, map_trafo3w = None, None, None
    branch_mask = ppci['internal']['branch_is']
    if "line" in net["_pd2ppc_lookups"]["branch"]:
        map_line = {line_ix: br_ix for line_ix, br_ix in
                    zip(net.line.index, range(*net["_pd2ppc_lookups"]["branch"]["line"])) if branch_mask[br_ix]}

    if "trafo" in net["_pd2ppc_lookups"]["branch"]:
        trafo_ix_start, trafo_ix_end = net["_pd2ppc_lookups"]["branch"]["trafo"]
        trafo_ix_offset = np.sum(~branch_mask[:trafo_ix_start])
        trafo_ix_start, trafo_ix_end = trafo_ix_start - trafo_ix_offset, trafo_ix_end - trafo_ix_offset
        map_trafo = {trafo_ix: br_ix for trafo_ix, br_ix in
                     zip(net.trafo.index, range(trafo_ix_start, trafo_ix_end))
                     if branch_mask[br_ix+trafo_ix_offset]}

    if "trafo3w" in net["_pd2ppc_lookups"]["branch"]:
        trafo3w_ix_start, trafo3w_ix_end = net["_pd2ppc_lookups"]["branch"]["trafo3w"]
        trafo3w_ix_offset = np.sum(~branch_mask[:trafo3w_ix_start])
        num_trafo3w = net.trafo3w.shape[0]
        trafo3w_ix_start, trafo3w_ix_end = trafo3w_ix_start - trafo3w_ix_offset, trafo3w_ix_end - trafo3w_ix_offset
        map_trafo3w = {trafo3w_ix: {'hv': br_ix, 'mv': br_ix+num_trafo3w, 'lv': br_ix+2*num_trafo3w}
                        for trafo3w_ix, br_ix in
                       zip(net.trafo3w.index, range(trafo3w_ix_start, trafo3w_ix_start+num_trafo3w))
                       if branch_mask[br_ix+trafo3w_ix_offset]}

    # set measurements for ppc format
    # add 9 columns to ppc[bus] for Vm, Vm std dev, P, P std dev, Q, Q std dev,
    # pandapower measurement indices V, P, Q
    bus_append = np.full((ppci["bus"].shape[0], bus_cols_se), np.nan, dtype=ppci["bus"].dtype)
    
    v_measurements = meas_bus[(meas_bus.measurement_type == "v")]
    if len(v_measurements):
        bus_positions = map_bus[v_measurements.element.values.astype(int)]
        bus_append[bus_positions, VM] = v_measurements.value.values
        bus_append[bus_positions, VM_STD] = v_measurements.std_dev.values
        bus_append[bus_positions, VM_IDX] = v_measurements.index.values

    p_measurements = meas_bus[(meas_bus.measurement_type == "p")]
    if len(p_measurements):
        bus_positions = map_bus[p_measurements.element.values.astype(int)]
        unique_bus_positions = np.unique(bus_positions)
        if len(unique_bus_positions) < len(bus_positions):
            std_logger.warning("P Measurement duplication will be automatically merged!")
            for bus in unique_bus_positions:
                p_meas_on_bus = p_measurements.iloc[np.argwhere(bus_positions==bus).ravel(), :]
                bus_append[bus, P] = p_meas_on_bus.value.sum()
                bus_append[bus, P_STD] = p_meas_on_bus.std_dev.max()
                bus_append[bus, P_IDX] = p_meas_on_bus.index[0]
        else:
            bus_append[bus_positions, P] = p_measurements.value.values
            bus_append[bus_positions, P_STD] = p_measurements.std_dev.values
            bus_append[bus_positions, P_IDX] = p_measurements.index.values

    q_measurements = meas_bus[(meas_bus.measurement_type == "q")]
    if len(q_measurements):
        bus_positions = map_bus[q_measurements.element.values.astype(int)]
        unique_bus_positions = np.unique(bus_positions)
        if len(unique_bus_positions) < len(bus_positions):
            std_logger.warning("Q Measurement duplication will be automatically merged!")
            for bus in unique_bus_positions:
                q_meas_on_bus = q_measurements.iloc[np.argwhere(bus_positions==bus).ravel(), :]
                bus_append[bus, Q] = q_meas_on_bus.value.sum()
                bus_append[bus, Q_STD] = q_meas_on_bus.std_dev.max()
                bus_append[bus, Q_IDX] = q_meas_on_bus.index[0]
        else:
            bus_positions = map_bus[q_measurements.element.values.astype(int)]
            bus_append[bus_positions, Q] = q_measurements.value.values
            bus_append[bus_positions, Q_STD] = q_measurements.std_dev.values
            bus_append[bus_positions, Q_IDX] = q_measurements.index.values

    #add zero injection measurement and labels defined in parameter zero_injection
    bus_append = _add_zero_injection(net, ppci, bus_append, zero_injection)
    # add virtual measurements for artificial buses, which were created because
    # of an open line switch. p/q are 0. and std dev is 1. (small value)
    new_in_line_buses = np.setdiff1d(np.arange(ppci["bus"].shape[0]), map_bus[map_bus >= 0])
    bus_append[new_in_line_buses, 2] = 0.
    bus_append[new_in_line_buses, 3] = 1.
    bus_append[new_in_line_buses, 4] = 0.
    bus_append[new_in_line_buses, 5] = 1.

    # add 15 columns to mpc[branch] for Im_from, Im_from std dev, Im_to, Im_to std dev,
    # P_from, P_from std dev, P_to, P_to std dev, Q_from, Q_from std dev,  Q_to, Q_to std dev,
    # pandapower measurement index I, P, Q
    branch_append = np.full((ppci["branch"].shape[0], branch_cols_se),
                            np.nan, dtype=ppci["branch"].dtype)

    if map_line is not None:
        i_measurements = meas[(meas.measurement_type == "i") & (meas.element_type == "line") &\
                              meas.element.isin(map_line)]
        if len(i_measurements):
            meas_from = i_measurements[(i_measurements.side.values.astype(int) ==
                                        net.line.from_bus[i_measurements.element]).values]
            meas_to = i_measurements[(i_measurements.side.values.astype(int) ==
                                      net.line.to_bus[i_measurements.element]).values]
            ix_from = [map_line[l] for l in meas_from.element.values.astype(int)]
            ix_to = [map_line[l] for l in meas_to.element.values.astype(int)]
            i_ka_to_pu_from = (net.bus.vn_kv[meas_from.side]).values * 1e3
            i_ka_to_pu_to = (net.bus.vn_kv[meas_to.side]).values * 1e3
            branch_append[ix_from, IM_FROM] = meas_from.value.values * i_ka_to_pu_from
            branch_append[ix_from, IM_FROM_STD] = meas_from.std_dev.values * i_ka_to_pu_from
            branch_append[ix_from, IM_FROM_IDX] = meas_from.index.values
            branch_append[ix_to, IM_TO] = meas_to.value.values * i_ka_to_pu_to
            branch_append[ix_to, IM_TO_STD] = meas_to.std_dev.values * i_ka_to_pu_to
            branch_append[ix_to, IM_TO_IDX] = meas_to.index.values

        p_measurements = meas[(meas.measurement_type == "p") & (meas.element_type == "line") &
                              meas.element.isin(map_line)]
        if len(p_measurements):
            meas_from = p_measurements[(p_measurements.side.values.astype(int) ==
                                        net.line.from_bus[p_measurements.element]).values]
            meas_to = p_measurements[(p_measurements.side.values.astype(int) ==
                                      net.line.to_bus[p_measurements.element]).values]
            ix_from = [map_line[l] for l in meas_from.element.values.astype(int)]
            ix_to = [map_line[l] for l in meas_to.element.values.astype(int)]
            branch_append[ix_from, P_FROM] = meas_from.value.values
            branch_append[ix_from, P_FROM_STD] = meas_from.std_dev.values
            branch_append[ix_from, P_FROM_IDX] = meas_from.index.values
            branch_append[ix_to, P_TO] = meas_to.value.values
            branch_append[ix_to, P_TO_STD] = meas_to.std_dev.values
            branch_append[ix_to, P_TO_IDX] = meas_to.index.values

        q_measurements = meas[(meas.measurement_type == "q") & (meas.element_type == "line") &
                              meas.element.isin(map_line)]
        if len(q_measurements):
            meas_from = q_measurements[(q_measurements.side.values.astype(int) ==
                                        net.line.from_bus[q_measurements.element]).values]
            meas_to = q_measurements[(q_measurements.side.values.astype(int) ==
                                      net.line.to_bus[q_measurements.element]).values]
            ix_from = [map_line[l] for l in meas_from.element.values.astype(int)]
            ix_to = [map_line[l] for l in meas_to.element.values.astype(int)]
            branch_append[ix_from, Q_FROM] = meas_from.value.values
            branch_append[ix_from, Q_FROM_STD] = meas_from.std_dev.values
            branch_append[ix_from, Q_FROM_IDX] = meas_from.index.values
            branch_append[ix_to, Q_TO] = meas_to.value.values
            branch_append[ix_to, Q_TO_STD] = meas_to.std_dev.values
            branch_append[ix_to, Q_TO_IDX] = meas_to.index.values

    # TODO review in 2019 -> is this a use case? create test with switches on lines
    # determine number of lines in ppci["branch"]
    # out of service lines and lines with open switches at both ends are not in the ppci
    # _is_elements = net["_is_elements"]
    # if "line" not in _is_elements:
    #     get_is_lines(net)
    # lines_is = _is_elements['line']
    # bus_is_idx = _is_elements['bus_is_idx']
    # slidx = (net["switch"]["closed"].values == 0) \
    #         & (net["switch"]["et"].values == "l") \
    #         & (np.in1d(net["switch"]["element"].values, lines_is.index)) \
    #         & (np.in1d(net["switch"]["bus"].values, bus_is_idx))
    # ppci_lines = len(lines_is) - np.count_nonzero(slidx)

    if map_trafo is not None:
        i_tr_measurements = meas[(meas.measurement_type == "i") & (meas.element_type == "trafo") &
                                 meas.element.isin(map_trafo)]
        if len(i_tr_measurements):
            meas_from = i_tr_measurements[(i_tr_measurements.side.values.astype(int) ==
                                           net.trafo.hv_bus[i_tr_measurements.element]).values]
            meas_to = i_tr_measurements[(i_tr_measurements.side.values.astype(int) ==
                                         net.trafo.lv_bus[i_tr_measurements.element]).values]
            ix_from = [map_trafo[t] for t in meas_from.element.values.astype(int)]
            ix_to = [map_trafo[t] for t in meas_to.element.values.astype(int)]
            i_ka_to_pu_from = (net.bus.vn_kv[meas_from.side]).values * 1e3
            i_ka_to_pu_to = (net.bus.vn_kv[meas_to.side]).values * 1e3
            branch_append[ix_from, IM_FROM] = meas_from.value.values * i_ka_to_pu_from
            branch_append[ix_from, IM_FROM_STD] = meas_from.std_dev.values * i_ka_to_pu_from
            branch_append[ix_from, IM_FROM_IDX] = meas_from.index.values
            branch_append[ix_to, IM_TO] = meas_to.value.values * i_ka_to_pu_to
            branch_append[ix_to, IM_TO_STD] = meas_to.std_dev.values * i_ka_to_pu_to
            branch_append[ix_to, IM_TO_IDX] = meas_to.index.values

        p_tr_measurements = meas[(meas.measurement_type == "p") & (meas.element_type == "trafo") &
                                 meas.element.isin(map_trafo)]
        if len(p_tr_measurements):
            meas_from = p_tr_measurements[(p_tr_measurements.side.values.astype(int) ==
                                           net.trafo.hv_bus[p_tr_measurements.element]).values]
            meas_to = p_tr_measurements[(p_tr_measurements.side.values.astype(int) ==
                                         net.trafo.lv_bus[p_tr_measurements.element]).values]
            ix_from = [map_trafo[t] for t in meas_from.element.values.astype(int)]
            ix_to = [map_trafo[t] for t in meas_to.element.values.astype(int)]
            branch_append[ix_from, P_FROM] = meas_from.value.values
            branch_append[ix_from, P_FROM_STD] = meas_from.std_dev.values
            branch_append[ix_from, P_FROM_IDX] = meas_from.index.values
            branch_append[ix_to, P_TO] = meas_to.value.values
            branch_append[ix_to, P_TO_STD] = meas_to.std_dev.values
            branch_append[ix_to, P_TO_IDX] = meas_to.index.values

        q_tr_measurements = meas[(meas.measurement_type == "q") & (meas.element_type == "trafo") &
                                 meas.element.isin(map_trafo)]
        if len(q_tr_measurements):
            meas_from = q_tr_measurements[(q_tr_measurements.side.values.astype(int) ==
                                           net.trafo.hv_bus[q_tr_measurements.element]).values]
            meas_to = q_tr_measurements[(q_tr_measurements.side.values.astype(int) ==
                                         net.trafo.lv_bus[q_tr_measurements.element]).values]
            ix_from = [map_trafo[t] for t in meas_from.element.values.astype(int)]
            ix_to = [map_trafo[t] for t in meas_to.element.values.astype(int)]
            branch_append[ix_from, Q_FROM] = meas_from.value.values
            branch_append[ix_from, Q_FROM_STD] = meas_from.std_dev.values
            branch_append[ix_from, Q_FROM_IDX] = meas_from.index.values
            branch_append[ix_to, Q_TO] = meas_to.value.values
            branch_append[ix_to, Q_TO_STD] = meas_to.std_dev.values
            branch_append[ix_to, Q_TO_IDX] = meas_to.index.values

    # Add measurements for trafo3w
    if map_trafo3w is not None:
        i_tr3w_measurements = meas[(meas.measurement_type == "i") & (meas.element_type == "trafo3w") &
                                   meas.element.isin(map_trafo3w)]
        if len(i_tr3w_measurements):
            meas_hv = i_tr3w_measurements[(i_tr3w_measurements.side.values.astype(int) ==
                                           net.trafo3w.hv_bus[i_tr3w_measurements.element]).values]
            meas_mv = i_tr3w_measurements[(i_tr3w_measurements.side.values.astype(int) ==
                                           net.trafo3w.mv_bus[i_tr3w_measurements.element]).values]
            meas_lv = i_tr3w_measurements[(i_tr3w_measurements.side.values.astype(int) ==
                                           net.trafo3w.lv_bus[i_tr3w_measurements.element]).values]
            ix_hv = [map_trafo3w[t]['hv'] for t in meas_hv.element.values.astype(int)]
            ix_mv = [map_trafo3w[t]['mv'] for t in meas_mv.element.values.astype(int)]
            ix_lv = [map_trafo3w[t]['lv'] for t in meas_lv.element.values.astype(int)]
            i_ka_to_pu_hv = (net.bus.vn_kv[meas_hv.side]).values
            i_ka_to_pu_mv = (net.bus.vn_kv[meas_mv.side]).values
            i_ka_to_pu_lv = (net.bus.vn_kv[meas_lv.side]).values
            branch_append[ix_hv, IM_FROM] = meas_hv.value.values * i_ka_to_pu_hv
            branch_append[ix_hv, IM_FROM_STD] = meas_hv.std_dev.values * i_ka_to_pu_hv
            branch_append[ix_hv, IM_FROM_IDX] = meas_hv.index.values
            branch_append[ix_mv, IM_TO] = meas_mv.value.values * i_ka_to_pu_mv
            branch_append[ix_mv, IM_TO_STD] = meas_mv.std_dev.values * i_ka_to_pu_mv
            branch_append[ix_mv, IM_TO_IDX] = meas_mv.index.values
            branch_append[ix_lv, IM_TO] = meas_lv.value.values * i_ka_to_pu_lv
            branch_append[ix_lv, IM_TO_STD] = meas_lv.std_dev.values * i_ka_to_pu_lv
            branch_append[ix_lv, IM_TO_IDX] = meas_lv.index.values

        p_tr3w_measurements = meas[(meas.measurement_type == "p") & (meas.element_type == "trafo3w") &
                                   meas.element.isin(map_trafo3w)]
        if len(p_tr3w_measurements):
            meas_hv = p_tr3w_measurements[(p_tr3w_measurements.side.values.astype(int) ==
                                           net.trafo3w.hv_bus[p_tr3w_measurements.element]).values]
            meas_mv = p_tr3w_measurements[(p_tr3w_measurements.side.values.astype(int) ==
                                           net.trafo3w.mv_bus[p_tr3w_measurements.element]).values]
            meas_lv = p_tr3w_measurements[(p_tr3w_measurements.side.values.astype(int) ==
                                           net.trafo3w.lv_bus[p_tr3w_measurements.element]).values]
            ix_hv = [map_trafo3w[t]['hv'] for t in meas_hv.element.values.astype(int)]
            ix_mv = [map_trafo3w[t]['mv'] for t in meas_mv.element.values.astype(int)]
            ix_lv = [map_trafo3w[t]['lv'] for t in meas_lv.element.values.astype(int)]
            branch_append[ix_hv, P_FROM] = meas_hv.value.values
            branch_append[ix_hv, P_FROM_STD] = meas_hv.std_dev.values
            branch_append[ix_hv, P_FROM_IDX] = meas_hv.index.values
            branch_append[ix_mv, P_TO] = meas_mv.value.values
            branch_append[ix_mv, P_TO_STD] = meas_mv.std_dev.values
            branch_append[ix_mv, P_TO_IDX] = meas_mv.index.values
            branch_append[ix_lv, P_TO] = meas_lv.value.values
            branch_append[ix_lv, P_TO_STD] = meas_lv.std_dev.values
            branch_append[ix_lv, P_TO_IDX] = meas_lv.index.values

        q_tr3w_measurements = meas[(meas.measurement_type == "q") & (meas.element_type == "trafo3w") &
                                   meas.element.isin(map_trafo3w)]
        if len(q_tr3w_measurements):
            meas_hv = q_tr3w_measurements[(q_tr3w_measurements.side.values.astype(int) ==
                                           net.trafo3w.hv_bus[q_tr3w_measurements.element]).values]
            meas_mv = q_tr3w_measurements[(q_tr3w_measurements.side.values.astype(int) ==
                                           net.trafo3w.mv_bus[q_tr3w_measurements.element]).values]
            meas_lv = q_tr3w_measurements[(q_tr3w_measurements.side.values.astype(int) ==
                                           net.trafo3w.lv_bus[q_tr3w_measurements.element]).values]
            ix_hv = [map_trafo3w[t]['hv'] for t in meas_hv.element.values.astype(int)]
            ix_mv = [map_trafo3w[t]['mv'] for t in meas_mv.element.values.astype(int)]
            ix_lv = [map_trafo3w[t]['lv'] for t in meas_lv.element.values.astype(int)]
            branch_append[ix_hv, Q_FROM] = meas_hv.value.values
            branch_append[ix_hv, Q_FROM_STD] = meas_hv.std_dev.values
            branch_append[ix_hv, Q_FROM_IDX] = meas_hv.index.values
            branch_append[ix_mv, Q_TO] = meas_mv.value.values
            branch_append[ix_mv, Q_TO_STD] = meas_mv.std_dev.values
            branch_append[ix_mv, Q_TO_IDX] = meas_mv.index.values
            branch_append[ix_lv, Q_TO] = meas_lv.value.values
            branch_append[ix_lv, Q_TO_STD] = meas_lv.std_dev.values
            branch_append[ix_lv, Q_TO_IDX] = meas_lv.index.values

    ppci["bus"] = np.hstack((ppci["bus"], bus_append))
    ppci["branch"] = np.hstack((ppci["branch"], branch_append))
    return ppci


def _add_zero_injection(net, ppci, bus_append, zero_injection): 
    """
    Add zero injection labels to the ppci structure and add virtual measurements to those buses
    :param net: pandapower net
    :param ppci: generated ppci
    :param bus_append: added columns to the ppci bus with zero injection label
    :param zero_injection: parameter to control which bus to be identified as zero injection
    :return bus_append: added columns
    """   
    bus_append[:, ZERO_INJ_FLAG] = False   
    if zero_injection is not None:
        # identify aux bus to zero injection
        if net._pd2ppc_lookups['aux']:
            aux_bus_lookup = np.concatenate([v for k,v in net._pd2ppc_lookups['aux'].items() if k != 'xward'])
            aux_bus = net._pd2ppc_lookups['bus'][aux_bus_lookup]
            bus_append[aux_bus, ZERO_INJ_FLAG] = True

        if isinstance(zero_injection, str):
            if zero_injection == 'auto':
                # identify bus without elements and pq measurements as zero injection
                zero_inj_bus_mask = (ppci["bus"][:, 1] == 1) & (ppci["bus"][:, 2:6]==0).all(axis=1) &\
                    np.isnan(bus_append[:, P:(Q_STD+1)]).all(axis=1)
                bus_append[zero_inj_bus_mask, ZERO_INJ_FLAG] = True
            elif zero_injection != "aux_bus":
                raise UserWarning("zero injection parameter is not correctly initialized")
        elif hasattr(zero_injection, '__iter__'):
            zero_inj_bus = net._pd2ppc_lookups['bus'][zero_injection]
            bus_append[zero_inj_bus, ZERO_INJ_FLAG] = True

        zero_inj_bus = np.argwhere(bus_append[:, ZERO_INJ_FLAG]).ravel()
        bus_append[zero_inj_bus, P] = 0
        bus_append[zero_inj_bus, P_STD] = 1
        bus_append[zero_inj_bus, Q] = 0
        bus_append[zero_inj_bus, Q_STD] = 1
    return bus_append


def _build_measurement_vectors(ppci):
    """
    Building measurement vector z, pandapower to ppci measurement mapping and covariance matrix R
    :param ppci: generated ppci which contains the measurement columns
    :param branch_cols: number of columns in original ppci["branch"] without measurements
    :param bus_cols: number of columns in original ppci["bus"] without measurements
    :return: both created vectors
    """
    p_bus_not_nan = ~np.isnan(ppci["bus"][:, bus_cols + P])
    p_line_f_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + P_FROM])
    p_line_t_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + P_TO])
    q_bus_not_nan = ~np.isnan(ppci["bus"][:, bus_cols + Q])
    q_line_f_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + Q_FROM])
    q_line_t_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + Q_TO])
    v_bus_not_nan = ~np.isnan(ppci["bus"][:, bus_cols + VM])
    i_line_f_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + IM_FROM])
    i_line_t_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + IM_TO])
    # piece together our measurement vector z
    z = np.concatenate((ppci["bus"][p_bus_not_nan, bus_cols + P],
                        ppci["branch"][p_line_f_not_nan, branch_cols + P_FROM],
                        ppci["branch"][p_line_t_not_nan, branch_cols + P_TO],
                        ppci["bus"][q_bus_not_nan, bus_cols + Q],
                        ppci["branch"][q_line_f_not_nan, branch_cols + Q_FROM],
                        ppci["branch"][q_line_t_not_nan, branch_cols + Q_TO],
                        ppci["bus"][v_bus_not_nan, bus_cols + VM],
                        ppci["branch"][i_line_f_not_nan, branch_cols + IM_FROM],
                        ppci["branch"][i_line_t_not_nan, branch_cols + IM_TO]
                        )).real.astype(np.float64)
    # conserve the pandapower indices of measurements in the ppci order
    pp_meas_indices = np.concatenate((ppci["bus"][p_bus_not_nan, bus_cols + P_IDX],
                                      ppci["branch"][p_line_f_not_nan, branch_cols + P_FROM_IDX],
                                      ppci["branch"][p_line_t_not_nan, branch_cols + P_TO_IDX],
                                      ppci["bus"][q_bus_not_nan, bus_cols + Q_IDX],
                                      ppci["branch"][q_line_f_not_nan, branch_cols + Q_FROM_IDX],
                                      ppci["branch"][q_line_t_not_nan, branch_cols + Q_TO_IDX],
                                      ppci["bus"][v_bus_not_nan, bus_cols + VM_IDX],
                                      ppci["branch"][i_line_f_not_nan, branch_cols + IM_FROM_IDX],
                                      ppci["branch"][i_line_t_not_nan, branch_cols + IM_TO_IDX]
                                      )).real.astype(int)
    # Covariance matrix R
    r_cov = np.concatenate((ppci["bus"][p_bus_not_nan, bus_cols + P_STD],
                            ppci["branch"][p_line_f_not_nan, branch_cols + P_FROM_STD],
                            ppci["branch"][p_line_t_not_nan, branch_cols + P_TO_STD],
                            ppci["bus"][q_bus_not_nan, bus_cols + Q_STD],
                            ppci["branch"][q_line_f_not_nan, branch_cols + Q_FROM_STD],
                            ppci["branch"][q_line_t_not_nan, branch_cols + Q_TO_STD],
                            ppci["bus"][v_bus_not_nan, bus_cols + VM_STD],
                            ppci["branch"][i_line_f_not_nan, branch_cols + IM_FROM_STD],
                            ppci["branch"][i_line_t_not_nan, branch_cols + IM_TO_STD]
                            )).real.astype(np.float64)
    return z, pp_meas_indices, r_cov
