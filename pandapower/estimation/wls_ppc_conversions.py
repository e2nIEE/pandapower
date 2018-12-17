# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
from pandapower.auxiliary import _select_is_elements_numba, _add_ppc_options, _add_auxiliary_elements
from pandapower.pd2ppc import _pd2ppc
from pandapower.estimation.idx_bus import *
from pandapower.estimation.idx_brch import *
from pandapower.idx_brch import branch_cols
from pandapower.idx_bus import bus_cols
from pandapower.pf.run_newton_raphson_pf import _run_dc_pf
from pandapower.build_branch import get_is_lines

try:
    import pplog as logging
except ImportError:
    import logging
std_logger = logging.getLogger(__name__)


def _init_ppc(net, v_start, delta_start, calculate_voltage_angles):
    # initialize ppc voltages
    net.res_bus.vm_pu = v_start
    net.res_bus.vm_pu[net.bus.index[net.bus.in_service == False]] = np.nan
    net.res_bus.va_degree = delta_start
    # select elements in service and convert pandapower ppc to ppc
    net._options = {}
    _add_ppc_options(net, check_connectivity=False, init_vm_pu="results", init_va_degree="results",
                     trafo_model="t", mode="pf", enforce_q_lims=False,
                     calculate_voltage_angles=calculate_voltage_angles, r_switch=0.0,
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


def _add_measurements_to_ppc(net, ppci):
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
    if (map_bus[meas_bus['element']] >= ppci["bus"].shape[0]).any():
        std_logger.warning("Measurement defined in pp-grid does not exist in ppci! Will be deleted!")
        meas_bus = meas_bus[map_bus[meas_bus['element']] < ppci["bus"].shape[0]]

    map_line, map_trafo, map_trafo3w = None, None, None
    # mapping to dict instead of np array ensures good performance for large indices
    # (e.g., 999999999 requires a large np array even if there are only 2 buses)
    # downside is loop comprehension to access the map
    branch_mask = ppci['internal']['branch_is']
    if "line" in net["_pd2ppc_lookups"]["branch"]:
        map_line = {line_ix:br_ix for line_ix, br_ix in\
                    zip(net.line.index, range(*net["_pd2ppc_lookups"]["branch"]["line"])) if branch_mask[br_ix]}
        
    if "trafo" in net["_pd2ppc_lookups"]["branch"]:
        trafo_ix_start, trafo_ix_end = net["_pd2ppc_lookups"]["branch"]["trafo"]
        trafo_ix_offset = np.sum(~branch_mask[:trafo_ix_start])
        trafo_ix_start, trafo_ix_end = trafo_ix_start - trafo_ix_offset, trafo_ix_end - trafo_ix_offset
        map_trafo = {trafo_ix:br_ix for trafo_ix, br_ix in\
                     zip(net.trafo.index, range(trafo_ix_start, trafo_ix_end)) if branch_mask[br_ix+trafo_ix_offset]}

    if "trafo3w" in net["_pd2ppc_lookups"]["branch"]:
        trafo3w_ix_start, trafo3w_ix_end = net["_pd2ppc_lookups"]["branch"]["trafo3w"]
        trafo3w_ix_offset = np.sum(~branch_mask[:trafo3w_ix_start])
        trafo3w_ix_start, trafo3w_ix_end = trafo3w_ix_start - trafo3w_ix_offset, trafo3w_ix_end - trafo3w_ix_offset
        map_trafo3w = {trafo3w_ix: {'hv': br_ix, 'mv': br_ix+1, 'lv': br_ix+2} for trafo3w_ix, br_ix in\
                       zip(net.trafo3w.index, range(trafo3w_ix_start, trafo3w_ix_end, 3)) if branch_mask[br_ix+trafo3w_ix_offset]}


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
        bus_append[bus_positions, P] = p_measurements.value.values
        bus_append[bus_positions, P_STD] = p_measurements.std_dev.values
        bus_append[bus_positions, P_IDX] = p_measurements.index.values

    q_measurements = meas_bus[(meas_bus.measurement_type == "q")]
    if len(q_measurements):
        bus_positions = map_bus[q_measurements.element.values.astype(int)]
        bus_append[bus_positions, Q] = q_measurements.value.values
        bus_append[bus_positions, Q_STD] = q_measurements.std_dev.values
        bus_append[bus_positions, Q_IDX] = q_measurements.index.values

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
    
        p_measurements = meas[(meas.measurement_type == "p") & (meas.element_type == "line") &\
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
    
        q_measurements = meas[(meas.measurement_type == "q") & (meas.element_type == "line")&\
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
        i_tr_measurements = meas[(meas.measurement_type == "i") & (meas.element_type == "trafo") &\
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
    
        p_tr_measurements = meas[(meas.measurement_type == "p") & (meas.element_type == "trafo")&\
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
    
        q_tr_measurements = meas[(meas.measurement_type == "q") & (meas.element_type == "trafo")&\
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
        i_tr3w_measurements = meas[(meas.measurement_type == "i") & (meas.element_type == "trafo3w")&\
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
    
        p_tr3w_measurements = meas[(meas.measurement_type == "p") & (meas.element_type == "trafo3w")&\
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
    
        q_tr3w_measurements = meas[(meas.measurement_type == "q") & (meas.element_type == "trafo3w")&\
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
