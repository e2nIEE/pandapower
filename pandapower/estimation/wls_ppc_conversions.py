# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from pandapower.auxiliary import _select_is_elements, _add_ppc_options
from pandapower.pd2ppc import _pd2ppc
from pandapower.estimation.idx_bus import *
from pandapower.estimation.idx_brch import *
from pandapower.idx_brch import branch_cols
from pandapower.idx_bus import bus_cols


def _init_ppc(net, v_start, delta_start, calculate_voltage_angles):
    # initialize ppc voltages
    net.res_bus.vm_pu = v_start
    net.res_bus.vm_pu[net.bus.index[net.bus.in_service == False]] = np.nan
    net.res_bus.va_degree = delta_start
    # select elements in service and convert pandapower ppc to ppc
    net._options = {}
    _add_ppc_options(net, check_connectivity=False, init="results", trafo_model="t",
                     copy_constraints_to_ppc=False, mode="pf", enforce_q_lims=False,
                     calculate_voltage_angles=calculate_voltage_angles, r_switch=0.0,
                     recycle=dict(_is_elements=False, ppc=False, Ybus=False))
    net["_is_elements"] = _select_is_elements(net)
    ppc, ppci = _pd2ppc(net)
    return ppc, ppci


def _add_measurements_to_ppc(net, mapping_table, ppci, s_ref):
    """
    Add pandapower measurements to the ppci structure by adding new columns
    :param net: pandapower net
    :param mapping_table: mapping table pd->ppc
    :param ppci: generated ppci 
    :param s_ref: reference power
    :return: ppc with added columns
    """
    # set measurements for ppc format
    # add 9 columns to ppc[bus] for Vm, Vm std dev, P, P std dev, Q, Q std dev,
    # pandapower measurement indices V, P, Q
    bus_append = np.full((ppci["bus"].shape[0], bus_cols_se), np.nan, dtype=ppci["bus"].dtype)

    v_measurements = net.measurement[(net.measurement.type == "v")
                                     & (net.measurement.element_type == "bus")]
    if len(v_measurements):
        bus_positions = mapping_table[v_measurements.bus.values.astype(int)]
        bus_append[bus_positions, VM] = v_measurements.value.values
        bus_append[bus_positions, VM_STD] = v_measurements.std_dev.values
        bus_append[bus_positions, VM_IDX] = v_measurements.index.values

    p_measurements = net.measurement[(net.measurement.type == "p")
                                     & (net.measurement.element_type == "bus")]
    if len(p_measurements):
        bus_positions = mapping_table[p_measurements.bus.values.astype(int)]
        bus_append[bus_positions, P] = p_measurements.value.values * 1e3 / s_ref
        bus_append[bus_positions, P_STD] = p_measurements.std_dev.values * 1e3 / s_ref
        bus_append[bus_positions, P_IDX] = p_measurements.index.values

    q_measurements = net.measurement[(net.measurement.type == "q")
                                     & (net.measurement.element_type == "bus")]
    if len(q_measurements):
        bus_positions = mapping_table[q_measurements.bus.values.astype(int)]
        bus_append[bus_positions, Q] = q_measurements.value.values * 1e3 / s_ref
        bus_append[bus_positions, Q_STD] = q_measurements.std_dev.values * 1e3 / s_ref
        bus_append[bus_positions, Q_IDX] = q_measurements.index.values

    # add virtual measurements for artificial buses, which were created because
    # of an open line switch. p/q are 0. and std dev is 1. (small value)
    new_in_line_buses = np.setdiff1d(np.arange(ppci["bus"].shape[0]),
                                     mapping_table[mapping_table >= 0])
    bus_append[new_in_line_buses, 2] = 0.
    bus_append[new_in_line_buses, 3] = 1.
    bus_append[new_in_line_buses, 4] = 0.
    bus_append[new_in_line_buses, 5] = 1.

    # add 15 columns to mpc[branch] for Im_from, Im_from std dev, Im_to, Im_to std dev,
    # P_from, P_from std dev, P_to, P_to std dev, Q_from, Q_from std dev,  Q_to, Q_to std dev,
    # pandapower measurement index I, P, Q
    branch_append = np.full((ppci["branch"].shape[0], branch_cols_se),
                            np.nan, dtype=ppci["branch"].dtype)

    i_measurements = net.measurement[(net.measurement.type == "i")
                                     & (net.measurement.element_type == "line")]
    if len(i_measurements):
        meas_from = i_measurements[(i_measurements.bus.values.astype(int) ==
                                    net.line.from_bus[i_measurements.element]).values]
        meas_to = i_measurements[(i_measurements.bus.values.astype(int) ==
                                  net.line.to_bus[i_measurements.element]).values]
        ix_from = meas_from.element.values.astype(int)
        ix_to = meas_to.element.values.astype(int)
        i_a_to_pu_from = (net.bus.vn_kv[meas_from.bus] * 1e3 / s_ref).values
        i_a_to_pu_to = (net.bus.vn_kv[meas_to.bus] * 1e3 / s_ref).values
        branch_append[ix_from, IM_FROM] = meas_from.value.values * i_a_to_pu_from
        branch_append[ix_from, IM_FROM_STD] = meas_from.std_dev.values * i_a_to_pu_from
        branch_append[ix_to, IM_TO] = meas_to.value.values * i_a_to_pu_to
        branch_append[ix_to, IM_TO_STD] = meas_to.std_dev.values * i_a_to_pu_to
        branch_append[meas_from.element.values.astype(int), IM_FROM_IDX] = meas_from.index.values
        branch_append[meas_to.element.values.astype(int), IM_TO_IDX] = meas_to.index.values

    p_measurements = net.measurement[(net.measurement.type == "p")
                                          & (net.measurement.element_type == "line")]
    if len(p_measurements):
        meas_from = p_measurements[(p_measurements.bus.values.astype(int) ==
                                    net.line.from_bus[p_measurements.element]).values]
        meas_to = p_measurements[(p_measurements.bus.values.astype(int) ==
                                  net.line.to_bus[p_measurements.element]).values]
        ix_from = meas_from.element.values.astype(int)
        ix_to = meas_to.element.values.astype(int)
        branch_append[ix_from, P_FROM] = meas_from.value.values * 1e3 / s_ref
        branch_append[ix_from, P_FROM_STD] = meas_from.std_dev.values * 1e3 / s_ref
        branch_append[ix_to, P_TO] = meas_to.value.values * 1e3 / s_ref
        branch_append[ix_to, P_TO_STD] = meas_to.std_dev.values * 1e3 / s_ref
        branch_append[meas_from.element.values.astype(int), P_FROM_IDX] = meas_from.index.values
        branch_append[meas_to.element.values.astype(int), P_TO_IDX] = meas_to.index.values

    q_measurements = net.measurement[(net.measurement.type == "q")
                                          & (net.measurement.element_type == "line")]
    if len(q_measurements):
        meas_from = q_measurements[(q_measurements.bus.values.astype(int) ==
                                    net.line.from_bus[q_measurements.element]).values]
        meas_to = q_measurements[(q_measurements.bus.values.astype(int) ==
                                  net.line.to_bus[q_measurements.element]).values]
        ix_from = meas_from.element.values.astype(int)
        ix_to = meas_to.element.values.astype(int)
        branch_append[ix_from, Q_FROM] = meas_from.value.values * 1e3 / s_ref
        branch_append[ix_from, Q_FROM_STD] = meas_from.std_dev.values * 1e3 / s_ref
        branch_append[ix_to, Q_TO] = meas_to.value.values * 1e3 / s_ref
        branch_append[ix_to, Q_TO_STD] = meas_to.std_dev.values * 1e3 / s_ref
        branch_append[meas_from.element.values.astype(int), Q_FROM_IDX] = meas_from.index.values
        branch_append[meas_to.element.values.astype(int), Q_TO_IDX] = meas_to.index.values

    # determine number of lines in ppci["branch"]
    # out of service lines and lines with open switches at both ends are not in the ppci
    _is_elements = net["_is_elements"]
    lines_is = _is_elements['line']
    bus_is_idx = _is_elements['bus_is_idx']
    slidx = (net["switch"]["closed"].values == 0) \
            & (net["switch"]["et"].values == "l") \
            & (np.in1d(net["switch"]["element"].values, lines_is.index)) \
            & (np.in1d(net["switch"]["bus"].values, bus_is_idx))
    ppci_lines = len(lines_is) - np.count_nonzero(slidx)

    i_tr_measurements = net.measurement[(net.measurement.type == "i")
                                             & (net.measurement.element_type ==
                                                "transformer")]
    if len(i_tr_measurements):
        meas_from = i_tr_measurements[(i_tr_measurements.bus.values.astype(int) ==
                                       net.trafo.hv_bus[i_tr_measurements.element]).values]
        meas_to = i_tr_measurements[(i_tr_measurements.bus.values.astype(int) ==
                                     net.trafo.lv_bus[i_tr_measurements.element]).values]
        ix_from = ppci_lines + meas_from.element.values.astype(int)
        ix_to = ppci_lines + meas_to.element.values.astype(int)
        i_a_to_pu_from = (net.bus.vn_kv[meas_from.bus] * 1e3 / s_ref).values
        i_a_to_pu_to = (net.bus.vn_kv[meas_to.bus] * 1e3 / s_ref).values
        branch_append[ix_from, IM_FROM] = meas_from.value.values * i_a_to_pu_from
        branch_append[ix_from, IM_FROM_STD] = meas_from.std_dev.values * i_a_to_pu_from
        branch_append[ix_to, IM_TO] = meas_to.value.values * i_a_to_pu_to
        branch_append[ix_to, IM_TO_STD] = meas_to.std_dev.values * i_a_to_pu_to
        branch_append[meas_from.element.values.astype(int), IM_FROM_IDX] = meas_from.index.values
        branch_append[meas_to.element.values.astype(int), IM_TO_IDX] = meas_to.index.values

    p_tr_measurements = net.measurement[(net.measurement.type == "p") &
                                             (net.measurement.element_type ==
                                              "transformer")]

    if len(p_tr_measurements):
        meas_from = p_tr_measurements[(p_tr_measurements.bus.values.astype(int) ==
                                       net.trafo.hv_bus[p_tr_measurements.element]).values]
        meas_to = p_tr_measurements[(p_tr_measurements.bus.values.astype(int) ==
                                     net.trafo.lv_bus[p_tr_measurements.element]).values]
        ix_from = ppci_lines + meas_from.element.values.astype(int)
        ix_to = ppci_lines + meas_to.element.values.astype(int)
        branch_append[ix_from, P_FROM] = meas_from.value.values * 1e3 / s_ref
        branch_append[ix_from, P_FROM_STD] = meas_from.std_dev.values * 1e3 / s_ref
        branch_append[ix_to, P_TO] = meas_to.value.values * 1e3 / s_ref
        branch_append[ix_to, P_TO_STD] = meas_to.std_dev.values * 1e3 / s_ref
        branch_append[meas_from.element.values.astype(int), P_FROM_IDX] = meas_from.index.values
        branch_append[meas_to.element.values.astype(int), P_TO_IDX] = meas_to.index.values

    q_tr_measurements = net.measurement[(net.measurement.type == "q") &
                                             (net.measurement.element_type ==
                                              "transformer")]
    if len(q_tr_measurements):
        meas_from = q_tr_measurements[(q_tr_measurements.bus.values.astype(int) ==
                                       net.trafo.hv_bus[q_tr_measurements.element]).values]
        meas_to = q_tr_measurements[(q_tr_measurements.bus.values.astype(int) ==
                                     net.trafo.lv_bus[q_tr_measurements.element]).values]
        ix_from = ppci_lines + meas_from.element.values.astype(int)
        ix_to = ppci_lines + meas_to.element.values.astype(int)
        branch_append[ix_from, Q_FROM] = meas_from.value.values * 1e3 / s_ref
        branch_append[ix_from, Q_FROM_STD] = meas_from.std_dev.values * 1e3 / s_ref
        branch_append[ix_to, Q_TO] = meas_to.value.values * 1e3 / s_ref
        branch_append[ix_to, Q_TO_STD] = meas_to.std_dev.values * 1e3 / s_ref
        branch_append[meas_from.element.values.astype(int), Q_FROM_IDX] = meas_from.index.values
        branch_append[meas_to.element.values.astype(int), Q_TO_IDX] = meas_to.index.values

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
