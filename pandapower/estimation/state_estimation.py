# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandas as pd
import warnings

from pandapower.estimation.wls_matrix_ops import wls_matrix_ops
from pandapower.pd2ppc import _pd2ppc
from pandapower.results import _set_buses_out_of_service, _extract_results, \
                        reset_results, _copy_results_ppci_to_ppc
from pandapower.auxiliary import get_values, _select_is_elements, \
                        calculate_line_results, _add_ppc_options, _add_pf_options
from pandapower.topology import estimate_voltage_vector
from pandapower.pypower_extensions.runpf import _get_pf_variables_from_ppci, \
                        _store_results_from_pf_in_ppci
from pypower.idx_brch import F_BUS, T_BUS, BR_STATUS, PF, PT, QF, QT
from pypower.int2ext import int2ext

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.stats import chi2


try:
    import pplog as logging
except ImportError:
    import logging
std_logger = logging.getLogger(__name__)


def estimate(net, init='flat', tolerance=1e-6, maximum_iterations=10,
             calculate_voltage_angles=True):
    """
    Wrapper function for WLS state estimation.

    INPUT:
        **net** - The net within this line should be created.

        **init** - (string) Initial voltage for the estimation. 'flat' sets 1.0 p.u. / 0° for all
        buses, 'results' uses the values from *res_bus_est* if available and 'slack' considers the
        slack bus voltage (and optionally, angle) as the initial values. Default is 'flat'.

    OPTIONAL:
        **tolerance** - (float) - When the maximum state change between iterations is less than
        tolerance, the process stops. Default is 1e-6.

        **maximum_iterations** - (integer) - Maximum number of iterations. Default is 10.

        **calculate_voltage_angles** - (boolean) - Take into account absolute voltage angles and phase
        shifts in transformers, if init is 'slack'. Default is True.

    OUTPUT:
        **successful** (boolean) - Was the state estimation successful?
    """
    wls = state_estimation(tolerance, maximum_iterations, net)
    v_start = None
    delta_start = None
    if init == 'results':
        v_start = net.res_bus_est.vm_pu
        delta_start = net.res_bus_est.va_degree
    elif init == 'slack':
        res_bus = estimate_voltage_vector(net)
        v_start = res_bus.vm_pu.values
        if calculate_voltage_angles:
            delta_start = res_bus.va_degree.values
    elif init != 'flat':
        raise UserWarning("Unsupported init value. Using flat initialization.")
    return wls.estimate(v_start, delta_start, calculate_voltage_angles)


def remove_bad_data(net, init='flat', tolerance=1e-6, maximum_iterations=10,
             calculate_voltage_angles=True, rn_max_threshold=3.0, chi2_prob_false=0.05):
    """
    Wrapper function for bad data removal.

    INPUT:
        **net** - The net within this line should be created.

        **init** - (string) Initial voltage for the estimation. 'flat' sets 1.0 p.u. / 0° for all
        buses, 'results' uses the values from *res_bus_est* if available and 'slack' considers the
        slack bus voltage (and optionally, angle) as the initial values. Default is 'flat'.

    OPTIONAL:
        **tolerance** - (float) - When the maximum state change between iterations is less than
        tolerance, the process stops. Default is 1e-6.

        **maximum_iterations** - (integer) - Maximum number of iterations. Default is 10.

        **calculate_voltage_angles** - (boolean) - Take into account absolute voltage angles and phase
        shifts in transformers, if init is 'slack'. Default is True.

        **rn_max_threshold** (float) - Identification threshold to determine
        if the largest normalized residual reflects a bad measurement
        (default value of 3.0)

        **chi2_prob_false** (float) - probability of error / false alarms
        (default value: 0.05)

    OUTPUT:
        **successful** (boolean) - Was the state estimation successful?
    """
    wls = state_estimation(tolerance, maximum_iterations, net)
    v_start = None
    delta_start = None
    if init == 'results':
        v_start = net.res_bus_est.vm_pu
        delta_start = net.res_bus_est.va_degree
    elif init == 'slack':
        res_bus = estimate_voltage_vector(net)
        v_start = res_bus.vm_pu.values
        if calculate_voltage_angles:
            delta_start = res_bus.va_degree.values
    elif init != 'flat':
        raise UserWarning("Unsupported init value. Using flat initialization.")
    return wls.perform_rn_max_test(v_start, delta_start, calculate_voltage_angles,
                                   rn_max_threshold, chi2_prob_false)


def chi2_analysis(net, init='flat', tolerance=1e-6, maximum_iterations=10,
             calculate_voltage_angles=True, chi2_prob_false=0.05):
    """
    Wrapper function for the chi-squared test.

    INPUT:
        **net** - The net within this line should be created.

        **init** - (string) Initial voltage for the estimation. 'flat' sets 1.0 p.u. / 0° for all
        buses, 'results' uses the values from *res_bus_est* if available and 'slack' considers the
        slack bus voltage (and optionally, angle) as the initial values. Default is 'flat'.

    OPTIONAL:
        **tolerance** - (float) - When the maximum state change between iterations is less than
        tolerance, the process stops. Default is 1e-6.

        **maximum_iterations** - (integer) - Maximum number of iterations. Default is 10.

        **calculate_voltage_angles** - (boolean) - Take into account absolute voltage angles and phase
        shifts in transformers, if init is 'slack'. Default is True.

        **chi2_prob_false** (float) - probability of error / false alarms
        (default value: 0.05)

    OUTPUT:
        **successful** (boolean) - Was the state estimation successful?
    """
    wls = state_estimation(tolerance, maximum_iterations, net)
    v_start = None
    delta_start = None
    if init == 'results':
        v_start = net.res_bus_est.vm_pu
        delta_start = net.res_bus_est.va_degree
    elif init == 'slack':
        res_bus = estimate_voltage_vector(net)
        v_start = res_bus.vm_pu.values
        if calculate_voltage_angles:
            delta_start = res_bus.va_degree.values
    elif init != 'flat':
        raise UserWarning("Unsupported init value. Using flat initialization.")
    return wls.perform_chi2_test(v_start, delta_start, calculate_voltage_angles,
                                 chi2_prob_false)


class state_estimation(object):
    """
    Any user of the estimation module only needs to use the class state_estimation. It contains all
    relevant functions to control and operator the module. Two functions are used to configure the
    system according to the users needs while one function is used for the actual estimation
    process.
    """
    def __init__(self, tolerance=1e-6, maximum_iterations=10, net=None, logger=None):
        self.logger = logger
        if self.logger is None:
            self.logger = std_logger
        self.tolerance = tolerance
        self.max_iterations = maximum_iterations
        self.net = net
        self.s_ref = 1e6
        self.s_node_powers = None
        # for chi square test
        self.hx = None
        self.R_inv = None
        self.H = None
        self.Ht = None
        self.Gm = None
        self.r = None
        self.V = None
        self.delta = None
        self.bad_data_present = None
        # offset to accommodate pypower - pandapower differences (additional columns)
        self.br_col_offset = 6
        self.mapping_table = None

    def estimate(self, v_start=None, delta_start=None, calculate_voltage_angles=True):
        """
        The function estimate is the main function of the module. It takes up to three input
        arguments: v_start, delta_start and calculate_voltage_angles. The first two are the initial
        state variables for the estimation process. Usually they can be initialized in a
        "flat-start" condition: All voltages being 1.0 pu and all voltage angles being 0 degrees.
        In this case, the parameters can be left at their default values (None). If the estimation
        is applied continuously, using the results from the last estimation as the starting
        condition for the current estimation can decrease the  amount of iterations needed to
        estimate the current state. The third parameter defines whether all voltage angles are
        calculated absolutely, including phase shifts from transformers. If only the relative
        differences between buses are required, this parameter can be set to False. Returned is a
        boolean value, which is true after a successful estimation and false otherwise.
        The resulting complex voltage will be written into the pandapower network. The result
        fields are found res_bus_est of the pandapower network.

        INPUT:
            **net** - The net within this line should be created

            **v_start** (np.array, shape=(1,), optional) - Vector with initial values for all
            voltage magnitudes in p.u. (sorted by bus index)

            **delta_start** (np.array, shape=(1,), optional) - Vector with initial values for all
            voltage angles in degrees (sorted by bus index)

        OPTIONAL:
            **calculate_voltage_angles** - (bool) - Take into account absolute voltage angles and
            phase shifts in transformers Default is True.

        OUTPUT:
            **successful** (boolean) - True if the estimation process was successful

        Optional estimation variables:
            The bus power injections can be accessed with *se.s_node_powers* and the estimated
            values corresponding to the (noisy) measurement values with *se.hx*. (*hx* denotes h(x))

        EXAMPLE:
            success = estimate(np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0]))

        """
        if self.net is None:
            raise UserWarning("Component was not initialized with a network.")

        # add initial values for V and delta
        # node voltages
        # V<delta
        if v_start is None:
            v_start = np.ones(self.net.bus.shape[0])
        if delta_start is None:
            delta_start = np.zeros(self.net.bus.shape[0])

        # initialize the ppc bus with the initial values given
        # initialize result tables if not existant
        if "res_bus" not in self.net or "res_line" not in self.net:
            reset_results(self.net)
        # create backup of current results
        vm_backup = self.net.res_bus.vm_pu.copy()
        va_backup = self.net.res_bus.va_degree.copy()
        # initialize ppc voltages
        self.net.res_bus.vm_pu = v_start
        self.net.res_bus.vm_pu[self.net.bus.index[self.net.bus.in_service == False]] = np.nan
        self.net.res_bus.va_degree = delta_start
        # select elements in service and convert pandapower ppc to ppc
        self.net._options = {}
        _add_ppc_options(self.net, check_connectivity=False, init="results", trafo_model="t",
                         copy_constraints_to_ppc=False, mode="pf", enforce_q_lims=False,
                         calculate_voltage_angles=calculate_voltage_angles, r_switch=0.0,
                         recycle=dict(_is_elements=False, ppc=False, Ybus=False))
        self.net["_is_elements"] = _select_is_elements(self.net)
        ppc, ppci = _pd2ppc(self.net)
        self.mapping_table = self.net["_pd2ppc_lookups"]["bus"]
        br_cols = ppci["branch"].shape[1]
        bs_cols = ppci["bus"].shape[1]
        # restore backup of previous results
        self.net.res_bus.vm_pu = vm_backup
        self.net.res_bus.va_degree = va_backup
        # set measurements for ppc format
        # add 6 columns to ppc[bus] for Vm, Vm std dev, P, P std dev, Q, Q std dev
        bus_append = np.full((ppci["bus"].shape[0], 6), np.nan, dtype=ppci["bus"].dtype)

        v_measurements = self.net.measurement[(self.net.measurement.type == "v")
                                              & (self.net.measurement.element_type == "bus")]
        if len(v_measurements):
            bus_positions = self.mapping_table[v_measurements.bus.values.astype(int)]
            bus_append[bus_positions, 0] = v_measurements.value.values
            bus_append[bus_positions, 1] = v_measurements.std_dev.values

        p_measurements = self.net.measurement[(self.net.measurement.type == "p")
                                              & (self.net.measurement.element_type == "bus")]
        if len(p_measurements):
            bus_positions = self.mapping_table[p_measurements.bus.values.astype(int)]
            bus_append[bus_positions, 2] = p_measurements.value.values * 1e3 / self.s_ref
            bus_append[bus_positions, 3] = p_measurements.std_dev.values * 1e3 / self.s_ref

        q_measurements = self.net.measurement[(self.net.measurement.type == "q")
                                              & (self.net.measurement.element_type == "bus")]
        if len(q_measurements):
            bus_positions = self.mapping_table[q_measurements.bus.values.astype(int)]
            bus_append[bus_positions, 4] = q_measurements.value.values * 1e3 / self.s_ref
            bus_append[bus_positions, 5] = q_measurements.std_dev.values * 1e3 / self.s_ref

        # add virtual measurements for artificial buses, which were created because
        # of an open line switch. p/q are 0. and std dev is 1. (small value)
        new_in_line_buses = np.setdiff1d(np.arange(ppci["bus"].shape[0]),
                                         self.mapping_table[self.mapping_table >= 0])
        bus_append[new_in_line_buses, 2] = 0.
        bus_append[new_in_line_buses, 3] = 1.
        bus_append[new_in_line_buses, 4] = 0.
        bus_append[new_in_line_buses, 5] = 1.

        # add 12 columns to mpc[branch] for Im_from, Im_from std dev, Im_to, Im_to std dev,
        # P_from, P_from std dev, P_to, P_to std dev, Q_from,Q_from std dev,  Q_to, Q_to std dev
        branch_append = np.full((ppci["branch"].shape[0], 12), np.nan, dtype=ppci["branch"].dtype)

        i_measurements = self.net.measurement[(self.net.measurement.type == "i")
                                              & (self.net.measurement.element_type == "line")]
        if len(i_measurements):
            meas_from = i_measurements[(i_measurements.bus.values.astype(int) ==
                                        self.net.line.from_bus[i_measurements.element]).values]
            meas_to = i_measurements[(i_measurements.bus.values.astype(int) ==
                                      self.net.line.to_bus[i_measurements.element]).values]
            ix_from = meas_from.element.values.astype(int)
            ix_to = meas_to.element.values.astype(int)
            i_a_to_pu_from = (self.net.bus.vn_kv[meas_from.bus] * 1e3 / self.s_ref).values
            i_a_to_pu_to = (self.net.bus.vn_kv[meas_to.bus] * 1e3 / self.s_ref).values
            branch_append[ix_from, 0] = meas_from.value.values * i_a_to_pu_from
            branch_append[ix_from, 1] = meas_from.std_dev.values * i_a_to_pu_from
            branch_append[ix_to, 2] = meas_to.value.values * i_a_to_pu_to
            branch_append[ix_to, 3] = meas_to.std_dev.values * i_a_to_pu_to

        p_measurements = self.net.measurement[(self.net.measurement.type == "p")
                                              & (self.net.measurement.element_type == "line")]
        if len(p_measurements):
            meas_from = p_measurements[(p_measurements.bus.values.astype(int) ==
                                        self.net.line.from_bus[p_measurements.element]).values]
            meas_to = p_measurements[(p_measurements.bus.values.astype(int) ==
                                      self.net.line.to_bus[p_measurements.element]).values]
            ix_from = meas_from.element.values.astype(int)
            ix_to = meas_to.element.values.astype(int)
            branch_append[ix_from, 4] = meas_from.value.values * 1e3 / self.s_ref
            branch_append[ix_from, 5] = meas_from.std_dev.values * 1e3 / self.s_ref
            branch_append[ix_to, 6] = meas_to.value.values * 1e3 / self.s_ref
            branch_append[ix_to, 7] = meas_to.std_dev.values * 1e3 / self.s_ref

        q_measurements = self.net.measurement[(self.net.measurement.type == "q")
                                              & (self.net.measurement.element_type == "line")]
        if len(q_measurements):
            meas_from = q_measurements[(q_measurements.bus.values.astype(int) ==
                                        self.net.line.from_bus[q_measurements.element]).values]
            meas_to = q_measurements[(q_measurements.bus.values.astype(int) ==
                                      self.net.line.to_bus[q_measurements.element]).values]
            ix_from = meas_from.element.values.astype(int)
            ix_to = meas_to.element.values.astype(int)
            branch_append[ix_from, 8] = meas_from.value.values * 1e3 / self.s_ref
            branch_append[ix_from, 9] = meas_from.std_dev.values * 1e3 / self.s_ref
            branch_append[ix_to, 10] = meas_to.value.values * 1e3 / self.s_ref
            branch_append[ix_to, 11] = meas_to.std_dev.values * 1e3 / self.s_ref

        i_tr_measurements = self.net.measurement[(self.net.measurement.type == "i")
                                                 & (self.net.measurement.element_type ==
                                                 "transformer")]
        if len(i_tr_measurements):
            meas_from = i_tr_measurements[(i_tr_measurements.bus.values.astype(int) ==
                                           self.net.trafo.hv_bus[i_tr_measurements.element]).values]
            meas_to = i_tr_measurements[(i_tr_measurements.bus.values.astype(int) ==
                                         self.net.trafo.lv_bus[i_tr_measurements.element]).values]
            ix_from = meas_from.element.values.astype(int)
            ix_to = meas_to.element.values.astype(int)
            i_a_to_pu_from = (self.net.bus.vn_kv[meas_from.bus] * 1e3 / self.s_ref).values
            i_a_to_pu_to = (self.net.bus.vn_kv[meas_to.bus] * 1e3 / self.s_ref).values
            branch_append[ix_from, 0] = meas_from.value.values * i_a_to_pu_from
            branch_append[ix_from, 1] = meas_from.std_dev.values * i_a_to_pu_from
            branch_append[ix_to, 2] = meas_to.value.values * i_a_to_pu_to
            branch_append[ix_to, 3] = meas_to.std_dev.values * i_a_to_pu_to

        p_tr_measurements = self.net.measurement[(self.net.measurement.type == "p") &
                                                 (self.net.measurement.element_type ==
                                                  "transformer")]
        if len(p_tr_measurements):
            meas_from = p_tr_measurements[(p_tr_measurements.bus.values.astype(int) ==
                                           self.net.trafo.hv_bus[p_tr_measurements.element]).values]
            meas_to = p_tr_measurements[(p_tr_measurements.bus.values.astype(int) ==
                                         self.net.trafo.lv_bus[p_tr_measurements.element]).values]
            ix_from = len(self.net.line) + meas_from.element.values.astype(int)
            ix_to = len(self.net.line) + meas_to.element.values.astype(int)
            branch_append[ix_from, 4] = meas_from.value.values * 1e3 / self.s_ref
            branch_append[ix_from, 5] = meas_from.std_dev.values * 1e3 / self.s_ref
            branch_append[ix_to, 6] = meas_to.value.values * 1e3 / self.s_ref
            branch_append[ix_to, 7] = meas_to.std_dev.values * 1e3 / self.s_ref

        q_tr_measurements = self.net.measurement[(self.net.measurement.type == "q") &
                                                 (self.net.measurement.element_type ==
                                                 "transformer")]
        if len(q_tr_measurements):
            meas_from = q_tr_measurements[(q_tr_measurements.bus.values.astype(int) ==
                                           self.net.trafo.hv_bus[q_tr_measurements.element]).values]
            meas_to = q_tr_measurements[(q_tr_measurements.bus.values.astype(int) ==
                                         self.net.trafo.lv_bus[q_tr_measurements.element]).values]
            ix_from = len(self.net.line) + meas_from.element.values.astype(int)
            ix_to = len(self.net.line) + meas_to.element.values.astype(int)
            branch_append[ix_from, 8] = meas_from.value.values * 1e3 / self.s_ref
            branch_append[ix_from, 9] = meas_from.std_dev.values * 1e3 / self.s_ref
            branch_append[ix_to, 10] = meas_to.value.values * 1e3 / self.s_ref
            branch_append[ix_to, 11] = meas_to.std_dev.values * 1e3 / self.s_ref

        ppci["bus"] = np.hstack((ppci["bus"], bus_append))
        ppci["branch"] = np.hstack((ppci["branch"], branch_append))

        p_bus_not_nan = ~np.isnan(ppci["bus"][:, bs_cols + 2])
        p_line_f_not_nan = ~np.isnan(ppci["branch"][:, br_cols + 4])
        p_line_t_not_nan = ~np.isnan(ppci["branch"][:, br_cols + 6])
        q_bus_not_nan = ~np.isnan(ppci["bus"][:, bs_cols + 4])
        q_line_f_not_nan = ~np.isnan(ppci["branch"][:, br_cols + 8])
        q_line_t_not_nan = ~np.isnan(ppci["branch"][:, br_cols + 10])
        v_bus_not_nan = ~np.isnan(ppci["bus"][:, bs_cols + 0])
        i_line_f_not_nan = ~np.isnan(ppci["branch"][:, br_cols + 0])
        i_line_t_not_nan = ~np.isnan(ppci["branch"][:, br_cols + 2])

        # piece together our measurement vector z
        z = np.concatenate((ppci["bus"][p_bus_not_nan, bs_cols + 2],
                            ppci["branch"][p_line_f_not_nan, br_cols + 4],
                            ppci["branch"][p_line_t_not_nan, br_cols + 6],
                            ppci["bus"][q_bus_not_nan, bs_cols + 4],
                            ppci["branch"][q_line_f_not_nan, br_cols + 8],
                            ppci["branch"][q_line_t_not_nan, br_cols + 10],
                            ppci["bus"][v_bus_not_nan, bs_cols + 0],
                            ppci["branch"][i_line_f_not_nan, br_cols + 0],
                            ppci["branch"][i_line_t_not_nan, br_cols + 2]
                            )).real.astype(np.float64)

        # number of nodes
        n_active = len(np.where(ppci["bus"][:, 1] != 4)[0])
        slack_buses = np.where(ppci["bus"][:, 1] == 3)[0]

        # Check if observability criterion is fulfilled and the state estimation is possible
        if len(z) < 2 * n_active - 1:
            self.logger.error("System is not observable (cancelling)")
            self.logger.error("Measurements available: %d. Measurements required: %d" %
                              (len(z), 2 * n_active - 1))
            return False

        # Set the starting values for all active buses
        v_m = ppci["bus"][:, 7]
        delta = ppci["bus"][:, 8] * np.pi / 180  # convert to rad
        delta_masked = np.ma.array(delta, mask=False)
        delta_masked.mask[slack_buses] = True
        non_slack_buses = np.arange(len(delta))[~delta_masked.mask]

        # Matrix calculation object
        sem = wls_matrix_ops(ppci, slack_buses, non_slack_buses, self.s_ref, bs_cols, br_cols)

        # state vector
        E = np.concatenate((delta_masked.compressed(), v_m))

        # Covariance matrix R
        r_cov = np.concatenate((ppci["bus"][p_bus_not_nan, bs_cols + 3],
                                ppci["branch"][p_line_f_not_nan, br_cols + 5],
                                ppci["branch"][p_line_t_not_nan, br_cols + 7],
                                ppci["bus"][q_bus_not_nan, bs_cols + 5],
                                ppci["branch"][q_line_f_not_nan, br_cols + 9],
                                ppci["branch"][q_line_t_not_nan, br_cols + 11],
                                ppci["bus"][v_bus_not_nan, bs_cols + 1],
                                ppci["branch"][i_line_f_not_nan, br_cols + 1],
                                ppci["branch"][i_line_t_not_nan, br_cols + 3]
                                )).real.astype(np.float64)

        r_inv = csr_matrix(np.linalg.inv(np.diagflat(r_cov) ** 2))

        current_error = 100
        current_iterations = 0

        while current_error > self.tolerance and current_iterations < self.max_iterations:
            self.logger.debug(" Starting iteration %d" % (1 + current_iterations))
            try:

                # create h(x) for the current iteration
                h_x = sem.create_hx(v_m, delta)

                # Residual r
                r = csr_matrix(z - h_x).T

                # Jacobian matrix H
                H = csr_matrix(sem.create_jacobian(v_m, delta))

                # if not np.linalg.cond(H) < 1 / sys.float_info.epsilon:
                #    self.logger.error("Error in matrix H")

                # Gain matrix G_m
                # G_m = H^t * R^-1 * H
                G_m = H.T * (r_inv * H)

                # State Vector difference d_E
                # d_E = G_m^-1 * (H' * R^-1 * r)
                d_E = spsolve(G_m, H.T * (r_inv * r))
                E += d_E

                # Update V/delta
                delta[non_slack_buses] = E[:len(non_slack_buses)]
                v_m = np.squeeze(E[len(non_slack_buses):])

                current_iterations += 1
                current_error = np.max(np.abs(d_E))
                self.logger.debug("Current error: %.4f" % current_error)
            except np.linalg.linalg.LinAlgError:
                self.logger.error("A problem appeared while using the linear algebra methods."
                                  "Check and change the measurement set.")
                return False
        # Print output for results
        if current_error <= self.tolerance:
            successful = True
            self.logger.info("WLS State Estimation successful (%d iterations)" % current_iterations)
        else:
            successful = False
            self.logger.info("WLS State Estimation not successful (%d/%d iterations" %
                             (current_iterations, self.max_iterations))

        # store results for all elements
        # write voltage into ppc
        ppci["bus"][:, 7] = v_m
        ppci["bus"][:, 8] = delta * 180 / np.pi  # convert to degree
        # calculate bus power injections
        v_cpx = v_m * np.exp(1j * delta)
        bus_powers_conj = np.zeros(len(v_cpx), dtype=np.complex128)
        for i in range(len(v_cpx)):
            bus_powers_conj[i] = np.dot(sem.Y_bus[i, :], v_cpx) * np.conjugate(v_cpx[i])
        ppci["bus"][:, 2] = bus_powers_conj.real  # saved in per unit
        ppci["bus"][:, 3] = - bus_powers_conj.imag  # saved in per unit
        # calculate line results (in ppc_i)
        baseMVA, bus, gen, branch = _get_pf_variables_from_ppci(ppci)[0:4]
        out = np.flatnonzero(branch[:, BR_STATUS] == 0)  # out-of-service branches
        br = np.flatnonzero(branch[:, BR_STATUS]).astype(int)  # in-service branches
        # complex power at "from" bus
        Sf = v_cpx[np.real(branch[br, F_BUS]).astype(int)] * np.conj(sem.Yf[br, :] * v_cpx) \
             * baseMVA
        # complex power injected at "to" bus
        St = v_cpx[np.real(branch[br, T_BUS]).astype(int)] * np.conj(sem.Yt[br, :] * v_cpx) \
             * baseMVA
        branch[np.ix_(br, [PF, QF, PT, QT])] = np.c_[Sf.real, Sf.imag, St.real, St.imag]
        branch[np.ix_(out, [PF, QF, PT, QT])] = np.zeros((len(out), 4))
        ppc_i = _store_results_from_pf_in_ppci(ppci, bus, gen, branch)
        # convert to pandapower indices
        ppc = _copy_results_ppci_to_ppc(ppci, ppc, mode="se")
        # extract results from ppc
        _extract_results_se(self, ppc)
        # Store some variables required for Chi^2 and r_N_max test:
        self.R_inv = r_inv.toarray()
        self.Gm = G_m.toarray()
        self.r = r.toarray()
        self.H = H.toarray()
        self.Ht = self.H.T
        self.hx = h_x
        self.V = v_m
        self.delta = delta
        return successful

    def perform_chi2_test(self, v_in_out=None, delta_in_out=None,
                          calculate_voltage_angles=True, chi2_prob_false=0.05):
        """
        The function perform_chi2_test performs a Chi^2 test for bad data and topology error
        detection. The function can be called with the optional input arguments v_in_out and
        delta_in_out. Then, the Chi^2 test is performed after calling the function estimate using
        them as input arguments. It can also be called without these arguments if it is called
        from the same object with which estimate had been called beforehand. Then, the Chi^2 test is
        performed for the states estimated by the funtion estimate and the result, the existence of bad data,
        is given back as a boolean. As a optional argument the probability
        of a false measurement can be provided additionally. For bad data detection, the function
        perform_rn_max_test is more powerful and should be the function of choice. For topology
        error detection, however, perform_chi2_test should be used.

        INPUT:
            **v_in_out** (np.array, shape=(1,), optional) - Vector with initial values for all
            voltage magnitudes in p.u. (sorted by bus index)

            **delta_in_out** (np.array, shape=(1,), optional) - Vector with initial values for all
            voltage angles in degrees (sorted by bus index)

        OPTIONAL:
            **calculate_voltage_angles** - (boolean) - Take into account absolute voltage angles and phase
            shifts in transformers, if init is 'slack'. Default is True.

            **chi2_prob_false** (float) - probability of error / false alarms (standard value: 0.05)

        OUTPUT:
            **successful** (boolean) - True if bad data has been detected

        EXAMPLE:
            perform_chi2_test(np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0]), 0.97)

        """
        # 'flat'-start conditions
        if v_in_out is None:
            v_in_out = np.ones(self.net.bus.shape[0])
        if delta_in_out is None:
            delta_in_out = np.zeros(self.net.bus.shape[0])

        # perform SE
        self.estimate(v_in_out, delta_in_out, calculate_voltage_angles)

        # Performance index J(hx)
        J = np.dot(self.r.T, np.dot(self.R_inv, self.r))

        # Number of measurements
        m = len(self.net.measurement)

        # Number of state variables (the -1 is due to the reference bus)
        n = len(self.V) + len(self.delta) - 1

        # Chi^2 test threshold
        test_thresh = chi2.ppf(1 - chi2_prob_false, m - n)

        # Print results
        self.logger.debug("Result of Chi^2 test:")
        self.logger.debug("Number of measurements: %d" % m)
        self.logger.debug("Number of state variables: %d" % n)
        self.logger.debug("Performance index: %.2f" % J)
        self.logger.debug("Chi^2 test threshold: %.2f" % test_thresh)

        if J <= test_thresh:
            self.bad_data_present = False
            self.logger.info("Chi^2 test passed. No bad data or topology error detected.")
        else:
            self.bad_data_present = True
            self.logger.info("Chi^2 test failed. Bad data or topology error detected.")
        successful = self.bad_data_present
        
        if (v_in_out is not None) and (delta_in_out is not None):
            return successful

    def perform_rn_max_test(self, v_in_out=None, delta_in_out=None,
                            calculate_voltage_angles=True, rn_max_threshold=3.0, chi2_prob_false=0.05):
        """
        The function perform_rn_max_test performs a largest normalized residual test for bad data
        identification and removal. It takes two input arguments: v_in_out and delta_in_out.
        These are the initial state variables for the combined estimation and bad data
        identification and removal process. They can be initialized as described above, e.g.,
        using a "flat" start. In an iterative process, the function performs a state estimation,
        identifies a bad data measurement, removes it from the set of measurements
        (only if the rn_max threshold is violated by the largest residual of all measurements,
        which can be modified), performs the state estimation again,
        and so on and so forth until no further bad data measurements are detected.

        INPUT:
            **v_in_out** (np.array, shape=(1,), optional) - Vector with initial values for all
            voltage magnitudes in p.u. (sorted by bus index)

            **delta_in_out** (np.array, shape=(1,), optional) - Vector with initial values for all
            voltage angles in degrees (sorted by bus index)

        OPTIONAL:
            **calculate_voltage_angles** - (boolean) - Take into account absolute voltage angles and phase
            shifts in transformers, if init is 'slack'. Default is True.

            **rn_max_threshold** (float) - Identification threshold to determine
            if the largest normalized residual reflects a bad measurement
            (standard value of 3.0)

            **chi2_prob_false** (float) - probability of error / false alarms
            (standard value: 0.05)

        OUTPUT:
            **successful** (boolean) - True if the estimation process was successful

        EXAMPLE:
            perform_rn_max_test(np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0]), 5.0, 0.05)

        """
        # 'flat'-start conditions
        if v_in_out is None:
            v_in_out = np.ones(self.net.bus.shape[0])
        if delta_in_out is None:
            delta_in_out = np.zeros(self.net.bus.shape[0])

        num_iterations = 0

        v_in = v_in_out
        delta_in = delta_in_out

        self.bad_data_present = True

        while self.bad_data_present and (num_iterations < 11):
            # Estimate the state with bad data identified in previous iteration
            # removed from set of measurements:
            successful = self.estimate(v_in, delta_in, calculate_voltage_angles)
            v_in_out = self.net.res_bus_est.vm_pu.values
            delta_in_out = self.net.res_bus_est.va_degree.values

            # Perform a Chi^2 test to determine whether bad data is to be removed.
            self.perform_chi2_test(v_in_out, delta_in_out, calculate_voltage_angles=calculate_voltage_angles,
                                   chi2_prob_false=chi2_prob_false)

            try:
                # Error covariance matrix:
                R = np.linalg.inv(self.R_inv)

                # Covariance matrix of the residuals: \Omega = S*R = R - H*G^(-1)*H^T
                # (S is the sensitivity matrix: r = S*e):
                Omega = R - np.dot(self.H, np.dot(np.linalg.inv(self.Gm), self.Ht))

                # Diagonalize \Omega:
                Omega = np.diag(np.diag(Omega))

                # Compute squareroot (|.| since some -0.0 produced nans):
                Omega = np.sqrt(np.absolute(Omega))

                OmegaInv = np.linalg.inv(Omega)

                # Compute normalized residuals (r^N_i = |r_i|/sqrt{Omega_ii}):
                rN = np.dot(OmegaInv, np.absolute(self.r))

                if max(rN) <= rn_max_threshold:
                    self.logger.info(
                        "Largest normalized residual test passed. No bad data detected.")
                else:
                    self.logger.info(
                        "Largest normalized residual test failed. Bad data identified.")

                    if self.bad_data_present:
                        # Identify bad data: Determine index corresponding to max(rN):
                        idx_rN = np.argsort(rN, axis=0)[len(rN) - 1]

                        # Sort measurement indexes:
                        sorted_meas_idxs = np.concatenate(
                            (self.net.measurement.loc[(self.net.measurement['type'] == 'p') & (
                                self.net.measurement['element_type'] == 'bus')].index,
                             self.net.measurement.loc[(self.net.measurement['type'] == 'p') & (
                                 self.net.measurement['element_type'] == 'line')].index,
                             self.net.measurement.loc[(self.net.measurement['type'] == 'q') & (
                                 self.net.measurement['element_type'] == 'bus')].index,
                             self.net.measurement.loc[(self.net.measurement['type'] == 'q') & (
                                 self.net.measurement['element_type'] == 'line')].index,
                             self.net.measurement.loc[(self.net.measurement['type'] == 'v') & (
                                 self.net.measurement['element_type'] == 'bus')].index,
                             self.net.measurement.loc[(self.net.measurement['type'] == 'i') & (
                                 self.net.measurement['element_type'] == 'line')].index))

                        # Determine index of measurement to be removed:
                        meas_idx = sorted_meas_idxs[idx_rN]

                        # Remove bad measurement:
                        self.net.measurement.drop(meas_idx, inplace=True)
                        self.logger.debug("Bad data removed from the set of measurements.")
                    else:
                        self.logger.debug("No bad data removed from the set of measurements.")
            except np.linalg.linalg.LinAlgError:
                self.logger.error("A problem appeared while using the linear algebra methods."
                                  "Check and change the measurement set.")
                return False

            self.logger.debug("rn_max identification threshold: %.2f" % rn_max_threshold)

            num_iterations += 1

        return successful


def _extract_results_se(self, ppc):
    """
    This function extracts all important results from 'ppc'. 
    It creates data frameworks for the SE's results in net
    and stores in these the new results of the performed SE.
    Thereby old results of a powerflow analysis will be protected against data loss.
    Note: You can find previous pf-results in the known data structure (res_xxx).
    """
    
    # create 'res_xxx_est' data frameworks
    self.net.res_bus_est = self.net._empty_res_bus.copy()
    self.net.res_line_est = self.net._empty_res_line.copy()
    self.net.res_dcline_est = self.net._empty_res_dcline.copy()
    self.net.res_ext_grid_est = self.net._empty_res_ext_grid.copy()
    self.net.res_trafo_est = self.net._empty_res_trafo.copy()
    self.net.res_trafo3w_est = self.net._empty_res_trafo3w.copy()
    self.net.res_load_est = self.net._empty_res_load.copy()
    self.net.res_gen_est = self.net._empty_res_gen.copy()
    self.net.res_sgen_est = self.net._empty_res_sgen.copy()
    self.net.res_impedance_est = self.net._empty_res_impedance.copy()
    self.net.res_shunt_est = self.net._empty_res_shunt.copy()
    self.net.res_ward_est = self.net._empty_res_ward.copy()
    self.net.res_xward_est = self.net._empty_res_xward.copy()
    
    # store bus results
    bus_idx = self.mapping_table[self.net["bus"].index.values]
    self.net["res_bus_est"]["vm_pu"] = ppc["bus"][bus_idx][:, 7]
    self.net["res_bus_est"]["va_degree"] = ppc["bus"][bus_idx][:, 8]

    self.net.res_bus_est.p_kw = - get_values(ppc["bus"][:, 2], self.net.bus.index,
                                             self.mapping_table) * self.s_ref / 1e3
    self.net.res_bus_est.q_kvar = - get_values(ppc["bus"][:, 3], self.net.bus.index,
                                               self.mapping_table) * self.s_ref / 1e3
    
    # store previous pf-results
    pre_results_bus = self.net.res_bus.copy()
    # check if previous pf-results are available 
    check_pre_pf_results = len(self.net.res_line) > 0
    if check_pre_pf_results:
        pre_results_line = self.net.res_line.copy()
        pre_results_dcline = self.net.res_dcline.copy()
        pre_results_ext_grid = self.net.res_ext_grid.copy()
        pre_results_trafo = self.net.res_trafo.copy()
        pre_results_trafo3w = self.net.res_trafo3w.copy()
        pre_results_load = self.net.res_load.copy()
        pre_results_gen = self.net.res_gen.copy()
        pre_results_sgen = self.net.res_sgen.copy()
        pre_results_impedance = self.net.res_impedance.copy()
        pre_results_shunt = self.net.res_shunt.copy()
        pre_results_ward = self.net.res_ward.copy()
        pre_results_xward = self.net.res_xward.copy()
    
    # get results
    _add_pf_options(self.net, tolerance_kva=1e-5, trafo_loading="current",
                    numba=True, ac=True, algorithm='nr', max_iteration="auto")
    _extract_results(self.net, ppc)
    
    # copy results to 'est'-dictionaries
    self.net.res_line_est = self.net.res_line
    self.net.res_dcline_est = self.net.res_dcline
    self.net.res_ext_grid_est = self.net.res_ext_grid
    self.net.res_trafo_est = self.net.res_trafo
    self.net.res_trafo3w_est = self.net.res_trafo3w
    self.net.res_load_est = self.net.res_load
    self.net.res_gen_est = self.net.res_gen
    self.net.res_sgen_est = self.net.res_sgen
    self.net.res_impedance_est = self.net.res_impedance
    self.net.res_shunt_est = self.net.res_shunt
    self.net.res_ward_est = self.net.res_ward
    self.net.res_xward_est = self.net.res_xward
    
    # reset 'res'-dictionaries to previous pf-results (if required)
    if check_pre_pf_results:
        self.net.res_line = pre_results_line
        self.net.res_dcline = pre_results_dcline
        self.net.res_ext_grid = pre_results_ext_grid
        self.net.res_trafo_est = pre_results_trafo
        self.net.res_trafo3w_est = pre_results_trafo3w
        self.net.res_load = pre_results_load
        self.net.res_gen = pre_results_gen
        self.net.res_sgen = pre_results_sgen
        self.net.res_impedance = pre_results_impedance
        self.net.res_shunt = pre_results_shunt
        self.net.res_ward = pre_results_ward
        self.net.res_xward = pre_results_xward
    else:
        reset_results(self.net)
        self.net.res_bus = pre_results_bus


if __name__ == "__main__":
    from pandapower.test.estimation.test_wls_estimation import test_3bus
    test_3bus()
