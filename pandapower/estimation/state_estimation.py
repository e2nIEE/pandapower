# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

import numpy as np
import logging
import pandas as pd
import warnings
from pandapower.estimation.wls_matrix_ops import wls_matrix_ops
from pandapower.pd2ppc import _pd2ppc
from pandapower.results import _set_buses_out_of_service
from pandapower.auxiliary import get_values, _select_is_elements, calculate_line_results
from pandapower.toolbox import convert_format
from pandapower.topology import estimate_voltage_vector
from pypower.ext2int import ext2int
from pypower.int2ext import int2ext
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def estimate(net, init='flat', tolerance=1e-6, maximum_iterations=10,
             calculate_voltage_angles=True):
    """
    Wrapper function for WLS state estimation.

    Input:
        **net** - The net within this line should be created.

        **init** - (string) Initial voltage for the estimation. 'flat' sets 1.0 p.u. / 0° for all
        buses, 'results' uses the values from *res_bus_est* if available and 'slack' considers the
        slack bus voltage (and optionally, angle) as the initial values. Default is 'flat'.

        **tolerance** - (float) - When the maximum state change between iterations is less than
        tolerance, the process stops. Default is 1e-6.

        **maximum_iterations** - (int) - Maximum number of iterations. Default is 10.

        **calculate_voltage_angles** - (bool) - Take into account absolute voltage angles and phase
        shifts in transformers, if init is 'slack'. Default is True.

    Return:
        (bool) Was the state estimation successful?
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
            self.logger = logging.getLogger("wls_se")
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

        Input:

            **net** - The net within this line should be created

            **v_start** (np.array, shape=(1,), optional) - Vector with initial values for all
            voltage magnitudes in p.u. (sorted by bus index)

            **delta_start** (np.array, shape=(1,), optional) - Vector with initial values for all
            voltage angles in degrees (sorted by bus index)

            **calculate_voltage_angles** - (bool) - Take into account absolute voltage angles and
            phase shifts in transformers Default is True.
        Return:

            **successful** (boolean) - True if the estimation process was successful

        Optional estimation variables:

            The bus power injections can be accessed with *se.s_node_powers* and the estimated
            values corresponding to the (noisy) measurement values with *se.hx*. (*hx* denotes h(x))

        Example:

            success = estimate(np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0]))

        """
        if self.net is None:
            raise Exception("Component was not initialized with a network.")

        # add initial values for V and delta
        # node voltages
        # V<delta
        if v_start is None:
            v_start = np.ones(self.net.bus.shape[0])
        if delta_start is None:
            delta_start = np.zeros(self.net.bus.shape[0])

        # initialize the ppc bus with the initial values given
        vm_backup, va_backup = self.net.res_bus.vm_pu.copy(), self.net.res_bus.va_degree.copy()
        self.net.res_bus.vm_pu = v_start
        self.net.res_bus.vm_pu[self.net.bus.index[self.net.bus.in_service == False]] = np.nan
        self.net.res_bus.va_degree = delta_start

        # select elements in service and convert pandapower ppc to ppc
        is_elems = _select_is_elements(self.net)
        ppc, _, mapping_table = _pd2ppc(self.net, is_elems, init_results=True,
                                        calculate_voltage_angles=calculate_voltage_angles)
        br_cols = ppc["branch"].shape[1]
        bs_cols = ppc["bus"].shape[1]

        self.net.res_bus.vm_pu = vm_backup
        self.net.res_bus.va_degree = va_backup

        # add 6 columns to ppc[bus] for Vm, Vm std dev, P, P std dev, Q, Q std dev
        bus_append = np.full((ppc["bus"].shape[0], 6), np.nan, dtype=ppc["bus"].dtype)

        v_measurements = self.net.measurement[(self.net.measurement.type == "v")
                                              & (self.net.measurement.element_type == "bus")]
        if len(v_measurements):
            bus_positions = mapping_table[v_measurements.bus.values.astype(int)]
            bus_append[bus_positions, 0] = v_measurements.value.values
            bus_append[bus_positions, 1] = v_measurements.std_dev.values

        p_measurements = self.net.measurement[(self.net.measurement.type == "p")
                                              & (self.net.measurement.element_type == "bus")]
        if len(p_measurements):
            bus_positions = mapping_table[p_measurements.bus.values.astype(int)]
            bus_append[bus_positions, 2] = p_measurements.value.values * 1e3 / self.s_ref
            bus_append[bus_positions, 3] = p_measurements.std_dev.values * 1e3 / self.s_ref

        q_measurements = self.net.measurement[(self.net.measurement.type == "q")
                                              & (self.net.measurement.element_type == "bus")]
        if len(q_measurements):
            bus_positions = mapping_table[q_measurements.bus.values.astype(int)]
            bus_append[bus_positions, 4] = q_measurements.value.values * 1e3 / self.s_ref
            bus_append[bus_positions, 5] = q_measurements.std_dev.values * 1e3 / self.s_ref

        # add virtual measurements for artificial buses, which were created because
        # of an open line switch. p/q are 0. and std dev is 1. (small value)
        new_in_line_buses = np.setdiff1d(np.arange(ppc["bus"].shape[0]),
                                         mapping_table[mapping_table >= 0])
        bus_append[new_in_line_buses, 2] = 0.
        bus_append[new_in_line_buses, 3] = 1.
        bus_append[new_in_line_buses, 4] = 0.
        bus_append[new_in_line_buses, 5] = 1.

        # add 12 columns to mpc[branch] for Im_from, Im_from std dev, Im_to, Im_to std dev,
        # P_from, P_from std dev, P_to, P_to std dev, Q_from,Q_from std dev,  Q_to, Q_to std dev
        branch_append = np.full((ppc["branch"].shape[0], 12), np.nan, dtype=ppc["branch"].dtype)

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

        ppc["bus"] = np.hstack((ppc["bus"], bus_append))
        ppc["branch"] = np.hstack((ppc["branch"], branch_append))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ppc_i = ext2int(ppc)

        p_bus_not_nan = ~np.isnan(ppc_i["bus"][:, bs_cols + 2])
        p_line_f_not_nan = ~np.isnan(ppc_i["branch"][:, br_cols + 4])
        p_line_t_not_nan = ~np.isnan(ppc_i["branch"][:, br_cols + 6])
        q_bus_not_nan = ~np.isnan(ppc_i["bus"][:, bs_cols + 4])
        q_line_f_not_nan = ~np.isnan(ppc_i["branch"][:, br_cols + 8])
        q_line_t_not_nan = ~np.isnan(ppc_i["branch"][:, br_cols + 10])
        v_bus_not_nan = ~np.isnan(ppc_i["bus"][:, bs_cols + 0])
        i_line_f_not_nan = ~np.isnan(ppc_i["branch"][:, br_cols + 0])
        i_line_t_not_nan = ~np.isnan(ppc_i["branch"][:, br_cols + 2])

        # piece together our measurement vector z
        z = np.concatenate((ppc_i["bus"][p_bus_not_nan, bs_cols + 2],
                            ppc_i["branch"][p_line_f_not_nan, br_cols + 4],
                            ppc_i["branch"][p_line_t_not_nan, br_cols + 6],
                            ppc_i["bus"][q_bus_not_nan, bs_cols + 4],
                            ppc_i["branch"][q_line_f_not_nan, br_cols + 8],
                            ppc_i["branch"][q_line_t_not_nan, br_cols + 10],
                            ppc_i["bus"][v_bus_not_nan, bs_cols + 0],
                            ppc_i["branch"][i_line_f_not_nan, br_cols + 0],
                            ppc_i["branch"][i_line_t_not_nan, br_cols + 2]
                            )).real.astype(np.float64)

        # number of nodes
        n_active = len(np.where(ppc_i["bus"][:, 1] != 4)[0])
        slack_buses = np.where(ppc_i["bus"][:, 1] == 3)[0]

        # Check if observability criterion is fulfilled and the state estimation is possible
        if len(z) < 2 * n_active - 1:
            self.logger.error("System is not observable (cancelling)")
            self.logger.error("Measurements available: %d. Measurements required: %d" %
                              (len(z), 2 * n_active - 1))
            return False

        # Set the starting values for all active buses
        v_m = ppc_i["bus"][:, 7]
        delta = ppc_i["bus"][:, 8] * np.pi / 180  # convert to rad
        delta_masked = np.ma.array(delta, mask=False)
        delta_masked.mask[slack_buses] = True
        non_slack_buses = np.arange(len(delta))[~delta_masked.mask]

        # Matrix calculation object
        sem = wls_matrix_ops(ppc_i, slack_buses, non_slack_buses, self.s_ref, bs_cols, br_cols)

        # state vector
        E = np.concatenate((delta_masked.compressed(), v_m))

        # Covariance matrix R
        r_cov = np.concatenate((ppc_i["bus"][p_bus_not_nan, bs_cols + 3],
                                ppc_i["branch"][p_line_f_not_nan, br_cols + 5],
                                ppc_i["branch"][p_line_t_not_nan, br_cols + 7],
                                ppc_i["bus"][q_bus_not_nan, bs_cols + 5],
                                ppc_i["branch"][q_line_f_not_nan, br_cols + 9],
                                ppc_i["branch"][q_line_t_not_nan, br_cols + 11],
                                ppc_i["bus"][v_bus_not_nan, bs_cols + 1],
                                ppc_i["branch"][i_line_f_not_nan, br_cols + 1],
                                ppc_i["branch"][i_line_t_not_nan, br_cols + 3]
                                )).real.astype(np.float64)

        r_inv = csr_matrix(np.linalg.inv(np.diagflat(r_cov) ** 2))

        current_error = 100
        current_iterations = 0

        while current_error > self.tolerance and current_iterations < self.max_iterations:
            self.logger.debug("Iteration %d" % (1 + current_iterations))

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
            self.logger.debug("Error: " + str(current_error))

        # Print output for results
        self.logger.debug("Finished (" + str(current_iterations) + "/" + str(self.max_iterations) +
                          " iterations)")

        if current_error <= self.tolerance:
            successful = True
            self.logger.debug("==> Successful")
        else:
            successful = False
            self.logger.debug("==> Not successful")

        # write voltage into ppc
        ppc_i["bus"][:, 7] = v_m
        ppc_i["bus"][:, 8] = delta * 180 / np.pi  # convert to degree

        # calculate bus powers
        v_cpx = v_m * np.exp(1j * delta)
        bus_powers_conj = np.zeros(len(v_cpx), dtype=np.complex128)
        for i in range(len(v_cpx)):
            bus_powers_conj[i] = np.dot(sem.Y_bus[i, :], v_cpx) * np.conjugate(v_cpx[i])
        ppc_i["bus"][:, 2] = bus_powers_conj.real  # saved in per unit
        ppc_i["bus"][:, 3] = - bus_powers_conj.imag  # saved in per unit

        # convert to pandapower indices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ppc = int2ext(ppc_i)
            _set_buses_out_of_service(ppc)

        # Store results, overwrite old results
        self.net.res_bus_est = pd.DataFrame(columns=["vm_pu", "va_degree", "p_kw", "q_kvar"],
                                            index=self.net.bus.index)
        self.net.res_line_est = pd.DataFrame(columns=["p_from_kw", "q_from_kvar", "p_to_kw",
                                                      "q_to_kvar", "pl_kw", "ql_kvar", "i_from_ka",
                                                      "i_to_ka", "i_ka", "loading_percent"],
                                             index=self.net.line.index)

        bus_idx = mapping_table[self.net["bus"].index.values]
        self.net["res_bus_est"]["vm_pu"] = ppc["bus"][bus_idx][:, 7]
        self.net["res_bus_est"]["va_degree"] = ppc["bus"][bus_idx][:, 8]

        self.net.res_bus_est.p_kw = -  get_values(ppc["bus"][:, 2], self.net.bus.index,
                                                  mapping_table) * self.s_ref / 1e3
        self.net.res_bus_est.q_kvar = - get_values(ppc["bus"][:, 3], self.net.bus.index,
                                                   mapping_table) * self.s_ref / 1e3
        self.net.res_line_est = calculate_line_results(self.net, use_res_bus_est=True)

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

if __name__ == "__main__":
    from pandapower.test.test_wls_estimation import test_3bus
    test_3bus()
