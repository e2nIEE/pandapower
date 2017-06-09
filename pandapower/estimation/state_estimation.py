# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import numpy as np


from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.stats import chi2

from pandapower.estimation.wls_ppc_conversions import _add_measurements_to_ppc, \
    _build_measurement_vectors, _init_ppc
from pandapower.estimation.results import _copy_power_flow_results, _rename_results
from pandapower.idx_brch import F_BUS, T_BUS, BR_STATUS, PF, PT, QF, QT
from pandapower.auxiliary import _add_pf_options, get_values
from pandapower.estimation.wls_matrix_ops import wls_matrix_ops
from pandapower.pf.runpf_pypower import _get_pf_variables_from_ppci, \
    _store_results_from_pf_in_ppci
from pandapower.results import _copy_results_ppci_to_ppc, _extract_results
from pandapower.topology import estimate_voltage_vector

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
        **bad_data_detected** (boolean) - Returns true if bad data has been detected
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
        # variables for chi^2 / rn_max tests
        self.hx = None
        self.R_inv = None
        self.H = None
        self.Ht = None
        self.Gm = None
        self.r = None
        self.V = None
        self.pp_meas_indices = None
        self.delta = None
        self.bad_data_present = None

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

        # initialize result tables if not existent
        _copy_power_flow_results(self.net)

        # initialize ppc
        ppc, ppci = _init_ppc(self.net, v_start, delta_start, calculate_voltage_angles)
        mapping_table = self.net["_pd2ppc_lookups"]["bus"]

        # add measurements to ppci structure
        ppci = _add_measurements_to_ppc(self.net, mapping_table, ppci, self.s_ref)

        # calculate relevant vectors from ppci measurements
        z, self.pp_meas_indices, r_cov = _build_measurement_vectors(ppci)

        # number of nodes
        n_active = len(np.where(ppci["bus"][:, 1] != 4)[0])
        slack_buses = np.where(ppci["bus"][:, 1] == 3)[0]

        # Check if observability criterion is fulfilled and the state estimation is possible
        if len(z) < 2 * n_active - 1:
            self.logger.error("System is not observable (cancelling)")
            self.logger.error("Measurements available: %d. Measurements required: %d" %
                              (len(z), 2 * n_active - 1))
            return False

        # set the starting values for all active buses
        v_m = ppci["bus"][:, 7]
        delta = ppci["bus"][:, 8] * np.pi / 180  # convert to rad
        delta_masked = np.ma.array(delta, mask=False)
        delta_masked.mask[slack_buses] = True
        non_slack_buses = np.arange(len(delta))[~delta_masked.mask]

        # matrix calculation object
        sem = wls_matrix_ops(ppci, slack_buses, non_slack_buses, self.s_ref)

        # state vector
        E = np.concatenate((delta_masked.compressed(), v_m))

        # invert covariance matrix
        r_inv = csr_matrix(np.linalg.inv(np.diagflat(r_cov) ** 2))

        current_error = 100.
        current_iterations = 0

        while current_error > self.tolerance and current_iterations < self.max_iterations:
            self.logger.debug(" Starting iteration %d" % (1 + current_iterations))
            try:
                # create h(x) for the current iteration
                h_x = sem.create_hx(v_m, delta)

                # residual r
                r = csr_matrix(z - h_x).T

                # jacobian matrix H
                H = csr_matrix(sem.create_jacobian(v_m, delta))

                # gain matrix G_m
                # G_m = H^t * R^-1 * H
                G_m = H.T * (r_inv * H)

                # state vector difference d_E
                # d_E = G_m^-1 * (H' * R^-1 * r)
                d_E = spsolve(G_m, H.T * (r_inv * r))
                E += d_E

                # update V/delta
                delta[non_slack_buses] = E[:len(non_slack_buses)]
                v_m = np.squeeze(E[len(non_slack_buses):])

                # prepare next iteration
                current_iterations += 1
                current_error = np.max(np.abs(d_E))
                self.logger.debug("Current error: %.7f" % current_error)

            except np.linalg.linalg.LinAlgError:
                self.logger.error("A problem appeared while using the linear algebra methods."
                                  "Check and change the measurement set.")
                return False

        # print output for results
        if current_error <= self.tolerance:
            successful = True
            self.logger.info("WLS State Estimation successful (%d iterations)" % current_iterations)
        else:
            successful = False
            self.logger.info("WLS State Estimation not successful (%d/%d iterations)" %
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
        s_ref, bus, gen, branch = _get_pf_variables_from_ppci(ppci)[0:4]
        out = np.flatnonzero(branch[:, BR_STATUS] == 0)  # out-of-service branches
        br = np.flatnonzero(branch[:, BR_STATUS]).astype(int)  # in-service branches
        # complex power at "from" bus
        Sf = v_cpx[np.real(branch[br, F_BUS]).astype(int)] * np.conj(sem.Yf[br, :] * v_cpx) * s_ref
        # complex power injected at "to" bus
        St = v_cpx[np.real(branch[br, T_BUS]).astype(int)] * np.conj(sem.Yt[br, :] * v_cpx) * s_ref
        branch[np.ix_(br, [PF, QF, PT, QT])] = np.c_[Sf.real, Sf.imag, St.real, St.imag]
        branch[np.ix_(out, [PF, QF, PT, QT])] = np.zeros((len(out), 4))
        ppci = _store_results_from_pf_in_ppci(ppci, bus, gen, branch)

        # convert to pandapower indices
        ppc = _copy_results_ppci_to_ppc(ppci, ppc, mode="se")

        # extract results from ppc
        _add_pf_options(self.net, tolerance_kva=1e-5, trafo_loading="current",
                        numba=True, ac=True, algorithm='nr', max_iteration="auto")
        _extract_results(self.net, ppc)

        # restore backup of previous results
        _rename_results(self.net)

        # additionally, write bus results (these are not written in _extract_results)
        self.net.res_bus_est.p_kw = - get_values(ppc["bus"][:, 2], self.net.bus.index,
                                                 mapping_table) * self.s_ref / 1e3
        self.net.res_bus_est.q_kvar = - get_values(ppc["bus"][:, 3], self.net.bus.index,
                                                   mapping_table) * self.s_ref / 1e3

        # store variables required for chi^2 and r_N_max test:
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

        if (v_in_out is not None) and (delta_in_out is not None):
            return self.bad_data_present

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
            **successful** (boolean) - True if all bad data could be removed

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

        while num_iterations <= 10:
            # Estimate the state with bad data identified in previous iteration
            # removed from set of measurements:
            _ = self.estimate(v_in, delta_in, calculate_voltage_angles)
            v_in_out = self.net.res_bus_est.vm_pu.values
            delta_in_out = self.net.res_bus_est.va_degree.values

            # Perform a Chi^2 test to determine whether bad data is to be removed.
            self.bad_data_present = self.perform_chi2_test(v_in_out, delta_in_out,
                                                           calculate_voltage_angles=
                                                           calculate_voltage_angles,
                                                           chi2_prob_false=chi2_prob_false)

            # If bad data was removed in the previous iterations, return True
            if not self.bad_data_present:
                return True

            # Try to remove the bad data
            try:
                # Error covariance matrix:
                R = np.linalg.inv(self.R_inv)

                # todo for future debugging: this line's results have changed with the ppc
                # overhaul in April 2017 after commit 9ae5b8f42f69ae39f8c8cf (which still works)
                # there are differences of < 1e-10 for the Omega entries which cause
                # the function to work far worse. As of now it is unclear if it's just numerical
                # accuracy to blame or an error in the code. a sort in the ppc creation function
                # was removed which caused this issue
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

                    # Identify bad data: Determine index corresponding to max(rN):
                    idx_rN = np.argsort(rN, axis=0)[-1]

                    # Determine pandapower index of measurement to be removed:
                    meas_idx = self.pp_meas_indices[idx_rN]

                    # Remove bad measurement:
                    self.logger.debug("Removing measurement: %s"
                                      % self.net.measurement.loc[meas_idx].values[0])
                    self.net.measurement.drop(meas_idx, inplace=True)
                    self.logger.debug("Bad data removed from the set of measurements.")

            except np.linalg.linalg.LinAlgError:
                self.logger.error("A problem appeared while using the linear algebra methods."
                                  "Check and change the measurement set.")
                return False

            self.logger.debug("rN_max identification threshold: %.2f" % rn_max_threshold)
            num_iterations += 1

        return not self.bad_data_present
