__author__ = 'menke, nbornhorst'
import numpy as np
import logging
import pandas as pd
import warnings
from scipy.stats import chi2
from pandapower.estimation.wls_matrix_ops import se_matrix
from pandapower.run import _pd2ppc, _select_is_elements
from pandapower.results import _set_buses_out_of_service
from pandapower.auxiliary import get_values
from pypower.ext2int import ext2int
from pypower.int2ext import int2ext
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class state_estimation:
    """
    Any user of the estimation module only needs to use the class state_estimation. It contains all
    relevant functions to control and operator the module. Two functions are used to configure the
    system according to the users needs while one function is used for the actual estimation
    process.
    """

    def __init__(self):
        self.logger = logging.getLogger("wls_se")
        self.tolerance = 1e-6
        self.max_iterations = 10
        self.net = None
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
        self.s_bus_kva = None
        self.bad_data_present = None
        # Offset to accomodate pypower <-> pandapower differences (additional columns)
        self.br_col_offset = 6

    def configure(self, tolerance, maximum_iterations, net=None):
        """
        The function configure takes up to 3 arguments and configures the process. The first
        argument tolerance sets the tolerance limit, at which the process has found a solution and
        is stopped. The largest difference in a state variable between the current iteration and the
        last iteration is compared to the tolerance value. If the largest difference is smaller or
        equal to the given tolerance, the process is successfully terminated. The second argument
        maximum_iterations sets a maximum amount of iterations done by the process. If the tolerance
        is not reached in the maximum amount of iterations, the process is unsuccessfully
        terminated. The third and optional argument net is the pandapower network on which the
        estimation process is conducted.

        Input:

            **tolerance** (float) - When the change between iterations is less than tolerance,
            the process stops. Default is 1e-6.

            **maximum_iterations** (int) - Maximum number of iterations. Default is 20.

        Optional:

            **net** - The net within this line should be created.

        Example:

            configure(1e-4, 10, net)

        """
        self.tolerance = tolerance
        self.max_iterations = maximum_iterations
        if net:
            self.set_grid(net)

    # Set grid data
    def set_grid(self, net, s_ref=None):
        """
        If the network is not set by using the configure-function, it can be set by the function
        set_grid. This function only takes two arguments, the first one being the pandapower
        network. The second argument s_ref is a reference apparent power value for the network in
        VA. It is used to convert from and to per-unit values.

        Input:

            **net** - The net within this line should be created

        Optional:

            **s_ref** (float) - Reference power for the network. Default is 1e6 VA.

        Example:

            set_grid(net, 1e6)

        """
        self.net = net
        if s_ref:
            self.s_ref = s_ref

    def estimate(self, u_in=None, delta_in=None):
        """
        The function estimate is the main function of the module. It takes two input arguments: u_in
        and delta_in. These are the initial state variables for the estimation process. Usually they
        can be initialized in a "flart-start" condition: All voltages being 1.0 pu and all voltage
        angles being 0 degrees. If the estimation is applied continously, using the results from the
        last estimation as the starting condition for the current estimation can decrease the amount
        of iterations needed to estimate the current state. Returned is a boolean value, which is
        true after a successful estimation and false otherwise.
        The resulting complex voltage will be written into the pandapower network. The result
        fields are found res_bus_est of the pandapower network.

        Input:

            **net** - The net within this line should be created

            **u_in** (np.array, shape=(1,)) - Vector with initial values for all voltage magnitudes
            in p.u. (sorted by bus index)

            **delta_in** (np.array, shape=(1,)) - Vector with initial values for all voltage angles
            in degrees (sorted by bus index)

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
        if u_in is None:
            u_in = np.ones(self.net.bus.shape[0])
        if delta_in is None:
            delta_in = np.zeros(self.net.bus.shape[0])

        # initialize the ppc bus with the initial values given
        vm_backup, va_backup = self.net.res_bus.vm_pu.copy(), self.net.res_bus.va_degree.copy()
        self.net.res_bus.vm_pu = u_in
        self.net.res_bus.va_degree = delta_in

        # select elements in service and convert pandapower ppc to ppc
        is_elems = _select_is_elements(self.net)
        ppc, _, mapping_table = _pd2ppc(self.net, is_elems, init_results=True)

        self.net.res_bus.vm_pu = vm_backup
        self.net.res_bus.va_degree = va_backup

        # add 6 columns to ppc[bus] for Vm, Vm std dev, P, P std dev, Q, Q std dev
        bus_append = np.full((ppc["bus"].shape[0], 6), np.nan, dtype=ppc["bus"].dtype)

        v_measurements = self.net.measurement[self.net.measurement.type == "vbus_pu"]
        if len(v_measurements):
            bus_positions = mapping_table[v_measurements.bus.values.astype(int)]
            bus_append[bus_positions, 0] = v_measurements.value.values
            bus_append[bus_positions, 1] = v_measurements.std_dev.values

        p_measurements = self.net.measurement[self.net.measurement.type == "pbus_kw"]
        if len(p_measurements):
            bus_positions = mapping_table[p_measurements.bus.values.astype(int)]
            bus_append[bus_positions, 2] = p_measurements.value.values * 1e3 / self.s_ref
            bus_append[bus_positions, 3] = p_measurements.std_dev.values * 1e3 / self.s_ref

        q_measurements = self.net.measurement[self.net.measurement.type == "qbus_kvar"]
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

        # add 9 columns to mpc[branch] for Im, Im position, Im std dev, P, P position, P std dev,
        #  Q, Q position, Q std dev
        branch_append = np.full((ppc["branch"].shape[0], 9), np.nan, dtype=ppc["branch"].dtype)

        i_measurements = self.net.measurement[self.net.measurement.type == "iline_a"]
        if len(i_measurements):
            i_a_to_pu = (self.net.bus.vn_kv[self.net.line.from_bus[i_measurements.line]] * 1e3
                         / self.s_ref).values
            ix = i_measurements.line.values.astype(int)
            branch_append[ix, 0] = i_measurements.value.values * i_a_to_pu
            branch_append[ix, 1] = mapping_table[i_measurements.bus.values.astype(int)]
            branch_append[ix, 2] = i_measurements.std_dev.values * i_a_to_pu

        p_measurements = self.net.measurement[self.net.measurement.type == "pline_kw"]
        if len(p_measurements):
            ix = p_measurements.line.values.astype(int)
            branch_append[ix, 3] = p_measurements.value.values * 1e3 / self.s_ref
            branch_append[ix, 4] = mapping_table[p_measurements.bus.values.astype(int)]
            branch_append[ix, 5] = p_measurements.std_dev.values * 1e3 / self.s_ref

        q_measurements = self.net.measurement[self.net.measurement.type == "qline_kvar"]
        if len(q_measurements):
            ix = q_measurements.line.values.astype(int)
            branch_append[ix, 6] = q_measurements.value.values * 1e3 / self.s_ref
            branch_append[ix, 7] = mapping_table[q_measurements.bus.values.astype(int)]
            branch_append[ix, 8] = q_measurements.std_dev.values * 1e3 / self.s_ref

        ppc["bus"] = np.hstack((ppc["bus"], bus_append))
        ppc["branch"] = np.hstack((ppc["branch"], branch_append))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ppc_i = ext2int(ppc)

        # adjust line measurement buses to internal index
        for ix in range(ppc_i["branch"].shape[0]):
            p = ppc_i["branch"][ix, 17 + self.br_col_offset].real
            q = ppc_i["branch"][ix, 20 + self.br_col_offset].real
            i = ppc_i["branch"][ix, 14 + self.br_col_offset].real
            if ~np.isnan(p):
                ppc_i["branch"][ix, 17 + self.br_col_offset] = ppc_i["order"]["bus"]["e2i"][int(p)]
            if ~np.isnan(q):
                ppc_i["branch"][ix, 20 + self.br_col_offset] = ppc_i["order"]["bus"]["e2i"][int(q)]
            if ~np.isnan(i):
                ppc_i["branch"][ix, 14 + self.br_col_offset] = ppc_i["order"]["bus"]["e2i"][int(i)]

        p_bus_not_nan = ~np.isnan(ppc_i["bus"][:, 15])
        p_line_not_nan = ~np.isnan(ppc_i["branch"][:, 16 + self.br_col_offset])
        q_bus_not_nan = ~np.isnan(ppc_i["bus"][:, 17])
        q_line_not_nan = ~np.isnan(ppc_i["branch"][:, 19 + self.br_col_offset])
        v_bus_not_nan = ~np.isnan(ppc_i["bus"][:, 13])
        i_line_not_nan = ~np.isnan(ppc_i["branch"][:, 13 + self.br_col_offset])
        z = np.concatenate((ppc_i["bus"][p_bus_not_nan, 15],
                            ppc_i["branch"][p_line_not_nan, 16 + self.br_col_offset],
                            ppc_i["bus"][q_bus_not_nan, 17],
                            ppc_i["branch"][q_line_not_nan, 19 + self.br_col_offset],
                            ppc_i["bus"][v_bus_not_nan, 13],
                            ppc_i["branch"][i_line_not_nan, 13 + self.br_col_offset]
                            )).real.astype(np.float64)

        # number of nodes
        n_active = len(np.where(ppc_i["bus"][:, 1] != 4)[0])
        slack_bus = np.where(ppc_i["bus"][:, 1] == 3)[0][0]

        # Check if observability criterion is fulfilled and the state estimation is possible
        if len(z) < 2 * n_active - 1:
            self.logger.error("System is not observable (cancelling)")
            self.logger.error("Measurements available: %d. Measurements required: %d" %
                          (len(z), 2 * n_active - 1))
            return False

        # Matrix calculation object
        sem = se_matrix(ppc_i, slack_bus, self.s_ref)

        # Set the starting values for all active buses
        v_m = ppc_i["bus"][:, 7]
        delta = ppc_i["bus"][:, 8] * np.pi / 180  # convert to rad

        # state vector
        E = np.concatenate((delta[:slack_bus], delta[slack_bus + 1:], v_m))

        # Covariance matrix R
        r_cov = np.concatenate((ppc_i["bus"][p_bus_not_nan, 16],
                                ppc_i["branch"][p_line_not_nan, 18 + self.br_col_offset],
                                ppc_i["bus"][q_bus_not_nan, 18],
                                ppc_i["branch"][q_line_not_nan, 21 + self.br_col_offset],
                                ppc_i["bus"][v_bus_not_nan, 14],
                                ppc_i["branch"][i_line_not_nan, 15 + self.br_col_offset]
                                )).real.astype(np.float64)

        r_inv = csr_matrix(np.linalg.inv(np.diagflat(r_cov) ** 2))

        current_error = 100
        current_iterations = 0

        while current_error > self.tolerance and current_iterations < self.max_iterations:
            self.logger.debug("Iteration %d" % (1 + current_iterations))

            # create h(x) for the current iteration
            h_x = sem.create_hx(v_m, delta)

            # Residual r
            r = csr_matrix(z - h_x)

            # Jacobian matrix H
            H = csr_matrix(sem.create_jacobian(v_m, delta))

            # if not np.linalg.cond(H) < 1 / sys.float_info.epsilon:
            #    self.logger.error("Error in matrix H")

            # Gain matrix G_m
            # G_m = H^t * R^-1 * H
            G_m = H.T * (r_inv * H)

            # State Vector difference d_E
            # d_E = G_m^-1 * (H' * R^-1 * r)
            d_E = spsolve(G_m, H.T * (r_inv * r.T))
            E += d_E

            # Update V/delta
            delta[:slack_bus] = E[:slack_bus]
            delta[slack_bus + 1:] = E[slack_bus:n_active - 1]
            v_m = np.squeeze(E[n_active - 1:])

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

        # Store bus powers in kVa:
        self.s_bus_kva = ppc["bus"][:, 2] + 1j * ppc["bus"][:, 3] * self.s_ref / 1e3

        # Store results, overwrite old results
        self.net.res_bus_est = pd.DataFrame(columns=["vm_pu", "va_degree", "p_kw", "q_kvar"],
                                            index=self.net.bus.index)

        bus_idx = mapping_table[self.net["bus"].index.values]
        self.net["res_bus_est"]["vm_pu"] = ppc["bus"][bus_idx][:, 7]
        self.net["res_bus_est"]["va_degree"] = ppc["bus"][bus_idx][:, 8]

        self.net.res_bus_est.p_kw = -  get_values(ppc["bus"][:, 2], self.net.bus.index,
                                                  mapping_table) * self.s_ref / 1e3
        self.net.res_bus_est.q_kvar = - get_values(ppc["bus"][:, 3], self.net.bus.index,
                                                   mapping_table) * self.s_ref / 1e3

        # Store some variables required for Chi^2 and r_N_max test:
        self.R_inv = r_inv
        self.hx = h_x
        self.H = H
        self.Ht = H.T
        self.Gm = G_m
        self.r = r
        self.V = v_m
        self.delta = delta

        return successful

    def perform_chi2_test(self, v_in_out=None, delta_in_out=None):
        """
        The function perform_chi2_test performs a Chi^2 test for bad data and topology error
        detection. The function can be called with the optional input arguments v_in_out and
        delta_in_out. Then, the Chi^2 test is performed after calling the function estimate using
        them as input arguments. It can also be called without these arguments if it is called
        from the same object with which estimate had been called beforehand. Then, the Chi^2 test is
        performed for the states estimated by the funtion estimate and stored internally in a
        member variable of the class state_estimation. For bad data detection, the function
        perform_rn_max_test is more powerful and should be the function of choice. For topology
        error detection, however, perform_chi2_test should be used.

        Input:

            **v_in_out** (np.array, shape=(1,)) - Vector with initial values for all voltage
            magnitudes in p.u. (sorted by bus index)

            **delta_in_out** (np.array, shape=(1,)) - Vector with initial values for all voltage
            angles in Rad (sorted by bus index)


        Return:

            **V** (np.array, shape=(1,)) - Vector with estimated values for all voltage
            magnitudes in p.u. (sorted by bus index)

            **delta** (np.array, shape=(1,)) - Vector with estimated values for all voltage
            angles in Rad (sorted by bus index)

            **successful** (boolean) - True if the estimation process was successful

        Example:

            perform_chi2_test(np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0]))

        """
        # If desired, estimate the state first:
        if (v_in_out is not None) and (delta_in_out is not None):
            successful = self.estimate(v_in_out, delta_in_out)
            v_in_out = self.net.res_bus_est.vm_pu.values
            delta_in_out = self.net.res_bus_est.va_degree.values * 180 / np.pi

        if ((v_in_out is not None) and (delta_in_out is None)) \
                or ((v_in_out is None) and (delta_in_out is not None)):
            self.logger.error("Both V and delta have to be defined or none of them! Cancelling...")
            return

        # Performance index J(hx)
        J = np.dot(np.transpose(self.r), np.dot(self.R_inv, self.r))

        # Number of measurements
        m = len(self.net.measurement)

        # Number of state variables (the -1 is due to the reference bus)
        n = len(self.V) + len(self.delta) - 1

        # Chi^2 test threshold
        test_thresh = chi2.ppf(0.95, m - n)

        # Print results
        self.logger.info("-----------------------")
        self.logger.info("Result of Chi^2 test:")
        self.logger.info("Number of measurements:")
        self.logger.info(m)
        self.logger.info("Number of state variables:")
        self.logger.info(n)
        self.logger.info("Performance index:")
        self.logger.info(J)
        self.logger.info("Chi^2 test threshold:")
        self.logger.info(test_thresh)

        if J <= test_thresh:
            self.bad_data_present = False
            self.logger.info("Chi^2 test passed --> no bad data or topology error detected.")
        else:
            self.bad_data_present = True
            self.logger.info("Chi^2 test failed --> bad data or topology error detected.")

        if (v_in_out is not None) and (delta_in_out is not None):
            return v_in_out, delta_in_out, successful

    def perform_rn_max_test(self, v_in_out, delta_in_out):
        """
        The function perform_rn_max_test performs a largest normalized residual test for bad data
        identification and removal. It takes two input arguments: v_in_out and delta_in_out.
        These are the initial state variables for the combined estimation and bad data
        identification and removal process. They can be initialized as described above, e.g.,
        using “flat start”. In an iterative process, the function performs a state estimation,
        identifies a bad data measurement, removes it from the set of measurements, performs the
        state estimation again, and so on and so forth until no further bad data measurements are
        detected. The return values are the same as for the function estimate.

        Input:

            **v_in_out** (np.array, shape=(1,)) - Vector with initial values for all voltage
            magnitudes in p.u. (sorted by bus index)

            **delta_in_out** (np.array, shape=(1,)) - Vector with initial values for all voltage
            angles in Rad (sorted by bus index)


        Return:

            **V** (np.array, shape=(1,)) - Vector with estimated values for all voltage magnitudes
            in p.u. (sorted by bus index)

            **delta** (np.array, shape=(1,)) - Vector with estimated values for all voltage angles
            in Rad (sorted by bus index)

            **successful** (boolean) - True if the estimation process was successful

        Example:

            perform_chi2_test(np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0]))

        """
        num_iterations = 0

        v_in = v_in_out
        delta_in = delta_in_out

        self.bad_data_present = True

        while self.bad_data_present and (num_iterations < 11):
            # Estimate the state with bad data identified in previous iteration
            # removed from set of measurements:
            successful = self.estimate(v_in, delta_in)
            v_in_out = self.net.res_bus_est.vm_pu.values
            delta_in_out = self.net.res_bus_est.va_degree.values * 180 / np.pi

            # Perform a Chi^2 test to determine whether bad data is to be removed.
            # Til now, r_N_max test is used only for the identification:
            self.perform_chi2_test()

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

            if max(rN) <= 5.0:
                self.logger.info("Largest normalized residual test passed "
                                 "--> no bad data detected.")
            else:
                self.logger.info("Largest normalized residual test failed --> bad data identified.")

            if self.bad_data_present:
                # Identify bad data: Determine index corresponding to max(rN):
                idx_rN = np.argsort(rN)[len(rN) - 1]

                # Sort measurement indexes:
                sorted_meas_idxs = np.concatenate(
                    (self.net.measurement.loc[self.net.measurement['type'] == 'pbus_kw'].index,
                     self.net.measurement.loc[self.net.measurement['type'] == 'pline_kw'].index,
                     self.net.measurement.loc[self.net.measurement['type'] == 'qbus_kvar'].index,
                     self.net.measurement.loc[self.net.measurement['type'] == 'qline_kvar'].index,
                     self.net.measurement.loc[self.net.measurement['type'] == 'vbus_pu'].index,
                     self.net.measurement.loc[self.net.measurement['type'] == 'iline_a'].index))

                # Determine index of measurement to be removed:
                meas_idx = sorted_meas_idxs[idx_rN]

                # Remove bad measurement:
                self.net.measurement.drop(meas_idx, inplace=True)
                self.logger.info("Bad data removed from the set of measurements.")
                self.logger.info("----------------------------------------------")
            else:
                self.logger.info("No bad data removed from the set of measurements.")
                self.logger.info("Finished, successful.")

            num_iterations += 1

        return v_in_out, delta_in_out, successful
