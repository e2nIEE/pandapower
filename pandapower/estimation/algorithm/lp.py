# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from scipy.optimize import linprog

# Make sure the library is installed
# Otherwise, load the scipy linprog
try:
    from ortools.linear_solver import pywraplp
    ortools_available = True
except ImportError:
    ortools_available = False
import warnings

from pandapower.estimation.ppc_conversion import ExtendedPPCI
from pandapower.estimation.algorithm.base import BaseAlgorithm
from pandapower.estimation.algorithm.matrix_base import BaseAlgebra


class LPAlgorithm(BaseAlgorithm):
    def estimate(self, eppci: ExtendedPPCI, with_ortools=True, **kwargs):
        if "estimator" in kwargs and kwargs["estimator"].lower() != "lav":  # pragma: no cover
            self.logger.warning("LP Algorithm supports only LAV Estimator!! Set to LAV!!")

        # matrix calculation object
        sem = BaseAlgebra(eppci)

        current_error, cur_it = 100., 0
        E = eppci.E
        while current_error > self.tolerance and cur_it < self.max_iterations:
            self.logger.debug("Starting iteration {:d}".format(1 + cur_it))
            try:
                # residual r
                r = sem.create_rx(E)

                # jacobian matrix H
                H = sem.create_hx_jacobian(E)

                # state vector difference d_E
                # d_E = G_m^-1 * (H' * R^-1 * r)
                d_E = self._solve_lp(H, E, r, with_ortools=with_ortools)
                E += d_E
                eppci.update_E(E)

                # prepare next iteration
                cur_it += 1
                current_error = np.max(np.abs(d_E))
                self.logger.debug("Current error: {:.7f}".format(current_error))
            except np.linalg.linalg.LinAlgError:  # pragma: no cover
                self.logger.error("A problem appeared while using the linear algebra methods."
                                  "Check and change the measurement set.")
                return False

        # check if the estimation is successfull
        self.check_result(current_error, cur_it)
        return eppci

    @staticmethod
    def _solve_lp(H, x, r, with_ortools):

        """Function to choose the best option based on the installed libraries to solve linear programming.

        Performance comparison(601 bus system/1204 random measurements):
        Scipy   : 269.20 seconds
        OR-Tools:   8.51 seconds +- 154 ms
        """

        n, m = H.shape[1], H.shape[0]
        zero_n = np.zeros((n, 1))
        one_m = np.ones((m, 1))
        Im = np.eye(m)

        c_T = np.r_[zero_n, zero_n, one_m, one_m]
        A = np.c_[H, -H, Im, -Im]

        if ortools_available and with_ortools:
            return LPAlgorithm._solve_or_tools(c_T.ravel(), A, r, n)
        else:
            return LPAlgorithm._solve_scipy(c_T.ravel(), A, r, n)

    @staticmethod
    def _solve_scipy(c_T, A, r, n):

        """The use of linprog function from the scipy library."""

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = linprog(c_T.ravel(), A_eq=A, b_eq=r,
                          method="simplex", options={'tol': 1e-5, 'disp': False, 'maxiter': 20000})
        if res.success:
            d_x = np.array(res['x'][:n]).ravel() - np.array(res['x'][n:2 * n]).ravel()
            return d_x
        else:  # pragma: no cover
            raise np.linalg.linalg.LinAlgError


    @staticmethod
    def _solve_or_tools(c_T, A, r, n):
        # Just to ensure floating point precision if there are any
        error_margin = 1e-10

        #'GLOP' fails with ortools version > 9.4.1874
        solver = pywraplp.Solver.CreateSolver('SCIP')


        # Create the states...
        x = []
        for counter in range(len(c_T)):
            x.append(solver.NumVar(0, solver.infinity(), 'x_' + str(counter)))


        # Give the equality constraints...
        for row_counter, row in enumerate(A):
            row_equality = 0
            for col_counter, col in enumerate(row):
                if abs(col) > error_margin:
                    row_equality = row_equality + col*x[col_counter]

            solver.Add(row_equality == r[row_counter])


        # What to optimize?
        to_minimize = 0
        for counter, coef in enumerate(c_T):
            to_minimize = to_minimize + coef*x[counter]

        solver.Minimize(to_minimize)

        # Solve the optimization problem
        status = solver.Solve()


        d_x = []
        if status == pywraplp.Solver.OPTIMAL or status == solver.FEASIBLE:
            # An optimal solution is found
            for counter, x_current in enumerate(x):
                d_x.append(x_current.solution_value())

            d_x = np.array(d_x)
            return d_x[:n] - d_x[n:2*n]
        else:
            # No solution found...
            raise np.linalg.linalg.LinAlgError


