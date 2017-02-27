# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import numpy as np
import warnings

from pypower.idx_brch import T_BUS, F_BUS
from pypower.idx_bus import BUS_TYPE, REF
from pandas import DataFrame
from scipy import sparse
import pandas as pd

from pandapower.pypower_extensions.makeYbus_pypower import makeYbus


def _make_objective(ppci, net, lambda_opf=1, p_nominal=None, **kwargs):
    """
    Implementaton of diverse objective functions for the OPF of the Form C{N}, C{fparm},
    C{H} and C{Cw}

    INPUT:
        **ppci** - Matpower case of the net

    OPTIONAL:

        **objectivetype** (string, "linear") - string with name of objective function

            - **"linear"** - Linear costs of the form  :math:`I\\cdot P_G`. :math:`P_G` represents
              the active power values of the generators. Target of this objectivefunction is to
              maximize the generator output.
              This then basically is this:

                  .. math::
                      max\{P_G\}

            - **"linear_minloss"** - Quadratic costs of the form
              :math:`I\\cdot P_G - dV_m^T Y_L dV_m`.
              :math:`P_G` represents the active power values of the generators,
              :math:`dV_m` the voltage drop for each line and :math:`Y_L` the line admittance
              matrix.
              Target of this objectivefunction is to maximize the generator output but minimize the
              linelosses.
              This then basically is this:

                  .. math::
                      max\{P_G - dVm^TY_{L}dVm\}

            .. note:: Both objective functions have the following constraints:

                .. math::
                    V_{m,min} < &V_m < V_{m,max}\\\\
                    P_{G,min} < &P_G < P_{G,max}\\\\
                    Q_{G,min} < &Q_G < Q_{G,max}\\\\
                    I < &I_{max}

        **net** (attrdict, None) - Pandapower network

    """

    ng = len(ppci["gen"])

    if (net.piecewise_linear_cost.type == "q").any() or (net.polynomial_cost.type == "q").any() :
        nconst = 2 * ng
    else:
        nconst = 1 * ng


    eg_idx = net._pd2ppc_lookups["ext_grid"] if "ext_grid" in net._pd2ppc_lookups else None
    gen_idx = net._pd2ppc_lookups["gen"] if "gen" in net._pd2ppc_lookups else None
    sgen_idx = net._pd2ppc_lookups["sgen_controllable"] if "sgen_controllable" in \
                    net._pd2ppc_lookups else None
    load_idx = net._pd2ppc_lookups["load_controllable"] if "load_controllable" in \
                    net._pd2ppc_lookups else None


    nref = sum(ppci["bus"][:, BUS_TYPE] == REF)
    if gen_idx is not None:
        ngen = len(gen_idx)
    else:
        ngen = 0

    if sgen_idx is not None:
        nsgen= len(sgen_idx)
    else:
        nsgen = 0

    if load_idx is not None:
        nload= len(load_idx)
    else:
        nload = 0


    # calculate lenght of gencost array
    if len(net.piecewise_linear_cost):
        n_coefficients = net.piecewise_linear_cost.p.values[0].shape[1]*2
    else:
        n_coefficients = 0
    if len(net.polynomial_cost):
       n_coefficients = max(n_coefficients,  net.polynomial_cost.c.values[0].shape[1])

    if n_coefficients:
        ppci["gencost"] = np.zeros((nconst, 4 + n_coefficients), dtype=float)
        ppci["gencost"][:, 0:4] = np.array([2, 0, 0, n_coefficients ])  # initialize as pol cost - otherwise we will get a user warning from pypower for unspecified costs.

        if len(net.piecewise_linear_cost):


            ppci["gencost"][:, 0:8] = np.array([1, 0, 0, 2, 0, 0, 1, 0])  # initializing gencost array

            if (net.piecewise_linear_cost.type == "p").any():
                p_costs = net.piecewise_linear_cost[net.piecewise_linear_cost.type == "p"]

                p = np.concatenate(p_costs.p)
                f = np.concatenate(p_costs.f.values)


                egel = p_costs.element[p_costs.element_type == "ext_grid"].values
                genel = p_costs.element[p_costs.element_type=="gen"].values + nref
                sgenel = p_costs.element[p_costs.element_type == "sgen"].values + nref + ngen
                loadel = p_costs.element[p_costs.element_type=="load"].values + nref + ngen + nsgen

                elements = np.append(egel, genel)
                elements = np.append(elements, sgenel)
                elements = pd.to_numeric(np.append(elements, loadel))

                ppci["gencost"][elements, 4::2] = p * 1e-3
                ppci["gencost"][elements, 5::2] = f
                ppci["gencost"][pd.to_numeric(loadel), 5::2] *=-1

            if (net.piecewise_linear_cost.type == "q").any():
                q_costs = net.piecewise_linear_cost[net.piecewise_linear_cost.type == "q"]

                p = np.concatenate(q_costs.p)
                f = np.concatenate(q_costs.f)

                egel = q_costs.element[q_costs.element_type == "ext_grid"].values + ng
                genel = q_costs.element[q_costs.element_type=="gen"].values + nref + ng
                sgenel = q_costs.element[q_costs.element_type == "sgen"].values + nref + ngen + ng
                loadel = q_costs.element[q_costs.element_type=="load"].values + nref + ngen + nsgen + ng

                elements = np.append(egel, genel)
                elements = np.append(elements, sgenel)
                elements = pd.to_numeric(np.append(elements, loadel))

                ppci["gencost"][elements, 4::2] = p * 1e-3
                ppci["gencost"][elements, 5::2] = f
                ppci["gencost"][pd.to_numeric(loadel), 5::2] *= -1

    else:
        ppci["gencost"] = np.zeros((nconst, 8), dtype=float)
        ppci["gencost"] = np.array([1, 0, 0, 2, 0, 0, 1, 0])  # initialize as pwl cost - otherwise we will get a user warning from pypower for unspecified costs.

    return True





    if len(net.polynomial_cost):

        if (net.polynomial_cost.type == "p").any():
            p_costs = net.polynomial_cost[net.polynomial_cost.type == "p"]

            c = np.concatenate(p_costs.c)
            c = c * np.power(1e3, np.array(range(c.shape[1]))[::-1])

            egel = p_costs.element[p_costs.element_type == "ext_grid"].values
            genel = p_costs.element[p_costs.element_type == "gen"].values + nref
            sgenel = p_costs.element[p_costs.element_type == "sgen"].values + nref + ngen
            loadel = p_costs.element[p_costs.element_type == "load"].values + nref + ngen + nsgen

            elements = np.append(egel, genel)
            elements = np.append(elements, sgenel)
            elements = pd.to_numeric(np.append(elements, loadel))

            gap = n_coefficients - c.shape[1]

            if gap:
                c = np.append(np.zeros(gap), c)

            ppci["gencost"][elements, 0:4] = np.array([2, 0, 0, n_coefficients])  # initializing gencost array for eg
            ppci["gencost"][elements, 4::] = c
            ppci["gencost"][pd.to_numeric(loadel), 4::] *= -1

        if (net.polynomial_cost.type == "q").any():
            q_costs = net.polynomial_cost[net.polynomial_cost.type == "q"]

            c = np.concatenate(q_costs.c)
            c = c * np.power(1e3, np.array(range(c.shape[1]))[::-1])

            egel = q_costs.element[p_costs.element_type == "ext_grid"].values + ng
            genel = q_costs.element[p_costs.element_type == "gen"].values + nref + ng
            sgenel = q_costs.element[p_costs.element_type == "sgen"].values + nref + ngen + ng
            loadel = q_costs.element[p_costs.element_type == "load"].values + nref + ngen + nsgen + ng

            elements = np.append(egel, genel)
            elements = np.append(elements, sgenel)
            elements = pd.to_numeric(np.append(elements, loadel))

            gap = n_coefficients - len(c)
            if gap:
                c = np.append(np.zeros(gap), c)

            ppci["gencost"][elements, 0:4] = np.array([2, 0, 0, n_coefficients])  # initializing gencost array for eg
            ppci["gencost"][elements, 4::] = c
            ppci["gencost"][pd.to_numeric(loadel), 4::] *= -1

    return True


    # if cost_function == "linear_minloss":
    #
    #     ppci["gencost"] = np.zeros((ng, 8), dtype=float)
    #     ppci["gencost"][:nref, :] = np.array([1, 0, 0, 2, 0, 0, 1, 1]) # initializing gencost array for eg
    #     ppci["gencost"][nref:ng, :] = np.array([1, 0, 0, 2, 0, 0, 1, 1])  # initializing gencost array
    #     ppci["gencost"][:, 7] = np.nan_to_num(gen_costs*1e3)
    #
    #     # Get additional counts
    #     dim = 2 * nb + 2 * ng + 2 * nl
    #
    #     # Get branch admitance matrices
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         Ybus, Yf, Yt = makeYbus(1, ppci["bus"], ppci["branch"])
    #
    #     #########################
    #     # Additional z variables
    #     #########################
    #
    #     # z_k = u_i - u_j
    #     # z_m = alpha_i - alpha_j
    #     # with i,j = start and endbus from lines
    #
    #     # Epsilon for z constraints
    #     eps = 0
    #
    #     # z constraints upper and lower bounds
    #     l = np.ones(2*nl)*eps
    #     u = np.ones(2*nl)*-eps
    #
    #     # Initialize A and H matrix
    #     H = sparse.csr_matrix((dim, dim), dtype=float)
    #     A = sparse.csr_matrix((2*nl, dim), dtype=float)
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         for i in range(nl):
    #             bus_f = int(ppci["branch"][i, F_BUS].real)
    #             bus_t = int(ppci["branch"][i, T_BUS].real)
    #             # minimization of potential difference between two buses
    #             H[dim-2*nl+i, dim-2*nl+i] = abs(Ybus[bus_f, bus_t]) * lambda_opf  # weigthing of minloss # NICHT BESSER REALTEIL?
    #             A[i, nb+bus_f] = 1
    #             A[i, nb+bus_t] = -1
    #             A[i, dim-2*nl+i] = 1
    #             # minimization of angle between two buses
    #             H[dim-nl+i, dim-nl+i] = abs(Ybus[bus_f, bus_t])# * 0# * 800 * lambda_opf  # weigthing of angles
    #             A[nl+i, bus_f] = 1
    #             A[nl+i, bus_t] = -1
    #             A[nl+i, dim-nl+i] = 1
    #
    #     # Linear transformation for new omega-vector
    #     N = sparse.lil_matrix((dim, dim), dtype=float)
    #     for i in range(dim - 2*nl, dim):
    #         N[i, i] = 1.0
    #
    #     # Cw = 0, no linear costs in additional costfunction
    #     Cw = np.zeros(dim, dtype=float)
    #     # d = 1
    #     d = np.ones((dim, 1), dtype=float)
    #     # r = 0
    #     r = np.zeros((dim, 1), dtype=float)
    #     # k = 0
    #     k = np.zeros((dim, 1), dtype=float)
    #     # m = 1
    #     m = np.ones((dim, 1), dtype=float)
    #
    #     # Set ppci matrices
    #     ppci["H"] = H
    #     ppci["Cw"] = Cw
    #     ppci["N"] = N
    #     ppci["A"] = A
    #     ppci["l"] = l
    #     ppci["u"] = u
    #     ppci["fparm"] = np.hstack((d, r, k, m))

    return ppci
