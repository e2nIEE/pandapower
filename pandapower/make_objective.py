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
                eg_index = p_costs.element[p_costs.element_type == "ext_grid"].index
                genel = p_costs.element[p_costs.element_type == "gen"].values + nref
                gen_index = p_costs.element[p_costs.element_type == "gen"].index
                sgenel = p_costs.element[p_costs.element_type == "sgen"].values + nref + ngen
                sgen_index = p_costs.element[p_costs.element_type == "sgen"].index
                loadel = p_costs.element[p_costs.element_type == "load"].values + nref + ngen + nsgen
                load_index = p_costs.element[p_costs.element_type == "load"].index

                elements = np.append(egel, genel)
                elements = np.append(elements, sgenel)
                elements = pd.to_numeric(np.append(elements, loadel))

                el_index= np.concatenate((eg_index,gen_index, sgen_index, load_index))

                ppci["gencost"][elements, 4::2] = p[el_index] * 1e-3
                ppci["gencost"][elements, 5::2] = f[el_index]
                ppci["gencost"][pd.to_numeric(loadel), 5::2] *=-1

            if (net.piecewise_linear_cost.type == "q").any():
                q_costs = net.piecewise_linear_cost[net.piecewise_linear_cost.type == "q"]

                p = np.concatenate(q_costs.p)
                f = np.concatenate(q_costs.f)

                egel = q_costs.element[q_costs.element_type == "ext_grid"].values + ng
                eg_index = q_costs.element[q_costs.element_type == "ext_grid"].index
                genel = q_costs.element[q_costs.element_type == "gen"].values + nref + ng
                gen_index = q_costs.element[q_costs.element_type == "gen"].index
                sgenel = q_costs.element[q_costs.element_type == "sgen"].values + nref + ngen + ng
                sgen_index = q_costs.element[q_costs.element_type == "sgen"].index
                loadel = q_costs.element[q_costs.element_type == "load"].values + nref + ngen + nsgen + ng
                load_index = q_costs.element[q_costs.element_type == "load"].index


                elements = np.append(egel, genel)
                elements = np.append(elements, sgenel)
                elements = pd.to_numeric(np.append(elements, loadel))

                el_index= np.concatenate((eg_index,gen_index, sgen_index, load_index))

                ppci["gencost"][elements, 4::2] = p[el_index] * 1e-3
                ppci["gencost"][elements, 5::2] = f[el_index]
                ppci["gencost"][pd.to_numeric(loadel), 5::2] *= -1


        if len(net.polynomial_cost):

            if (net.polynomial_cost.type == "p").any():
                p_costs = net.polynomial_cost[net.polynomial_cost.type == "p"]

                c = np.concatenate(p_costs.c)
                c = c * np.power(1e3, np.array(range(c.shape[1]))[::-1])

                egel = p_costs.element[p_costs.element_type == "ext_grid"].values
                eg_index = p_costs.element[p_costs.element_type == "ext_grid"].index
                genel = p_costs.element[p_costs.element_type == "gen"].values + nref
                gen_index = p_costs.element[p_costs.element_type == "gen"].index
                sgenel = p_costs.element[p_costs.element_type == "sgen"].values + nref + ngen
                sgen_index = p_costs.element[p_costs.element_type == "sgen"].index
                loadel = p_costs.element[p_costs.element_type == "load"].values + nref + ngen + nsgen
                load_index = p_costs.element[p_costs.element_type == "load"].index

                elements = np.append(egel, genel)
                elements = np.append(elements, sgenel)
                elements = pd.to_numeric(np.append(elements, loadel))

                el_index= np.concatenate((eg_index,gen_index, sgen_index, load_index))

                gap = n_coefficients - c.shape[1]

                if gap:
                    c = np.append(np.zeros(gap), c)

                ppci["gencost"][elements, 0:4] = np.array([2, 0, 0, n_coefficients])  # initializing gencost array for eg
                ppci["gencost"][elements, 4::] = c[el_index]
                ppci["gencost"][pd.to_numeric(loadel), 4::] *= -1

            if (net.polynomial_cost.type == "q").any():
                q_costs = net.polynomial_cost[net.polynomial_cost.type == "q"]

                c = np.concatenate(q_costs.c)
                c = c * np.power(1e3, np.array(range(c.shape[1]))[::-1])

                egel = q_costs.element[q_costs.element_type == "ext_grid"].values  + ng
                eg_index = q_costs.element[q_costs.element_type == "ext_grid"].index
                genel = q_costs.element[q_costs.element_type == "gen"].values + nref + ng
                gen_index = q_costs.element[q_costs.element_type == "gen"].index
                sgenel = q_costs.element[q_costs.element_type == "sgen"].values + nref +  + ng
                sgen_index = q_costs.element[q_costs.element_type == "sgen"].index
                loadel = q_costs.element[q_costs.element_type == "load"].values + nref + ngen + nsgen + ng
                load_index = q_costs.element[q_costs.element_type == "load"].index

                elements = np.append(egel, genel)
                elements = np.append(elements, sgenel)
                elements = pd.to_numeric(np.append(elements, loadel))

                el_index= np.concatenate((eg_index,gen_index, sgen_index, load_index))

                gap = n_coefficients - len(c)
                if gap:
                    c = np.append(np.zeros(gap), c)

                ppci["gencost"][elements, 0:4] = np.array([2, 0, 0, n_coefficients])  # initializing gencost array for eg
                ppci["gencost"][elements, 4::] = c[el_index]
                ppci["gencost"][pd.to_numeric(loadel), 4::] *= -1
    else:
        ppci["gencost"] = np.zeros((nconst, 8), dtype=float)
        ppci["gencost"][:,:] = np.array([1, 0, 0, 2, 0, 0, 1, 1000])  # initialize as pwl cost - otherwise we will get a user warning from pypower for unspecified costs.


    return ppci
