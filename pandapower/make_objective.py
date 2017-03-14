# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

from numpy import zeros, array, concatenate, power
from pypower.idx_cost import MODEL, NCOST, COST


def _make_objective(ppci, net):
    """
    Implementaton of objective functions for the OPF

    Limitations:
    - Polynomial reactive power costs can only be quadratic, linear or constant

    INPUT:
        **net** - The pandapower format network
        **ppci** - The "internal" pypower format network for PF calculations

    OUTPUT:
        **ppci** - The "internal" pypower format network for PF calculations
    """

    ng = len(ppci["gen"])

    # Determine length of gencost array
    if (net.piecewise_linear_cost.type == "q").any() or (net.polynomial_cost.type == "q").any():
        len_gencost = 2 * ng
    else:
        len_gencost = 1 * ng

    # get indices
    eg_idx = net._pd2ppc_lookups["ext_grid"] if "ext_grid" in net._pd2ppc_lookups else None
    gen_idx = net._pd2ppc_lookups["gen"] if "gen" in net._pd2ppc_lookups else None
    sgen_idx = net._pd2ppc_lookups["sgen_controllable"] if "sgen_controllable" in \
        net._pd2ppc_lookups else None
    load_idx = net._pd2ppc_lookups["load_controllable"] if "load_controllable" in \
        net._pd2ppc_lookups else None
    dc_gens = net.gen.index[(len(net.gen) - len(net.dcline) * 2):]
    from_gens = net.gen.loc[dc_gens[1::2]]
    if gen_idx is not None:
        dcline_idx = gen_idx[from_gens.index]

    # calculate size of gencost array
    if len(net.piecewise_linear_cost):
        n_coefficients = net.piecewise_linear_cost.p.values[0].shape[1] * 2
    else:
        n_coefficients = 0
    if len(net.polynomial_cost):
        n_coefficients = max(n_coefficients,  net.polynomial_cost.c.values[0].shape[1], 4)

    if n_coefficients:
        # initialize array
        ppci["gencost"] = zeros((len_gencost, 4 + n_coefficients), dtype=float)
        ppci["gencost"][:, MODEL:COST + 4] = array([1, 0, 0, 2, 0, 0, 1, 0])

        if len(net.piecewise_linear_cost):

            for type in ["p", "q"]:
                if (net.piecewise_linear_cost.type == type).any():
                    costs = net.piecewise_linear_cost[net.piecewise_linear_cost.type == type]
                    p = concatenate(costs.p)
                    f = concatenate(costs.f)

                    if type == "q":
                        shift_idx = ng
                    else:
                        shift_idx = 0

                    for el in ["gen", "sgen", "ext_grid", "load", "dcline"]:

                        if not costs.element[costs.element_type == el].empty:
                            if el == "gen":
                                idx = gen_idx
                            if el == "sgen":
                                idx = sgen_idx
                            if el == "ext_grid":
                                idx = eg_idx
                            if el == "load":
                                idx = load_idx
                            if el == "dcline":
                                idx = dcline_idx

                        if not costs.element[costs.element_type == el].empty:
                            elements = idx[costs.element[costs.element_type ==
                                                         el].values.astype(int)] + shift_idx
                            ppci["gencost"][elements, COST::2] = p[
                                costs.index[costs.element_type == el]]
                            if el in ["load", "dcline"]:
                                ppci["gencost"][elements, COST + 1::2] = - \
                                    f[costs.index[costs.element_type == el]] * 1e3
                            else:
                                ppci["gencost"][elements, COST + 1::2] = f[
                                    costs.index[costs.element_type == el]] * 1e3

                            ppci["gencost"][elements, NCOST] = n_coefficients / 2
                            ppci["gencost"][elements, MODEL] = 1

        if len(net.polynomial_cost):

            for type in ["p", "q"]:
                if (net.polynomial_cost.type == type).any():
                    costs = net.polynomial_cost[net.polynomial_cost.type == type]
                    c = concatenate(costs.c)
                    n_c = c.shape[1]
                    c = c * power(1e3, array(range(n_c))[::-1])

                    if type == "q":
                        shift_idx = ng
                    else:
                        shift_idx = 0

                    for el in ["gen", "sgen", "ext_grid", "load", "dcline"]:

                        if not costs.element[costs.element_type == el].empty:
                            if el == "gen":
                                idx = gen_idx
                            if el == "sgen":
                                idx = sgen_idx
                            if el == "ext_grid":
                                idx = eg_idx
                            if el == "load":
                                idx = load_idx
                            if el == "dcline":
                                idx = dcline_idx

                            elements = idx[costs.element[costs.element_type ==
                                                         el].values.astype(int)] + shift_idx
                            if el in ["load", "dcline"]:
                                ppci["gencost"][elements, COST:(COST + n_c):] = - \
                                    c[costs.index[costs.element_type == el]]
                            else:
                                ppci["gencost"][elements, COST:(
                                    COST + n_c):] = c[costs.index[costs.element_type == el]]

                            ppci["gencost"][elements, NCOST] = n_c
                            ppci["gencost"][elements, MODEL] = 2

    else:
        ppci["gencost"] = zeros((len_gencost, 8), dtype=float)
        # initialize as pwl cost - otherwise we will get a user warning from
        # pypower for unspecified costs.
        ppci["gencost"][:, :] = array([1, 0, 0, 2, 0, 0, 1, 1000])

    return ppci
