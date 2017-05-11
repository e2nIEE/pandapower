# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

from numpy import zeros, array, concatenate, power
from pandapower.idx_cost import MODEL, NCOST, COST


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
    # Determine duplicated cost data
    all_costs = net.polynomial_cost[['type', 'element', 'element_type']].append(
            net.piecewise_linear_cost[['type', 'element', 'element_type']])
    duplicates = all_costs.loc[all_costs.duplicated()]
    if duplicates.shape[0]:
        raise ValueError("There are elements with multipy costs.\nelement_types: %s\n"
                         "element: %s\ntypes: %s" % (duplicates.element_type.values,
                                                     duplicates.element.values,
                                                     duplicates.type.values))

    # Determine length of gencost array
    ng = len(ppci["gen"])
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
    else:
        dcline_idx = None

    # calculate size of gencost array
    if len(net.piecewise_linear_cost):
        n_piece_lin_coefficients = net.piecewise_linear_cost.p.values[0].shape[1] * 2
    else:
        n_piece_lin_coefficients = 0
    if len(net.polynomial_cost):
        n_coefficients = max(n_piece_lin_coefficients,  net.polynomial_cost.c.values[0].shape[1])
        if (n_piece_lin_coefficients > 0) & (n_coefficients % 2):
            # avoid uneven n_coefficient in case of (n_piece_lin_coefficients>0)
            n_coefficients += 1
    else:
        n_coefficients = n_piece_lin_coefficients

    if n_coefficients:
        # initialize array
        ppci["gencost"] = zeros((len_gencost, 4 + n_coefficients), dtype=float)
        ppci["gencost"][:, MODEL:COST] = array([2, 0, 0, n_coefficients])

        for type in ["p", "q"]:

            if type == "q":
                shift_idx = ng
                sign_corr = -1
            else:
                shift_idx = 0
                sign_corr = 1

            for el in ["gen", "sgen", "ext_grid", "load", "dcline"]:

                if el == "gen":
                    idx = gen_idx
                elif el == "sgen":
                    idx = sgen_idx
                elif el == "ext_grid":
                    idx = eg_idx
                elif el == "load":
                    idx = load_idx
                elif el == "dcline":
                    idx = dcline_idx

                if len(net.piecewise_linear_cost):

                    if (net.piecewise_linear_cost.type == type).any():
                        costs = net.piecewise_linear_cost[
                                net.piecewise_linear_cost.type == type].reset_index(drop=True)

                        if not costs.element[costs.element_type == el].empty:


                            # cost data to write into gencost
                            el_is = net[el].loc[(net[el].in_service) & net[el].index.isin(
                                    costs.loc[costs.element_type == el].element)].index
                            p = costs.loc[(costs.element_type == el) & (
                                    costs.element.isin(el_is))].p.reset_index(drop=True)
                            f = costs.loc[(costs.element_type == el) & (
                                    costs.element.isin(el_is))].f.reset_index(drop=True)
                            if len(p) > 0:
                                p = concatenate(p)
                                f = concatenate(f)
                                # gencost indices

                                elements = idx[el_is] + shift_idx

                                ppci["gencost"][elements, COST:COST+n_piece_lin_coefficients:2] = p

                                if el in ["load", "dcline"]:
                                    ppci["gencost"][elements, COST+1:COST +
                                                    n_piece_lin_coefficients+1:2] = - f * 1e3
                                else:
                                    ppci["gencost"][elements, COST+1:COST+n_piece_lin_coefficients +
                                                    1:2] = f * 1e3 * sign_corr

                                ppci["gencost"][elements, NCOST] = n_coefficients / 2
                                ppci["gencost"][elements, MODEL] = 1


        if len(net.polynomial_cost):

            for type in ["p", "q"]:
                if (net.polynomial_cost.type == type).any():
                    costs = net.polynomial_cost[net.polynomial_cost.type == type].reset_index(drop=True)

                    if type == "q":
                        shift_idx = ng
                        sign_corr = -1
                    else:
                        shift_idx = 0
                        sign_corr = 1

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

                            el_is = net[el].loc[(net[el].in_service) & net[el].index.isin(\
                                     costs.loc[costs.element_type == el].element)].index
                            c = costs.loc[(costs.element_type == el) & (costs.element.isin(el_is))].c.reset_index(drop=True)

                            if len(c) > 0:
                                c = concatenate(c)
                                n_c = c.shape[1]
                                c = c * power(1e3, array(range(n_c))[::-1])
                                # gencost indices
                                elements = idx[el_is] + shift_idx
                                n_gencost = ppci["gencost"].shape[1]

                            elcosts = costs[costs.element_type == el]
                            elcosts.index = elcosts.element
                            if el in ["load", "dcline"]:
                                ppci["gencost"][elements, COST:(COST + n_c):] = - c
                            else:
                                ppci["gencost"][elements, -n_c:n_gencost] = c * sign_corr

                            ppci["gencost"][elements, NCOST] = n_coefficients
                            ppci["gencost"][elements, MODEL] = 2

    else:
        ppci["gencost"] = zeros((len_gencost, 8), dtype=float)
        # initialize as pwl cost - otherwise we will get a user warning from
        # pypower for unspecified costs.
        ppci["gencost"][:, :] = array([1, 0, 0, 2, 0, 0, 1, 1000])

    return ppci
