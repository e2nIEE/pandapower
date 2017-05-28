# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import copy

import numpy as np

from pandapower.idx_area import PRICE_REF_BUS
from pandapower.idx_brch import F_BUS, T_BUS, BR_STATUS
from pandapower.idx_bus import NONE, BUS_I, BUS_TYPE
from pandapower.idx_gen import GEN_BUS, GEN_STATUS

try:
    from pypower.run_userfcn import run_userfcn
except ImportError:
    # ToDo: Error only for OPF functions if PYPOWER is not installed
    pass

import pandapower.auxiliary as aux
from pandapower.build_branch import _build_branch_ppc, _switch_branches, _branches_with_oos_buses, \
    _update_trafo_trafo3w_ppc
from pandapower.build_bus import _build_bus_ppc, _calc_loads_and_add_on_ppc, \
    _calc_shunts_and_add_on_ppc, _add_gen_impedances_ppc, _add_motor_impedances_ppc
from pandapower.build_gen import _build_gen_ppc, _update_gen_ppc
from pandapower.opf.make_objective import _make_objective


def _pd2ppc(net):
    """
    Converter Flow:
        1. Create an empty pypower datatructure
        2. Calculate loads and write the bus matrix
        3. Build the gen (Infeeder)- Matrix
        4. Calculate the line parameter and the transformer parameter,
           and fill it in the branch matrix.
           Order: 1st: Line values, 2nd: Trafo values
        5. if opf: make opf objective (gencost)
        6. convert internal ppci format for pypower powerflow / opf without out of service elements and rearanged buses

    INPUT:
        **net** - The pandapower format network

    OUTPUT:
        **ppc** - The simple matpower format network. Which consists of:
                  ppc = {
                        "baseMVA": 1., *float*
                        "version": 2,  *int*
                        "bus": np.array([], dtype=float),
                        "branch": np.array([], dtype=np.complex128),
                        "gen": np.array([], dtype=float),
                        "gencost" =  np.array([], dtype=float), only for OPF
                        "internal": {
                              "Ybus": np.array([], dtype=np.complex128)
                              , "Yf": np.array([], dtype=np.complex128)
                              , "Yt": np.array([], dtype=np.complex128)
                              , "branch_is": np.array([], dtype=bool)
                              , "gen_is": np.array([], dtype=bool)
                              }
        **ppci** - The "internal" pypower format network for PF calculations
    """
    # select elements in service (time consuming, so we do it once)
    use_numba = ("numba" in net["_options"] and net["_options"]["numba"] and
                 (net["_options"]["recycle"] is None or
                  not net["_options"]["recycle"]["_is_elements"]))
    # use_numba = False
    if use_numba:
        net["_is_elements"] = aux._select_is_elements_numba(net)
    else:
        net["_is_elements"] = aux._select_is_elements(net)

    # get options
    mode = net["_options"]["mode"]
    check_connectivity = net["_options"]["check_connectivity"]

    ppc = _init_ppc(net)

    if mode == "opf":
        # additional fields in ppc
        ppc["gencost"] = np.array([], dtype=float)

    # init empty ppci
    ppci = copy.deepcopy(ppc)
    # generate ppc['bus'] and the bus lookup
    _build_bus_ppc(net, ppc)
    # generate ppc['gen'] and fills ppc['bus'] with generator values (PV, REF nodes)
    _build_gen_ppc(net, ppc)
    # generate ppc['branch'] and directly generates branch values
    _build_branch_ppc(net, ppc)
    # adds P and Q for loads / sgens in ppc['bus'] (PQ nodes)
    if mode == "sc":
        _add_gen_impedances_ppc(net, ppc)
        _add_motor_impedances_ppc(net, ppc)
    else:
        _calc_loads_and_add_on_ppc(net, ppc)
        # adds P and Q for shunts, wards and xwards (to PQ nodes)
        _calc_shunts_and_add_on_ppc(net, ppc)

    # adds auxilary buses for open switches at branches
    _switch_branches(net, ppc)

    # add auxilary buses for out of service buses at in service lines.
    # Also sets lines out of service if they are connected to two out of service buses
    _branches_with_oos_buses(net, ppc)

    # sets buses out of service, which aren't connected to branches / REF buses
    if check_connectivity:
        isolated_nodes, _, _ = aux._check_connectivity(ppc)
        net["_is_elements"] = aux._select_is_elements_numba(net, isolated_nodes)
    else:
        aux._set_isolated_buses_out_of_service(net, ppc)

    # generates "internal" ppci format (for powerflow calc) from "external" ppc format and updates the bus lookup
    # Note: Also reorders buses and gens in ppc
    ppci = _ppc2ppci(ppc, ppci, net)

    if mode == "opf":
        # make opf objective
        ppci = _make_objective(ppci, net)

    return ppc, ppci


def _init_ppc(net):
    # init empty ppc
    ppc = {"baseMVA": net.sn_kva * 1e-3
        , "version": 2
        , "bus": np.array([], dtype=float)
        , "branch": np.array([], dtype=np.complex128)
        , "gen": np.array([], dtype=float)
        , "internal": {
            "Ybus": np.array([], dtype=np.complex128)
            , "Yf": np.array([], dtype=np.complex128)
            , "Yt": np.array([], dtype=np.complex128)
            , "branch_is": np.array([], dtype=bool)
            , "gen_is": np.array([], dtype=bool)

            , "DLF": np.array([], dtype=np.complex128)
            , "buses_ord_bfs_nets": np.array([], dtype=float)
        }
           }
    net["_ppc"] = ppc
    return ppc


def _ppc2ppci(ppc, ppci, net):
    # BUS Sorting and lookups
    # get bus_lookup
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    # get OOS busses and place them at the end of the bus array (there are no OOS busses in the ppci)
    oos_busses = ppc['bus'][:, BUS_TYPE] == NONE
    ppci['bus'] = ppc['bus'][~oos_busses]
    # in ppc the OOS busses are included and at the end of the array
    ppc['bus'] = np.r_[ppc['bus'][~oos_busses], ppc['bus'][oos_busses]]

    # generate bus_lookup_ppc_ppci (ppc -> ppci lookup)
    ppc_former_order = (ppc['bus'][:, BUS_I]).astype(int)
    aranged_buses = np.arange(len(ppc["bus"]))

    # lookup ppc former order -> consecutive order
    e2i = np.zeros(len(ppc["bus"]), dtype=int)
    e2i[ppc_former_order] = aranged_buses

    # save consecutive indices in ppc and ppci
    ppc['bus'][:, BUS_I] = aranged_buses
    ppci['bus'][:, BUS_I] = ppc['bus'][:len(ppci['bus']), BUS_I]

    # update lookups (pandapower -> ppci internal)
    _update_lookup_entries(net, bus_lookup, e2i, "bus")

    if 'areas' in ppc:
        if len(ppc["areas"]) == 0:  # if areas field is empty
            del ppc['areas']  # delete it (so it's ignored)

    # bus types
    bt = ppc["bus"][:, BUS_TYPE]

    # update branch, gen and areas bus numbering
    ppc['gen'][:, GEN_BUS] = e2i[np.real(ppc["gen"][:, GEN_BUS]).astype(int)].copy()
    ppc["branch"][:, F_BUS] = e2i[np.real(ppc["branch"][:, F_BUS]).astype(int)].copy()
    ppc["branch"][:, T_BUS] = e2i[np.real(ppc["branch"][:, T_BUS]).astype(int)].copy()

    # Note: The "update branch, gen and areas bus numbering" does the same as this:
    # ppc['gen'][:, GEN_BUS] = get_indices(ppc['gen'][:, GEN_BUS], bus_lookup_ppc_ppci)
    # ppc["branch"][:, F_BUS] = get_indices(ppc["branch"][:, F_BUS], bus_lookup_ppc_ppci)
    # ppc["branch"][:, T_BUS] = get_indices( ppc["branch"][:, T_BUS], bus_lookup_ppc_ppci)
    # but faster...

    if 'areas' in ppc:
        ppc["areas"][:, PRICE_REF_BUS] = \
            e2i[np.real(ppc["areas"][:, PRICE_REF_BUS]).astype(int)].copy()

    # reorder gens (and gencosts) in order of increasing bus number
    sort_gens = ppc['gen'][:, GEN_BUS].argsort()
    new_gen_positions = np.arange(len(sort_gens))
    new_gen_positions[sort_gens] = np.arange(len(sort_gens))
    ppc['gen'] = ppc['gen'][sort_gens,]

    # update gen lookups
    _is_elements = net["_is_elements"]
    eg_end = np.sum(_is_elements['ext_grid'])
    gen_end = eg_end + np.sum(_is_elements['gen'])
    sgen_end = len(_is_elements["sgen_controllable"]) + gen_end if "sgen_controllable" in _is_elements else gen_end
    load_end = len(_is_elements["load_controllable"]) + sgen_end if "load_controllable" in _is_elements else sgen_end

    if eg_end > 0:
        _build_gen_lookups(net, "ext_grid", 0, eg_end, new_gen_positions)
    if gen_end > eg_end:
        _build_gen_lookups(net, "gen", eg_end, gen_end, new_gen_positions)
    if sgen_end > gen_end:
        _build_gen_lookups(net, "sgen_controllable", gen_end, sgen_end, new_gen_positions)
    if load_end > sgen_end:
        _build_gen_lookups(net, "load_controllable", sgen_end, load_end, new_gen_positions)

    # determine which buses, branches, gens are connected and
    # in-service
    n2i = ppc["bus"][:, BUS_I].astype(int)
    bs = (bt != NONE)  # bus status

    gs = ((ppc["gen"][:, GEN_STATUS] > 0) &  # gen status
          bs[n2i[np.real(ppc["gen"][:, GEN_BUS]).astype(int)]])
    ppci["internal"]["gen_is"] = gs

    brs = (np.real(ppc["branch"][:, BR_STATUS]).astype(int) &  # branch status
           bs[n2i[np.real(ppc["branch"][:, F_BUS]).astype(int)]] &
           bs[n2i[np.real(ppc["branch"][:, T_BUS]).astype(int)]]).astype(bool)
    ppci["internal"]["branch_is"] = brs

    if 'areas' in ppc:
        ar = bs[n2i[ppc["areas"][:, PRICE_REF_BUS].astype(int)]]
        # delete out of service areas
        ppci["areas"] = ppc["areas"][ar]

    # select in service elements from ppc and put them in ppci
    ppci["branch"] = ppc["branch"][brs]

    ppci["gen"] = ppc["gen"][gs]

    if 'dcline' in ppc:
        ppci['dcline'] = ppc['dcline']
    # execute userfcn callbacks for 'ext2int' stage
    if 'userfcn' in ppci:
        ppci = run_userfcn(ppci['userfcn'], 'ext2int', ppci)

    return ppci


def _update_lookup_entries(net, lookup, e2i, element):
    valid_bus_lookup_entries = lookup >= 0
    # update entries
    lookup[valid_bus_lookup_entries] = e2i[lookup[valid_bus_lookup_entries]]
    aux._write_lookup_to_net(net, element, lookup)


def _build_gen_lookups(net, element, ppc_start_index, ppc_end_index, sort_gens):
    # get buses from pandapower and ppc
    _is_elements = net["_is_elements"]
    if element in ["sgen_controllable", "load_controllable"]:
        pandapower_index = net["_is_elements"][element].index.values
    else:
        pandapower_index = net[element].index.values[_is_elements[element]]
    ppc_index = sort_gens[ppc_start_index: ppc_end_index]

    # init lookup
    lookup = -np.ones(max(pandapower_index) + 1, dtype=int)

    # update lookup
    lookup[pandapower_index] = ppc_index
    aux._write_lookup_to_net(net, element, lookup)


def _update_ppc(net):
    """
    Updates P, Q values of the ppc with changed values from net

    @param _is_elements:
    @return:
    """
    # select elements in service (time consuming, so we do it once)
    if net["_options"]["numba"]:
        net["_is_elements"] = aux._select_is_elements_numba(net)
    else:
        net["_is_elements"] = aux._select_is_elements(net)

    recycle = net["_options"]["recycle"]
    # get the old ppc and lookup
    ppc = net["_ppc"]
    ppci = copy.deepcopy(ppc)
    # adds P and Q for loads / sgens in ppc['bus'] (PQ nodes)
    _calc_loads_and_add_on_ppc(net, ppc)
    # adds P and Q for shunts, wards and xwards (to PQ nodes)
    _calc_shunts_and_add_on_ppc(net, ppc)
    # updates values for gen
    _update_gen_ppc(net, ppc)
    if not recycle["Ybus"]:
        # updates trafo and trafo3w values
        _update_trafo_trafo3w_ppc(net, ppc)

    # get OOS busses and place them at the end of the bus array (so that: 3
    # (REF), 2 (PV), 1 (PQ), 4 (OOS))
    oos_busses = ppc['bus'][:, BUS_TYPE] == NONE
    # there are no OOS busses in the ppci
    ppci['bus'] = ppc['bus'][~oos_busses]
    # select in service elements from ppc and put them in ppci
    brs = ppc["internal"]["branch_is"]
    gs = ppc["internal"]["gen_is"]
    ppci["branch"] = ppc["branch"][brs]
    ppci["gen"] = ppc["gen"][gs]

    return ppc, ppci
