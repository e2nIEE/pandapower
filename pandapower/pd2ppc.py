# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import numpy as np
import copy

from pypower.idx_bus import NONE, BUS_I, BUS_TYPE
from pypower.idx_gen import GEN_BUS, GEN_STATUS
from pypower.idx_brch import F_BUS, T_BUS, BR_STATUS
from pypower.idx_area import PRICE_REF_BUS
from pypower.run_userfcn import run_userfcn

from pandapower.build_branch import _build_branch_ppc, _switch_branches, _branches_with_oos_buses, \
                        _update_trafo_trafo3w_ppc
from pandapower.build_bus import _build_bus_ppc, _calc_loads_and_add_on_ppc, \
    _calc_shunts_and_add_on_ppc
from pandapower.build_gen import _build_gen_ppc, _update_gen_ppc
from pandapower.auxiliary import _set_isolated_buses_out_of_service
from pandapower.make_objective import _make_objective

def _pd2ppc(net, is_elems, calculate_voltage_angles=False, enforce_q_lims=False,
            trafo_model="pi", init_results=False, copy_constraints_to_ppc=False,
            opf=False, cost_function=None):
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
        **net** - The Pandapower format network
        **is_elems** - In service elements from the network (see _select_is_elements())

    OPTIONAL PARAMETERS:
        **calculate_voltage_angles** (bool, False) - consider voltage angles in powerflow calculation
            (see the description of runpp())
        **enforce_q_lims** (bool, False) - respect generator reactive power limits (see description of runpp())
        **trafo_model** (str,pi) - transformer equivalent circuit model (see description of runpp())
        **init_results** (bool, False) - initialization method of the loadflow (see description of runpp())
        **copy_constraints_to_ppc** (bool, False) - additional constraints
            (like voltage boundaries, maximum thermal capacity of branches rateA and generator P and Q limits
             will be copied to the ppc). This is necessary for the OPF as well as the converter functions
        **opf** (bool, False) - changes to the ppc are necessary if OPF is calculated instead of PF
        **cost_function** (obj, None) - The OPF cost function


    RETURN:
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
        **bus_lookup** - Lookup Pandapower -> ppc / ppci indices
    """

    # init empty ppc
    ppc = {"baseMVA": 1.
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
                  }
           }

    if opf:
        # additional fields in ppc
        ppc["gencost"] =  np.array([], dtype=float)

    # init empty ppci
    ppci = copy.deepcopy(ppc)
    # generate ppc['bus'] and the bus lookup
    bus_lookup = _build_bus_ppc(net, ppc, is_elems, init_results, copy_constraints_to_ppc=copy_constraints_to_ppc)
    # generate ppc['gen'] and fills ppc['bus'] with generator values (PV, REF nodes)
    _build_gen_ppc(net, ppc, is_elems, bus_lookup, enforce_q_lims, calculate_voltage_angles,
                   copy_constraints_to_ppc = False, opf=opf)
    # generate ppc['branch'] and directly generates branch values
    _build_branch_ppc(net, ppc, is_elems, bus_lookup, calculate_voltage_angles, trafo_model,
                      copy_constraints_to_ppc=copy_constraints_to_ppc)
    # adds P and Q for loads / sgens in ppc['bus'] (PQ nodes)
    _calc_loads_and_add_on_ppc(net, ppc, is_elems, bus_lookup, opf=opf)
    # adds P and Q for shunts, wards and xwards (to PQ nodes)
    _calc_shunts_and_add_on_ppc(net, ppc, is_elems, bus_lookup)
    # adds auxilary buses for open switches at branches
    _switch_branches(net, ppc, is_elems, bus_lookup)
    # add auxilary buses for out of service buses at in service lines.
    # Also sets lines out of service if they are connected to two out of service buses
    _branches_with_oos_buses(net, ppc, is_elems, bus_lookup)
    # sets buses out of service, which aren't connected to branches / REF buses
    _set_isolated_buses_out_of_service(net, ppc)
    if opf:
        # make opf objective
        ppc = _make_objective(ppc, net, is_elems, cost_function)

    # generates "internal" ppci format (for powerflow calc) from "external" ppc format and updates the bus lookup
    # Note: Also reorders buses and gens in ppc
    ppci, bus_lookup = _ppc2ppci(ppc, ppci, bus_lookup)

    return ppc, ppci, bus_lookup


def _ppc2ppci(ppc, ppci, bus_lookup):
    # BUS Sorting and lookup
    # sort busses in descending order of column 1 (namely: 4 (OOS), 3 (REF), 2 (PV), 1 (PQ))
    ppc_buses = ppc["bus"]
    ppc['bus'] = ppc_buses[ppc_buses[:, BUS_TYPE].argsort(axis=0)[::-1][:], ]
    # get OOS busses and place them at the end of the bus array (so that: 3
    # (REF), 2 (PV), 1 (PQ), 4 (OOS))
    oos_busses = ppc['bus'][:, BUS_TYPE] == NONE
    # there are no OOS busses in the ppci
    ppci['bus'] = ppc['bus'][~oos_busses]
    # in ppc the OOS busses are included and at the end of the array
    ppc['bus'] = np.r_[ppc['bus'][~oos_busses], ppc['bus'][oos_busses]]
    # generate bus_lookup_ppc_ppci (ppc -> ppci lookup)
    ppc_former_order = (ppc['bus'][:, BUS_I]).astype(int)
    aranged_buses = np.arange(len(ppc_buses))

    # lookup ppc former order -> consecutive order
    e2i = np.zeros(len(ppc_buses), dtype=int)
    e2i[ppc_former_order] = aranged_buses

    # save consecutive indices in ppc and ppci
    ppc['bus'][:, BUS_I] = aranged_buses
    ppci['bus'][:, BUS_I] = ppc['bus'][:len(ppci['bus']), BUS_I]

    # update bus_lookup (pandapower -> ppci internal)
    valid_bus_lookup_entries = bus_lookup >= 0
    bus_lookup[valid_bus_lookup_entries] = e2i[bus_lookup[valid_bus_lookup_entries]]

    if 'areas' in ppc:
        if len(ppc["areas"]) == 0:  # if areas field is empty
            del ppc['areas']  # delete it (so it's ignored)

    # bus types
    bt = ppc["bus"][:, BUS_TYPE]

    # update branch, gen and areas bus numbering
    ppc['gen'][:, GEN_BUS] = \
        e2i[np.real(ppc["gen"][:, GEN_BUS]).astype(int)].copy()
    ppc["branch"][:, F_BUS] = \
        e2i[np.real(ppc["branch"][:, F_BUS]).astype(int)].copy()
    ppc["branch"][:, T_BUS] = \
        e2i[np.real(ppc["branch"][:, T_BUS]).astype(int)].copy()

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
    ppc['gen'] = ppc['gen'][sort_gens, ]
    if 'gencost' in ppc:
        ppc['gencost'] = ppc['gencost'][sort_gens, ]

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

    if 'gencost' in ppc:
        ppci["gencost"] = ppc["gencost"][gs]
    if 'dcline' in ppc:
        ppci['dcline'] = ppc['dcline']
    # execute userfcn callbacks for 'ext2int' stage
    if 'userfcn' in ppci:
        ppci = run_userfcn(ppci['userfcn'], 'ext2int', ppci)

    return ppci, bus_lookup


def _update_ppc(net, is_elems, recycle, calculate_voltage_angles=False, enforce_q_lims=False, 
                trafo_model="pi"):
    """
    Updates P, Q values of the ppc with changed values from net

    @param is_elems:
    @return:
    """    
    # get the old ppc and lookup
    ppc = net["_ppc"]
    ppci = copy.deepcopy(ppc)
    bus_lookup = net["_bus_lookup"]
    # adds P and Q for loads / sgens in ppc['bus'] (PQ nodes)
    _calc_loads_and_add_on_ppc(net, ppc, is_elems, bus_lookup)
    # adds P and Q for shunts, wards and xwards (to PQ nodes)
    _calc_shunts_and_add_on_ppc(net, ppc, is_elems, bus_lookup)
    # updates values for gen
    _update_gen_ppc(net, ppc, is_elems, bus_lookup, enforce_q_lims, calculate_voltage_angles)
    if not recycle["Ybus"]:
        # updates trafo and trafo3w values
        _update_trafo_trafo3w_ppc(net, ppc, bus_lookup, calculate_voltage_angles, trafo_model)

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

    return ppc, ppci, bus_lookup
