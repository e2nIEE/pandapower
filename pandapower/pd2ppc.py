# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pandapower.auxiliary as aux
from pandapower.build_branch import _switch_branches, _branches_with_oos_buses, \
    _build_branch_ppc, _build_tcsc_ppc, _build_branch_dc_ppc
from pandapower.build_bus import _build_bus_ppc, _calc_pq_elements_and_add_on_ppc, \
    _calc_shunts_and_add_on_ppc, _add_ext_grid_sc_impedance, _add_motor_impedances_ppc, \
    _build_svc_ppc, _add_load_sc_impedances_ppc, _build_ssc_ppc, _build_vsc_ppc, _build_bus_dc_ppc
from pandapower.build_gen import _build_gen_ppc, _check_voltage_setpoints_at_same_bus, \
    _check_voltage_angles_at_same_bus, _check_for_reference_bus
from pandapower.opf.make_objective import _make_objective
from pandapower.pypower.idx_area import PRICE_REF_BUS
from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_STATUS
from pandapower.pypower.idx_brch_dc import DC_F_BUS, DC_T_BUS, DC_BR_STATUS
from pandapower.pypower.idx_bus import NONE, BUS_I, BUS_TYPE, REF
from pandapower.pypower.idx_bus_dc import DC_BUS_I, DC_BUS_TYPE, DC_NONE, DC_B2B
from pandapower.pypower.idx_gen import GEN_BUS, GEN_STATUS
from pandapower.pypower.idx_ssc import SSC_STATUS, SSC_BUS, SSC_INTERNAL_BUS
from pandapower.pypower.idx_tcsc import TCSC_STATUS, TCSC_F_BUS, TCSC_T_BUS
from pandapower.pypower.idx_svc import SVC_STATUS, SVC_BUS
from pandapower.pypower.idx_vsc import VSC_BUS, VSC_INTERNAL_BUS, VSC_BUS_DC, VSC_STATUS, VSC_MODE_AC, VSC_MODE_AC_V, \
    VSC_MODE_AC_Q, VSC_MODE_AC_SL, VSC_INTERNAL_BUS_DC
from pandapower.pypower.run_userfcn import run_userfcn
from itertools import combinations


def _pd2ppc_recycle(net, sequence, recycle):
    # todo for FACTS and DC elements: SVC, TCSC, SSC, VSC
    key = "_ppc" if sequence is None else "_ppc%d" % sequence
    if not recycle or not net.get(key, None):
        return _pd2ppc(net, sequence=sequence)

    ppc = net[key]
    ppc["success"] = False
    ppc["iterations"] = 0.
    ppc["et"] = 0.

    if "bus_pq" in recycle and recycle["bus_pq"]:
        # update pq values in bus
        _calc_pq_elements_and_add_on_ppc(net, ppc, sequence=sequence)

    # if "trafo" in recycle and recycle["trafo"]:
    #     # update trafo in branch and Ybus
    #     lookup = net._pd2ppc_lookups["branch"]
    #     if "trafo" in lookup:
    #         _calc_trafo_parameter(net, ppc)
    #     if "trafo3w" in lookup:
    #         _calc_trafo3w_parameter(net, ppc)

    if "gen" in recycle and recycle["gen"]:
        # updates the ppc["gen"] part
        _build_gen_ppc(net, ppc)
        ppc["gen"] = np.nan_to_num(ppc["gen"])

    ppci = _ppc2ppci(ppc, net)
    ppci["internal"] = net[key]["internal"]
    net[key] = ppc

    return ppc, ppci


def _check_line_dc_at_b2b_buses(ppci):
    b2b_buses = ppci["bus_dc"][ppci["bus_dc"][:, DC_BUS_TYPE] == DC_B2B, DC_BUS_I].astype(np.int64)
    intersect_from = np.intersect1d(ppci["branch_dc"][:, DC_F_BUS].astype(np.int64), b2b_buses)
    intersect_to = np.intersect1d(ppci["branch_dc"][:, DC_T_BUS].astype(np.int64), b2b_buses)
    if len(intersect_from) != 0 or len(intersect_to) != 0:
        raise NotImplementedError("Found DC lines connected to Back-To-Back VSC converter configuration - "
                                  "not implemented. DC lines can only connect to the DC buses that are not "
                                  "part of a Back-To-Back configuration.")


def _check_vsc_different_ac_control_modes_at_same_bus(ppci):
    ac_vm_pu_buses = ppci["vsc"][ppci["vsc"][:, VSC_MODE_AC] == VSC_MODE_AC_V, VSC_BUS]
    ac_q_mvar_buses = ppci["vsc"][ppci["vsc"][:, VSC_MODE_AC] == VSC_MODE_AC_Q, VSC_BUS]
    ac_slack_buses = ppci["vsc"][ppci["vsc"][:, VSC_MODE_AC] == VSC_MODE_AC_SL, VSC_BUS]
    ac_bus_intersection = np.hstack([np.intersect1d(a, b) for a, b in combinations([ac_vm_pu_buses, ac_q_mvar_buses, ac_slack_buses], r=2)])
    if len(ac_bus_intersection) != 0:
        raise NotImplementedError("Found multiple VSC converters that share the same AC bus and have "
                                  "different AC control modes - not implemented. VSC converters can only "
                                  "have the same AC control mode if they share the same AC bus.")


def _check_slack_at_vsc_bus(ppci):
    vsc_buses = ppci["vsc"][:, VSC_BUS]
    ac_slack_buses = ppci["gen"][ppci["internal"]["ref_gens"], GEN_BUS]
    ac_bus_intersection = np.intersect1d(vsc_buses, ac_slack_buses)
    if len(ac_bus_intersection) != 0:
        raise NotImplementedError("Found VSC elements connected to AC slack buses - "
                                  "this configuration is not implemented.")


def _pd2ppc(net, sequence=None, **kwargs):
    """
    Converter Flow:
        1. Create an empty pypower datatructure
        2. Calculate loads and write the bus matrix
        3. Build the gen (Infeeder)- Matrix
        4. Calculate the line parameter and the transformer parameter,
           and fill it in the branch matrix.
           Order: 1st: Line values, 2nd: Trafo values
        5. if opf: make opf objective (gencost)
        6. convert internal ppci format for pypower powerflow /
        opf without out of service elements and rearanged buses

    INPUT:
        **net** - The pandapower format network
        **sequence** - Used for three phase analysis
        ( 0 - Zero Sequence
          1 - Positive Sequence
          2 - Negative Sequence
        )

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
    net["_is_elements"] = aux._select_is_elements_numba(net, sequence=sequence)

    # Gets network configurations
    mode = net["_options"]["mode"]
    check_connectivity = net["_options"]["check_connectivity"]
    calculate_voltage_angles = net["_options"]["calculate_voltage_angles"]

    ppc = _init_ppc(net, mode=mode, sequence=sequence)

    # generate ppc['bus'] and the bus lookup
    _build_bus_ppc(net, ppc, sequence=sequence)
    _build_bus_dc_ppc(net, ppc)
    if sequence == 0:
        from pandapower.pd2ppc_zero import _add_ext_grid_sc_impedance_zero, _build_branch_ppc_zero
        # Adds external grid impedance for 3ph and sc calculations in ppc0
        _add_ext_grid_sc_impedance_zero(net, ppc)
        # Calculates ppc0 branch impedances from branch elements
        _build_branch_ppc_zero(net, ppc)
    else:
        # Calculates ppc1/ppc2 branch impedances from branch elements
        _build_branch_ppc(net, ppc)
    _build_branch_dc_ppc(net, ppc)

    _build_tcsc_ppc(net, ppc, mode)
    _build_svc_ppc(net, ppc, mode)
    _build_ssc_ppc(net, ppc, mode)
    _build_vsc_ppc(net, ppc, mode)

    # Adds P and Q for loads / sgens in ppc['bus'] (PQ nodes)
    if mode == "sc":
        _add_ext_grid_sc_impedance(net, ppc)
        # Generator impedance are seperately added in sc module
        _add_motor_impedances_ppc(net, ppc)
        if net._options.get("use_pre_fault_voltage", False):
            _add_load_sc_impedances_ppc(net, ppc)  # add SC impedances for loads

    else:
        _calc_pq_elements_and_add_on_ppc(net, ppc, sequence=sequence)
        # adds P and Q for shunts, wards and xwards (to PQ nodes)
        _calc_shunts_and_add_on_ppc(net, ppc)

    # adds auxilary buses for open switches at branches
    _switch_branches(net, ppc)

    # Adds auxilary buses for in service lines with out of service buses.
    # Also deactivates lines if they are connected to two out of service buses
    _branches_with_oos_buses(net, ppc)

    if check_connectivity:
        if sequence in [None, 1, 2]:
            # sets islands (multiple isolated nodes) out of service
            if mode == "opf":
                net["_isolated_buses"], _, _ = aux._check_connectivity_opf(ppc)
                net["_isolated_buses_dc"] = np.array([], dtype=np.int64)
            else:
                net["_isolated_buses"], _, _, net["_isolated_buses_dc"], _, _ = aux._check_connectivity(ppc)
            net["_is_elements_final"] = aux._select_is_elements_numba(net, net._isolated_buses,
                                                                      net._isolated_buses_dc, sequence)
        else:
            ppc["bus"][net._isolated_buses, BUS_TYPE] = NONE
        net["_is_elements"] = net["_is_elements_final"]
    else:
        # sets buses out of service, which aren't connected to branches / REF buses
        aux._set_isolated_buses_out_of_service(net, ppc)

    # we need to check this after checking connectivity (isolated vsc as DC slack cause change of DC_REF to DC_P)
    if "pf" in mode or "se" in mode:
        _check_for_reference_bus(ppc)

    _build_gen_ppc(net, ppc)

    aux._replace_nans_with_default_limits(net, ppc)

    # generates "internal" ppci format (for powerflow calc)
    # from "external" ppc format and updates the bus lookup
    # Note: Also reorders buses and gens in ppc
    ppci = _ppc2ppci(ppc, net)

    _check_line_dc_at_b2b_buses(ppci)
    _check_vsc_different_ac_control_modes_at_same_bus(ppci)
    # _check_slack_at_vsc_bus(ppci)

    if mode == "pf":
        # check if any generators connected to the same bus have different voltage setpoints
        _check_voltage_setpoints_at_same_bus(ppc)
        if calculate_voltage_angles:
            _check_voltage_angles_at_same_bus(net, ppci)

    if mode == "opf":
        # make opf objective
        ppci = _make_objective(ppci, net)

    return ppc, ppci


def _init_ppc(net, mode="pf", sequence=None):
    # init empty ppc
    ppc = {"baseMVA": net.sn_mva,
           "version": 2,
           "bus": np.array([], dtype=float),
           "bus_dc": np.array([], dtype=np.float64),
           "branch": np.array([], dtype=np.complex128),
           "branch_dc": np.array([], dtype=np.float64),
           "tcsc": np.array([], dtype=np.complex128),
           "svc": np.array([], dtype=np.complex128),
           "ssc": np.array([], dtype=np.complex128),
           "vsc": np.array([], dtype=np.float64),
           "gen": np.array([], dtype=float),
           "internal": {
               "Ybus": np.array([], dtype=np.complex128),
               "Yf": np.array([], dtype=np.complex128),
               "Yt": np.array([], dtype=np.complex128),
               "branch_is": np.array([], dtype=bool),
               "branch_dc_is": np.array([], dtype=bool),
               "gen_is": np.array([], dtype=bool),
               "DLF": np.array([], dtype=np.complex128),
               "buses_ord_bfs_nets": np.array([], dtype=float)
               }
           }
    if mode == "opf":
        # additional fields in ppc
        ppc["gencost"] = np.array([], dtype=float)
    net["_ppc"] = ppc

    if sequence is None:
        net["_ppc"] = ppc
    else:
        ppc["sequence"] = int(sequence)
        net["_ppc%s" % sequence] = ppc
    return ppc


def _ppc2ppci(ppc, net, ppci=None):
    """
    Creates the ppci which is used to run the power flow / OPF...
    The ppci is similar to the ppc except that:
    1. it contains no out of service elements
    2. buses are sorted

    Parameters
    ----------
    ppc - the ppc
    net - the pandapower net

    Returns
    -------
    ppci - the "internal" ppc

    """
    # get empty ppci
    if ppci is None:
        ppci = _init_ppc(net, mode=net["_options"]["mode"])
    # BUS Sorting and lookups
    # get bus_lookup
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    bus_dc_lookup = net["_pd2ppc_lookups"]["bus_dc"]
    # get OOS busses and place them at the end of the bus array
    # (there are no OOS busses in the ppci)
    oos_buses = ppc['bus'][:, BUS_TYPE] == NONE
    oos_buses_dc = ppc['bus_dc'][:, BUS_TYPE] == DC_NONE
    ppci['bus'] = ppc['bus'][~oos_buses]
    ppci['bus_dc'] = ppc['bus_dc'][~oos_buses_dc]
    # in ppc the OOS busses are included and at the end of the array
    ppc['bus'] = np.vstack([ppc['bus'][~oos_buses], ppc['bus'][oos_buses]])
    ppc['bus_dc'] = np.vstack([ppc['bus_dc'][~oos_buses_dc], ppc['bus_dc'][oos_buses_dc]])

    # generate bus_lookup_ppc_ppci (ppc -> ppci lookup)
    ppc_former_order = (ppc['bus'][:, BUS_I]).astype(np.int64)
    aranged_buses = np.arange(len(ppc["bus"]))

    # generate bus_dc_lookup_ppc_ppci (ppc -> ppci lookup)
    ppc_former_order_dc = (ppc['bus_dc'][:, DC_BUS_I]).astype(np.int64)
    aranged_buses_dc = np.arange(len(ppc["bus_dc"]))

    # lookup ppc former order -> consecutive order
    e2i = np.zeros(len(ppc["bus"]), dtype=np.int64)
    e2i[ppc_former_order] = aranged_buses

    # dc lookup ppc former order -> consecutive order
    e2i_dc = np.zeros(len(ppc["bus_dc"]), dtype=np.int64)
    e2i_dc[ppc_former_order_dc] = aranged_buses_dc

    # save consecutive indices in ppc and ppci
    ppc['bus'][:, BUS_I] = aranged_buses
    ppci['bus'][:, BUS_I] = ppc['bus'][:len(ppci['bus']), BUS_I]

    # save consecutive indices in ppc and ppci
    ppc['bus_dc'][:, DC_BUS_I] = aranged_buses_dc
    ppci['bus_dc'][:, DC_BUS_I] = ppc['bus_dc'][:len(ppci['bus_dc']), DC_BUS_I]

    # update lookups (pandapower -> ppci internal)
    _update_lookup_entries(net, bus_lookup, e2i, "bus")

    # update lookups (pandapower -> ppci internal)
    _update_lookup_entries(net, bus_dc_lookup, e2i_dc, "bus_dc")

    if 'areas' in ppc:
        if len(ppc["areas"]) == 0:  # if areas field is empty
            del ppc['areas']  # delete it (so it's ignored)

    # bus types
    bt = ppc["bus"][:, BUS_TYPE]
    bt_dc = ppc["bus_dc"][:, DC_BUS_TYPE]

    # update branch, gen and areas bus numbering
    ppc['gen'][:, GEN_BUS] = e2i[np.real(ppc["gen"][:, GEN_BUS]).astype(np.int64)].copy()
    ppc['svc'][:, SVC_BUS] = e2i[np.real(ppc["svc"][:, SVC_BUS]).astype(np.int64)].copy()
    ppc['ssc'][:, SSC_BUS] = e2i[np.real(ppc["ssc"][:, SSC_BUS]).astype(np.int64)].copy()
    ppc['ssc'][:, SSC_INTERNAL_BUS] = e2i[np.real(ppc["ssc"][:, SSC_INTERNAL_BUS]).astype(np.int64)].copy()
    ppc['vsc'][:, VSC_BUS] = e2i[np.real(ppc["vsc"][:, VSC_BUS]).astype(np.int64)].copy()
    ppc['vsc'][:, VSC_INTERNAL_BUS] = e2i[np.real(ppc["vsc"][:, VSC_INTERNAL_BUS]).astype(np.int64)].copy()
    ppc['vsc'][:, VSC_BUS_DC] = e2i_dc[np.real(ppc["vsc"][:, VSC_BUS_DC]).astype(np.int64)].copy()
    ppc['vsc'][:, VSC_INTERNAL_BUS_DC] = e2i_dc[np.real(ppc["vsc"][:, VSC_INTERNAL_BUS_DC]).astype(np.int64)].copy()
    ppc["branch"][:, F_BUS] = e2i[np.real(ppc["branch"][:, F_BUS]).astype(np.int64)].copy()
    ppc["branch"][:, T_BUS] = e2i[np.real(ppc["branch"][:, T_BUS]).astype(np.int64)].copy()
    ppc["branch_dc"][:, DC_F_BUS] = e2i_dc[np.real(ppc["branch_dc"][:, DC_F_BUS]).astype(np.int64)].copy()
    ppc["branch_dc"][:, DC_T_BUS] = e2i_dc[np.real(ppc["branch_dc"][:, DC_T_BUS]).astype(np.int64)].copy()
    ppc["tcsc"][:, TCSC_F_BUS] = e2i[np.real(ppc["tcsc"][:, TCSC_F_BUS]).astype(np.int64)].copy()
    ppc["tcsc"][:, TCSC_T_BUS] = e2i[np.real(ppc["tcsc"][:, TCSC_T_BUS]).astype(np.int64)].copy()

    # Note: The "update branch, gen and areas bus numbering" does the same as:
    # ppc['gen'][:, GEN_BUS] = get_indices(ppc['gen'][:, GEN_BUS], bus_lookup_ppc_ppci)
    # ppc["branch"][:, F_BUS] = get_indices(ppc["branch"][:, F_BUS], bus_lookup_ppc_ppci)
    # ppc["branch"][:, T_BUS] = get_indices( ppc["branch"][:, T_BUS], bus_lookup_ppc_ppci)
    # but faster...

    if 'areas' in ppc:
        ppc["areas"][:, PRICE_REF_BUS] = \
            e2i[np.real(ppc["areas"][:, PRICE_REF_BUS]).astype(np.int64)].copy()

    # initialize gen lookups
    for element, (f, t) in net._gen_order.items():
        _build_gen_lookups(net, element, f, t)

    # determine which buses, branches, gens are connected and
    # in-service
    n2i = ppc["bus"][:, BUS_I].astype(np.int64)
    bs = (bt != NONE)  # bus status

    n2i_dc = ppc["bus_dc"][:, DC_BUS_I].astype(np.int64)
    bs_dc = (bt_dc != DC_NONE)  # bus status

    gs = ((ppc["gen"][:, GEN_STATUS] > 0) &  # gen status
          bs[n2i[np.real(ppc["gen"][:, GEN_BUS]).astype(np.int64)]])
    ppci["internal"]["gen_is"] = gs

    svcs = ((ppc["svc"][:, SVC_STATUS] > 0) &  # gen status
          bs[n2i[np.real(ppc["svc"][:, SVC_BUS]).astype(np.int64)]])
    ppci["internal"]["svc_is"] = svcs

    sscs = ((ppc["ssc"][:, SSC_STATUS] > 0) &  # ssc status
          bs[n2i[np.real(ppc["ssc"][:, SSC_BUS]).astype(np.int64)]] &
          bs[n2i[np.real(ppc["ssc"][:, SSC_INTERNAL_BUS]).astype(np.int64)]])
    ppci["internal"]["ssc_is"] = sscs

    vscs = ((ppc["vsc"][:, VSC_STATUS] > 0) &  # vsc status
          bs[n2i[np.real(ppc["vsc"][:, VSC_BUS]).astype(np.int64)]] &
          bs[n2i[np.real(ppc["vsc"][:, VSC_INTERNAL_BUS]).astype(np.int64)]] &
          bs_dc[n2i_dc[np.real(ppc["vsc"][:, VSC_BUS_DC]).astype(np.int64)]] &
          bs_dc[n2i_dc[np.real(ppc["vsc"][:, VSC_INTERNAL_BUS_DC]).astype(np.int64)]])
    ppci["internal"]["vsc_is"] = vscs

    brs = (np.real(ppc["branch"][:, BR_STATUS]).astype(np.int64) &  # branch status
           bs[n2i[np.real(ppc["branch"][:, F_BUS]).astype(np.int64)]] &
           bs[n2i[np.real(ppc["branch"][:, T_BUS]).astype(np.int64)]]).astype(bool)
    ppci["internal"]["branch_is"] = brs

    brs_dc = (np.real(ppc["branch_dc"][:, DC_BR_STATUS]).astype(np.int64) &  # branch status
           bs_dc[n2i_dc[np.real(ppc["branch_dc"][:, DC_F_BUS]).astype(np.int64)]] &
           bs_dc[n2i_dc[np.real(ppc["branch_dc"][:, DC_T_BUS]).astype(np.int64)]]).astype(bool)
    ppci["internal"]["branch_dc_is"] = brs_dc

    trs = (np.real(ppc["tcsc"][:, TCSC_STATUS]).astype(np.int64) &  # branch status
           bs[n2i[np.real(ppc["tcsc"][:, TCSC_F_BUS]).astype(np.int64)]] &
           bs[n2i[np.real(ppc["tcsc"][:, TCSC_T_BUS]).astype(np.int64)]]).astype(bool)
    ppci["internal"]["tcsc_is"] = trs

    if 'areas' in ppc:
        ar = bs[n2i[ppc["areas"][:, PRICE_REF_BUS].astype(np.int64)]]
        # delete out of service areas
        ppci["areas"] = ppc["areas"][ar]

    # select in service elements from ppc and put them in ppci
    ppci["branch"] = ppc["branch"][brs]
    ppci["branch_dc"] = ppc["branch_dc"][brs_dc]
    ppci["tcsc"] = ppc["tcsc"][trs]

    ppci["gen"] = ppc["gen"][gs]
    ppci["svc"] = ppc["svc"][svcs]
    ppci["ssc"] = ppc["ssc"][sscs]
    ppci["vsc"] = ppc["vsc"][vscs]

    if 'dcline' in ppc:
        ppci['dcline'] = ppc['dcline']
    # execute userfcn callbacks for 'ext2int' stage
    if 'userfcn' in ppci:
        ppci = run_userfcn(ppci['userfcn'], 'ext2int', ppci)

    if net._pd2ppc_lookups["ext_grid"] is not None:
        ref_gens = np.setdiff1d(net._pd2ppc_lookups["ext_grid"], np.array([-1]))
    else:
        ref_gens = np.array([])
    if np.any(net.gen.slack.values[net._is_elements["gen"]]):
        slack_gens = np.array(net.gen.index)[net._is_elements["gen"]
                                             & net.gen["slack"].values]
        ref_gens = np.append(ref_gens, net._pd2ppc_lookups["gen"][slack_gens])
    ppci["internal"]["ref_gens"] = ref_gens.astype(np.int64)
    return ppci


def _update_lookup_entries(net, lookup, e2i, element):
    valid_bus_lookup_entries = lookup >= 0
    # update entries
    lookup[valid_bus_lookup_entries] = e2i[lookup[valid_bus_lookup_entries]]
    aux._write_lookup_to_net(net, element, lookup)


def _build_gen_lookups(net, element, f, t):
    in_service = net._is_elements[element]
    if "controllable" in element:
        pandapower_index = net[element.split("_")[0]].index.values[in_service]
    else:
        pandapower_index = net[element].index.values[in_service]
    ppc_index = np.arange(f, t)
    if len(pandapower_index) > 0:
        _init_lookup(net, element, pandapower_index, ppc_index)


def _init_lookup(net, lookup_name, pandapower_index, ppc_index):
    # init lookup
    lookup = -np.ones(max(pandapower_index) + 1, dtype=np.int64)

    # update lookup
    lookup[pandapower_index] = ppc_index

    aux._write_lookup_to_net(net, lookup_name, lookup)
