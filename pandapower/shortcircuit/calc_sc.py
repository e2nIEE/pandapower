# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

try:
    import pplog as logging
except:
    import logging

logger = logging.getLogger(__name__)
#import time

from pandapower.auxiliary import _clean_up, _add_ppc_options, _add_sc_options
from pandapower.pd2ppc import _pd2ppc
from pandapower.powerflow import _add_auxiliary_elements
from pandapower.results import _copy_results_ppci_to_ppc
from pandapower.shortcircuit.currents import _calc_ikss, _calc_ip, _calc_ith, _calc_branch_currents
from pandapower.shortcircuit.impedance import _calc_zbus, _calc_ybus, _calc_rx
from pandapower.shortcircuit.kappa import _add_kappa_to_ppc
from pandapower.shortcircuit.results import _extract_results


def calc_sc(net, fault="3ph", case='max', lv_tol_percent=10, topology="auto", ip=False,
          ith=False, tk_s=1., kappa_method="C", r_fault_ohm=0., x_fault_ohm=0., branch_results=True):

    """
    Calculates minimal or maximal symmetrical short-circuit currents.
    The calculation is based on the method of the equivalent voltage source
    according to DIN/IEC EN 60909.
    The initial short-circuit alternating current *ikss* is the basis of the short-circuit
    calculation and is therefore always calculated.
    Other short-circuit currents can be calculated from *ikss* with the conversion factors defined
    in DIN/IEC EN 60909.

    The output is stored in the net.res_bus_sc table as a short_circuit current
    for each bus.

    INPUT:
        **net** (pandapowerNet) pandapower Network

        ***fault** (str, 3ph) type of fault

            - "3ph" for three-phase

            - "2ph" for two-phase short-circuits

        **case** (str, "max")

            - "max" for maximal current calculation

            - "min" for minimal current calculation

        **lv_tol_percent** (int, 10) voltage tolerance in low voltage grids

            - 6 for 6% voltage tolerance

            - 10 for 10% voltage olerance

        **ip** (bool, False) if True, calculate aperiodic short-circuit current

        **Ith** (bool, False) if True, calculate equivalent thermical short-circuit current Ith

        **topology** (str, "auto") define option for meshing (only relevant for ip and ith)

            - "meshed" - it is assumed all buses are supplied over multiple paths

            - "radial" - it is assumed all buses are supplied over exactly one path

            - "auto" - topology check for each bus is performed to see if it is supplied over multiple paths

        **tk_s** (float, 1) failure clearing time in seconds (only relevant for ith)

        **r_fault_ohm** (float, 0) fault resistance in Ohm

        **x_fault_ohm** (float, 0) fault reactance in Ohm

        **consider_sgens** (bool, True) defines if short-circuit contribution of static generators should be considered or not


    OUTPUT:

    EXAMPLE:
        calc_sc(net)

        print(net.res_bus_sc)
    """
    if fault not in ["3ph", "2ph"]:
        raise NotImplementedError("Only 3ph and 2ph short-circuit currents implemented")

    if len(net.gen) and (ip or ith):
        logger.warning("aperiodic and thermal short-circuit currents are only implemented for faults far from generators!")

    if case not in ['max', 'min']:
        raise ValueError('case can only be "min" or "max" for minimal or maximal short "\
                                "circuit current')
    if topology not in ["meshed", "radial", "auto"]:
        raise ValueError('specify network structure as "meshed", "radial" or "auto"')

    kappa = ith or ip
    net["_options"] = {}
    _add_ppc_options(net, calculate_voltage_angles=False,
                             trafo_model="pi", check_connectivity=False,
                             mode="sc", copy_constraints_to_ppc=False,
                             r_switch=0.0, init="flat", enforce_q_lims=False, recycle=None)
    _add_sc_options(net, fault=fault, case=case, lv_tol_percent=lv_tol_percent, tk_s=tk_s,
                    topology=topology, r_fault_ohm=r_fault_ohm, kappa_method=kappa_method,
                    x_fault_ohm=x_fault_ohm, kappa=kappa, ip=ip, ith=ith,
                    consider_sgens=False, branch_results=branch_results)
    _calc_sc(net)

def _calc_sc(net):
#    t0 = time.perf_counter()
    _add_auxiliary_elements(net)
    ppc, ppci = _pd2ppc(net)
#    t1 = time.perf_counter()
    _calc_ybus(ppci)
#    t2 = time.perf_counter()
    _calc_zbus(ppci)
    _calc_rx(net, ppci)
#    t3 = time.perf_counter()
    _add_kappa_to_ppc(net, ppci)
#    t4 = time.perf_counter()
    _calc_ikss(net, ppci)
    if net["_options"]["ip"]:
        _calc_ip(net, ppci)
    if net["_options"]["ith"]:
        _calc_ith(net, ppci)
    if net._options["branch_results"]:
        _calc_branch_currents(net, ppci)
    ppc = _copy_results_ppci_to_ppc(ppci, ppc, "sc")
    _extract_results(net, ppc)
    _clean_up(net)
#    t5 = time.perf_counter()
#    net._et = {"sum": t5-t0, "model": t1-t0, "ybus": t2-t1, "zbus": t3-t2, "kappa": t4-t3,
#               "currents": t5-t4}