# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import warnings
from pandapower.build_bus import _add_load_sc_impedances_ppc
import numpy as np
from scipy.sparse.linalg import factorized
from numbers import Number
from pandapower.auxiliary import _clean_up, _add_ppc_options, _add_sc_options, _add_auxiliary_elements
from pandapower.pd2ppc import _pd2ppc, _ppc2ppci
from pandapower.pd2ppc_zero import _pd2ppc_zero
from pandapower.results import _copy_results_ppci_to_ppc
from pandapower.shortcircuit.currents import _calc_ikss, _calc_ip, _calc_ith
from pandapower.shortcircuit.impedance import _calc_zbus, _calc_ybus, _calc_rx
from pandapower.shortcircuit.ppc_conversion import _create_ppc, _create_k_updated_ppci, _get_is_ppci_bus
from pandapower.shortcircuit.kappa import _add_kappa_to_ppc
from pandapower.shortcircuit.results import _extract_net_results, _extract_bus_results
from pandapower.results import init_results
from pandapower.pypower.idx_brch_sc import K_ST
import logging

logger = logging.getLogger(__name__)


def calc_sc(net, fault_bus=None, fault="LLL", case='max', lv_tol_percent=10, 
            topology="auto", ip=False, ith=False, tk_s=1., kappa_method="C", 
            r_fault_ohm=0., x_fault_ohm=0., branch_results=True, 
            check_connectivity=True, use_pre_fault_voltage=False):

    """
    Calculates minimum or maximum symmetrical short-circuit currents.
    The calculation is based on the method of the equivalent voltage source
    according to DIN/IEC EN 60909.
    The initial short-circuit alternating current *ikss* is the basis of the short-circuit
    calculation and is therefore always calculated.
    Other short-circuit currents can be calculated from *ikss* with the conversion factors defined
    in DIN/IEC EN 60909.

    INPUT:
        **net** (pandapowerNet) pandapower Network

        **bus** (int, list, np.array, None) defines if short-circuit calculations should only be calculated for defined bus

        ***fault** (str, LLL) type of fault

            - "3ph" or "LLL" for three-phase faults

            - "2ph" or "LL" for two-phase (phase-to-phase) faults

            - "2ph-g" or "LLG" for two-phase-to-ground faults

            - "1ph" or "LG" for single-phase-to-ground faults

        **case** (str, "max")

            - "max" for maximum current calculation

            - "min" for minimum current calculation

        **lv_tol_percent** (int, 10) voltage tolerance in low voltage grids

            - 6 for 6% voltage tolerance

            - 10 for 10% voltage olerance

        **ip** (bool, False) if True, calculate aperiodic short-circuit current

        **ith** (bool, False) if True, calculate equivalent thermical short-circuit current Ith

        **topology** (str, "auto") define option for meshing (only relevant for ip and ith)

            - "meshed" - it is assumed all buses are supplied over multiple paths

            - "radial" - it is assumed all buses are supplied over exactly one path

            - "auto" - topology check for each bus is performed to see if it is supplied over multiple paths

        **tk_s** (float, 1) failure clearing time in seconds (only relevant for ith)

        **r_fault_ohm** (float, 0) fault resistance in Ohm

        **x_fault_ohm** (float, 0) fault reactance in Ohm

        **branch_results** (bool, False) defines if short-circuit results are calculated on all branches, namely on the entire grid

        **use_pre_fault_voltage** (bool, False) whether to consider the pre-fault grid state (superposition method, "Type C"). The user must first execute pp.runpp(net) before executing sc.calc_sc in this case

    OUTPUT:

    EXAMPLE:
        calc_sc(net)

        print(net.res_bus_sc)
        print(net.res_line_sc)
        ...
    """
    if fault in ["3ph", "2ph", "1ph", "2ph-g"]:
        msg = ("Short-circuit fault types 3ph, 2ph, 2ph-g and 1ph have been renamed to LLL, LL, LLG and LG, "
            "please use the new naming convention as the old convention will be removed in future pandapower versions.")
        warnings.warn(msg, DeprecationWarning)
        mapping = {"3ph": "LLL", "2ph": "LL", "1ph": "LG", "2ph-g": "LLG"}
        fault = mapping[fault]

    if fault not in ["LLL", "LL", "LG", "LLG"]:
        raise NotImplementedError(
            "Only LLL, LL, LLG and LG short-circuit faults implemented")

    if len(net.gen) and (ip or ith):
        logger.warning("aperiodic, thermal short-circuit currents are only implemented for "
                       "faults far from generators!")

    if case not in ['max', 'min']:
        raise ValueError('case can only be "min" or "max" for minimal or maximal short "\
                                "circuit current')

    if topology not in ["meshed", "radial", "auto"]:
        raise ValueError(
            'specify network structure as "meshed", "radial" or "auto"')

    # NOTE: Type-C short circuit calculation is currently not fully supported
    if use_pre_fault_voltage:
        init_vm_pu = init_va_degree = "results"
        trafo_model = net._options["trafo_model"] # trafo model for SC must match the trafo model for PF calculation
        if not isinstance(fault_bus, Number) and len(net.sgen.query("in_service")) > 0:
            raise NotImplementedError("Short-circuit with Type C method and sgen is only implemented for a single bus")
    else:
        init_vm_pu = init_va_degree = "flat"
        trafo_model = "pi"

    # Convert bus to numpy array
    if fault_bus is None:
        fault_bus = net.bus.index.values
    else:
        fault_bus = np.array([fault_bus]).ravel()
        fault_bus.sort()
    if len(fault_bus) > 1:
        branch_results = False

    kappa = ith or ip

    # Convert fault impedance
    base_r = np.square(net.bus["vn_kv"][fault_bus].values) / net.sn_mva
    fault_impedance = (r_fault_ohm + x_fault_ohm * 1j) / base_r

    net["_options"] = {}
    _add_ppc_options(net, calculate_voltage_angles=False, trafo_model=trafo_model,
                     check_connectivity=check_connectivity, mode="sc", switch_rx_ratio=2,
                     init_vm_pu=init_vm_pu, init_va_degree=init_va_degree, enforce_q_lims=False,
                     recycle=None)
    _add_sc_options(net, fault=fault, case=case, lv_tol_percent=lv_tol_percent, tk_s=tk_s, topology=topology,
                    z_fault_pu=fault_impedance, kappa=kappa, ip=ip, ith=ith, branch_results=branch_results, 
                    kappa_method=kappa_method, use_pre_fault_voltage=use_pre_fault_voltage)
    init_results(net, "sc")

    if fault in ("LLL", "LG", "LLG", "LL"):
        _calc_sc(net, fault_bus)
    else:
        raise ValueError("Invalid fault %s" % fault)


def _calc_sc(net, bus):
    """
    calculation method for phase to ground short-circuit currents
    """
    # TODO: check if necessary, this is related to dclines that are not yet supported in short circuit
    _add_auxiliary_elements(net)

    # positive sequence bus impedance
    ppc_1, ppci_1 = _create_ppc(net)
    ppci_bus = _get_is_ppci_bus(net, bus)
    _, ppci_1, _ = _create_k_updated_ppci(net, ppci_1, ppci_bus=ppci_bus)

    # negative sequence bus impedance
    ppc_2, ppci_2 = _create_ppc(net, sequence=2)
    _, ppci_2, _ = _create_k_updated_ppci(net, ppci_2, ppci_bus=ppci_bus)

    # zero seq bus impedance
    ppc_0, ppci_0 = _pd2ppc_zero(net, ppc_1['branch'][:, K_ST])

    # placing this here allows saving the calculation of Ybus if not type C
    # NOTE: this is used only with Type-C calculation, which currently is not fully supported
    if net._options.get("use_pre_fault_voltage", False):
        _add_load_sc_impedances_ppc(net, ppc_1)  # add SC impedances for sgens and loads
        ppci_1 = _ppc2ppci(ppc_1, net)
        _calc_ybus(ppci_1)

        _add_load_sc_impedances_ppc(net, ppc_2, relevant_elements=("load",))  # add SC impedances for loads
        ppci_2 = _ppc2ppci(ppc_2, net)
        _calc_ybus(ppci_2)

    # calculation of admittance matrices
    _calc_ybus(ppci_1)
    _calc_ybus(ppci_2)
    _calc_ybus(ppci_0)

    # calculation of grid impedance matrices Zbus
    _calc_zbus(net, ppci_0)
    _calc_zbus(net, ppci_1)
    _calc_zbus(net, ppci_2)

    # consideration of the fault impedance
    _calc_rx(net, ppci_0, ppci_bus, 0)
    _calc_rx(net, ppci_1, ppci_bus, 1)
    _calc_rx(net, ppci_2, ppci_bus, 2)

    # calculation of symmetric short circuit current
    _calc_ikss(net, ppci_0, ppci_1, ppci_2, ppci_bus)

    _add_kappa_to_ppc(net, ppci_1)  # todo add kappa only to ppci_1?
    
    # extraction of the results
    ppc_0 = _copy_results_ppci_to_ppc(ppci_0, ppc_0, "sc")
    ppc_1 = _copy_results_ppci_to_ppc(ppci_1, ppc_1, "sc")
    ppc_2 = _copy_results_ppci_to_ppc(ppci_2, ppc_2, "sc")
    
    if net._options["branch_results"]:
        _extract_net_results(net, ppc_0, ppc_1, ppc_2, bus)
    else: 
        _extract_bus_results(net, ppc_0, ppc_1, ppc_2, bus)
    _clean_up(net)
