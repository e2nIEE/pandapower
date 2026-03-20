import pytest
from copy import deepcopy
import logging
from itertools import product

from pandapower.grid_equivalents.get_equivalent import get_equivalent
from pandapower.run import runpp
from pandapower.create import create_switch

from pandapower.networks.power_system_test_cases import case9, case30, case39, case118

epsilon = 1e-4
"""
Attention:

the epsilon value depends on the "tolerance_mva" by the power flow calculation.

if tolerance_mva = 1e-8, this test should work with epsilon=1e-6 .
if tolerance_mva = 1e-6, we should use a bigger value, e.g. epsilon=1e-5
"""

test_data = {
    "c9-0": (case9, [3], [0], True, {}),
    "c9-1": (case9, [4, 8], [0], True, {}),
    "c9-2": (case9, [4, 8], [0], True, {"buses_out_of_service": [6]}),
    "c9-3": (case9, [3, 4], [0], True, {"switch_changes": [['l', 5, 4]]}),
    "c9-4": (case9, [3, 8], [], False, {}),
    "c9-5": (case9, [4, 6], [0], True, {"switch_changes": [["b", 4, 8]]}),
    "c9-6": (case9, [3, 4, 6], [1], False, {}),
    "c9-7": (case9, [3, 5], [2, 4, 0], True, {}),

    "c30-0": (case30, [8], [0], True, {"buses_out_of_service": [9]}),
    "c30-1": (case30, [1, 2], [0], True, {}),
    "c30-2": (case30, [3, 9, 22], [0], True, {"switch_changes": [['b', 11, 19]]}),
    "c30-3": (case30, [21, 22, 26], [0, 20], True, {"switch_changes": [['b', 22, 18], ['l', 21, 30]]}),
    "c30-4": (case30, [3, 16, 19, 22], [0, 20], True, {}),
    "c30-5": (case30, [5, 16, 18, 23, 27], [0, 24], True, {"buses_out_of_service": [9, 28]}),

    "c39-0": (case39, [1, 7], [0], False, {"buses_out_of_service": [4, 8], "switch_changes": [['b', 2, 25]]}),
    "c39-1": (case39, [15, 25], [30], True, {"switch_changes": [['t', 11, 4], ['t', 11, 3]]}),

    # FIXME: Some set from case118 crash the pipeline (c118-5)
    "c118-0": (case118, [7], [0], True, {}),
    "c118-1": (case118, [4, 14, 15], [0], True, {}),
    "c118-2": (case118, [4, 14, 15], [68], True, {}),
    "c118-3": (case118, [18, 22, 37, 64], [68], True, {"buses_out_of_service": [32]}),
    "c118-4": (case118, [18, 20, 25, 26, 29, 31], [68], True, {}),
    "c118-5": (case118, [70, 69, 67, 48, 44], [68], True, {}),
    "c118-6": (case118, [39, 42, 48, 65, 69, 70], [68], True, {})
}


@pytest.mark.parametrize("eq_type, sn_mva", list(product(["xward", "rei", "ward"], [1.0, 23.0, 89.0])))
@pytest.mark.parametrize(
    "net_func, boundary_buses, internal_buses, return_internal, kwargs", test_data.values(), ids=test_data.keys()
)
def test_networks(eq_type, sn_mva, net_func, boundary_buses, internal_buses, return_internal, kwargs):
    logging.debug(f'test with {net_func.__name__}:')
    net = net_func()
    net.sn_mva = sn_mva

    va_degree = net_func.__name__ == "case118" and eq_type != "xward"

    max_error, related_values = get_max_error(
        net, eq_type, boundary_buses, internal_buses, return_internal, va_degree=va_degree, **kwargs
    )
    assert max_error < epsilon


def get_max_error(net, eq_type, boundary_buses, internal_buses, return_internal,
                  buses_out_of_service=None, switch_changes=(), **kwargs):
    # --- topology adaption
    if switch_changes:
        for i in range(len(switch_changes)):
            create_switch(net, bus=switch_changes[i][1], element=switch_changes[i][2], et=switch_changes[i][0])
    runpp(net)
    #  --- get net_eq
    net_eq = get_equivalent(
        net, eq_type, boundary_buses, internal_buses, return_internal=return_internal, calculate_voltage_angles=True
    )

    # --- calulate max. error
    max_error, related_values = calc_max_error(net, net_eq, return_internal, **kwargs)
    return max_error, related_values


def calc_max_error(net_org, net_eq, return_internal, va_degree=True):
    max_error = 0
    related_values = ""
    i_buses = net_eq.bus_lookups["origin_all_internal_buses"]
    reserved_buses = net_eq.bus.index.tolist()
    res_bus_parameter_to_compare = ["vm_pu", "p_mw", "q_mvar"]
    if va_degree:
        res_bus_parameter_to_compare += ["va_degree"]

    if return_internal and len(set(reserved_buses) & set(i_buses)):
        related_buses = net_eq.bus_lookups["origin_all_internal_buses"]

        if len(related_buses):
            for para in res_bus_parameter_to_compare:
                max_para_error = max(
                    abs(net_eq.res_bus[para][related_buses].values - net_org.res_bus[para][related_buses].values)
                )
                if max_para_error > max_error:
                    max_error = max_para_error
                    related_values = para
    else:
        related_buses = net_eq.bus_lookups["bus_lookup_pd"]["b_area_buses"]
        for para in ["vm_pu"]:
            max_para_error = max(
                abs(net_eq.res_bus[para][related_buses].values - net_org.res_bus[para][related_buses].values)
            )
            if max_para_error > max_error:
                max_error = max_para_error
                related_values = para

    return max_error, related_values


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
