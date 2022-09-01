import pytest
from copy import deepcopy
import pandapower as pp
import pandapower.networks as pn
import pandapower.grid_equivalents
import logging


def test_networks():
    epsilon = 1e-4
    """
    Attention:

    the epsilon value depends on the "tolerance_mva" by the power flow calculation.
    (please confirm the tolerance_mva-value in the function "try_runpp" of
    "get_equivalent.py")

    if tolerance_mva = 1e-8, this test mit epsilon=1e-6 should work.
    if tolerance_mva = 1e-6, we should hier give a bigger value, e.g. epsilon = 1e-5

    """

    for eq_type in ["xward", "rei", "ward"]:
        # case9
        for sn_mva in [1.0, 23.0, 89.0]:
            net = pn.case9()
            net.sn_mva = sn_mva
            pp.runpp(net)
            logging.debug('test with case9:')

            max_error, related_values = get_max_error(
                net, eq_type, boundary_buses=[3], internal_buses=[0], return_internal=True)
            assert max_error < epsilon

            max_error, related_values = get_max_error(
                net, eq_type, boundary_buses=[4, 8], internal_buses=[0], return_internal=True)
            assert max_error < epsilon

            max_error, related_values = get_max_error(
                net, eq_type, boundary_buses=[4, 8], internal_buses=[0], return_internal=True,
                buses_out_of_service=[6])
            assert max_error < epsilon

            max_error, related_values = get_max_error(
                net, eq_type, boundary_buses=[3, 4], internal_buses=[0], return_internal=True,
                switch_changes=[['l', 5, 4]])
            assert max_error < epsilon

            max_error, related_values = get_max_error(
                net, eq_type, boundary_buses=[3, 8], internal_buses=[], return_internal=False)
            assert max_error < epsilon

            max_error, related_values = get_max_error(
                net, eq_type, boundary_buses=[4, 6], internal_buses=[0], return_internal=True,
                switch_changes=[["b", 4, 8]])
            assert max_error < epsilon

            max_error, related_values = get_max_error(
                net, eq_type, boundary_buses=[3, 4, 6], internal_buses=[1], return_internal=False)
            assert max_error < epsilon

            max_error, related_values = get_max_error(
                net, eq_type, boundary_buses=[3, 5], internal_buses=[2, 4, 0], return_internal=True)
            assert max_error < epsilon

            # case30
            logging.debug('test with case30:')
            net = pn.case30()
            net.sn_mva = sn_mva
            pp.runpp(net)

            max_error, related_values = get_max_error(
                net, eq_type, boundary_buses=[8], internal_buses=[0], return_internal=True,
                buses_out_of_service=[9])
            assert max_error < epsilon

            max_error, related_values = get_max_error(
                net, eq_type, boundary_buses=[1, 2], internal_buses=[0], return_internal=True)
            assert max_error <= epsilon

            max_error, related_values = get_max_error(
                net, eq_type, boundary_buses=[3, 9, 22], internal_buses=[0], return_internal=True,
                switch_changes=[['b', 11, 19]])
            assert max_error <= epsilon

            max_error, related_values = get_max_error(
                net, eq_type, boundary_buses=[21, 22, 26], internal_buses=[0, 20], return_internal=True,
                switch_changes=[['b', 22, 18], ['l', 21, 30]])
            assert max_error <= epsilon

            max_error, related_values = get_max_error(
                net, eq_type, boundary_buses=[3, 16, 19, 22], internal_buses=[0, 20],
                return_internal=True)
            assert max_error <= epsilon

            max_error, related_values = get_max_error(
                net, eq_type, boundary_buses=[5, 16, 18, 23, 27], internal_buses=[0, 24],
                return_internal=True, buses_out_of_service=[9, 28])
            assert max_error <= epsilon

            # case39
            logging.debug('test with case39:')
            net = pn.case39()
            net.sn_mva = sn_mva
            pp.runpp(net)
            max_error, related_values = get_max_error(
                net, eq_type, boundary_buses=[1, 7], internal_buses=[0], return_internal=False,
                buses_out_of_service=[4, 8], switch_changes=[['b', 2, 25]])
            assert max_error <= epsilon

            max_error, related_values = get_max_error(
                net, eq_type, boundary_buses=[15, 25], internal_buses=[30], return_internal=True,
                switch_changes=[['t', 11, 4], ['t', 11, 3]])
            assert max_error <= epsilon

            # case118
            logging.debug('case118:')
            net = pn.case118()
            net.sn_mva = sn_mva
            pp.runpp(net)
            va_degree = eq_type != "xward"
            max_error, related_values = get_max_error(
                net, eq_type, boundary_buses=[7], internal_buses=[0], return_internal=True,
                va_degree=va_degree)
            assert max_error <= epsilon

            max_error, related_values = get_max_error(
                net, eq_type, boundary_buses=[4, 14, 15], internal_buses=[0], return_internal=True,
                va_degree=va_degree)
            assert max_error <= epsilon

            max_error, related_values = get_max_error(
                net, eq_type, boundary_buses=[4, 14, 15], internal_buses=[68], return_internal=True,
                va_degree=va_degree)
            assert max_error <= epsilon

            max_error, related_values = get_max_error(
                net, eq_type, boundary_buses=[18, 22, 37, 64], internal_buses=[68],
                return_internal=True, buses_out_of_service=[32], va_degree=va_degree)
            assert max_error <= epsilon

            max_error, related_values = get_max_error(
                net, eq_type, boundary_buses=[18, 20, 25, 26, 29, 31], internal_buses=[68],
                return_internal=True, va_degree=va_degree)
            assert max_error <= epsilon

            max_error, related_values = get_max_error(
                net, eq_type, boundary_buses=[70, 69, 67, 48, 44], internal_buses=[68],
                return_internal=True, va_degree=va_degree)
            assert max_error <= epsilon

            max_error, related_values = get_max_error(
                net, eq_type, boundary_buses=[39, 42, 48, 65, 69, 70], internal_buses=[68],
                return_internal=True, va_degree=va_degree)
            assert max_error <= epsilon


def get_max_error(net, eq_type, boundary_buses, internal_buses, return_internal,
                  buses_out_of_service=[], switch_changes=[], **kwargs):
    # --- topology adaption
    if switch_changes:
        net = deepcopy(net)
        for i in range(len(switch_changes)):
            pp.create_switch(net, bus=switch_changes[i][1], element=switch_changes[i][2],
                             et=switch_changes[i][0])
    pp.runpp(net)
    #  --- get net_eq
    net_eq = pp.grid_equivalents.get_equivalent(net, eq_type, boundary_buses, internal_buses,
                                                return_internal=return_internal,
                                                calculate_voltage_angles=True)

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
                max_para_error = max(abs(net_eq.res_bus[para][related_buses].values -
                                         net_org.res_bus[para][related_buses].values))
                if max_para_error > max_error:
                    max_error = max_para_error
                    related_values = para
    else:
        related_buses = net_eq.bus_lookups["bus_lookup_pd"]["b_area_buses"]
        for para in ["vm_pu"]:
            max_para_error = max(abs(net_eq.res_bus[para][related_buses].values -
                                     net_org.res_bus[para][related_buses].values))
            if max_para_error > max_error:
                max_error = max_para_error
                related_values = para

    return max_error, related_values


if __name__ == "__main__":
    if 0:
        pytest.main(['-x', __file__])
    else:
        test_networks()
    pass
