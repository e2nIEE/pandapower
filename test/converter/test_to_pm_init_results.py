import math

import pytest

import pandapower as pp
from pandapower.converter.pandamodels import convert_pp_to_pm


def test_convert_pp_to_pm_init_results_propagates_bus_vm_and_va():
    """Ensures that init_vm_pu/init_va_degree='results' is not wiped by to_pm conversion.
    """

    net = pp.create_empty_network(sn_mva=100.0)
    b_slack = pp.create_bus(net, vn_kv=110.0)
    b_load = pp.create_bus(net, vn_kv=110.0)
    pp.create_ext_grid(net, b_slack, vm_pu=1.02)
    pp.create_line_from_parameters(
        net,
        from_bus=b_slack,
        to_bus=b_load,
        length_km=1.0,
        r_ohm_per_km=0.01,
        x_ohm_per_km=0.05,
        c_nf_per_km=0.0,
        max_i_ka=1.0,
    )
    pp.create_load(net, b_load, p_mw=10.0, q_mvar=2.0)

    pp.runpp(net, calculate_voltage_angles=True)
    net.res_bus.at[b_load, "vm_pu"] = 0.987
    net.res_bus.at[b_load, "va_degree"] = 7.123

    pm = convert_pp_to_pm(
        net,
        calculate_voltage_angles=True,
        init_vm_pu="results",
        init_va_degree="results",
    )

    pm_bus_slack = int(net._pd2pm_lookups["bus"][b_slack])
    pm_bus_load = int(net._pd2pm_lookups["bus"][b_load])

    assert pm["bus"][str(pm_bus_slack)]["vm"] == pytest.approx(net.res_bus.at[b_slack, "vm_pu"])
    assert pm["bus"][str(pm_bus_slack)]["va"] == pytest.approx(
        math.radians(net.res_bus.at[b_slack, "va_degree"])
    )

    assert pm["bus"][str(pm_bus_load)]["vm"] == pytest.approx(0.987)
    assert pm["bus"][str(pm_bus_load)]["va"] == pytest.approx(math.radians(7.123))