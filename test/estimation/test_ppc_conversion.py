import numpy as np
import pytest

from pandapower.create import (
    create_bus,
    create_buses,
    create_ext_grid,
    create_gens,
    create_lines,
    create_loads,
    create_measurement,
    create_sgens,
    create_switch,
    create_transformer3w,
)
from pandapower.estimation.idx_brch import P_TO, P_TO_STD
from pandapower.estimation.idx_bus import P, P_IDX, Q, Q_IDX, ZERO_INJ_FLAG
from pandapower.estimation.ppc_conversion import pp2eppci
from pandapower.network import pandapowerNet
from pandapower.pypower.idx_brch import branch_cols
from pandapower.pypower.idx_bus import bus_cols


@pytest.fixture
def zero_injection_net():
    net = pandapowerNet(name="zero_injection")
    create_buses(net, 4, vn_kv=20, index=[10, 20, 30, 40], in_service=[True, True, True, False])
    create_ext_grid(net, 10)
    create_lines(
        net,
        from_buses=[10, 20, 30],
        to_buses=[20, 30, 40],
        length_km=1,
        line_params={"r_ohm_per_km": 0.1, "x_ohm_per_km": 0.1, "c_nf_per_km": 0, "max_i_ka": 1},
    )
    for bus in [10, 20, 30]:
        create_measurement(net, "v", "bus", 1.0, 0.001, bus)
    return net


@pytest.mark.parametrize("create_elements", [create_loads, create_sgens, create_gens])
@pytest.mark.parametrize(
    "buses, indices, in_service, expected",
    [
        ([20], [0], [True], [False, False, True]),
        ([20], [20], [True], [False, False, True]),
        ([20, 30], [5, 0], [True, False], [False, False, True]),
        ([20, 30], [30, 20], [True, False], [False, False, True]),
        ([20, 20], [5, 0], [False, True], [False, False, True]),
        ([20, 20], [5, 0], [False, False], [False, True, True]),
        ([40], [0], [True], [False, True, True]),
    ],
)
def test_zero_injection_element_indices(zero_injection_net, create_elements, buses, indices, in_service, expected):
    net = zero_injection_net
    create_elements(net, buses, p_mw=0, index=indices, in_service=in_service)

    _, _, eppci = pp2eppci(net, zero_injection="no_inj_bus")
    rows = net._pd2ppc_lookups["bus"][[10, 20, 30]]
    np.testing.assert_array_equal(eppci["bus"][rows, bus_cols + ZERO_INJ_FLAG], expected)
    for row, is_zero in zip(rows, expected):
        if is_zero:
            np.testing.assert_array_equal(eppci["bus"][row, [bus_cols + P, bus_cols + Q]], [0, 0])
            np.testing.assert_array_equal(eppci["bus"][row, [bus_cols + P_IDX, bus_cols + Q_IDX]], [-1, -1])
        else:
            assert np.isnan(eppci["bus"][row, [bus_cols + P, bus_cols + Q]]).all()


def test_zero_injection_fused_bus(zero_injection_net):
    net = zero_injection_net
    create_loads(net, [20], p_mw=0, index=[5])
    create_switch(net, 20, 30, et="b", closed=True)

    _, _, eppci = pp2eppci(net, zero_injection="no_inj_bus")
    rows = net._pd2ppc_lookups["bus"][[20, 30]]
    assert rows[0] == rows[1]
    np.testing.assert_array_equal(eppci["bus"][rows, bus_cols + ZERO_INJ_FLAG], [False, False])


@pytest.mark.parametrize("create_elements", [create_loads, create_sgens])
@pytest.mark.parametrize("index", [0, 50])
def test_zero_injection_disconnected_bus(zero_injection_net, create_elements, index):
    net = zero_injection_net
    create_buses(net, 1, vn_kv=20, index=[50])
    create_elements(net, [50], p_mw=0, index=[index])

    _, _, eppci = pp2eppci(net, zero_injection="no_inj_bus")
    rows = net._pd2ppc_lookups["bus"][[10, 20, 30]]
    np.testing.assert_array_equal(eppci["bus"][rows, bus_cols + ZERO_INJ_FLAG], [False, True, True])


def test_zero_power_bus_with_active_load(zero_injection_net):
    net = zero_injection_net
    create_loads(net, [20], p_mw=0, index=[5])

    _, _, eppci = pp2eppci(net, zero_injection="zero_pwr_bus")
    rows = net._pd2ppc_lookups["bus"][[20, 30]]
    np.testing.assert_array_equal(eppci["bus"][rows, bus_cols + ZERO_INJ_FLAG], [True, True])


def test_duplicate_measurements_at_trafo3w():
    # Create an empty network
    net = pandapowerNet(name="test_duplicate_measurements_at_trafo3w")

    # Create buses for two three-winding transformers (HV, MV, LV each)
    hv_bus1 = create_bus(net, vn_kv=110, name="HV Bus 1")
    mv_bus1 = create_bus(net, vn_kv=20, name="MV Bus 1")
    lv_bus1 = create_bus(net, vn_kv=10, name="LV Bus 1")

    hv_bus2 = create_bus(net, vn_kv=110, name="HV Bus 2")
    mv_bus2 = create_bus(net, vn_kv=20, name="MV Bus 2")
    lv_bus2 = create_bus(net, vn_kv=10, name="LV Bus 2")

    # Create external grids for slack reference at HV sides
    create_ext_grid(net, bus=hv_bus1, vm_pu=1.02, name="Grid Slack 1")
    create_ext_grid(net, bus=hv_bus2, vm_pu=1.02, name="Grid Slack 2")

    # Create two three-winding transformers using standard types
    trafo1 = create_transformer3w(
        net, hv_bus=hv_bus1, mv_bus=mv_bus1, lv_bus=lv_bus1, std_type="63/25/38 MVA 110/20/10 kV", name="3W Trafo 1"
    )

    trafo2 = create_transformer3w(
        net, hv_bus=hv_bus2, mv_bus=mv_bus2, lv_bus=lv_bus2, std_type="63/25/38 MVA 110/20/10 kV", name="3W Trafo 2"
    )

    # Add measurements: measure active (p) and reactive (q) power on each side of each transformer
    for trafo in [trafo1, trafo2]:
        for side in ["hv", "mv"]:
            create_measurement(
                net,
                meas_type="p",
                element_type="trafo3w",
                element=trafo,
                side=side,
                value=2.0,
                std_dev=1,
                name=f"P_{side.upper()}_Trafo{trafo}",
            )
            create_measurement(
                net,
                meas_type="p",
                element_type="trafo3w",
                element=trafo,
                side=side,
                value=4.0,
                std_dev=1,
                name=f"P2_{side.upper()}_Trafo{trafo}",
            )

    _, _, eppci = pp2eppci(net, v_start="flat", delta_start="flat", zero_injection="aux_bus")
    vals = eppci.data["branch"][[2, 3], branch_cols + P_TO]
    np.testing.assert_array_equal(vals, [3, 3])

    std_dev = eppci.data["branch"][[2, 3], branch_cols + P_TO_STD]
    # 0.707107 =  ((1^2 + 1^2)^0.5)/2
    np.testing.assert_array_almost_equal(std_dev, [0.707107, 0.707107], decimal=6)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
