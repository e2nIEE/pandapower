import numpy as np

from pandapower.create import (
    create_bus,
    create_ext_grid,
    create_transformer3w,
    create_measurement
)
from pandapower.estimation.idx_brch import P_TO, P_TO_STD
from pandapower.estimation.ppc_conversion import pp2eppci
from pandapower.network import pandapowerNet
from pandapower.pypower.idx_brch import branch_cols
import pytest


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
        net,
        hv_bus=hv_bus1,
        mv_bus=mv_bus1,
        lv_bus=lv_bus1,
        std_type="63/25/38 MVA 110/20/10 kV",
        name="3W Trafo 1"
    )

    trafo2 = create_transformer3w(
        net,
        hv_bus=hv_bus2,
        mv_bus=mv_bus2,
        lv_bus=lv_bus2,
        std_type="63/25/38 MVA 110/20/10 kV",
        name="3W Trafo 2"
    )

    # Add measurements: measure active (p) and reactive (q) power on each side of each transformer
    for trafo in [trafo1, trafo2]:
        for side in ["hv", "mv"]:
            create_measurement(net,
                                  meas_type="p",
                                  element_type="trafo3w",
                                  element=trafo,
                                  side=side,
                                  value=2.0,
                                  std_dev=1,
                                  name=f"P_{side.upper()}_Trafo{trafo}")
            create_measurement(net,
                                  meas_type="p",
                                  element_type="trafo3w",
                                  element=trafo,
                                  side=side,
                                  value=4.0,
                                  std_dev=1,
                                  name=f"P2_{side.upper()}_Trafo{trafo}")

    _, _, eppci = pp2eppci(net, v_start="flat", delta_start="flat", zero_injection="aux_bus")
    vals = eppci.data["branch"][[2, 3], branch_cols + P_TO]
    np.testing.assert_array_equal(vals, [3, 3])

    std_dev = eppci.data["branch"][[2, 3], branch_cols + P_TO_STD]
    # 0.707107 =  ((1^2 + 1^2)^0.5)/2
    np.testing.assert_array_almost_equal(std_dev, [0.707107, 0.707107], decimal=6)

if __name__ == '__main__':
    pytest.main([__file__, "-xs"])