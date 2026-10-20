# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import logging

import pytest
from numpy import all, allclose

from pandapower.create import (
    create_buses,
    create_ext_grid,
    create_gen,
    create_lines,
    create_load,
    create_poly_cost,
    create_transformer,
)
from pandapower.create._utils import add_column_to_df
from pandapower.network import pandapowerNet
from pandapower.run import runopp

logger = logging.getLogger(__name__)


def test_minimize_active_power_curtailment():
    net = pandapowerNet(name="test_minimize_active_power_curtailment")

    # create buses
    bus1, bus2, bus3, bus4 = create_buses(net, 4, vn_kv=[220.0, 110.0, 110.0, 110.0])

    # create 220/110 kV transformer
    create_transformer(net, bus1, bus2, std_type="100 MVA 220/110 kV")

    # create 110 kV lines
    create_lines(
        net, [bus2, bus3, bus4], [bus3, bus4, bus2], length_km=[70.0, 50.0, 40.0], line_params="149-AL1/24-ST1A 110.0"
    )

    # create loads
    create_load(net, bus2, p_mw=60, controllable=False)
    create_load(net, bus3, p_mw=70, controllable=False)
    create_load(net, bus4, p_mw=10, controllable=False)

    # create generators
    create_ext_grid(net, bus1)
    create_gen(net, bus3, p_mw=80.0, max_p_mw=80.0, min_p_mw=0.0, vm_pu=1.01, controllable=True)
    create_gen(net, bus4, p_mw=0.1, max_p_mw=100.0, min_p_mw=0.0, vm_pu=1.01, controllable=True)

    add_column_to_df(net, "gen", "min_q_mvar")
    add_column_to_df(net, "gen", "max_q_mvar")

    net.trafo["max_loading_percent"] = 50.0
    net.line["max_loading_percent"] = 50.0

    net.bus["min_vm_pu"] = 1.0
    net.bus["max_vm_pu"] = 1.02

    # use cheapest gen first, avoid ext_grid as much as possible (until vm_pu max is reached for gens)
    create_poly_cost(net, 0, "gen", cp1_eur_per_mw=0.02)
    create_poly_cost(net, 1, "gen", cp1_eur_per_mw=0.0)
    create_poly_cost(net, 0, "ext_grid", cp1_eur_per_mw=0.3)
    runopp(net, calculate_voltage_angles=True)
    assert net["OPF_converged"]
    # checks limits in general
    assert all(net.res_line.loading_percent < 50.01)
    assert all(net.res_bus.vm_pu < 1.02)
    assert all(net.res_bus.vm_pu > 1.0)
    # checks if the cheaper gen is rather used then the more expensive one to cover the load
    assert net.res_gen.at[1, "p_mw"] > net.res_gen.at[0, "p_mw"]
    # check if they use their maximum voltage
    assert allclose(net.res_gen.loc[:, "vm_pu"], 1.02)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
