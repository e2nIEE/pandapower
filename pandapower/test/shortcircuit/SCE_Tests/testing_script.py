import pandapower as pp
import pandapower.shortcircuit as sc
from pandapower.shortcircuit.calc_sc import calc_sc
import numpy as np
from pandapower.create import create_empty_network, create_bus, create_ext_grid, create_line, create_sgen, \
    create_transformer_from_parameters, create_transformers_from_parameters, create_line_from_parameters, create_buses, \
    create_lines_from_parameters, create_switch, create_load, create_shunt, create_ward, create_xward
import pandas as pd
import numpy as np
from pandapower.shortcircuit.calc_sc import calc_sc
from pandapower.file_io import from_json
import pytest
import re
import copy
import os
from pandapower import pp_dir

"""def three_bus_example():
    net = create_empty_network(sn_mva=56)
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)
    b3 = create_bus(net, 110)

    create_ext_grid(net, b1, s_sc_max_mva=100., s_sc_min_mva=80., rx_min=0.4, rx_max=0.4)
    create_line(net, b1, b2, std_type="305-AL1/39-ST1A 110.0", length_km=20.)
    create_line(net, b2, b3, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV", length_km=15.)
    net.line["endtemp_degree"] = 80

    create_sgen(net, b2, sn_mva=2, p_mw=0, k=1.2)

    net.ext_grid['x0x_min'] = 0.1
    net.ext_grid['r0x0_min'] = 0.1
    net.ext_grid['x0x_max'] = 0.1
    net.ext_grid['r0x0_max'] = 0.1

    net.line['r0_ohm_per_km'] = 0.1
    net.line['x0_ohm_per_km'] = 0.1
    net.line['c0_nf_per_km'] = 0.1
    net.line["endtemp_degree"] = 80
    return net

#
# eg--0---l0---1---l1---2
#              |
#              g
#
# With generator
net = three_bus_example()
calc_sc(net, case="max", fault='LG', branch_results=True, return_all_currents=True)
i_bus_with_sgen = net.res_bus_sc.copy()
i_line_with_gen = net.res_line_sc.copy()

# Without generator
net = three_bus_example()
net.sgen.in_service = False
calc_sc(net, case="max", fault='LG')
i_bus_without_sgen = net.res_bus_sc.copy()

# Isolate sgen contrib
i_bus_only_sgen = i_bus_with_sgen - i_bus_without_sgen

assert np.isclose(i_line_with_gen.ikss_ka.loc[(0, 0)], i_bus_only_sgen.ikss_ka.at[0], atol=1e-4)
assert np.isclose(i_line_with_gen.ikss_ka.loc[(0, 1)], i_bus_without_sgen.ikss_ka.at[1], atol=1e-4)
assert np.isclose(i_line_with_gen.ikss_ka.loc[(0, 2)], i_bus_without_sgen.ikss_ka.at[2] - (
        i_bus_only_sgen.ikss_ka.at[1] - i_bus_only_sgen.ikss_ka.at[2]), atol=1e-4)
assert np.isclose(i_line_with_gen.ikss_ka.loc[(1, 0)], 0., atol=1e-4)
assert np.isclose(i_line_with_gen.ikss_ka.loc[(1, 1)], 0., atol=1e-4)
assert np.isclose(i_line_with_gen.ikss_ka.loc[(1, 2)], i_bus_with_sgen.ikss_ka.at[2], atol=1e-4)"""

net = from_json(os.path.join(pp_dir, "test", "shortcircuit", "SCE_Tests", "4_bus_radial_grid.json"))
calc_sc(net, fault="LG", case="max", branch_results=True, ip=False, r_fault_ohm=0,
        x_fault_ohm=0)
print(net.res_bus_sc)
print(net.res_line_sc)