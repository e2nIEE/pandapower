import copy

import numpy as np
import pytest

from pandapower import create_line_dc_from_parameters
from pandapower.create import create_buses, create_bus, create_empty_network, create_line_from_parameters, \
    create_load, create_ext_grid, create_bus_dc, create_b2b_vsc, create_line_dc

from pandapower.run import runpp
from pandapower.test.consistency_checks import runpp_with_consistency_checks
from pandapower.test.loadflow.test_facts import copy_with_impedance, facts_case_study_grid, compare_ssc_impedance_gen


def test_b2b_vsc():
    """
    Test to test a simple bipolar vsc setup, without a metallic return line:
       +-------+        +-------+
       |       |--------|       |
    ---| BiVsc |        | BiVsc |--- Load
       |       |--------|       |
       +-------+        +-------+

    """
    net = create_empty_network()

    # AC part
    #create_buses(net, 4, 110, geodata=[(0, 0), (100, 0), (200, 0), (300, 0)])
    create_buses(net, 4, 380, geodata=[(0, 0), (100, 0), (200, 0), (300, 0)])
    create_line_from_parameters(net, 0, 1, 1, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 2, 3, 1, 0.0487, 0.13823, 160, 0.664)

    create_ext_grid(net, 1)
    create_load(net, 2, 100, q_mvar=0)

    # DC part
    create_bus_dc(net, 380, 'A', geodata=(100, 10))  # 0
    create_bus_dc(net, 380, 'B', geodata=(200, 10))  # 1

    create_bus_dc(net, 380, 'C', geodata=(100, -10)) # 2
    create_bus_dc(net, 380, 'D', geodata=(200, -10)) # 3

    # DC Lines
    create_line_dc_from_parameters(net, 0, 1, length_km=100, r_ohm_per_km=0.0212, max_i_ka=0.963)
    create_line_dc_from_parameters(net, 2, 3, length_km=100, r_ohm_per_km=0.0212, max_i_ka=0.963)

    # b2b VSC, first one regulates the voltages
    create_b2b_vsc(net, 1, 0, 2, 0.006, 1, 0.1,
                   control_mode_ac='vm_pu', control_value_ac=1, control_mode_dc="vm_pu", control_value_dc=1.)

    # second one draws constant power and works as a slack on the ac side
    create_b2b_vsc(net, 2, 1, 3, 0.006, 1, 0.1,
                   control_mode_ac='slack', control_value_ac=1, control_mode_dc="p_mw", control_value_dc=0.)

    runpp_with_consistency_checks(net)


def test_b2b_vsc_with_long_lines():
    """
    Test to test a simple bipolar vsc setup:
       +-------+        +-------+
       |       |--------|       |
    ---| BiVsc |        | BiVsc |--- Load
       |       |--------|       |
       +-------+        +-------+

    """
    net = create_empty_network()

    # AC part
    create_buses(net, 4, 110, geodata=[(0, 0), (100, 0), (200, 0), (300, 0)])
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 2, 3, 30, 0.0487, 0.13823, 160, 0.664)

    create_ext_grid(net, 0)
    create_load(net, 3, p_mw=10, q_mvar=5)

    # DC part
    create_bus_dc(net, 110, 'A', geodata=(100, 10))  # 0
    create_bus_dc(net, 110, 'B', geodata=(200, 10))  # 1

    create_bus_dc(net, 110, 'C', geodata=(100, -10)) # 2
    create_bus_dc(net, 110, 'D', geodata=(200, -10)) # 3

    # DC Lines
    create_line_dc(net, 0, 1, 100, std_type="2400-CU")
    create_line_dc(net, 2, 3, 100, std_type="2400-CU")

    # b2b VSC, first one regulates the voltages
    create_b2b_vsc(net, 1, 0, 2, 0.2, 10, 0.3,
                   control_mode_ac='vm_pu', control_value_ac=1, control_mode_dc="vm_pu", control_value_dc=1.02)

    # second one draws constant power and works as a slack on the ac side
    create_b2b_vsc(net, 2, 1, 3, 0.2, 10, 0.3,
                   control_mode_ac='slack', control_value_ac=1., control_mode_dc="p_mw", control_value_dc=0.)

    runpp_with_consistency_checks(net)


def test_grounded_b2b_vsc():
    """
    Test to test a simple bipolar vsc setup, without a metallic return line, but the second line grounded:
       +-------+        +-------+
       |       |--------|       |
    ---| BiVsc |        | BiVsc |--- Load
       |       |---+----|       |
       +-------+   |    +-------+
                  ---
    """
    net = create_empty_network()

    # AC part
    create_buses(net, 4, 110, geodata=[(0, 0), (100, 0), (200, 0), (300, 0)])
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 2, 3, 30, 0.0487, 0.13823, 160, 0.664)

    create_ext_grid(net, 0)
    create_load(net, 3, 10, q_mvar=5)

    # DC part
    create_bus_dc(net, 110, 'A', geodata=(100, 10))  # 0
    create_bus_dc(net, 110, 'B', geodata=(200, 10))  # 1

    create_bus_dc(net, 110, 'C', geodata=(100, -10)) # 2
    create_bus_dc(net, 110, 'D', geodata=(200, -10)) # 3

    # DC Lines
    create_line_dc(net, 0, 1, 100, std_type="2400-CU")
    create_line_dc(net, 2, 3, 100, std_type="2400-CU", in_service=False)

    # b2b VSC, first one regulates the voltages
    create_b2b_vsc(net, 1, 0, 2, 0.2, 10, 0.3,
                   control_mode_ac='vm_pu', control_value_ac=1, control_mode_dc="vm_pu", control_value_dc=1.02)

    # second one draws constant power and works as a slack on the ac side
    create_b2b_vsc(net, 2, 1, 3, 0.2, 10, 0.3,
                   control_mode_ac='slack', control_value_ac=1, control_mode_dc="p_mw", control_value_dc=0.)

    runpp_with_consistency_checks(net)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
