import copy

import numpy as np
import pytest

from pandapower import create_line_dc_from_parameters
from pandapower.create import (
    create_buses, create_bus, create_line_from_parameters, create_load, create_ext_grid, create_bus_dc, create_b2b_vsc,
    create_line_dc, create_source_dc, create_load_dc
)
from pandapower.network import pandapowerNet
from pandapower.run import runpp
from pandapower.test.consistency_checks import runpp_with_consistency_checks
import pandapower.control as control


def test_hvdc_interconnect_with_dmr():
    """
    Test to check the dmr (the portion in the middle) of an hvdc interconnect.
             +-------+        +-------+
             |       |--------|       |
    Ext -----| BiVsc |        | BiVsc |--- Load
             |       |-+    +-|       |
             +-------+ |    | +-------+
                       +----+
             +-------+ |    | +-------+
             |       |-+    +-|       |
    Ext -----| BiVsc |        | BiVsc |--- Load
             |       |--------|       |
             +-------+        +-------+

    The solution is to add a dmr control, which will calculate the current in the dmr line.
    """
    net = pandapowerNet(name="test_hvdc_interconnect_with_dmr")

    create_buses(net, 8, 380)#, geodata=[(0, 0), (100, 0), (200, 0), (300, 0)])

    create_ext_grid(net, bus=0, vm_pu=1.0)
    create_ext_grid(net, bus=1, vm_pu=1.0)

    create_line_from_parameters(net, 0, 2, 1, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 3, 1, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 4, 6, 1, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 5, 7, 1, 0.0487, 0.13823, 160, 0.664)

    create_load(net, bus=6, p_mw=100.)
    create_load(net, bus=7, p_mw=150.)

    # DC part
    create_bus_dc(net, 380., 'A', geodata=(100,  10))  # 0
    create_bus_dc(net, 380., 'B', geodata=(100,   0))  # 1
    create_bus_dc(net, 380., 'C', geodata=(100, -10))  # 2

    create_bus_dc(net, 380., 'D', geodata=(200,  10))  # 3
    create_bus_dc(net, 380., 'E', geodata=(200,   0))  # 4
    create_bus_dc(net, 380., 'F', geodata=(200, -10))  # 5

    dcp = create_line_dc_from_parameters(net, 0, 3, length_km=100, r_ohm_per_km=0.0212, max_i_ka=0.963)
    dcm = create_line_dc_from_parameters(net, 2, 5, length_km=100, r_ohm_per_km=0.0212, max_i_ka=0.963)
    # DMR Line
    dmr = create_line_dc_from_parameters(net, 1, 4, length_km=100, r_ohm_per_km=0.0212, max_i_ka=0.963, in_service=False)

    # Left side
    create_b2b_vsc(net, 2, 0, 1, 0.2, 10, 0.3,
                   control_mode_ac='vm_pu', control_value_ac=1, control_mode_dc="vm_pu", control_value_dc=1.)
    create_b2b_vsc(net, 3, 1, 2, 0.2, 10, 0.3,
                   control_mode_ac='vm_pu', control_value_ac=1, control_mode_dc="vm_pu", control_value_dc=1.)

    # Right side
    create_b2b_vsc(net, 4, 3, 4, 0.2, 10, 0.3,
                   control_mode_ac='slack', control_value_ac=1, control_mode_dc="p_mw", control_value_dc=1.5)
    create_b2b_vsc(net, 5, 4, 5, 0.2, 10, 0.3,
                   control_mode_ac='slack', control_value_ac=1, control_mode_dc="p_mw", control_value_dc=0.5)

    # dmr current calculator
    control.DmrControl(net, dmr_line=dmr, dc_minus_line=dcm, dc_plus_line=dcp)

    runpp(net, run_control=True)

    # check results, -0.133 was calculated using powerfactory
    assert np.isclose(net.res_line_dc.loc[dmr, 'i_ka'], 0.133, atol=0.001)


def test_source_dc():
    net = pandapowerNet(name="test_source_dc")
    create_bus(net, 380)
    create_bus(net, 380)
    create_ext_grid(net, bus=0, vm_pu=1.0)
    create_load(net, bus=1, p_mw=15.)
    create_line_from_parameters(net, 0, 1, 1, 0.0487, 0.13823, 160, 0.664)

    create_bus_dc(net, 380., 'A', geodata=(100, 10))  # 0
    create_bus_dc(net, 380., 'B', geodata=(200, 10))  # 1
    create_line_dc_from_parameters(net, 0, 1, length_km=100, r_ohm_per_km=0.0212, max_i_ka=0.963)

    create_source_dc(net, bus_dc=0, vm_pu=.5)
    create_load_dc(net, bus_dc=1, p_dc_mw=10)

    runpp(net)
    pass


@pytest.mark.xfail
def test_b2b_vsc_shorted():
    """
    Test to test a simple bipolar vsc setup, without a metallic return line:
       +-------+        +-------+
       |       |--------|       |
    ---| BiVsc |        | BiVsc |--- Load
       |       |---+----|       |
       +-------+   |    +-------+
                  ---

    For reasons I do not understand, this test fails on the github server, but runs locally.
    """
    net = pandapowerNet(name="test_b2b_vsc_shorted")

    # AC part
    ext_bus = 10
    create_buses(net, 4, 380, geodata=[(0, 0), (100, 0), (200, 0), (300, 0)], index=[ext_bus, 1, 2, 3])
    create_line_from_parameters(net, ext_bus, 1, 1, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 2, 3, 1, 0.0487, 0.13823, 160, 0.664)

    create_load(net, 2, 100, q_mvar=0)
    create_ext_grid(net, ext_bus)

    # DC part
    create_bus_dc(net, 380, 'A', geodata=(100, 10))  # 0
    create_bus_dc(net, 380, 'B', geodata=(200, 10))  # 1

    create_bus_dc(net, 380, 'C', geodata=(100, -10)) # 2
    create_bus_dc(net, 380, 'D', geodata=(200, -10)) # 3

    # Earthing
    #create_bus_dc(net, 380, 'Earth', geodata=(200, -20)) # 4
    # create_bus_dc(net, -380, 'Earth', geodata=(200, -20)) # 5
    #create_line_dc_from_parameters(net, 1, 4, length_km=10, r_ohm_per_km=0.0212, max_i_ka=0.963)
    # create_line_dc_from_parameters(net, 0, 5, length_km=10, r_ohm_per_km=0.0212, max_i_ka=0.963)
    # create_source_dc(net, bus_dc=4, vm_pu=0.1, in_service=True)
    #create_load_dc(net, bus_dc=4, p_dc_mw=10.)

    # DC Lines
    create_line_dc_from_parameters(net, 0, 1, length_km=100, r_ohm_per_km=0.0212, max_i_ka=0.963)
    create_line_dc_from_parameters(net, 2, 3, length_km=100, r_ohm_per_km=0.0212, max_i_ka=0.963)

    # b2b VSC, first one regulates the voltages
    create_b2b_vsc(net, 1, 0, 2, 0.006, 1., 0.1,
                   control_mode_ac='vm_pu', control_value_ac=1., control_mode_dc="vm_pu", control_value_dc=1.)

    # second one draws constant power and works as a slack on the ac side
    create_b2b_vsc(net, 2, 1, 3, 0.006, 1., 0.1,
                   control_mode_ac='slack', control_value_ac=1., control_mode_dc="p_mw", control_value_dc=0.)

    runpp_with_consistency_checks(net)


def test_b2b_vsc_with_long_lines():
    """
    Test to test a simple bipolar vsc setup:
               +-------+        +-------+
               |       |--------|       |
    ext_grid---| BiVsc |        | BiVsc |--- Load
               |       |--------|       |
               +-------+        +-------+
    """
    net = pandapowerNet(name="test_b2b_vsc_with_long_lines")

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
       |       |--------|       |
       +-------+        +-------+
    """
    net = pandapowerNet(name="test_grounded_b2b_vsc")

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
                   control_mode_ac='slack', control_value_ac=1, control_mode_dc="p_mw", control_value_dc=10.)

    runpp_with_consistency_checks(net)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
