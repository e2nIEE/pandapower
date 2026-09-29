import copy

import numpy as np
import pytest

from pandapower.create import (
    create_buses, create_bus, create_empty_network, create_line_from_parameters, create_load, create_ext_grid,
    create_bus_dc, create_vsc_stacked, create_line_dc, create_source_dc, create_load_dc, create_line_dc_from_parameters,
    create_vsc_bipolar, create_vsc
)

from pandapower.run import runpp
from pandapower.test.consistency_checks import runpp_with_consistency_checks
import pandapower.control as control


def _bipolar_dmr_net(load_plus=100., load_minus=150., dmr_in_service=True, ground=True):
    """
    Bipolar HVDC interconnect with dedicated metallic return (DMR). Every station consists of two VSC, the positive
    pole between the positive bus and the neutral bus, the negative pole between the neutral bus and the negative bus:

                  +pole  +-----+  A (+1 pu)      D  +-----+  +pole
          Ext ---------- | VSC |-------------------| VSC |---------- Load +
                         +-----+  |             |  +-----+
                                  B (0 pu) --DMR-- E
                         +-----+  |             |  +-----+
          Ext ---------- | VSC |-------------------| VSC |---------- Load -
                  -pole  +-----+  C (-1 pu)      F  +-----+  -pole

    The neutral bus B is grounded. The DMR line carries the unbalance current of the poles.
    """
    net = create_empty_network()

    create_buses(net, 8, 380)

    create_ext_grid(net, bus=0, vm_pu=1.0)
    create_ext_grid(net, bus=1, vm_pu=1.0)

    create_line_from_parameters(net, 0, 2, 1, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 3, 1, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 4, 6, 1, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 5, 7, 1, 0.0487, 0.13823, 160, 0.664)

    create_load(net, bus=6, p_mw=load_plus)
    create_load(net, bus=7, p_mw=load_minus)

    # DC part
    create_bus_dc(net, 380., 'A', geodata=(100,  10))  # 0
    create_bus_dc(net, 380., 'B', geodata=(100,   0))  # 1
    create_bus_dc(net, 380., 'C', geodata=(100, -10))  # 2

    create_bus_dc(net, 380., 'D', geodata=(200,  10))  # 3
    create_bus_dc(net, 380., 'E', geodata=(200,   0))  # 4
    create_bus_dc(net, 380., 'F', geodata=(200, -10))  # 5

    create_line_dc_from_parameters(net, 0, 3, length_km=100, r_ohm_per_km=0.0212, max_i_ka=0.963, name="dc_plus")
    create_line_dc_from_parameters(net, 2, 5, length_km=100, r_ohm_per_km=0.0212, max_i_ka=0.963, name="dc_minus")
    create_line_dc_from_parameters(net, 1, 4, length_km=100, r_ohm_per_km=0.0212, max_i_ka=0.963, name="dmr",
                                   in_service=dmr_in_service)

    # Left side: controls the pole-to-neutral voltages
    create_vsc_bipolar(net, 2, 0, 1, 0.2, 10, 0.3, name="left +",
                       control_mode_ac='vm_pu', control_value_ac=1, control_mode_dc="vm_pu", control_value_dc=1.)
    create_vsc_bipolar(net, 3, 1, 2, 0.2, 10, 0.3, name="left -",
                       control_mode_ac='vm_pu', control_value_ac=1, control_mode_dc="vm_pu", control_value_dc=1.)

    # Right side: slack for the AC grids of the loads
    create_vsc_bipolar(net, 4, 3, 4, 0.2, 10, 0.3, name="right +",
                       control_mode_ac='slack', control_value_ac=1, control_mode_dc="p_mw", control_value_dc=1.5)
    create_vsc_bipolar(net, 5, 4, 5, 0.2, 10, 0.3, name="right -",
                       control_mode_ac='slack', control_value_ac=1, control_mode_dc="p_mw", control_value_dc=0.5)

    if ground:
        create_source_dc(net, bus_dc=1, vm_pu=0.)
    return net


def test_hvdc_interconnect_with_dmr():
    """
    Unbalanced bipolar system: the DMR current is calculated by the power flow, no DmrControl is needed.
    """
    net = _bipolar_dmr_net(load_plus=100., load_minus=150.)
    runpp_with_consistency_checks(net)

    dcp, dcm, dmr = net.line_dc.index[net.line_dc.name.isin(["dc_plus", "dc_minus", "dmr"])]
    res = net.res_line_dc

    # check results, 0.133 was calculated using powerfactory
    assert np.isclose(res.at[dmr, 'i_ka'], 0.133, atol=0.001)
    # the DMR carries the unbalance of the poles
    assert np.isclose(res.at[dmr, 'i_ka'], abs(res.at[dcm, 'i_ka'] - res.at[dcp, 'i_ka']), rtol=0, atol=1e-9)
    # poles and neutral
    assert np.allclose(net.res_bus_dc.vm_pu.values[[0, 1, 2]], [1., 0., -1.], rtol=0, atol=1e-9)
    assert net.res_bus_dc.at[3, "vm_pu"] > 0.99
    assert net.res_bus_dc.at[5, "vm_pu"] < -0.99
    assert abs(net.res_bus_dc.at[4, "vm_pu"]) > 1e-4  # voltage drop over the DMR
    # the converter DC currents equal the currents of the pole lines
    assert np.allclose(np.abs(net.res_vsc_bipolar.i_dc_ka.values),
                       res.loc[[dcp, dcm, dcp, dcm], "i_ka"].values, rtol=0, atol=1e-9)
    assert np.allclose(net.res_load.p_mw.values, [100., 150.])
    # auxiliary elements are removed after the power flow
    assert len(net.vsc) == 0 and len(net.res_vsc) == 0
    assert "bus_dc_minus" not in net.vsc.columns


def test_hvdc_interconnect_with_dmr_balanced():
    net = _bipolar_dmr_net(load_plus=100., load_minus=100.)
    runpp_with_consistency_checks(net)
    dmr = net.line_dc.index[net.line_dc.name == "dmr"][0]
    assert np.isclose(net.res_line_dc.at[dmr, 'i_ka'], 0., rtol=0, atol=1e-9)
    assert np.allclose(net.res_bus_dc.vm_pu.values[[1, 4]], 0., rtol=0, atol=1e-9)
    # symmetrical poles
    assert np.allclose(net.res_bus_dc.vm_pu.values[[0, 3]], -net.res_bus_dc.vm_pu.values[[2, 5]], rtol=0, atol=1e-9)


def test_hvdc_interconnect_with_dmr_monopolar_operation():
    """
    The positive poles are out of service, the whole current of the negative pole returns through the DMR.
    """
    net = _bipolar_dmr_net(load_plus=0., load_minus=150.)
    net.vsc_bipolar.loc[net.vsc_bipolar.name.isin(["left +", "right +"]), "in_service"] = False
    net.load.loc[0, "in_service"] = False
    runpp_with_consistency_checks(net)
    dcm, dmr = net.line_dc.index[net.line_dc.name.isin(["dc_minus", "dmr"])]
    assert net.res_line_dc.at[dcm, 'i_ka'] > 0.3
    assert np.isclose(net.res_line_dc.at[dmr, 'i_ka'], net.res_line_dc.at[dcm, 'i_ka'], rtol=0, atol=1e-9)
    assert np.isclose(net.res_load.at[1, "p_mw"], 150.)
    out_of_service = net.vsc_bipolar.name == "left +"
    assert (net.res_vsc_bipolar.loc[out_of_service, "p_dc_mw"] == 0).all()
    assert net.res_vsc_bipolar.loc[out_of_service, "vm_dc_pu_m"].isna().all()


def test_hvdc_interconnect_without_ground():
    """
    A DC system which consists only of bipolar VSC needs one grounded bus, otherwise the voltages are undefined.
    """
    net = _bipolar_dmr_net(ground=False)
    with pytest.raises(UserWarning, match="no voltage reference"):
        runpp(net)
    # auxiliary elements are removed also if the power flow fails
    assert len(net.vsc) == 0
    assert "bus_dc_minus" not in net.vsc.columns


def test_vsc_bipolar_with_monopolar_vsc():
    """
    Bipolar VSC and monopolar VSC in one net, results are written to the correct tables. Second power flow is
    initialized with the results.
    """
    net = _bipolar_dmr_net()
    # an additional monopolar HVDC link
    create_buses(net, 2, 380)
    create_line_from_parameters(net, 0, 8, 1, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 4, 9, 1, 0.0487, 0.13823, 160, 0.664)
    b1 = create_bus_dc(net, 380., 'G')
    b2 = create_bus_dc(net, 380., 'H')
    create_line_dc_from_parameters(net, b1, b2, length_km=100, r_ohm_per_km=0.0212, max_i_ka=0.963)
    create_vsc(net, 8, b1, 0.2, 10, 0.3, control_mode_ac="vm_pu", control_value_ac=1., control_mode_dc="vm_pu",
               control_value_dc=1.)
    create_vsc(net, 9, b2, 0.2, 10, 0.3, control_mode_ac="q_mvar", control_value_ac=0., control_mode_dc="p_mw",
               control_value_dc=20.)
    vsc_before = net.vsc.copy()

    runpp_with_consistency_checks(net)
    res_bipolar = net.res_vsc_bipolar.copy()
    assert len(net.res_vsc) == 2 and len(net.res_vsc_bipolar) == 4
    assert np.isclose(net.res_vsc.at[1, "p_dc_mw"], 20.)
    assert net.vsc.equals(vsc_before)

    runpp_with_consistency_checks(net, init="results")
    assert np.allclose(net.res_vsc_bipolar.values, res_bipolar.values, rtol=0, atol=1e-6, equal_nan=True)
    assert net.vsc.equals(vsc_before)


def test_hvdc_interconnect_with_dmr_control_legacy():
    """
    Deprecated modelling with vsc_stacked and DmrControl, kept for backwards compatibility.
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
    net = create_empty_network()

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
    create_vsc_stacked(net, 2, 0, 1, 0.2, 10, 0.3,
                   control_mode_ac='vm_pu', control_value_ac=1, control_mode_dc="vm_pu", control_value_dc=1.)
    create_vsc_stacked(net, 3, 1, 2, 0.2, 10, 0.3,
                   control_mode_ac='vm_pu', control_value_ac=1, control_mode_dc="vm_pu", control_value_dc=1.)

    # Right side
    create_vsc_stacked(net, 4, 3, 4, 0.2, 10, 0.3,
                   control_mode_ac='slack', control_value_ac=1, control_mode_dc="p_mw", control_value_dc=1.5)
    create_vsc_stacked(net, 5, 4, 5, 0.2, 10, 0.3,
                   control_mode_ac='slack', control_value_ac=1, control_mode_dc="p_mw", control_value_dc=0.5)

    # dmr current calculator
    with pytest.warns(DeprecationWarning):
        control.DmrControl(net, dmr_line=dmr, dc_minus_line=dcm, dc_plus_line=dcp)

    runpp(net, run_control=True)

    # check results, -0.133 was calculated using powerfactory
    assert np.isclose(net.res_line_dc.loc[dmr, 'i_ka'], 0.133, atol=0.001)


def test_source_dc():
    net = create_empty_network()
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
def test_vsc_stacked_shorted():
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
    net = create_empty_network()

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
    create_vsc_stacked(net, 1, 0, 2, 0.006, 1., 0.1,
                   control_mode_ac='vm_pu', control_value_ac=1., control_mode_dc="vm_pu", control_value_dc=1.)

    # second one draws constant power and works as a slack on the ac side
    create_vsc_stacked(net, 2, 1, 3, 0.006, 1., 0.1,
                   control_mode_ac='slack', control_value_ac=1., control_mode_dc="p_mw", control_value_dc=0.)

    runpp_with_consistency_checks(net)


def test_vsc_stacked_with_long_lines():
    """
    Test to test a simple bipolar vsc setup:
               +-------+        +-------+
               |       |--------|       |
    ext_grid---| BiVsc |        | BiVsc |--- Load
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
    create_vsc_stacked(net, 1, 0, 2, 0.2, 10, 0.3,
                   control_mode_ac='vm_pu', control_value_ac=1, control_mode_dc="vm_pu", control_value_dc=1.02)

    # second one draws constant power and works as a slack on the ac side
    create_vsc_stacked(net, 2, 1, 3, 0.2, 10, 0.3,
                   control_mode_ac='slack', control_value_ac=1., control_mode_dc="p_mw", control_value_dc=0.)

    runpp_with_consistency_checks(net)


def test_grounded_vsc_stacked():
    """
    Test to test a simple bipolar vsc setup, without a metallic return line, but the second line grounded:
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
    create_vsc_stacked(net, 1, 0, 2, 0.2, 10, 0.3,
                   control_mode_ac='vm_pu', control_value_ac=1, control_mode_dc="vm_pu", control_value_dc=1.02)

    # second one draws constant power and works as a slack on the ac side
    create_vsc_stacked(net, 2, 1, 3, 0.2, 10, 0.3,
                   control_mode_ac='slack', control_value_ac=1, control_mode_dc="p_mw", control_value_dc=10.)

    runpp_with_consistency_checks(net)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
