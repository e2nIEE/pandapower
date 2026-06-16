# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pytest

from pandapower.create import (
    create_bus,
    create_buses,
    create_ext_grid,
    create_impedance,
    create_impedances,
    create_tcsc,
    create_series_reactor_as_impedance,
)
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network


def test_create_impedance():
    """Test creating a single impedance element."""
    net = pandapowerNet(name="test_create_impedance")

    # Create buses
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)

    # Test basic creation with required parameters only
    idx = create_impedance(net, from_bus=b1, to_bus=b2, rft_pu=0.1, xft_pu=0.1, sn_mva=100)

    assert idx == 0
    assert len(net.impedance) == 1
    assert net.impedance.at[idx, "from_bus"] == b1
    assert net.impedance.at[idx, "to_bus"] == b2
    assert np.isclose(net.impedance.at[idx, "rft_pu"], 0.1)
    assert np.isclose(net.impedance.at[idx, "xft_pu"], 0.1)
    assert np.isclose(net.impedance.at[idx, "rtf_pu"], 0.1)  # defaults to rft_pu
    assert np.isclose(net.impedance.at[idx, "xtf_pu"], 0.1)  # defaults to xft_pu
    assert np.isclose(net.impedance.at[idx, "gf_pu"], 0.0)  # default value
    assert np.isclose(net.impedance.at[idx, "bf_pu"], 0.0)  # default value
    assert np.isclose(net.impedance.at[idx, "gt_pu"], 0.0)  # defaults to gf_pu
    assert np.isclose(net.impedance.at[idx, "bt_pu"], 0.0)  # defaults to bf_pu
    assert net.impedance.at[idx, "sn_mva"] == 100
    assert net.impedance.at[idx, "in_service"]

    # Test creation with all optional parameters
    b3 = create_bus(net, 110)
    b4 = create_bus(net, 110)

    idx2 = create_impedance(
        net,
        from_bus=b3,
        to_bus=b4,
        rft_pu=0.2,
        xft_pu=0.3,
        sn_mva=50,
        rtf_pu=0.15,
        xtf_pu=0.25,
        name="test_impedance",
        in_service=False,
        rft0_pu=0.01,
        xft0_pu=0.02,
        rtf0_pu=0.01,
        xtf0_pu=0.02,
        gf_pu=0.001,
        bf_pu=0.002,
        gt_pu=0.003,
        bt_pu=0.004,
        gf0_pu=0.005,
        bf0_pu=0.006,
        gt0_pu=0.007,
        bt0_pu=0.008,
    )

    assert idx2 == 1
    assert net.impedance.at[idx2, "name"] == "test_impedance"
    assert not net.impedance.at[idx2, "in_service"]
    assert np.isclose(net.impedance.at[idx2, "rft_pu"], 0.2)
    assert np.isclose(net.impedance.at[idx2, "xft_pu"], 0.3)
    assert np.isclose(net.impedance.at[idx2, "rtf_pu"], 0.15)
    assert np.isclose(net.impedance.at[idx2, "xtf_pu"], 0.25)
    assert np.isclose(net.impedance.at[idx2, "rft0_pu"], 0.01)
    assert np.isclose(net.impedance.at[idx2, "xft0_pu"], 0.02)
    assert np.isclose(net.impedance.at[idx2, "gf_pu"], 0.001)
    assert np.isclose(net.impedance.at[idx2, "bf_pu"], 0.002)
    assert np.isclose(net.impedance.at[idx2, "gt_pu"], 0.003)
    assert np.isclose(net.impedance.at[idx2, "bt_pu"], 0.004)
    assert np.isclose(net.impedance.at[idx2, "gf0_pu"], 0.005)
    assert np.isclose(net.impedance.at[idx2, "bf0_pu"], 0.006)
    assert np.isclose(net.impedance.at[idx2, "gt0_pu"], 0.007)
    assert np.isclose(net.impedance.at[idx2, "bt0_pu"], 0.008)

    validate_network(net)


def test_create_impedances():
    """Test creating multiple impedance elements at once."""
    net = pandapowerNet(name="test_create_impedances")

    # Create buses
    buses = create_buses(net, 4, 110)

    # Test basic creation with scalar values
    idx = create_impedances(
        net,
        from_buses=[buses[0], buses[1]],
        to_buses=[buses[1], buses[2]],
        rft_pu=0.1,
        xft_pu=0.1,
        sn_mva=100,
    )

    assert len(idx) == 2
    assert len(net.impedance) == 2

    # First impedance
    assert net.impedance.at[idx[0], "from_bus"] == buses[0]
    assert net.impedance.at[idx[0], "to_bus"] == buses[1]
    assert np.isclose(net.impedance.at[idx[0], "rft_pu"], 0.1)

    # Second impedance
    assert net.impedance.at[idx[1], "from_bus"] == buses[1]
    assert net.impedance.at[idx[1], "to_bus"] == buses[2]
    assert np.isclose(net.impedance.at[idx[1], "rft_pu"], 0.1)

    # Test with list/array values
    idx2 = create_impedances(
        net,
        from_buses=[buses[0], buses[2]],
        to_buses=[buses[2], buses[3]],
        rft_pu=[0.2, 0.3],
        xft_pu=[0.2, 0.3],
        sn_mva=[50, 75],
        rtf_pu=[0.15, 0.25],
        xtf_pu=[0.15, 0.25],
        name=["imp1", "imp2"],
        in_service=[True, False],
    )

    assert len(idx2) == 2
    assert net.impedance.at[idx2[0], "name"] == "imp1"
    assert net.impedance.at[idx2[0], "in_service"]
    assert np.isclose(net.impedance.at[idx2[0], "rft_pu"], 0.2)
    assert net.impedance.at[idx2[1], "name"] == "imp2"
    assert not net.impedance.at[idx2[1], "in_service"]
    assert np.isclose(net.impedance.at[idx2[1], "rft_pu"], 0.3)

    # Test with zero-sequence parameters
    idx3 = create_impedances(
        net,
        from_buses=[buses[0]],
        to_buses=[buses[3]],
        rft_pu=0.1,
        xft_pu=0.1,
        sn_mva=100,
        rft0_pu=0.01,
        xft0_pu=0.02,
    )

    assert np.isclose(net.impedance.at[idx3[0], "rft0_pu"], 0.01)
    assert np.isclose(net.impedance.at[idx3[0], "xft0_pu"], 0.02)
    assert np.isclose(net.impedance.at[idx3[0], "rtf0_pu"], 0.01)  # defaults to rft0_pu
    assert np.isclose(net.impedance.at[idx3[0], "xtf0_pu"], 0.02)  # defaults to xft0_pu

    validate_network(net)


def test_create_tcsc():
    """Test creating a TCSC (Thyristor Controlled Series Compensator) element."""
    net = pandapowerNet(name="test_create_tcsc")

    # Create buses
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)
    create_ext_grid(net, b1)

    # Test basic creation with required parameters
    idx = create_tcsc(
        net,
        from_bus=b1,
        to_bus=b2,
        x_l_ohm=1.0,
        x_cvar_ohm=-10.0,
        set_p_to_mw=50.0,
        thyristor_firing_angle_degree=140.0,
    )

    assert idx == 0
    assert len(net.tcsc) == 1
    assert net.tcsc.at[idx, "from_bus"] == b1
    assert net.tcsc.at[idx, "to_bus"] == b2
    assert np.isclose(net.tcsc.at[idx, "x_l_ohm"], 1.0)
    assert np.isclose(net.tcsc.at[idx, "x_cvar_ohm"], -10.0)
    assert np.isclose(net.tcsc.at[idx, "set_p_to_mw"], 50.0)
    assert np.isclose(net.tcsc.at[idx, "thyristor_firing_angle_degree"], 140.0)
    assert net.tcsc.at[idx, "in_service"]
    assert net.tcsc.at[idx, "controllable"]

    # Test creation with all optional parameters
    b3 = create_bus(net, 110)
    b4 = create_bus(net, 110)

    idx2 = create_tcsc(
        net,
        from_bus=b3,
        to_bus=b4,
        x_l_ohm=2.0,
        x_cvar_ohm=-20.0,
        set_p_to_mw=100.0,
        thyristor_firing_angle_degree=150.0,
        name="test_tcsc",
        controllable=False,
        in_service=False,
        min_angle_degree=90.0,
        max_angle_degree=180.0,
    )

    assert idx2 == 1
    assert net.tcsc.at[idx2, "name"] == "test_tcsc"
    assert not net.tcsc.at[idx2, "controllable"]
    assert not net.tcsc.at[idx2, "in_service"]
    assert np.isclose(net.tcsc.at[idx2, "min_angle_degree"], 90.0)
    assert np.isclose(net.tcsc.at[idx2, "max_angle_degree"], 180.0)


def test_create_series_reactor_as_impedance():
    """Test creating a series reactor as per-unit impedance."""
    net = pandapowerNet(name="test_create_series_reactor_as_impedance")

    # Create buses with same voltage level
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)

    # Test basic creation with required parameters
    idx = create_series_reactor_as_impedance(
        net,
        from_bus=b1,
        to_bus=b2,
        r_ohm=10.0,
        x_ohm=20.0,
        sn_mva=100,
    )

    assert idx == 0
    assert len(net.impedance) == 1
    assert net.impedance.at[idx, "from_bus"] == b1
    assert net.impedance.at[idx, "to_bus"] == b2
    assert net.impedance.at[idx, "sn_mva"] == 100
    assert net.impedance.at[idx, "in_service"]

    # Verify conversion from Ohm to per unit
    # base_z = vn_kv^2 / sn_mva = 110^2 / 100 = 121 ohm
    # rft_pu = r_ohm / base_z = 10 / 121 ≈ 0.08264
    # xft_pu = x_ohm / base_z = 20 / 121 ≈ 0.16529
    base_z = 110**2 / 100
    expected_rft_pu = 10.0 / base_z
    expected_xft_pu = 20.0 / base_z

    assert np.isclose(net.impedance.at[idx, "rft_pu"], expected_rft_pu)
    assert np.isclose(net.impedance.at[idx, "xft_pu"], expected_xft_pu)

    # Test with name and in_service parameters
    b3 = create_bus(net, 20)
    b4 = create_bus(net, 20)

    idx2 = create_series_reactor_as_impedance(
        net,
        from_bus=b3,
        to_bus=b4,
        r_ohm=1.0,
        x_ohm=2.0,
        sn_mva=10,
        name="test_series_reactor",
        in_service=False,
    )

    assert net.impedance.at[idx2, "name"] == "test_series_reactor"
    assert not net.impedance.at[idx2, "in_service"]

    # Verify conversion for 20 kV base
    base_z_20 = 20**2 / 10
    expected_rft_pu_20 = 1.0 / base_z_20
    expected_xft_pu_20 = 2.0 / base_z_20

    assert np.isclose(net.impedance.at[idx2, "rft_pu"], expected_rft_pu_20)
    assert np.isclose(net.impedance.at[idx2, "xft_pu"], expected_xft_pu_20)

    # Test with zero-sequence parameters
    b5 = create_bus(net, 110)
    b6 = create_bus(net, 110)

    idx3 = create_series_reactor_as_impedance(
        net,
        from_bus=b5,
        to_bus=b6,
        r_ohm=5.0,
        x_ohm=10.0,
        sn_mva=50,
        r0_ohm=15.0,
        x0_ohm=30.0,
    )

    base_z_110 = 110**2 / 50
    expected_rft0_pu = 15.0 / base_z_110
    expected_xft0_pu = 30.0 / base_z_110

    assert np.isclose(net.impedance.at[idx3, "rft0_pu"], expected_rft0_pu)
    assert np.isclose(net.impedance.at[idx3, "xft0_pu"], expected_xft0_pu)

    # Test error case with different voltage levels
    b7 = create_bus(net, 110)
    b8 = create_bus(net, 20)

    with pytest.raises(UserWarning, match="Unable to infer rated voltage"):
        create_series_reactor_as_impedance(
            net,
            from_bus=b7,
            to_bus=b8,
            r_ohm=1.0,
            x_ohm=2.0,
            sn_mva=10,
        )

    validate_network(net)