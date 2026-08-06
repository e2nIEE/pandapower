# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
import numpy as np

from pandapower.create import (
    create_bus,
    create_bus_dc,
    create_shunt,
    create_shunts,
    create_shunt_as_capacitor,
    create_svc,
    create_ssc,
    create_vsc,
    create_vsc_bipolar,
    create_vsc_stacked
)
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network


def test_create_shunt():
    net = pandapowerNet(name="test_create_shunt")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)

    # Test basic shunt creation with required parameters
    idx = create_shunt(net, bus=b1, q_mvar=-20.0)
    assert net.shunt.at[idx, "bus"] == b1
    assert np.isclose(net.shunt.at[idx, "q_mvar"], -20.0)
    assert np.isclose(net.shunt.at[idx, "p_mw"], 0.0)  # default value

    # Test shunt with all optional parameters
    idx2 = create_shunt(
        net, bus=b2, q_mvar=-10.0, p_mw=0.5, vn_kv=110.0,
        step=2, max_step=5, name="test_shunt",
        step_dependency_table=True, id_characteristic_table=1,
        in_service=False, test_kwargs="dummy_string"
    )
    assert net.shunt.at[idx2, "bus"] == b2
    assert np.isclose(net.shunt.at[idx2, "q_mvar"], -10.0)
    assert np.isclose(net.shunt.at[idx2, "p_mw"], 0.5)
    assert np.isclose(net.shunt.at[idx2, "vn_kv"], 110.0)
    assert net.shunt.at[idx2, "step"] == 2
    assert net.shunt.at[idx2, "max_step"] == 5
    assert net.shunt.at[idx2, "name"] == "test_shunt"
    assert net.shunt.at[idx2, "step_dependency_table"] == True
    assert net.shunt.at[idx2, "id_characteristic_table"] == 1
    assert net.shunt.at[idx2, "in_service"] == False
    assert net.shunt.at[idx2, "test_kwargs"] == "dummy_string"

    # Test that vn_kv defaults to bus voltage when not provided
    b3 = create_bus(net, 20.0)
    idx3 = create_shunt(net, bus=b3, q_mvar=-5.0)
    assert np.isclose(net.shunt.at[idx3, "vn_kv"], 20.0)

    # Test custom index
    idx4 = create_shunt(net, bus=b1, q_mvar=-15.0, index=10)
    assert idx4 == 10
    assert np.isclose(net.shunt.at[10, "q_mvar"], -15.0)

    validate_network(net)


def test_create_shunts():
    net = pandapowerNet(name="test_create_shunts")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)
    b3 = create_bus(net, 20)

    # Test creating multiple shunts with scalar values
    indices = create_shunts(net, buses=[b1, b2], q_mvar=-20.0, p_mw=1.0)
    assert len(indices) == 2
    assert net.shunt.at[indices[0], "bus"] == b1
    assert np.isclose(net.shunt.at[indices[0], "q_mvar"], -20.0)
    assert net.shunt.at[indices[1], "bus"] == b2
    assert np.isclose(net.shunt.at[indices[1], "q_mvar"], -20.0)

    # Test creating multiple shunts with list values
    indices2 = create_shunts(
        net, buses=[b1, b2, b3],
        q_mvar=[-10.0, -15.0, -5.0],
        p_mw=[0.5, 0.8, 0.2],
        step=[1, 2, 3],
        max_step=[5, 5, 3],
        name=["shunt1", "shunt2", "shunt3"],
        step_dependency_table=[True, False, True],
        in_service=[True, False, True],
        test_kwargs=["dummy1", "dummy2", "dummy3"],
    )
    assert len(indices2) == 3
    assert np.isclose(net.shunt.at[indices2[0], "q_mvar"], -10.0)
    assert np.isclose(net.shunt.at[indices2[0], "p_mw"], 0.5)
    assert net.shunt.at[indices2[0], "step"] == 1
    assert net.shunt.at[indices2[0], "max_step"] == 5
    assert net.shunt.at[indices2[0], "name"] == "shunt1"
    assert net.shunt.at[indices2[0], "step_dependency_table"] == True
    assert net.shunt.at[indices2[0], "in_service"] == True
    assert net.shunt.at[indices2[0], "test_kwargs"] == "dummy1"

    assert np.isclose(net.shunt.at[indices2[1], "q_mvar"], -15.0)
    assert net.shunt.at[indices2[1], "in_service"] == False
    assert net.shunt.at[indices2[1], "test_kwargs"] == "dummy2"

    assert np.isclose(net.shunt.at[indices2[2], "vn_kv"], 20.0)  # defaults to bus voltage
    assert net.shunt.at[indices2[2], "test_kwargs"] == "dummy3"

    validate_network(net)


def test_create_shunt_as_capacitor():
    net = pandapowerNet(name="test_create_shunt_as_capacitor")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)

    # Test basic capacitor bank creation
    # q_mvar should be negative for capacitor, p_mw = abs(q_mvar * loss_factor)
    idx = create_shunt_as_capacitor(net, bus=b1, q_mvar=10.0, loss_factor=0.01)
    assert net.shunt.at[idx, "bus"] == b1
    assert np.isclose(net.shunt.at[idx, "q_mvar"], -10.0)  # always negative
    assert net.shunt.at[idx, "p_mw"] == abs(10 * 0.01)

    # Test with negative q_mvar input (should use absolute value)
    idx2 = create_shunt_as_capacitor(net, bus=b1, q_mvar=-20.0, loss_factor=0.02)
    assert np.isclose(net.shunt.at[idx2, "q_mvar"], -20.0)
    assert net.shunt.at[idx2, "p_mw"] == abs(-20 * 0.02)

    # Test with additional kwargs passed to create_shunt
    idx3 = create_shunt_as_capacitor(
        net, bus=b2, q_mvar=5.0, loss_factor=0.005,
        name="capacitor_bank", vn_kv=20.0, step=3, max_step=10,
        in_service=False, test_kwargs="dummy_string"
    )
    assert net.shunt.at[idx3, "bus"] == b2
    assert np.isclose(net.shunt.at[idx3, "q_mvar"], -5.0)
    assert np.isclose(net.shunt.at[idx3, "p_mw"], 0.025)  # abs(5 * 0.005)
    assert net.shunt.at[idx3, "name"] == "capacitor_bank"
    assert np.isclose(net.shunt.at[idx3, "vn_kv"], 20.0)
    assert net.shunt.at[idx3, "step"] == 3
    assert net.shunt.at[idx3, "max_step"] == 10
    assert net.shunt.at[idx3, "in_service"] == False
    assert net.shunt.at[idx3, "test_kwargs"] == "dummy_string"

    # Test with different loss factors
    idx4 = create_shunt_as_capacitor(net, bus=b1, q_mvar=100.0, loss_factor=0.0)
    assert np.isclose(net.shunt.at[idx4, "q_mvar"], -100.0)
    assert np.isclose(net.shunt.at[idx4, "p_mw"], 0.0)  # no losses

    validate_network(net)


def test_create_shunt_nonexistent_bus():
    net = pandapowerNet(name="test_create_shunt_nonexistent_bus")
    create_bus(net, 110)

    # Test that creating shunt on non-existent bus raises error
    with pytest.raises(UserWarning, match=r"Cannot attach to bus 5, 5 does not exist"):
        create_shunt(net, bus=5, q_mvar=-10.0)


def test_create_shunts_nonexistent_buses():
    net = pandapowerNet(name="test_create_shunts_nonexistent_buses")
    b1 = create_bus(net, 110)

    # Test that creating shunts on non-existent buses raises error
    with pytest.raises(UserWarning, match=r"Cannot attach to buses {2, 3}, they do not exist"):
        create_shunts(net, buses=[b1, 2, 3], q_mvar=-10.0)


def test_create_shunt_index_conflict():
    net = pandapowerNet(name="test_create_shunt_index_conflict")
    b1 = create_bus(net, 110)

    # Create first shunt
    idx1 = create_shunt(net, bus=b1, q_mvar=-10.0)
    assert idx1 == 0

    # Test that creating shunt with same index raises error
    with pytest.raises(UserWarning, match=r"A shunt with the id \d already exists"):
        create_shunt(net, bus=b1, q_mvar=-20.0, index=0)

    # Test that custom index works when available
    idx2 = create_shunt(net, bus=b1, q_mvar=-15.0, index=5)
    assert idx2 == 5


def test_create_svc():
    net = pandapowerNet(name="test_create_svc")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)

    # Test basic SVC creation with required parameters
    idx = create_svc(
        net, bus=b1, x_l_ohm=10.0, x_cvar_ohm=-20.0,
        set_vm_pu=1.0, thyristor_firing_angle_degree=120.0
    )
    assert net.svc.at[idx, "bus"] == b1
    assert np.isclose(net.svc.at[idx, "x_l_ohm"], 10.0)
    assert np.isclose(net.svc.at[idx, "x_cvar_ohm"], -20.0)
    assert np.isclose(net.svc.at[idx, "set_vm_pu"], 1.0)
    assert np.isclose(net.svc.at[idx, "thyristor_firing_angle_degree"], 120.0)

    # Test SVC with all optional parameters
    idx2 = create_svc(
        net, bus=b2, x_l_ohm=5.0, x_cvar_ohm=-15.0,
        set_vm_pu=1.02, thyristor_firing_angle_degree=135.0,
        name="test_svc", controllable=False, in_service=False,
        min_angle_degree=100.0, max_angle_degree=170.0,
        test_kwargs="dummy_string"
    )
    assert net.svc.at[idx2, "bus"] == b2
    assert np.isclose(net.svc.at[idx2, "x_l_ohm"], 5.0)
    assert np.isclose(net.svc.at[idx2, "x_cvar_ohm"], -15.0)
    assert np.isclose(net.svc.at[idx2, "set_vm_pu"], 1.02)
    assert np.isclose(net.svc.at[idx2, "thyristor_firing_angle_degree"], 135.0)
    assert net.svc.at[idx2, "name"] == "test_svc"
    assert net.svc.at[idx2, "controllable"] == False
    assert net.svc.at[idx2, "in_service"] == False
    assert np.isclose(net.svc.at[idx2, "min_angle_degree"], 100.0)
    assert np.isclose(net.svc.at[idx2, "max_angle_degree"], 170.0)
    assert net.svc.at[idx2, "test_kwargs"] == "dummy_string"

    # Test default values
    b3 = create_bus(net, 20)
    idx3 = create_svc(
        net, bus=b3, x_l_ohm=2.0, x_cvar_ohm=-5.0,
        set_vm_pu=1.0, thyristor_firing_angle_degree=90.0
    )
    assert net.svc.at[idx3, "controllable"] == True  # default
    assert net.svc.at[idx3, "in_service"] == True   # default
    assert np.isclose(net.svc.at[idx3, "min_angle_degree"], 90.0)   # default
    assert np.isclose(net.svc.at[idx3, "max_angle_degree"], 180.0)  # default

    # Test custom index
    idx4 = create_svc(
        net, bus=b1, x_l_ohm=3.0, x_cvar_ohm=-8.0,
        set_vm_pu=1.0, thyristor_firing_angle_degree=100.0, index=10
    )
    assert idx4 == 10

    validate_network(net)


def test_create_ssc():
    net = pandapowerNet(name="test_create_ssc")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)

    # Test basic SSC creation with required parameters
    idx = create_ssc(net, bus=b1, r_ohm=1.0, x_ohm=-10.0)
    assert np.isclose(net.ssc.at[idx, "bus"], b1)
    assert np.isclose(net.ssc.at[idx, "r_ohm"], 1.0)
    assert np.isclose(net.ssc.at[idx, "x_ohm"], -10.0)

    # Test SSC with all optional parameters
    idx2 = create_ssc(
        net, bus=b2, r_ohm=0.5, x_ohm=-5.0,
        set_vm_pu=1.02, vm_internal_pu=1.05, va_internal_degree=5.0,
        name="test_ssc", controllable=False, in_service=False,
        test_kwargs="dummy_string"
    )
    assert net.ssc.at[idx2, "bus"] == b2
    assert np.isclose(net.ssc.at[idx2, "r_ohm"], 0.5)
    assert np.isclose(net.ssc.at[idx2, "x_ohm"], -5.0)
    assert np.isclose(net.ssc.at[idx2, "set_vm_pu"], 1.02)
    assert np.isclose(net.ssc.at[idx2, "vm_internal_pu"], 1.05)
    assert np.isclose(net.ssc.at[idx2, "va_internal_degree"], 5.0)
    assert net.ssc.at[idx2, "name"] == "test_ssc"
    assert net.ssc.at[idx2, "controllable"] == False
    assert net.ssc.at[idx2, "in_service"] == False
    assert net.ssc.at[idx2, "test_kwargs"] == "dummy_string"

    # Test default values
    b3 = create_bus(net, 110)
    idx3 = create_ssc(net, bus=b3, r_ohm=0.1, x_ohm=-2.0)
    assert np.isclose(net.ssc.at[idx3, "set_vm_pu"], 1.0)
    assert np.isclose(net.ssc.at[idx3, "vm_internal_pu"], 1.0)
    assert np.isclose(net.ssc.at[idx3, "va_internal_degree"], 0.0)
    assert net.ssc.at[idx3, "controllable"] == True
    assert net.ssc.at[idx3, "in_service"] == True

    # Test custom index
    idx4 = create_ssc(net, bus=b1, r_ohm=0.2, x_ohm=-3.0, index=15)
    assert idx4 == 15

    validate_network(net)


def test_create_svc_nonexistent_bus():
    net = pandapowerNet(name="test_create_svc_nonexistent_bus")
    create_bus(net, 110)

    # Test that creating SVC on non-existent bus raises error
    with pytest.raises(UserWarning, match=r"Cannot attach to bus 5, 5 does not exist"):
        create_svc(
            net, bus=5, x_l_ohm=10.0, x_cvar_ohm=-20.0,
            set_vm_pu=1.0, thyristor_firing_angle_degree=120.0
        )


def test_create_ssc_nonexistent_bus():
    net = pandapowerNet(name="test_create_ssc_nonexistent_bus")
    create_bus(net, 110)

    # Test that creating SSC on non-existent bus raises error
    with pytest.raises(UserWarning, match=r"Cannot attach to bus 5, 5 does not exist"):
        create_ssc(net, bus=5, r_ohm=1.0, x_ohm=10.0)


def test_create_svc_index_conflict():
    net = pandapowerNet(name="test_create_svc_index_conflict")
    b1 = create_bus(net, 110)

    # Create first SVC
    idx1 = create_svc(
        net, bus=b1, x_l_ohm=10.0, x_cvar_ohm=-20.0,
        set_vm_pu=1.0, thyristor_firing_angle_degree=120.0
    )
    assert idx1 == 0

    # Test that creating SVC with same index raises error
    with pytest.raises(UserWarning, match=r"A svc with the id \d already exists"):
        create_svc(
            net, bus=b1, x_l_ohm=5.0, x_cvar_ohm=-10.0,
            set_vm_pu=1.0, thyristor_firing_angle_degree=120.0, index=0
        )

    # Test that custom index works when available
    idx2 = create_svc(
        net, bus=b1, x_l_ohm=3.0, x_cvar_ohm=-8.0,
        set_vm_pu=1.0, thyristor_firing_angle_degree=100.0, index=5
    )
    assert idx2 == 5


def test_create_ssc_index_conflict():
    net = pandapowerNet(name="test_create_ssc_index_conflict")
    b1 = create_bus(net, 110)

    # Create first SSC
    idx1 = create_ssc(net, bus=b1, r_ohm=1.0, x_ohm=10.0)
    assert idx1 == 0

    # Test that creating SSC with same index raises error
    with pytest.raises(UserWarning, match=r"A ssc with the id \d already exists"):
        create_ssc(net, bus=b1, r_ohm=0.5, x_ohm=5.0, index=0)

    # Test that custom index works when available
    idx2 = create_ssc(net, bus=b1, r_ohm=0.2, x_ohm=3.0, index=5)
    assert idx2 == 5


def test_create_vsc_stacked():
    net = pandapowerNet(name="test_create_vsc_stacked")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)
    b_dc_plus = create_bus_dc(net, 100)
    b_dc_minus = create_bus_dc(net, 0.1)

    # Test basic VSC stacked creation with required parameters
    idx = create_vsc_stacked(
        net, bus=b1, bus_dc_plus=b_dc_plus, bus_dc_minus=b_dc_minus,
        r_ohm=1.0, x_ohm=10.0, r_dc_ohm=0.5
    )
    assert net.vsc_stacked.at[idx, "bus"] == b1
    assert net.vsc_stacked.at[idx, "bus_dc_plus"] == b_dc_plus
    assert net.vsc_stacked.at[idx, "bus_dc_minus"] == b_dc_minus
    assert np.isclose(net.vsc_stacked.at[idx, "r_ohm"], 1.0)
    assert np.isclose(net.vsc_stacked.at[idx, "x_ohm"], 10.0)
    assert np.isclose(net.vsc_stacked.at[idx, "r_dc_ohm"], 0.5)

    # Test VSC stacked with all optional parameters
    idx2 = create_vsc_stacked(
        net, bus=b2, bus_dc_plus=b_dc_plus, bus_dc_minus=b_dc_minus,
        r_ohm=0.5, x_ohm=5.0, r_dc_ohm=0.2,
        pl_dc_mw=0.1, control_mode_ac="vm_pu", control_value_ac=1.02,
        control_mode_dc="p_mw", control_value_dc=50.0,
        name="test_vsc_stacked", controllable=False, in_service=False,
        test_kwargs="dummy_string"
    )
    assert net.vsc_stacked.at[idx2, "bus"] == b2
    assert np.isclose(net.vsc_stacked.at[idx2, "r_ohm"], 0.5)
    assert np.isclose(net.vsc_stacked.at[idx2, "x_ohm"], 5.0)
    assert np.isclose(net.vsc_stacked.at[idx2, "r_dc_ohm"], 0.2)
    assert np.isclose(net.vsc_stacked.at[idx2, "pl_dc_mw"], 0.1)
    assert net.vsc_stacked.at[idx2, "control_mode_ac"] == "vm_pu"
    assert np.isclose(net.vsc_stacked.at[idx2, "control_value_ac"], 1.02)
    assert net.vsc_stacked.at[idx2, "control_mode_dc"] == "p_mw"
    assert np.isclose(net.vsc_stacked.at[idx2, "control_value_dc"], 50.0)
    assert net.vsc_stacked.at[idx2, "name"] == "test_vsc_stacked"
    assert net.vsc_stacked.at[idx2, "controllable"] == False
    assert net.vsc_stacked.at[idx2, "in_service"] == False
    assert net.vsc_stacked.at[idx2, "test_kwargs"] == "dummy_string"

    # Test default values
    b3 = create_bus(net, 20)
    idx3 = create_vsc_stacked(
        net, bus=b3, bus_dc_plus=b_dc_plus, bus_dc_minus=b_dc_minus,
        r_ohm=0.1, x_ohm=2.0, r_dc_ohm=0.1
    )
    assert np.isclose(net.vsc_stacked.at[idx3, "pl_dc_mw"], 0.0)  # default
    assert net.vsc_stacked.at[idx3, "control_mode_ac"] == "p_mw"  # default
    assert np.isclose(net.vsc_stacked.at[idx3, "control_value_ac"], 1.0)  # default
    assert net.vsc_stacked.at[idx3, "control_mode_dc"] == "p_mw"  # default
    assert np.isclose(net.vsc_stacked.at[idx3, "control_value_dc"], 0.0)  # default
    assert net.vsc_stacked.at[idx3, "controllable"] == True  # default
    assert net.vsc_stacked.at[idx3, "in_service"] == True   # default

    # Test custom index
    idx4 = create_vsc_stacked(
        net, bus=b1, bus_dc_plus=b_dc_plus, bus_dc_minus=b_dc_minus,
        r_ohm=0.2, x_ohm=3.0, r_dc_ohm=0.1, index=10
    )
    assert idx4 == 10

    validate_network(net)


def test_create_vsc_bipolar():
    net = pandapowerNet(name="test_create_vsc_bipolar")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    b_dc_plus = create_bus_dc(net, 100)
    b_dc_minus = create_bus_dc(net, 0.1)

    # Test basic VSC bipolar creation with required parameters
    idx = create_vsc_bipolar(
        net, bus=b1, bus_dc_plus=b_dc_plus, bus_dc_minus=b_dc_minus,
        r_ohm=1.0, x_ohm=10.0, r_dc_ohm=0.5
    )
    assert net.vsc_bipolar.at[idx, "bus"] == b1
    assert net.vsc_bipolar.at[idx, "bus_dc_plus"] == b_dc_plus
    assert net.vsc_bipolar.at[idx, "bus_dc_minus"] == b_dc_minus
    assert np.isclose(net.vsc_bipolar.at[idx, "r_ohm"], 1.0)
    assert np.isclose(net.vsc_bipolar.at[idx, "x_ohm"], 10.0)
    assert np.isclose(net.vsc_bipolar.at[idx, "r_dc_ohm"], 0.5)

    # Test VSC bipolar with all optional parameters
    idx2 = create_vsc_bipolar(
        net, bus=b2, bus_dc_plus=b_dc_plus, bus_dc_minus=b_dc_minus,
        r_ohm=0.5, x_ohm=5.0, r_dc_ohm=0.2,
        pl_dc_mw=0.1, control_mode="Vac_phi", control_value_1=1.02, control_value_2=5.0,
        name="test_vsc_bipolar", controllable=False, in_service=False,
        test_kwargs="dummy_string"
    )
    assert net.vsc_bipolar.at[idx2, "bus"] == b2
    assert np.isclose(net.vsc_bipolar.at[idx2, "r_ohm"], 0.5)
    assert np.isclose(net.vsc_bipolar.at[idx2, "x_ohm"], 5.0)
    assert np.isclose(net.vsc_bipolar.at[idx2, "r_dc_ohm"], 0.2)
    assert np.isclose(net.vsc_bipolar.at[idx2, "pl_dc_mw"], 0.1)
    assert net.vsc_bipolar.at[idx2, "control_mode"] == "Vac_phi"
    assert np.isclose(net.vsc_bipolar.at[idx2, "control_value_1"], 1.02)
    assert np.isclose(net.vsc_bipolar.at[idx2, "control_value_2"], 5.0)
    assert net.vsc_bipolar.at[idx2, "name"] == "test_vsc_bipolar"
    assert net.vsc_bipolar.at[idx2, "controllable"] == False
    assert net.vsc_bipolar.at[idx2, "in_service"] == False
    assert net.vsc_bipolar.at[idx2, "test_kwargs"] == "dummy_string"

    # Test default values
    b3 = create_bus(net, 110)
    idx3 = create_vsc_bipolar(
        net, bus=b3, bus_dc_plus=b_dc_plus, bus_dc_minus=b_dc_minus,
        r_ohm=0.1, x_ohm=2.0, r_dc_ohm=0.1
    )
    assert np.isclose(net.vsc_bipolar.at[idx3, "pl_dc_mw"], 0.0)  # default
    assert net.vsc_bipolar.at[idx3, "control_mode"] == "Vac_phi"  # default
    assert np.isclose(net.vsc_bipolar.at[idx3, "control_value_1"], 1.0)  # default
    assert np.isclose(net.vsc_bipolar.at[idx3, "control_value_2"], 0.0)  # default
    assert net.vsc_bipolar.at[idx3, "controllable"] == True  # default
    assert net.vsc_bipolar.at[idx3, "in_service"] == True   # default

    # Test custom index
    idx4 = create_vsc_bipolar(
        net, bus=b1, bus_dc_plus=b_dc_plus, bus_dc_minus=b_dc_minus,
        r_ohm=0.2, x_ohm=3.0, r_dc_ohm=0.1, index=10
    )
    assert idx4 == 10

    validate_network(net)


def test_create_vsc():
    net = pandapowerNet(name="test_create_vsc")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    b_dc = create_bus_dc(net, 100)

    # Test basic VSC creation with required parameters
    idx = create_vsc(
        net, bus=b1, bus_dc=b_dc, r_ohm=1.0, x_ohm=10.0, r_dc_ohm=0.5
    )
    assert net.vsc.at[idx, "bus"] == b1
    assert net.vsc.at[idx, "bus_dc"] == b_dc
    assert np.isclose(net.vsc.at[idx, "r_ohm"], 1.0)
    assert np.isclose(net.vsc.at[idx, "x_ohm"], 10.0)
    assert np.isclose(net.vsc.at[idx, "r_dc_ohm"], 0.5)

    # Test VSC with all optional parameters
    idx2 = create_vsc(
        net, bus=b2, bus_dc=b_dc, r_ohm=0.5, x_ohm=5.0, r_dc_ohm=0.2,
        pl_dc_mw=0.1, control_mode_ac="vm_pu", control_value_ac=1.02,
        control_mode_dc="p_mw", control_value_dc=50.0,
        name="test_vsc", controllable=False, in_service=False,
        ref_bus=0, test_kwargs="dummy_string"
    )
    assert net.vsc.at[idx2, "bus"] == b2
    assert net.vsc.at[idx2, "bus_dc"] == b_dc
    assert np.isclose(net.vsc.at[idx2, "r_ohm"], 0.5)
    assert np.isclose(net.vsc.at[idx2, "x_ohm"], 5.0)
    assert np.isclose(net.vsc.at[idx2, "pl_dc_mw"], 0.1)
    assert np.isclose(net.vsc.at[idx2, "r_dc_ohm"], 0.2)
    assert net.vsc.at[idx2, "control_mode_ac"] == "vm_pu"
    assert np.isclose(net.vsc.at[idx2, "control_value_ac"], 1.02)
    assert net.vsc.at[idx2, "control_mode_dc"] == "p_mw"
    assert np.isclose(net.vsc.at[idx2, "control_value_dc"], 50.0)
    assert net.vsc.at[idx2, "name"] == "test_vsc"
    assert net.vsc.at[idx2, "controllable"] == False
    assert net.vsc.at[idx2, "in_service"] == False
    assert net.vsc.at[idx2, "ref_bus"] == 0
    assert net.vsc.at[idx2, "test_kwargs"] == "dummy_string"

    # Test default values
    b3 = create_bus(net, 110)
    idx3 = create_vsc(
        net, bus=b3, bus_dc=b_dc, r_ohm=0.1, x_ohm=2.0, r_dc_ohm=0.1
    )
    assert np.isclose(net.vsc.at[idx3, "pl_dc_mw"], 0.0)  # default
    assert net.vsc.at[idx3, "control_mode_ac"] == "vm_pu"  # default
    assert np.isclose(net.vsc.at[idx3, "control_value_ac"], 1.0)  # default
    assert net.vsc.at[idx3, "control_mode_dc"] == "p_mw"  # default
    assert np.isclose(net.vsc.at[idx3, "control_value_dc"], 0.0)  # default
    assert net.vsc.at[idx3, "controllable"] == True  # default
    assert net.vsc.at[idx3, "in_service"] == True   # default

    # Test custom index
    idx4 = create_vsc(
        net, bus=b1, bus_dc=b_dc, r_ohm=0.2, x_ohm=3.0, r_dc_ohm=0.1, index=10
    )
    assert idx4 == 10

    validate_network(net)


def test_create_vsc_stacked_nonexistent_bus():
    net = pandapowerNet(name="test_create_vsc_stacked_nonexistent_bus")
    create_bus(net, 110)
    b_dc_plus = create_bus_dc(net, 100)
    b_dc_minus = create_bus_dc(net, -100)

    # Test that creating VSC on non-existent AC bus raises error
    with pytest.raises(UserWarning, match=r"Cannot attach to bus 5, 5 does not exist"):
        create_vsc_stacked(
            net, bus=5, bus_dc_plus=b_dc_plus, bus_dc_minus=b_dc_minus,
            r_ohm=1.0, x_ohm=10.0, r_dc_ohm=0.5
        )


def test_create_vsc_stacked_nonexistent_dc_buses():
    net = pandapowerNet(name="test_create_vsc_stacked_nonexistent_dc_buses")
    b1 = create_bus(net, 110)

    # Test that creating VSC on non-existent DC buses raises error
    with pytest.raises(UserWarning, match=r"Cannot attach to bus_dc 5, 5 does not exist"):
        create_vsc_stacked(
            net, bus=b1, bus_dc_plus=5, bus_dc_minus=6,
            r_ohm=1.0, x_ohm=10.0, r_dc_ohm=0.5
        )


def test_create_vsc_bipolar_nonexistent_bus():
    net = pandapowerNet(name="test_create_vsc_bipolar_nonexistent_bus")
    create_bus(net, 110)
    b_dc_plus = create_bus_dc(net, 100)
    b_dc_minus = create_bus_dc(net, -100)

    # Test that creating VSC bipolar on non-existent AC bus raises error
    with pytest.raises(UserWarning, match=r"Cannot attach to bus 5, 5 does not exist"):
        create_vsc_bipolar(
            net, bus=5, bus_dc_plus=b_dc_plus, bus_dc_minus=b_dc_minus,
            r_ohm=1.0, x_ohm=10.0, r_dc_ohm=0.5
        )


def test_create_vsc_bipolar_nonexistent_dc_buses():
    net = pandapowerNet(name="test_create_vsc_bipolar_nonexistent_dc_buses")
    b1 = create_bus(net, 110)

    # Test that creating VSC bipolar on non-existent DC buses raises error
    with pytest.raises(UserWarning, match=r"Cannot attach to bus_dc 5, 5 does not exist"):
        create_vsc_bipolar(
            net, bus=b1, bus_dc_plus=5, bus_dc_minus=6,
            r_ohm=1.0, x_ohm=10.0, r_dc_ohm=0.5
        )


def test_create_vsc_nonexistent_bus():
    net = pandapowerNet(name="test_create_vsc_nonexistent_bus")
    b_dc = create_bus_dc(net, 100)

    # Test that creating VSC on non-existent AC bus raises error
    with pytest.raises(UserWarning, match=r"Cannot attach to bus 5, 5 does not exist"):
        create_vsc(net, bus=5, bus_dc=b_dc, r_ohm=1.0, x_ohm=10.0, r_dc_ohm=0.5)


def test_create_vsc_nonexistent_dc_bus():
    net = pandapowerNet(name="test_create_vsc_nonexistent_dc_bus")
    b1 = create_bus(net, 110)

    # Test that creating VSC on non-existent DC bus raises error
    with pytest.raises(UserWarning, match=r"Cannot attach to bus_dc 5, 5 does not exist"):
        create_vsc(net, bus=b1, bus_dc=5, r_ohm=1.0, x_ohm=10.0, r_dc_ohm=0.5)


def test_create_vsc_stacked_index_conflict():
    net = pandapowerNet(name="test_create_vsc_stacked_index_conflict")
    b1 = create_bus(net, 110)
    b_dc_plus = create_bus_dc(net, 100)
    b_dc_minus = create_bus_dc(net, -100)

    # Create first VSC stacked
    idx1 = create_vsc_stacked(
        net, bus=b1, bus_dc_plus=b_dc_plus, bus_dc_minus=b_dc_minus,
        r_ohm=1.0, x_ohm=10.0, r_dc_ohm=0.5
    )
    assert idx1 == 0

    # Test that creating with same index raises error
    with pytest.raises(UserWarning, match=r"A vsc_stacked with the id \d already exists"):
        create_vsc_stacked(
            net, bus=b1, bus_dc_plus=b_dc_plus, bus_dc_minus=b_dc_minus,
            r_ohm=0.5, x_ohm=5.0, r_dc_ohm=0.2, index=0
        )

    # Test that custom index works when available
    idx2 = create_vsc_stacked(
        net, bus=b1, bus_dc_plus=b_dc_plus, bus_dc_minus=b_dc_minus,
        r_ohm=0.2, x_ohm=3.0, r_dc_ohm=0.1, index=5
    )
    assert idx2 == 5


def test_create_vsc_bipolar_index_conflict():
    net = pandapowerNet(name="test_create_vsc_bipolar_index_conflict")
    b1 = create_bus(net, 110)
    b_dc_plus = create_bus_dc(net, 100)
    b_dc_minus = create_bus_dc(net, -100)

    # Create first VSC bipolar
    idx1 = create_vsc_bipolar(
        net, bus=b1, bus_dc_plus=b_dc_plus, bus_dc_minus=b_dc_minus,
        r_ohm=1.0, x_ohm=10.0, r_dc_ohm=0.5
    )
    assert idx1 == 0

    # Test that creating with same index raises error
    with pytest.raises(UserWarning, match=r"A vsc_bipolar with the id \d already exists"):
        create_vsc_bipolar(
            net, bus=b1, bus_dc_plus=b_dc_plus, bus_dc_minus=b_dc_minus,
            r_ohm=0.5, x_ohm=5.0, r_dc_ohm=0.2, index=0
        )

    # Test that custom index works when available
    idx2 = create_vsc_bipolar(
        net, bus=b1, bus_dc_plus=b_dc_plus, bus_dc_minus=b_dc_minus,
        r_ohm=0.2, x_ohm=3.0, r_dc_ohm=0.1, index=5
    )
    assert idx2 == 5


def test_create_vsc_index_conflict():
    net = pandapowerNet(name="test_create_vsc_index_conflict")
    b1 = create_bus(net, 110)
    b_dc = create_bus_dc(net, 100)

    # Create first VSC
    idx1 = create_vsc(
        net, bus=b1, bus_dc=b_dc, r_ohm=1.0, x_ohm=10.0, r_dc_ohm=0.5
    )
    assert idx1 == 0

    # Test that creating with same index raises error
    with pytest.raises(UserWarning, match=r"A vsc with the id \d already exists"):
        create_vsc(
            net, bus=b1, bus_dc=b_dc, r_ohm=0.5, x_ohm=5.0, r_dc_ohm=0.2, index=0
        )

    # Test that custom index works when available
    idx2 = create_vsc(
        net, bus=b1, bus_dc=b_dc, r_ohm=0.2, x_ohm=3.0, r_dc_ohm=0.1, index=5
    )
    assert idx2 == 5
