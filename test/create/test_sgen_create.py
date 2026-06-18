# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
import numpy as np

from pandapower.create import (
    create_bus, create_sgen, create_sgens, create_asymmetric_sgen, create_sgen_from_cosphi
)
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network


def test_create_sgen():
    net = pandapowerNet(name="test_create_sgen")

    # Basic sgen creation
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)

    # Test minimal parameters
    sgen_id = create_sgen(net, b1, p_mw=50)
    assert sgen_id == 0
    assert net.sgen.at[sgen_id, "bus"] == b1
    assert net.sgen.at[sgen_id, "p_mw"] == 50
    assert net.sgen.at[sgen_id, "q_mvar"] == 0  # default

    # Test with all parameters
    sgen_id2 = create_sgen(
        net,
        bus=b2,
        p_mw=100,
        q_mvar=20,
        sn_mva=120,
        name="test_sgen",
        scaling=0.8,
        type="wye",
        in_service=False,
        max_p_mw=150,
        min_p_mw=10,
        max_q_mvar=30,
        min_q_mvar=5,
        controllable=True,
        k=1.2,
        rx=0.5,
        id_q_capability_characteristic=1,
        reactive_capability_curve=True,
        curve_style="straightLineYValues",
        current_source=True,
        generator_type="current_source",
        max_ik_ka=2.5,
        kappa=1.5,
        lrc_pu=0.5,
        test_kwargs="dummy_string",
    )

    assert sgen_id2 == 1
    assert net.sgen.at[sgen_id2, "bus"] == b2
    assert net.sgen.at[sgen_id2, "p_mw"] == 100
    assert net.sgen.at[sgen_id2, "q_mvar"] == 20
    assert net.sgen.at[sgen_id2, "sn_mva"] == 120
    assert net.sgen.at[sgen_id2, "name"] == "test_sgen"
    assert np.isclose(net.sgen.at[sgen_id2, "scaling"], 0.8)
    assert net.sgen.at[sgen_id2, "type"] == "wye"
    assert net.sgen.at[sgen_id2, "in_service"] == False
    assert net.sgen.at[sgen_id2, "max_p_mw"] == 150
    assert net.sgen.at[sgen_id2, "min_p_mw"] == 10
    assert net.sgen.at[sgen_id2, "max_q_mvar"] == 30
    assert net.sgen.at[sgen_id2, "min_q_mvar"] == 5
    assert net.sgen.at[sgen_id2, "controllable"] == True
    assert np.isclose(net.sgen.at[sgen_id2, "k"], 1.2)
    assert np.isclose(net.sgen.at[sgen_id2, "rx"], 0.5)
    assert net.sgen.at[sgen_id2, "id_q_capability_characteristic"] == 1
    assert net.sgen.at[sgen_id2, "reactive_capability_curve"] == True
    assert net.sgen.at[sgen_id2, "curve_style"] == "straightLineYValues"
    assert net.sgen.at[sgen_id2, "current_source"] == True
    assert net.sgen.at[sgen_id2, "generator_type"] == "current_source"
    assert np.isclose(net.sgen.at[sgen_id2, "kappa"], 1.5)
    assert "max_ik_ka" not in net.sgen.columns  # max_ik_ka should only be set for generator_type "async_doubly_fed"
    assert "lrc_pu" not in net.sgen.columns  # lrc_pu should only be set for generator_type "async"
    assert net.sgen.at[sgen_id2, "test_kwargs"] == "dummy_string"

    # Test generator_type="async"
    b3 = create_bus(net, 110)
    sgen_async = create_sgen(
        net,
        bus=b3,
        p_mw=50,
        generator_type="async",
        lrc_pu=0.6,
    )
    assert net.sgen.at[sgen_async, "generator_type"] == "async"
    assert np.isclose(net.sgen.at[sgen_async, "lrc_pu"], 0.6)

    # Test generator_type="async_doubly_fed"
    b4 = create_bus(net, 110)
    sgen_dfig = create_sgen(
        net,
        bus=b4,
        p_mw=50,
        generator_type="async_doubly_fed",
        max_ik_ka=3.0,
        kappa=1.8,
    )
    assert net.sgen.at[sgen_dfig, "generator_type"] == "async_doubly_fed"
    assert np.isclose(net.sgen.at[sgen_dfig, "max_ik_ka"], 3.0)
    assert np.isclose(net.sgen.at[sgen_dfig, "kappa"], 1.8)

    # Test with custom index
    b5 = create_bus(net, 110)
    sgen_custom_idx = create_sgen(net, b5, p_mw=25, index=100)
    assert sgen_custom_idx == 100
    assert net.sgen.at[100, "bus"] == b5
    assert net.sgen.at[100, "p_mw"] == 25

    validate_network(net)


def test_create_sgens():
    net = pandapowerNet(name="test_create_sgens")
    # standard
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)
    b3 = create_bus(net, 110)
    create_sgens(
        net,
        buses=[b1, b2, b3],
        p_mw=[0, 0, 1],
        q_mvar=0.0,
        controllable=[True, False, False],
        id_q_capability_characteristic=[0, 1, 2],
        reactive_capability_curve=False,
        curve_style=["straightLineYValues", "straightLineYValues", "straightLineYValues"],
        max_p_mw=0.2,
        min_p_mw=[0, 0.1, 0],
        max_q_mvar=0.2,
        min_q_mvar=[0, 0.1, 0],
        k=1.3,
        rx=0.4,
        current_source=True,
        test_kwargs="dummy_string",
    )

    assert net.sgen.bus.at[0] == b1
    assert net.sgen.bus.at[1] == b2
    assert net.sgen.bus.at[2] == b3
    assert net.sgen.p_mw.at[0] == 0
    assert net.sgen.p_mw.at[1] == 0
    assert net.sgen.p_mw.at[2] == 1
    assert net.sgen.q_mvar.at[0] == 0
    assert net.sgen.q_mvar.at[1] == 0
    assert net.sgen.q_mvar.at[2] == 0
    assert net.sgen.controllable.dtype == bool
    assert net.sgen.controllable.at[0]
    assert not net.sgen.controllable.at[1]
    assert not net.sgen.controllable.at[2]
    assert np.allclose(net.sgen.max_p_mw, 0.2)
    assert all(net.sgen.min_p_mw.values == [0, 0.1, 0])
    assert np.allclose(net.sgen.max_q_mvar, 0.2)
    assert all(net.sgen.min_q_mvar.values == [0, 0.1, 0])
    assert np.allclose(net.sgen.k.values, 1.3)
    assert np.allclose(net.sgen.rx, 0.4)
    assert all(net.sgen.current_source)
    assert all(net.sgen.test_kwargs == "dummy_string")
    assert all(net.sgen.id_q_capability_characteristic.values == [0, 1, 2])
    assert all(net.sgen.curve_style == "straightLineYValues")
    assert all(net.sgen.reactive_capability_curve == [False, False, False])


def test_create_sgen_controllable():
    net = pandapowerNet(name="test_create_sgen_controllable")

    b1 = create_bus(net, 110)
    s1 = create_sgen(net, b1, 50)
    # controllable column should not exist
    assert 'controllable' not in net.sgen.columns
    s2 = create_sgen(net, b1, 50, controllable=True)
    # controllable should be created with default value False
    assert not net.sgen.loc[s1, 'controllable']
    assert net.sgen.loc[s2, 'controllable']


def test_create_sgens_controllable():
    net = pandapowerNet(name="test_create_sgens_controllable")

    b1 = create_bus(net, 110)
    s1 = create_sgens(net, [b1], 50)[0]
    # controllable column should not exist
    assert 'controllable' not in net.sgen.columns
    s2 = create_sgens(net, [b1], 50, controllable=True)[0]
    # controllable should be created with default value False
    assert not net.sgen.loc[s1, 'controllable']
    assert net.sgen.loc[s2, 'controllable']

    validate_network(net)


def test_create_sgens_raise_errorexcept():
    net = pandapowerNet(name="test_create_sgens_raise_errorexcept")
    # standard
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)
    b3 = create_bus(net, 110)

    with pytest.raises(
            UserWarning, match=r"Cannot attach to buses \{3, 4, 5\}, they do not exist"
    ):
        create_sgens(
            net,
            buses=[3, 4, 5],
            p_mw=[0, 0, 1],
            q_mvar=0.0,
            controllable=[True, False, False],
            max_p_mw=0.2,
            min_p_mw=[0, 0.1, 0],
            max_q_mvar=0.2,
            min_q_mvar=[0, 0.1, 0],
            k=1.3,
            rx=0.4,
            current_source=True,
        )
    sg = create_sgens(
        net,
        buses=[b1, b2, b3],
        p_mw=[0, 0, 1],
        q_mvar=0.0,
        controllable=[True, False, False],
        max_p_mw=0.2,
        min_p_mw=[0, 0.1, 0],
        max_q_mvar=0.2,
        min_q_mvar=[0, 0.1, 0],
        k=1.3,
        rx=0.4,
        current_source=True,
    )
    with pytest.raises(
            UserWarning, match=r"Sgens with indexes \[0 1 2\] already exist"
    ):
        create_sgens(
            net,
            buses=[b1, b2, b3],
            p_mw=[0, 0, 1],
            q_mvar=0.0,
            controllable=[True, False, False],
            max_p_mw=0.2,
            min_p_mw=[0, 0.1, 0],
            max_q_mvar=0.2,
            min_q_mvar=[0, 0.1, 0],
            k=1.3,
            rx=0.4,
            current_source=True,
            index=sg,
        )

    validate_network(net)


def test_create_asymmetric_sgen():
    net = pandapowerNet(name="test_create_asymmetric_sgen")

    # Basic asymmetric sgen creation
    b1 = create_bus(net, 110)

    # Test with all parameters for all phases
    asym_sgen_id = create_asymmetric_sgen(
        net,
        bus=b1,
        p_a_mw=-10,
        p_b_mw=-20,
        p_c_mw=-30,
        q_a_mvar=1,
        q_b_mvar=2,
        q_c_mvar=3,
        sn_a_mva=15,
        sn_b_mva=25,
        sn_c_mva=35,
        sn_mva=100,
        name="asymmetric_test",
        scaling=0.9,
        type="delta",
        in_service=False,
        test_kwargs="dummy_string",
    )

    assert asym_sgen_id == 0
    assert net.asymmetric_sgen.at[asym_sgen_id, "bus"] == b1
    assert net.asymmetric_sgen.at[asym_sgen_id, "p_a_mw"] == -10
    assert net.asymmetric_sgen.at[asym_sgen_id, "p_b_mw"] == -20
    assert net.asymmetric_sgen.at[asym_sgen_id, "p_c_mw"] == -30
    assert net.asymmetric_sgen.at[asym_sgen_id, "q_a_mvar"] == 1
    assert net.asymmetric_sgen.at[asym_sgen_id, "q_b_mvar"] == 2
    assert net.asymmetric_sgen.at[asym_sgen_id, "q_c_mvar"] == 3
    assert net.asymmetric_sgen.at[asym_sgen_id, "sn_a_mva"] == 15
    assert net.asymmetric_sgen.at[asym_sgen_id, "sn_b_mva"] == 25
    assert net.asymmetric_sgen.at[asym_sgen_id, "sn_c_mva"] == 35
    assert net.asymmetric_sgen.at[asym_sgen_id, "sn_mva"] == 100
    assert net.asymmetric_sgen.at[asym_sgen_id, "name"] == "asymmetric_test"
    assert np.isclose(net.asymmetric_sgen.at[asym_sgen_id, "scaling"], 0.9)
    assert net.asymmetric_sgen.at[asym_sgen_id, "type"] == "delta"
    assert net.asymmetric_sgen.at[asym_sgen_id, "in_service"] == False
    assert net.asymmetric_sgen.at[asym_sgen_id, "test_kwargs"] == "dummy_string"

    # Test with only one phase specified (others default to 0)
    b2 = create_bus(net, 110)
    asym_sgen_id2 = create_asymmetric_sgen(net, bus=b2, p_b_mw=-12)

    assert net.asymmetric_sgen.at[asym_sgen_id2, "p_a_mw"] == 0  # default
    assert net.asymmetric_sgen.at[asym_sgen_id2, "p_b_mw"] == -12
    assert net.asymmetric_sgen.at[asym_sgen_id2, "p_c_mw"] == 0  # default
    assert net.asymmetric_sgen.at[asym_sgen_id2, "q_a_mvar"] == 0  # default
    assert net.asymmetric_sgen.at[asym_sgen_id2, "q_b_mvar"] == 0  # default
    assert net.asymmetric_sgen.at[asym_sgen_id2, "q_c_mvar"] == 0  # default

    # Test default values
    b3 = create_bus(net, 110)
    asym_sgen_id3 = create_asymmetric_sgen(net, bus=b3)

    assert net.asymmetric_sgen.at[asym_sgen_id3, "bus"] == b3
    assert net.asymmetric_sgen.at[asym_sgen_id3, "p_a_mw"] == 0
    assert net.asymmetric_sgen.at[asym_sgen_id3, "p_b_mw"] == 0
    assert net.asymmetric_sgen.at[asym_sgen_id3, "p_c_mw"] == 0
    assert net.asymmetric_sgen.at[asym_sgen_id3, "q_a_mvar"] == 0
    assert net.asymmetric_sgen.at[asym_sgen_id3, "q_b_mvar"] == 0
    assert net.asymmetric_sgen.at[asym_sgen_id3, "q_c_mvar"] == 0
    assert net.asymmetric_sgen.at[asym_sgen_id3, "in_service"] == True  # default
    assert np.isclose(net.asymmetric_sgen.at[asym_sgen_id3, "scaling"], 1.0)  # default

    # Test with custom index
    b4 = create_bus(net, 110)
    asym_sgen_id4 = create_asymmetric_sgen(net, bus=b4, p_a_mw=-5, index=50)
    assert asym_sgen_id4 == 50

    validate_network(net)


def test_create_sgen_from_cosphi():
    net = pandapowerNet(name="test_create_sgen_from_cosphi")
    b1 = create_bus(net, 110)

    # Test overexcited mode (Q injection, positive q_mvar)
    sgen_over = create_sgen_from_cosphi(
        net,
        bus=b1,
        sn_mva=10,
        cos_phi=0.95,
        mode="overexcited",
    )

    # For generation: p = s * cosphi = 10 * 0.95 = 9.5
    # q = sqrt(s^2 - p^2) = sqrt(100 - 90.25) = sqrt(9.75) ≈ 3.122
    assert net.sgen.at[sgen_over, "p_mw"] == pytest.approx(9.5, abs=1e-6)
    assert net.sgen.at[sgen_over, "q_mvar"] == pytest.approx(3.122, abs=1e-3)
    assert net.sgen.at[sgen_over, "sn_mva"] == 10

    # Test underexcited mode (Q absorption)
    b2 = create_bus(net, 110)
    sgen_under = create_sgen_from_cosphi(
        net,
        bus=b2,
        sn_mva=5,
        cos_phi=0.9,
        mode="underexcited",
    )

    # For generation with underexcited: q is negative (absorption)
    p_expected = 5 * 0.9  # 4.5
    assert net.sgen.at[sgen_under, "p_mw"] == pytest.approx(p_expected, abs=1e-6)
    assert net.sgen.at[sgen_under, "q_mvar"] < 0  # underexcited means Q absorption
    assert net.sgen.at[sgen_under, "sn_mva"] == 5

    # Test with additional parameters passed through
    b3 = create_bus(net, 110)
    sgen_with_kwargs = create_sgen_from_cosphi(
        net,
        bus=b3,
        sn_mva=8,
        cos_phi=0.85,
        mode="overexcited",
        name="cosphi_sgen",
        in_service=False,
        scaling=0.5,
        test_kwargs="dummy_string",
    )

    assert net.sgen.at[sgen_with_kwargs, "name"] == "cosphi_sgen"
    assert net.sgen.at[sgen_with_kwargs, "in_service"] == False
    assert np.isclose(net.sgen.at[sgen_with_kwargs, "scaling"], 0.5)
    assert net.sgen.at[sgen_with_kwargs, "test_kwargs"] == "dummy_string"

    # Verify the power calculation: p = s * cosphi, q = s * sin(arccos(cosphi))
    # For cos_phi = 0.85, sin_phi = sqrt(1 - 0.85^2) = sqrt(1 - 0.7225) = sqrt(0.2775) ≈ 0.5268
    # q = 8 * 0.5268 ≈ 4.214
    assert net.sgen.at[sgen_with_kwargs, "p_mw"] == pytest.approx(8 * 0.85, abs=1e-6)
    assert net.sgen.at[sgen_with_kwargs, "q_mvar"] == pytest.approx(4.214, abs=1e-3)

    validate_network(net)
