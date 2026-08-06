# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
import numpy as np
import pandas as pd

from pandapower.create import (
    create_bus, create_transformer_from_parameters, create_transformer, create_transformers,
    create_transformers_from_parameters, create_transformers3w_from_parameters, create_transformers3w, load_std_type,
    create_transformer3w, create_transformer3w_from_parameters
)
from pandapower.std_types import create_std_type
from pandapower.toolbox import dataframes_equal
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network


def test_tap_changer_type_default():
    net = pandapowerNet(name="test_tap_changer_type_default")
    create_bus(net, 110)
    create_bus(net, 20)
    data = load_std_type(net, "25 MVA 110/20 kV", "trafo")
    if "tap_changer_type" in data:
        del data["tap_changer_type"]
    create_std_type(net, data, "without_tap_shifter_info", "trafo")
    create_transformer_from_parameters(net, 0, 1, 25e3, 110, 20, 0.4, 12, 20, 0.07)
    create_transformer(net, 0, 1, "without_tap_shifter_info")
    if 'tap_changer_type' in net.trafo.columns:
        assert (net.trafo.tap_changer_type.isna()).all()

    validate_network(net)


def test_create_transformer_from_parameters():
    # Test basic transformer creation from parameters
    net = pandapowerNet(name="test_create_transformer_from_parameters0")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    t = create_transformer_from_parameters(
        net,
        hv_bus=b1,
        lv_bus=b2,
        sn_mva=40,
        vn_hv_kv=110,
        vn_lv_kv=20,
        vkr_percent=0.5,
        vk_percent=10,
        pfe_kw=30,
        i0_percent=0.1,
        name="test_trafo",
    )

    assert len(net.trafo) == 1
    assert net.trafo.at[t, "name"] == "test_trafo"
    assert net.trafo.at[t, "hv_bus"] == b1
    assert net.trafo.at[t, "lv_bus"] == b2
    assert net.trafo.at[t, "sn_mva"] == 40
    assert net.trafo.at[t, "vn_hv_kv"] == 110
    assert net.trafo.at[t, "vn_lv_kv"] == 20
    assert net.trafo.at[t, "vk_percent"] == 10
    assert np.isclose(net.trafo.at[t, "vkr_percent"], 0.5)
    assert net.trafo.at[t, "pfe_kw"] == 30
    assert np.isclose(net.trafo.at[t, "i0_percent"], 0.1)

    # Test with tap changer
    net = pandapowerNet(name="test_create_transformer_from_parameters1")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    t = create_transformer_from_parameters(
        net,
        hv_bus=b1,
        lv_bus=b2,
        sn_mva=40,
        vn_hv_kv=110,
        vn_lv_kv=20,
        vkr_percent=0.5,
        vk_percent=10,
        pfe_kw=30,
        i0_percent=0.1,
        tap_side="hv",
        tap_pos=5,
        tap_neutral=0,
        tap_max=10,
        tap_min=-10,
        tap_step_percent=1.0,
    )

    assert net.trafo.at[t, "tap_side"] == "hv"
    assert net.trafo.at[t, "tap_pos"] == 5
    assert net.trafo.at[t, "tap_neutral"] == 0
    assert net.trafo.at[t, "tap_max"] == 10
    assert net.trafo.at[t, "tap_min"] == -10
    assert np.isclose(net.trafo.at[t, "tap_step_percent"], 1.0)

    # Test with zero sequence parameters
    net = pandapowerNet(name="test_create_transformer_from_parameters2")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    t = create_transformer_from_parameters(
        net,
        hv_bus=b1,
        lv_bus=b2,
        sn_mva=40,
        vn_hv_kv=110,
        vn_lv_kv=20,
        vkr_percent=0.5,
        vk_percent=10,
        pfe_kw=30,
        i0_percent=0.1,
        vk0_percent=10,
        vkr0_percent=0.5,
        mag0_percent=100,
        mag0_rx=0.1,
        vector_group="Dyn",
    )

    assert net.trafo.at[t, "vk0_percent"] == 10
    assert np.isclose(net.trafo.at[t, "vkr0_percent"], 0.5)
    assert net.trafo.at[t, "mag0_percent"] == 100
    assert np.isclose(net.trafo.at[t, "mag0_rx"], 0.1)
    assert net.trafo.at[t, "vector_group"] == "Dyn"

    # Test with in_service=False
    net = pandapowerNet(name="test_create_transformer_from_parameters3")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    t = create_transformer_from_parameters(
        net,
        hv_bus=b1,
        lv_bus=b2,
        sn_mva=40,
        vn_hv_kv=110,
        vn_lv_kv=20,
        vkr_percent=0.5,
        vk_percent=10,
        pfe_kw=30,
        i0_percent=0.1,
        in_service=False,
    )
    assert not net.trafo.at[t, "in_service"]

    validate_network(net)

def test_create_transformers_from_parameters():
    # standard
    net = pandapowerNet(name="test_create_transformers_from_parameters0")
    b1 = create_bus(net, 15)
    b2 = create_bus(net, 0.4)
    index = create_transformers_from_parameters(
        net,
        [b1, b1],
        [b2, b2],
        vn_hv_kv=[15.0, 15.0],
        vn_lv_kv=[0.45, 0.45],
        sn_mva=[0.5, 0.7],
        vk_percent=[1.0, 1.0],
        vkr_percent=[0.3, 0.3],
        pfe_kw=0.2,
        i0_percent=0.3,
        foo=2,
    )
    with pytest.raises(UserWarning):
        create_transformers_from_parameters(
            net,
            [b1, b1],
            [b2, b2],
            vn_hv_kv=[15.0, 15.0],
            vn_lv_kv=[0.45, 0.45],
            sn_mva=[0.5, 0.7],
            vk_percent=[1.0, 1.0],
            vkr_percent=[0.3, 0.3],
            pfe_kw=0.2,
            i0_percent=0.3,
            foo=2,
            index=index
        )
    assert len(net.trafo) == 2
    assert len(net.trafo.vk_percent) == 2
    assert len(net.trafo.vkr_percent) == 2
    assert len(net.trafo.pfe_kw) == 2
    assert len(net.trafo.i0_percent) == 2
    assert len(net.trafo.df) == 2
    assert len(net.trafo.foo) == 2

    validate_network(net)

    # setting params as single value
    net = pandapowerNet(name="test_create_transformers_from_parameters1")
    b1 = create_bus(net, 15)
    b2 = create_bus(net, 0.4)
    create_transformers_from_parameters(
        net,
        hv_buses=[b1, b1],
        lv_buses=[b2, b2],
        vn_hv_kv=15.0,
        vn_lv_kv=0.45,
        sn_mva=0.5,
        vk_percent=1.0,
        vkr_percent=0.3,
        pfe_kw=0.2,
        i0_percent=0.3,
        vk0_percent=0.4,
        vkr0_percent=1.7,
        mag0_rx=0.4,
        mag0_percent=30,
        vector_group="Dyn",
        si0_hv_partial=0.1,
        max_loading_percent=80,
        test_kwargs="dummy_string",
    )
    assert len(net.trafo) == 2
    assert all(net.trafo.hv_bus == 0)
    assert all(net.trafo.lv_bus == 1)
    assert np.allclose(net.trafo.sn_mva, 0.5)
    assert np.allclose(net.trafo.vn_hv_kv, 15.0)
    assert np.allclose(net.trafo.vn_lv_kv, 0.45)
    assert np.allclose(net.trafo.vk_percent, 1.0)
    assert np.allclose(net.trafo.vkr_percent, 0.3)
    assert np.allclose(net.trafo.pfe_kw, 0.2)
    assert np.allclose(net.trafo.i0_percent, 0.3)
    assert np.allclose(net.trafo.vk0_percent, 0.4)
    assert np.allclose(net.trafo.mag0_rx, 0.4)
    assert all(net.trafo.mag0_percent == 30)
    assert all(net.trafo.vector_group.values == "Dyn")
    assert np.allclose(net.trafo.max_loading_percent, 80.0)
    assert np.allclose(net.trafo.si0_hv_partial, 0.1)
    assert all(net.trafo.test_kwargs == "dummy_string")

    validate_network(net)

    # setting params as array
    net = pandapowerNet(name="test_create_transformers_from_parameters2")
    b1 = create_bus(net, 10)
    b2 = create_bus(net, 10)
    create_transformers_from_parameters(
        net,
        hv_buses=[b1, b1],
        lv_buses=[b2, b2],
        vn_hv_kv=[15.0, 15.0],
        sn_mva=[0.6, 0.6],
        vn_lv_kv=[0.45, 0.45],
        vk_percent=[1.0, 1.0],
        vkr_percent=[0.3, 0.3],
        pfe_kw=[0.2, 0.2],
        i0_percent=[0.3, 0.3],
        vk0_percent=[0.4, 0.4],
        mag0_rx=[0.4, 0.4],
        mag0_percent=[30, 30],
        test_kwargs=["dummy_string", "dummy_string"],
    )

    assert len(net.trafo) == 2
    assert all(net.trafo.hv_bus == 0)
    assert all(net.trafo.lv_bus == 1)
    assert np.allclose(net.trafo.vn_hv_kv, 15.0)
    assert np.allclose(net.trafo.vn_lv_kv, 0.45)
    assert np.allclose(net.trafo.sn_mva, 0.6)
    assert np.allclose(net.trafo.vk_percent, 1.0)
    assert np.allclose(net.trafo.vkr_percent, 0.3)
    assert np.allclose(net.trafo.pfe_kw, 0.2)
    assert np.allclose(net.trafo.i0_percent, 0.3)
    assert np.allclose(net.trafo.vk0_percent, 0.4)
    assert np.allclose(net.trafo.mag0_rx, 0.4)
    assert all(net.trafo.mag0_percent == 30)
    assert all(net.trafo.test_kwargs == "dummy_string")

    validate_network(net)


def test_create_transformers_raise_errorexcept():
    # standard
    net = pandapowerNet(name="test_create_transformers_raise_errorexcept")
    b1 = create_bus(net, 10)
    b2 = create_bus(net, 10)
    create_transformers_from_parameters(
        net,
        [b1, b1],
        [b2, b2],
        vn_hv_kv=[15.0, 15.0],
        vn_lv_kv=[0.45, 0.45],
        sn_mva=[0.5, 0.7],
        vk_percent=[1.0, 1.0],
        vkr_percent=[0.3, 0.3],
        pfe_kw=0.2,
        i0_percent=0.3,
        foo=2,
    )

    with pytest.raises(UserWarning, match=r"Trafos with indexes \[1\] already exist."):
        create_transformers_from_parameters(
            net,
            [b1, b1],
            [b2, b2],
            vn_hv_kv=[15.0, 15.0],
            vn_lv_kv=[0.45, 0.45],
            sn_mva=[0.5, 0.7],
            vk_percent=[1.0, 1.0],
            vkr_percent=[0.3, 0.3],
            pfe_kw=0.2,
            i0_percent=0.3,
            index=[2, 1],
        )
    validate_network(net)


def test_create_transformer_raises_errorexcept1():
    net = pandapowerNet(name="test_create_transformer_raises_errorexcept1")
    b1 = create_bus(net, 10)
    b2 = create_bus(net, 10)
    create_transformers_from_parameters(
        net,
        [b1, b1],
        [b2, b2],
        vn_hv_kv=[15.0, 15.0],
        vn_lv_kv=[0.45, 0.45],
        sn_mva=[0.5, 0.7],
        vk_percent=[1.0, 1.0],
        vkr_percent=[0.3, 0.3],
        pfe_kw=0.2,
        i0_percent=0.3,
        foo=2,
    )
    with pytest.raises(
            UserWarning, match=r"Transformers trying to attach to non existing buses \{2\}"
    ):
        create_transformers_from_parameters(
            net,
            [b1, 2],
            [b2, b2],
            vn_hv_kv=[15.0, 15.0],
            vn_lv_kv=[0.45, 0.45],
            sn_mva=[0.5, 0.7],
            vk_percent=[1.0, 1.0],
            vkr_percent=[0.3, 0.3],
            pfe_kw=0.2,
            i0_percent=0.3,
            foo=2,
        )
    with pytest.raises(
            UserWarning, match=r"Transformers trying to attach to non existing buses \{3\}"
    ):
        create_transformers_from_parameters(
            net,
            [b1, b1],
            [b2, 3],
            vn_hv_kv=[15.0, 15.0],
            vn_lv_kv=[0.45, 0.45],
            sn_mva=[0.5, 0.7],
            vk_percent=[1.0, 1.0],
            vkr_percent=[0.3, 0.3],
            pfe_kw=0.2,
            i0_percent=0.3,
            foo=2,
        )

    validate_network(net)


def test_trafo_2_tap_changers():
    net = pandapowerNet(name="test_trafo_2_tap_changers")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    create_transformer(net, b1, b2, "40 MVA 110/20 kV")

    tap2_data = {"tap2_side": "hv",
                 "tap2_neutral": 0,
                 "tap2_max": 10,
                 "tap2_min": -10,
                 "tap2_step_percent": 1,
                 "tap2_step_degree": 0,
                 "tap2_changer_type": "Ratio"}

    for c in tap2_data:
        assert c not in net.trafo.columns

    std_type = load_std_type(net, "40 MVA 110/20 kV", "trafo")

    create_std_type(net, {**std_type, **tap2_data}, "test_trafo_type", "trafo")

    t = create_transformer(net, b1, b2, "test_trafo_type")

    for c in tap2_data:
        assert c in net.trafo.columns
        assert net.trafo.at[t, c] == tap2_data[c]

    validate_network(net)


def test_trafo_2_tap_changers_parameters():
    net = pandapowerNet(name="test_trafo_2_tap_changers_parameters")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)

    std_type = load_std_type(net, "40 MVA 110/20 kV", "trafo")
    tap2_data = {"tap2_side": "hv",
                 "tap2_neutral": 0,
                 "tap2_max": 10,
                 "tap2_min": -10,
                 "tap2_step_percent": 1,
                 "tap2_step_degree": 0,
                 "tap2_changer_type": "Ratio"}

    create_transformer_from_parameters(net, b1, b2, **std_type)

    for c in tap2_data:
        assert c not in net.trafo.columns

    t = create_transformer_from_parameters(net, b1, b2, **std_type, **tap2_data)

    for c in tap2_data:
        assert c in net.trafo.columns
        assert net.trafo.at[t, c] == tap2_data[c]

    validate_network(net)


def test_trafos_2_tap_changers_parameters():
    net = pandapowerNet(name="test_trafos_2_tap_changers_parameters")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)

    std_type = load_std_type(net, "40 MVA 110/20 kV", "trafo")
    tap2_data = {"tap2_side": "hv",
                 "tap2_neutral": 0,
                 "tap2_max": 10,
                 "tap2_min": -10,
                 "tap2_step_percent": 1,
                 "tap2_step_degree": 0,
                 "tap2_changer_type": "Ratio"}

    std_type_p = {k: np.array([v, v]) if not isinstance(v, str) else v for k, v in std_type.items()}

    create_transformers_from_parameters(net, [b1, b1], [b2, b2], **std_type_p)

    for c in tap2_data:
        assert c not in net.trafo.columns

    t = create_transformer_from_parameters(net, b1, b2, **std_type, **tap2_data)

    for c in tap2_data:
        assert c in net.trafo.columns
        assert net.trafo.at[t, c] == tap2_data[c]

    validate_network(net)


def test_create_transformer():
    # Test basic transformer creation from std_type
    net = pandapowerNet(name="test_create_transformer0")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    t = create_transformer(net, hv_bus=b1, lv_bus=b2, std_type="40 MVA 110/20 kV", name="test_trafo")

    assert len(net.trafo) == 1
    assert net.trafo.at[t, "name"] == "test_trafo"
    assert net.trafo.at[t, "hv_bus"] == b1
    assert net.trafo.at[t, "lv_bus"] == b2
    assert net.trafo.at[t, "std_type"] == "40 MVA 110/20 kV"

    std_type = load_std_type(net, "40 MVA 110/20 kV", "trafo")
    assert net.trafo.at[t, "sn_mva"] == std_type["sn_mva"]
    assert net.trafo.at[t, "vn_hv_kv"] == std_type["vn_hv_kv"]
    assert net.trafo.at[t, "vn_lv_kv"] == std_type["vn_lv_kv"]
    assert net.trafo.at[t, "vk_percent"] == std_type["vk_percent"]

    # Test with custom index
    net = pandapowerNet(name="test_create_transformer1")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    t = create_transformer(net, hv_bus=b1, lv_bus=b2, std_type="40 MVA 110/20 kV", index=5)
    assert t == 5
    assert 5 in net.trafo.index

    # Test with in_service=False
    net = pandapowerNet(name="test_create_transformer2")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    t = create_transformer(net, hv_bus=b1, lv_bus=b2, std_type="40 MVA 110/20 kV", in_service=False)
    assert not net.trafo.at[t, "in_service"]

    # Test with tap_pos
    net = pandapowerNet(name="test_create_transformer3")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    t = create_transformer(net, hv_bus=b1, lv_bus=b2, std_type="40 MVA 110/20 kV", tap_pos=5)
    assert net.trafo.at[t, "tap_pos"] == 5

    # Test with max_loading_percent
    net = pandapowerNet(name="test_create_transformer4")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    t = create_transformer(net, hv_bus=b1, lv_bus=b2, std_type="40 MVA 110/20 kV", max_loading_percent=80)
    assert net.trafo.at[t, "max_loading_percent"] == 80

    # Test error case - non-existent bus
    net = pandapowerNet(name="test_create_transformer5")
    b1 = create_bus(net, 110)
    create_bus(net, 20)
    with pytest.raises(UserWarning, match=r"Trafo \d tries to attach to non-existing bus\(es\) \{\d\}"):
        create_transformer(net, hv_bus=b1, lv_bus=5, std_type="40 MVA 110/20 kV")

    validate_network(net)

def test_create_transformers():
    net = pandapowerNet(name="test_create_transformers")
    b1 = create_bus(net, 10)
    b2 = create_bus(net, .4)
    b3 = create_bus(net, .4)
    create_transformers(
        net,
        hv_buses=[b1, b1],
        lv_buses=[b2, b3],
        std_type="0.4 MVA 10/0.4 kV",
        name=["trafo1", "trafo2"],
        test_kwargs="TestKW"
    )
    res_df = pd.DataFrame(
        {
            "name": pd.Series(["trafo1", "trafo2"], dtype=pd.StringDtype()),
            "std_type": pd.Series(["0.4 MVA 10/0.4 kV", "0.4 MVA 10/0.4 kV"], dtype=pd.StringDtype()),
            "hv_bus": pd.Series([0, 0], dtype=np.int64),
            "lv_bus": pd.Series([1, 2], dtype=np.int64),
            "sn_mva": pd.Series([0.4, 0.4], dtype=np.float64),
            "vn_hv_kv": pd.Series([10.0, 10.0], dtype=np.float64),
            "vn_lv_kv": pd.Series([0.4, 0.4], dtype=np.float64),
            "vk_percent": pd.Series([4.0, 4.0], dtype=np.float64),
            "vkr_percent": pd.Series([1.325, 1.325], dtype=np.float64),
            "pfe_kw": pd.Series([0.95, 0.95], dtype=np.float64),
            "i0_percent": pd.Series([0.2375, 0.2375], dtype=np.float64),
            "shift_degree": pd.Series([150.0, 150.0], dtype=np.float64),
            "tap_side": pd.Series(["hv", "hv"], dtype=pd.StringDtype()),
            "tap_neutral": pd.Series([0.0, 0.0], dtype=np.float64),
            "tap_min": pd.Series([-2.0, -2.0], dtype=np.float64),
            "tap_max": pd.Series([2.0, 2.0], dtype=np.float64),
            "tap_step_percent": pd.Series([2.5, 2.5], dtype=np.float64),
            "tap_step_degree": pd.Series([0.0, 0.0], dtype=np.float64),
            "tap_pos": pd.Series([0.0, 0.0], dtype=np.float64),
            "tap_changer_type": pd.Series(["Ratio", "Ratio"], dtype=pd.StringDtype()),
            "parallel": pd.Series([1, 1], dtype=np.int64),
            "df": pd.Series([1.0, 1.0], dtype=np.float64),
            "in_service": pd.Series([True, True], dtype=bool),
            # 'oltc': [False, False],
            "test_kwargs": ["TestKW", "TestKW"],
            "trafo_characteristic_table": [False, False],
            "vector_group": pd.Series(["Dyn5", "Dyn5"], dtype=pd.StringDtype()),
        }
    )
    for column in res_df:
        assert net.trafo[column].equals(res_df[column])
    assert dataframes_equal(net.trafo, res_df)


def test_create_transformers_for_single():
    net = pandapowerNet(name="test_create_transformers_for_single")
    b1 = create_bus(net, 10)
    b2 = create_bus(net, .4)
    create_transformers(
        net,
        hv_buses=[b1],
        lv_buses=[b2],
        std_type="0.4 MVA 10/0.4 kV",
        name="trafo1",
        test_kwargs="TestKW",
        sn_mva=.4
    )
    res_df = pd.DataFrame(
        {
            "name": pd.Series(["trafo1"], dtype=pd.StringDtype()),
            "std_type": pd.Series(["0.4 MVA 10/0.4 kV"], dtype=pd.StringDtype()),
            "hv_bus": pd.Series([0], dtype=np.int64),
            "lv_bus": pd.Series([1], dtype=np.int64),
            "sn_mva": pd.Series([0.4], dtype=np.float64),
            "vn_hv_kv": pd.Series([10.0], dtype=np.float64),
            "vn_lv_kv": pd.Series([0.4], dtype=np.float64),
            "vk_percent": pd.Series([4.0], dtype=np.float64),
            "vkr_percent": pd.Series([1.325], dtype=np.float64),
            "pfe_kw": pd.Series([0.95], dtype=np.float64),
            "i0_percent": pd.Series([0.2375], dtype=np.float64),
            "shift_degree": pd.Series([150.0], dtype=np.float64),
            "tap_side": pd.Series(["hv"], dtype=pd.StringDtype()),
            "tap_neutral": pd.Series([0.0], dtype=np.float64),
            "tap_min": pd.Series([-2.0], dtype=np.float64),
            "tap_max": pd.Series([2.0], dtype=np.float64),
            "tap_step_percent": pd.Series([2.5], dtype=np.float64),
            "tap_step_degree": pd.Series([0.0], dtype=np.float64),
            "tap_pos": pd.Series([0.0], dtype=np.float64),
            "tap_changer_type": pd.Series(["Ratio"], dtype=pd.StringDtype()),
            "parallel": pd.Series([1], dtype=np.int64),
            "df": pd.Series([1.0], dtype=np.float64),
            "in_service": pd.Series([True], dtype=bool),
            # "oltc": [False],
            "test_kwargs": ["TestKW"],
            "trafo_characteristic_table": [False],
            "vector_group": pd.Series(["Dyn5"], dtype=pd.StringDtype()),
        }
    )
    for column in res_df:
        assert net.trafo[column].equals(res_df[column])
    assert dataframes_equal(net.trafo, res_df)

    validate_network(net)


def test_create_transformer3w():
    # Test basic 3-winding transformer creation from std_type
    net = pandapowerNet(name="test_create_transformer3w0")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    b3 = create_bus(net, 10)
    t = create_transformer3w(
        net,
        hv_bus=b1,
        mv_bus=b2,
        lv_bus=b3,
        std_type="63/25/38 MVA 110/20/10 kV",
        name="test_trafo3w",
    )

    assert len(net.trafo3w) == 1
    assert net.trafo3w.at[t, "name"] == "test_trafo3w"
    assert net.trafo3w.at[t, "hv_bus"] == b1
    assert net.trafo3w.at[t, "mv_bus"] == b2
    assert net.trafo3w.at[t, "lv_bus"] == b3
    assert net.trafo3w.at[t, "std_type"] == "63/25/38 MVA 110/20/10 kV"

    std_type = load_std_type(net, "63/25/38 MVA 110/20/10 kV", "trafo3w")
    assert net.trafo3w.at[t, "sn_hv_mva"] == std_type["sn_hv_mva"]
    assert net.trafo3w.at[t, "sn_mv_mva"] == std_type["sn_mv_mva"]
    assert net.trafo3w.at[t, "sn_lv_mva"] == std_type["sn_lv_mva"]
    assert net.trafo3w.at[t, "vn_hv_kv"] == std_type["vn_hv_kv"]
    assert net.trafo3w.at[t, "vn_mv_kv"] == std_type["vn_mv_kv"]
    assert net.trafo3w.at[t, "vn_lv_kv"] == std_type["vn_lv_kv"]

    # Test with custom index
    net = pandapowerNet(name="test_create_transformer3w1")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    b3 = create_bus(net, 10)
    t = create_transformer3w(
        net,
        hv_bus=b1,
        mv_bus=b2,
        lv_bus=b3,
        std_type="63/25/38 MVA 110/20/10 kV",
        index=10,
    )
    assert t == 10
    assert 10 in net.trafo3w.index

    # Test with in_service=False
    net = pandapowerNet(name="test_create_transformer3w2")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    b3 = create_bus(net, 10)
    t = create_transformer3w(
        net,
        hv_bus=b1,
        mv_bus=b2,
        lv_bus=b3,
        std_type="63/25/38 MVA 110/20/10 kV",
        in_service=False,
    )
    assert not net.trafo3w.at[t, "in_service"]

    # Test with tap_pos
    net = pandapowerNet(name="test_create_transformer3w3")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    b3 = create_bus(net, 10)
    t = create_transformer3w(
        net,
        hv_bus=b1,
        mv_bus=b2,
        lv_bus=b3,
        std_type="63/25/38 MVA 110/20/10 kV",
        tap_pos=5,
    )
    assert net.trafo3w.at[t, "tap_pos"] == 5

    # Test with max_loading_percent
    net = pandapowerNet(name="test_create_transformer3w4")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    b3 = create_bus(net, 10)
    t = create_transformer3w(
        net,
        hv_bus=b1,
        mv_bus=b2,
        lv_bus=b3,
        std_type="63/25/38 MVA 110/20/10 kV",
        max_loading_percent=80,
    )
    assert net.trafo3w.at[t, "max_loading_percent"] == 80

    # Test error case - non-existent bus
    net = pandapowerNet(name="test_create_transformer3w5")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    create_bus(net, 10)
    with pytest.raises(UserWarning, match=r"Trafo tries to attach to bus 5"):
        create_transformer3w(
            net,
            hv_bus=b1,
            mv_bus=b2,
            lv_bus=5,
            std_type="63/25/38 MVA 110/20/10 kV",
        )

    validate_network(net)

def test_create_transformers3w():
    net = pandapowerNet(name="test_create_transformers3w")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    b3 = create_bus(net, 20)
    b4 = create_bus(net, 10)
    b5 = create_bus(net, 10)
    create_transformers3w(
        net=net,
        hv_buses=[b1, b1],
        mv_buses=[b2, b3],
        lv_buses=[b4, b5],
        std_type="63/25/38 MVA 110/20/10 kV",
        name=["t3w-1", "t3w-2"],
        in_service=[True, False],
        index=[5, 6],
    )
    res_df = pd.DataFrame({
        'name': pd.Series(['t3w-1', 't3w-2'], dtype=pd.StringDtype()),
        'std_type': pd.Series(['63/25/38 MVA 110/20/10 kV', '63/25/38 MVA 110/20/10 kV'], dtype=pd.StringDtype()),
        'hv_bus': pd.Series([0, 0], dtype=np.int64),
        'mv_bus': pd.Series([1, 2], dtype=np.int64),
        'lv_bus': pd.Series([3, 4], dtype=np.int64),
        'sn_hv_mva': pd.Series([63.0, 63.0], dtype=np.float64),
        'sn_mv_mva': pd.Series([25.0, 25.0], dtype=np.float64),
        'sn_lv_mva': pd.Series([38.0, 38.0], dtype=np.float64),
        'vn_hv_kv': pd.Series([110.0, 110.0], dtype=np.float64),
        'vn_mv_kv': pd.Series([20.0, 20.0], dtype=np.float64),
        'vn_lv_kv': pd.Series([10.0, 10.0], dtype=np.float64),
        'vk_hv_percent': pd.Series([10.4, 10.4], dtype=np.float64),
        'vk_mv_percent': pd.Series([10.4, 10.4], dtype=np.float64),
        'vk_lv_percent': pd.Series([10.4, 10.4], dtype=np.float64),
        'vkr_hv_percent': pd.Series([0.28, 0.28], dtype=np.float64),
        'vkr_mv_percent': pd.Series([0.32, 0.32], dtype=np.float64),
        'vkr_lv_percent': pd.Series([0.35, 0.35], dtype=np.float64),
        'pfe_kw': pd.Series([35.0, 35.0], dtype=np.float64),
        'i0_percent': pd.Series([0.89, 0.89], dtype=np.float64),
        'shift_mv_degree': pd.Series([0.0, 0.0], dtype=np.float64),
        'shift_lv_degree': pd.Series([0.0, 0.0], dtype=np.float64),
        'tap_side': pd.Series(['hv', 'hv'], dtype=pd.StringDtype()),
        'tap_neutral': pd.Series([0.0, 0.0], dtype=np.float64),
        'tap_min': pd.Series([-10.0, -10.0], dtype=np.float64),
        'tap_max': pd.Series([10.0, 10.0], dtype=np.float64),
        'tap_step_percent': pd.Series([1.2, 1.2], dtype=np.float64),
        'tap_step_degree': pd.Series([0.0, 0.0], dtype=np.float64),
        'tap_pos': pd.Series([0.0, 0.0], dtype=np.float64),
        'tap_at_star_point': pd.Series([False, False], dtype=pd.BooleanDtype()),
        # 'tap_changer_type': ['Ratio', 'Ratio'],
        # 'id_characteristic_table': pd.Series([pd.NA, pd.NA], dtype=pd.Int64Dtype),
        # 'tap_dependency_table': [False, False],
        'in_service': pd.Series([True, False], dtype=bool)
    }).set_index(pd.Index([5, 6]))
    for colum in res_df:
        assert net.trafo3w[colum].equals(res_df[colum])
    assert dataframes_equal(net.trafo3w, res_df)

    validate_network(net)


def net_transformer3w_from_parameters(**kwargs):
    net = pandapowerNet(name="net_transformer3w_from_parameters")
    b1 = create_bus(net, 15)
    b2 = create_bus(net, 0.4)
    b3 = create_bus(net, 0.9)
    create_transformers3w_from_parameters(
        net,
        hv_buses=[b1, b1],
        mv_buses=[b3, b3],
        lv_buses=[b2, b2],
        vn_hv_kv=15.0,
        vn_mv_kv=0.9,
        vn_lv_kv=0.45,
        sn_hv_mva=0.6,
        sn_mv_mva=0.5,
        sn_lv_mva=0.4,
        vk_hv_percent=1.0,
        vk_mv_percent=1.0,
        vk_lv_percent=1.0,
        vkr_hv_percent=0.3,
        vkr_mv_percent=0.3,
        vkr_lv_percent=0.3,
        pfe_kw=0.2,
        i0_percent=0.3,
        # tap_neutral=0.0, FIXME either remove this line or add tap_side and tap_pos
        mag0_rx=0.4,
        mag0_percent=30,
        **kwargs,
    )
    return net, b1, b2, b3


def test_create_transformer3w_from_parameters():
    # Test basic 3-winding transformer creation from parameters
    net = pandapowerNet(name="test_create_transformer3w_from_parameters0")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    b3 = create_bus(net, 10)
    t = create_transformer3w_from_parameters(
        net,
        hv_bus=b1,
        mv_bus=b2,
        lv_bus=b3,
        sn_hv_mva=63,
        sn_mv_mva=25,
        sn_lv_mva=38,
        vn_hv_kv=110,
        vn_mv_kv=20,
        vn_lv_kv=10,
        vk_hv_percent=10.4,
        vk_mv_percent=10.4,
        vk_lv_percent=10.4,
        vkr_hv_percent=0.28,
        vkr_mv_percent=0.32,
        vkr_lv_percent=0.35,
        pfe_kw=35,
        i0_percent=0.89,
        name="test_trafo3w",
    )

    assert len(net.trafo3w) == 1
    assert net.trafo3w.at[t, "name"] == "test_trafo3w"
    assert net.trafo3w.at[t, "hv_bus"] == b1
    assert net.trafo3w.at[t, "mv_bus"] == b2
    assert net.trafo3w.at[t, "lv_bus"] == b3
    assert net.trafo3w.at[t, "sn_hv_mva"] == 63
    assert net.trafo3w.at[t, "sn_mv_mva"] == 25
    assert net.trafo3w.at[t, "sn_lv_mva"] == 38
    assert net.trafo3w.at[t, "vn_hv_kv"] == 110
    assert net.trafo3w.at[t, "vn_mv_kv"] == 20
    assert net.trafo3w.at[t, "vn_lv_kv"] == 10
    assert np.allclose(net.trafo3w.at[t, "vk_hv_percent"], 10.4)
    assert np.allclose(net.trafo3w.at[t, "vk_mv_percent"], 10.4)
    assert np.allclose(net.trafo3w.at[t, "vk_lv_percent"], 10.4)
    assert np.allclose(net.trafo3w.at[t, "vkr_hv_percent"], 0.28)
    assert np.allclose(net.trafo3w.at[t, "vkr_mv_percent"], 0.32)
    assert np.allclose(net.trafo3w.at[t, "vkr_lv_percent"], 0.35)
    assert net.trafo3w.at[t, "pfe_kw"] == 35
    assert np.allclose(net.trafo3w.at[t, "i0_percent"], 0.89)

    # Test with shift angles
    net = pandapowerNet(name="test_create_transformer3w_from_parameters1")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    b3 = create_bus(net, 10)
    t = create_transformer3w_from_parameters(
        net,
        hv_bus=b1,
        mv_bus=b2,
        lv_bus=b3,
        sn_hv_mva=63,
        sn_mv_mva=25,
        sn_lv_mva=38,
        vn_hv_kv=110,
        vn_mv_kv=20,
        vn_lv_kv=10,
        vk_hv_percent=10.4,
        vk_mv_percent=10.4,
        vk_lv_percent=10.4,
        vkr_hv_percent=0.28,
        vkr_mv_percent=0.32,
        vkr_lv_percent=0.35,
        pfe_kw=35,
        i0_percent=0.89,
        shift_mv_degree=30,
        shift_lv_degree=150,
    )

    assert net.trafo3w.at[t, "shift_mv_degree"] == 30
    assert net.trafo3w.at[t, "shift_lv_degree"] == 150

    # Test with tap changer
    net = pandapowerNet(name="test_create_transformer3w_from_parameters2")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    b3 = create_bus(net, 10)
    t = create_transformer3w_from_parameters(
        net,
        hv_bus=b1,
        mv_bus=b2,
        lv_bus=b3,
        sn_hv_mva=63,
        sn_mv_mva=25,
        sn_lv_mva=38,
        vn_hv_kv=110,
        vn_mv_kv=20,
        vn_lv_kv=10,
        vk_hv_percent=10.4,
        vk_mv_percent=10.4,
        vk_lv_percent=10.4,
        vkr_hv_percent=0.28,
        vkr_mv_percent=0.32,
        vkr_lv_percent=0.35,
        pfe_kw=35,
        i0_percent=0.89,
        tap_side="hv",
        tap_pos=5,
        tap_neutral=0,
        tap_max=10,
        tap_min=-10,
        tap_step_percent=1.0,
    )

    assert net.trafo3w.at[t, "tap_side"] == "hv"
    assert net.trafo3w.at[t, "tap_pos"] == 5
    assert net.trafo3w.at[t, "tap_neutral"] == 0
    assert net.trafo3w.at[t, "tap_max"] == 10
    assert net.trafo3w.at[t, "tap_min"] == -10
    assert np.isclose(net.trafo3w.at[t, "tap_step_percent"], 1.0)

    # Test with zero sequence parameters
    net = pandapowerNet(name="test_create_transformer3w_from_parameters3")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    b3 = create_bus(net, 10)
    t = create_transformer3w_from_parameters(
        net,
        hv_bus=b1,
        mv_bus=b2,
        lv_bus=b3,
        sn_hv_mva=63,
        sn_mv_mva=25,
        sn_lv_mva=38,
        vn_hv_kv=110,
        vn_mv_kv=20,
        vn_lv_kv=10,
        vk_hv_percent=10.4,
        vk_mv_percent=10.4,
        vk_lv_percent=10.4,
        vkr_hv_percent=0.28,
        vkr_mv_percent=0.32,
        vkr_lv_percent=0.35,
        pfe_kw=35,
        i0_percent=0.89,
        vk0_hv_percent=10,
        vk0_mv_percent=10,
        vk0_lv_percent=10,
        vkr0_hv_percent=0.28,
        vkr0_mv_percent=0.32,
        vkr0_lv_percent=0.35,
        vector_group="YNd11",
    )

    assert net.trafo3w.at[t, "vk0_hv_percent"] == 10
    assert net.trafo3w.at[t, "vk0_mv_percent"] == 10
    assert net.trafo3w.at[t, "vk0_lv_percent"] == 10
    assert np.isclose(net.trafo3w.at[t, "vkr0_hv_percent"], 0.28)
    assert np.isclose(net.trafo3w.at[t, "vkr0_mv_percent"], 0.32)
    assert np.isclose(net.trafo3w.at[t, "vkr0_lv_percent"], 0.35)
    assert net.trafo3w.at[t, "vector_group"] == "YNd11"

    # Test with in_service=False
    net = pandapowerNet(name="test_create_transformer3w_from_parameters4")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    b3 = create_bus(net, 10)
    t = create_transformer3w_from_parameters(
        net,
        hv_bus=b1,
        mv_bus=b2,
        lv_bus=b3,
        sn_hv_mva=63,
        sn_mv_mva=25,
        sn_lv_mva=38,
        vn_hv_kv=110,
        vn_mv_kv=20,
        vn_lv_kv=10,
        vk_hv_percent=10.4,
        vk_mv_percent=10.4,
        vk_lv_percent=10.4,
        vkr_hv_percent=0.28,
        vkr_mv_percent=0.32,
        vkr_lv_percent=0.35,
        pfe_kw=35,
        i0_percent=0.89,
        in_service=False,
    )
    assert not net.trafo3w.at[t, "in_service"]

    # Test error case - non-existent bus
    net = pandapowerNet(name="test_create_transformer3w_from_parameters5")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 20)
    create_bus(net, 10)
    with pytest.raises(UserWarning, match=r"Trafo tries to attach to non-existent bus 5"):
        create_transformer3w_from_parameters(
            net,
            hv_bus=b1,
            mv_bus=b2,
            lv_bus=5,
            sn_hv_mva=63,
            sn_mv_mva=25,
            sn_lv_mva=38,
            vn_hv_kv=110,
            vn_mv_kv=20,
            vn_lv_kv=10,
            vk_hv_percent=10.4,
            vk_mv_percent=10.4,
            vk_lv_percent=10.4,
            vkr_hv_percent=0.28,
            vkr_mv_percent=0.32,
            vkr_lv_percent=0.35,
            pfe_kw=35,
            i0_percent=0.89,
        )

    validate_network(net)

def test_create_transformers3w_from_parameters():
    # setting params as single value
    net, *_ = net_transformer3w_from_parameters(test_kwargs="dummy_string")
    assert len(net.trafo3w) == 2
    assert all(net.trafo3w.hv_bus == 0)
    assert all(net.trafo3w.lv_bus == 1)
    assert all(net.trafo3w.mv_bus == 2)
    assert np.allclose(net.trafo3w.sn_hv_mva, 0.6)
    assert np.allclose(net.trafo3w.sn_mv_mva, 0.5)
    assert np.allclose(net.trafo3w.sn_lv_mva, 0.4)
    assert np.allclose(net.trafo3w.vn_hv_kv, 15.0)
    assert np.allclose(net.trafo3w.vn_mv_kv, 0.9)
    assert np.allclose(net.trafo3w.vn_lv_kv, 0.45)
    assert np.allclose(net.trafo3w.vk_hv_percent, 1.0)
    assert np.allclose(net.trafo3w.vk_mv_percent, 1.0)
    assert np.allclose(net.trafo3w.vk_lv_percent, 1.0)
    assert np.allclose(net.trafo3w.vkr_hv_percent, 0.3)
    assert np.allclose(net.trafo3w.vkr_mv_percent, 0.3)
    assert np.allclose(net.trafo3w.vkr_lv_percent, 0.3)
    assert np.allclose(net.trafo3w.pfe_kw, 0.2)
    assert np.allclose(net.trafo3w.i0_percent, 0.3)
    assert np.allclose(net.trafo3w.mag0_rx, 0.4)
    assert all(net.trafo3w.mag0_percent == 30)
    assert all(net.trafo3w.test_kwargs == "dummy_string")

    # setting params as array
    net = pandapowerNet(name="test_create_transformers3w_from_parameters")
    b1 = create_bus(net, 10)
    b2 = create_bus(net, 0.4)
    b3 = create_bus(net, 0.9)
    create_transformers3w_from_parameters(
        net,
        hv_buses=[b1, b1],
        mv_buses=[b3, b3],
        lv_buses=[b2, b2],
        vn_hv_kv=[15.0, 14.5],
        vn_mv_kv=[0.9, 0.7],
        vn_lv_kv=[0.45, 0.5],
        sn_hv_mva=[0.6, 0.7],
        sn_mv_mva=[0.5, 0.4],
        sn_lv_mva=[0.4, 0.3],
        vk_hv_percent=[1.0, 1.0],
        vk_mv_percent=[1.0, 1.0],
        vk_lv_percent=[1.0, 1.0],
        vkr_hv_percent=[0.3, 0.3],
        vkr_mv_percent=[0.3, 0.3],
        vkr_lv_percent=[0.3, 0.3],
        pfe_kw=[0.2, 0.1],
        i0_percent=[0.3, 0.2],
        in_service=[True, False],
        test_kwargs=["foo", "bar"],
    )
    assert len(net.trafo3w) == 2
    assert all(net.trafo3w.hv_bus == 0)
    assert all(net.trafo3w.lv_bus == 1)
    assert all(net.trafo3w.mv_bus == 2)
    assert all(net.trafo3w.sn_hv_mva == [0.6, 0.7])
    assert all(net.trafo3w.sn_mv_mva == [0.5, 0.4])
    assert all(net.trafo3w.sn_lv_mva == [0.4, 0.3])
    assert all(net.trafo3w.vn_hv_kv == [15.0, 14.5])
    assert all(net.trafo3w.vn_mv_kv == [0.9, 0.7])
    assert all(net.trafo3w.vn_lv_kv == [0.45, 0.5])
    assert np.allclose(net.trafo3w.vk_hv_percent, 1.0)
    assert np.allclose(net.trafo3w.vk_mv_percent, 1.0)
    assert np.allclose(net.trafo3w.vk_lv_percent, 1.0)
    assert np.allclose(net.trafo3w.vkr_hv_percent, 0.3)
    assert np.allclose(net.trafo3w.vkr_mv_percent, 0.3)
    assert np.allclose(net.trafo3w.vkr_lv_percent, 0.3)
    assert all(net.trafo3w.pfe_kw == [0.2, 0.1])
    assert all(net.trafo3w.i0_percent == [0.3, 0.2])
    assert all(net.trafo3w.in_service == [True, False])
    assert all(net.trafo3w.test_kwargs == ["foo", "bar"])

    validate_network(net)


def test_create_transformers3w_raise_errorexcept():
    # standard
    net, b1, b2, b3 = net_transformer3w_from_parameters()
    with pytest.raises(
            UserWarning,
            match=r"Three winding transformers with indexes \[1\] already exist.",
    ):
        create_transformers3w_from_parameters(
            net,
            hv_buses=[b1, b1],
            mv_buses=[b3, b3],
            lv_buses=[b2, b2],
            vn_hv_kv=15.0,
            vn_mv_kv=0.9,
            vn_lv_kv=0.45,
            sn_hv_mva=0.6,
            sn_mv_mva=0.5,
            sn_lv_mva=0.4,
            vk_hv_percent=1.0,
            vk_mv_percent=1.0,
            vk_lv_percent=1.0,
            vkr_hv_percent=0.3,
            vkr_mv_percent=0.3,
            vkr_lv_percent=0.3,
            pfe_kw=0.2,
            i0_percent=0.3,
            mag0_rx=0.4,
            mag0_percent=30,
            index=[2, 1],
        )
    validate_network(net)
    net = pandapowerNet(name="test_create_transformers3w_raise_errorexcept")
    b1 = create_bus(net, 15)
    b2 = create_bus(net, 0.4)
    b3 = create_bus(net, 0.9)
    with pytest.raises(
            UserWarning, match=r"Transformers trying to attach to non existing buses \{6\}"
    ):
        create_transformers3w_from_parameters(
            net,
            hv_buses=[6, b1],
            mv_buses=[b3, b3],
            lv_buses=[b2, b2],
            vn_hv_kv=15.0,
            vn_mv_kv=0.9,
            vn_lv_kv=0.45,
            sn_hv_mva=0.6,
            sn_mv_mva=0.5,
            sn_lv_mva=0.4,
            vk_hv_percent=1.0,
            vk_mv_percent=1.0,
            vk_lv_percent=1.0,
            vkr_hv_percent=0.3,
            vkr_mv_percent=0.3,
            vkr_lv_percent=0.3,
            pfe_kw=0.2,
            i0_percent=0.3,
            tap_neutral=0.0,
            mag0_rx=0.4,
            mag0_percent=30,
            index=[0, 1],
        )
    with pytest.raises(
            UserWarning, match=r"Transformers trying to attach to non existing buses \{3\}"
    ):
        create_transformers3w_from_parameters(
            net,
            hv_buses=[b1, b1],
            mv_buses=[b3, 3],
            lv_buses=[b2, b2],
            vn_hv_kv=15.0,
            vn_mv_kv=0.9,
            vn_lv_kv=0.45,
            sn_hv_mva=0.6,
            sn_mv_mva=0.5,
            sn_lv_mva=0.4,
            vk_hv_percent=1.0,
            vk_mv_percent=1.0,
            vk_lv_percent=1.0,
            vkr_hv_percent=0.3,
            vkr_mv_percent=0.3,
            vkr_lv_percent=0.3,
            pfe_kw=0.2,
            i0_percent=0.3,
            tap_neutral=0.0,
            mag0_rx=0.4,
            mag0_percent=30,
        )
    with pytest.raises(
            UserWarning,
            match=r"Transformers trying to attach to non existing buses \{3, 4\}",
    ):
        create_transformers3w_from_parameters(
            net,
            hv_buses=[b1, b1],
            mv_buses=[b3, b3],
            lv_buses=[4, 3],
            vn_hv_kv=15.0,
            vn_mv_kv=0.9,
            vn_lv_kv=0.45,
            sn_hv_mva=0.6,
            sn_mv_mva=0.5,
            sn_lv_mva=0.4,
            vk_hv_percent=1.0,
            vk_mv_percent=1.0,
            vk_lv_percent=1.0,
            vkr_hv_percent=0.3,
            vkr_mv_percent=0.3,
            vkr_lv_percent=0.3,
            pfe_kw=0.2,
            i0_percent=0.3,
            tap_neutral=0.0,
            mag0_rx=0.4,
            mag0_percent=30,
        )

    validate_network(net)
