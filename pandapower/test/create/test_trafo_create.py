# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
import numpy as np
import pandas as pd

from pandapower.create import (
    create_empty_network, create_bus, create_transformer_from_parameters, create_transformer, create_transformers,
    create_transformers_from_parameters, create_transformers3w_from_parameters, create_transformers3w, load_std_type,
)
from pandapower.std_types import create_std_type
from pandapower.toolbox import dataframes_equal
from pandapower.network_schema.tools.validation.network_validation import validate_network


def test_tap_changer_type_default():
    net = create_empty_network()
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


def test_create_transformer_from_parameters(): raise NotImplementedError()


def test_create_transformers_from_parameters():
    # standard
    net = create_empty_network()
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
    net = create_empty_network()
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
        # tap_neutral=0.0, FIXME add tap_side and tap_pos or remove
        vector_group="Dyn",
        si0_hv_partial=0.1,
        max_loading_percent=80,
        test_kwargs="dummy_string",
    )
    assert len(net.trafo) == 2
    assert all(net.trafo.hv_bus == 0)
    assert all(net.trafo.lv_bus == 1)
    assert all(net.trafo.sn_mva == 0.5)
    assert all(net.trafo.vn_hv_kv == 15.0)
    assert all(net.trafo.vn_lv_kv == 0.45)
    assert all(net.trafo.vk_percent == 1.0)
    assert all(net.trafo.vkr_percent == 0.3)
    assert all(net.trafo.pfe_kw == 0.2)
    assert all(net.trafo.i0_percent == 0.3)
    assert all(net.trafo.vk0_percent == 0.4)
    assert all(net.trafo.mag0_rx == 0.4)
    assert all(net.trafo.mag0_percent == 30)
    # assert all(net.trafo.tap_neutral == 0.0) FIXME either add tap_side or remove this
    # assert all(net.trafo.tap_pos == 0.0)
    assert all(net.trafo.vector_group.values == "Dyn")
    assert all(net.trafo.max_loading_percent == 80.0)
    assert all(net.trafo.si0_hv_partial == 0.1)
    assert all(net.trafo.test_kwargs == "dummy_string")

    validate_network(net)

    # setting params as array
    net = create_empty_network()
    b1 = create_bus(net, 10)
    b2 = create_bus(net, 10)
    t = create_transformers_from_parameters(
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
        # tap_neutral=[0.0, 1.0], FIXME add tap_side or remove
        # tap_pos=[-1, 4],
        test_kwargs=["dummy_string", "dummy_string"],
    )

    assert len(net.trafo) == 2
    assert all(net.trafo.hv_bus == 0)
    assert all(net.trafo.lv_bus == 1)
    assert all(net.trafo.vn_hv_kv == 15.0)
    assert all(net.trafo.vn_lv_kv == 0.45)
    assert all(net.trafo.sn_mva == 0.6)
    assert all(net.trafo.vk_percent == 1.0)
    assert all(net.trafo.vkr_percent == 0.3)
    assert all(net.trafo.pfe_kw == 0.2)
    assert all(net.trafo.i0_percent == 0.3)
    assert all(net.trafo.vk0_percent == 0.4)
    assert all(net.trafo.mag0_rx == 0.4)
    assert all(net.trafo.mag0_percent == 30)
    assert all(net.trafo.test_kwargs == "dummy_string")
    # assert net.trafo.tap_neutral.at[t[0]] == 0 FIXME add tap_side or remove
    # assert net.trafo.tap_neutral.at[t[1]] == 1
    # assert net.trafo.tap_pos.at[t[0]] == -1
    # assert net.trafo.tap_pos.at[t[1]] == 4

    validate_network(net)


def test_create_transformers_raise_errorexcept():
    # standard
    net = create_empty_network()
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
    net = create_empty_network()
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
    net = create_empty_network()
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
    net = create_empty_network()
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
    net = create_empty_network()
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


def test_create_transformer(): raise NotImplementedError()


def test_create_transformers():
    net = create_empty_network()
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
    res_df = pd.DataFrame({
        'name': pd.Series(['trafo1', 'trafo2'], dtype=pd.StringDtype),
        'std_type': pd.Series(['0.4 MVA 10/0.4 kV', '0.4 MVA 10/0.4 kV'], dtype=pd.StringDtype),
        'hv_bus': pd.Series([0, 0], dtype=np.int64),
        'lv_bus': pd.Series([1, 2], dtype=np.int64),
        'sn_mva': pd.Series([0.4, 0.4], dtype=np.float64),
        'vn_hv_kv': pd.Series([10.0, 10.0], dtype=np.float64),
        'vn_lv_kv': pd.Series([0.4, 0.4], dtype=np.float64),
        'vk_percent': pd.Series([4.0, 4.0], dtype=np.float64),
        'vkr_percent': pd.Series([1.325, 1.325], dtype=np.float64),
        'pfe_kw': pd.Series([0.95, 0.95], dtype=np.float64),
        'i0_percent': pd.Series([0.2375, 0.2375], dtype=np.float64),
        'shift_degree': pd.Series([0.0, 0.0], dtype=np.float64),
        # 'tap_side': ['', ''],
        # 'tap_neutral': [nan, nan],
        # 'tap_min': [nan, nan],
        # 'tap_max': [nan, nan],
        # 'tap_step_percent': [nan, nan],
        # 'tap_step_degree': [nan, nan],
        # 'tap_pos': [nan, nan],
        # 'tap_changer_type': [pd.NA, pd.NA],
        # 'id_characteristic_table': pd.Series([pd.NA, pd.NA], dtype=pd.Int64Dtype),
        # 'tap_dependency_table': [False, False],
        'parallel': pd.Series([1, 1], dtype=np.int64),
        'df': pd.Series([1.0, 1.0], dtype=np.float64),
        'in_service': pd.Series([True, True], dtype=bool),
        # 'oltc': [False, False],
        'test_kwargs': ['TestKW', 'TestKW'],
        'vector_group': pd.Series(['Dyn5', 'Dyn5'], dtype=pd.StringDtype),
    })
    for colum in res_df:
        assert net.trafo[colum].equals(res_df[colum])
    assert dataframes_equal(net.trafo, res_df)


def test_create_transformers_for_single():
    net = create_empty_network()
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
    res_df = pd.DataFrame({
        'name': pd.Series(['trafo1'], dtype=pd.StringDtype),
        'std_type': pd.Series(['0.4 MVA 10/0.4 kV'], dtype=pd.StringDtype),
        'hv_bus': pd.Series([0], dtype=np.int64),
        'lv_bus': pd.Series([1], dtype=np.int64),
        'sn_mva': pd.Series([0.4], dtype=np.float64),
        'vn_hv_kv': pd.Series([10.0], dtype=np.float64),
        'vn_lv_kv': pd.Series([0.4], dtype=np.float64),
        'vk_percent': pd.Series([4.0], dtype=np.float64),
        'vkr_percent': pd.Series([1.325], dtype=np.float64),
        'pfe_kw': pd.Series([0.95], dtype=np.float64),
        'i0_percent': pd.Series([0.2375], dtype=np.float64),
        'shift_degree': pd.Series([0.0], dtype=np.float64),
        # 'tap_side': [''],
        # 'tap_neutral': [nan],
        # 'tap_min': [nan],
        # 'tap_max': [nan],
        # 'tap_step_percent': [nan],
        # 'tap_step_degree': [nan],
        # 'tap_pos': [nan],
        # 'tap_changer_type': [''],
        # 'id_characteristic_table': pd.Series([pd.NA], dtype=pd.Int64Dtype),
        # 'tap_dependency_table': [False],
        'parallel': pd.Series([1], dtype=np.int64),
        'df': pd.Series([1.0], dtype=np.float64),
        'in_service': pd.Series([True], dtype=bool),
        # 'oltc': [False],
        'test_kwargs': ['TestKW'],
        'vector_group': pd.Series(['Dyn5'], dtype=pd.StringDtype),
    })
    assert dataframes_equal(net.trafo, res_df)

    validate_network(net)


def test_create_transformer3w(): raise NotImplementedError()


def test_create_transformers3w():
    net = create_empty_network()
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
        'name': pd.Series(['t3w-1', 't3w-2'], dtype=pd.StringDtype),
        'std_type': pd.Series(['63/25/38 MVA 110/20/10 kV', '63/25/38 MVA 110/20/10 kV'], dtype=pd.StringDtype),
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
        'tap_side': pd.Series(['hv', 'hv'], dtype=pd.StringDtype),
        'tap_neutral': pd.Series([0.0, 0.0], dtype=np.float64),
        'tap_min': pd.Series([-10.0, -10.0], dtype=np.float64),
        'tap_max': pd.Series([10.0, 10.0], dtype=np.float64),
        'tap_step_percent': pd.Series([1.2, 1.2], dtype=np.float64),
        'tap_step_degree': pd.Series([0.0, 0.0], dtype=np.float64),
        'tap_pos': pd.Series([0.0, 0.0], dtype=np.float64),
        'tap_at_star_point': pd.Series([False, False], dtype=pd.BooleanDtype),
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
    net = create_empty_network()
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


def test_create_transformer3w_from_parameters(): raise NotImplementedError()


def test_create_transformers3w_from_parameters():
    # setting params as single value
    net, *_ = net_transformer3w_from_parameters(test_kwargs="dummy_string")
    assert len(net.trafo3w) == 2
    assert all(net.trafo3w.hv_bus == 0)
    assert all(net.trafo3w.lv_bus == 1)
    assert all(net.trafo3w.mv_bus == 2)
    assert all(net.trafo3w.sn_hv_mva == 0.6)
    assert all(net.trafo3w.sn_mv_mva == 0.5)
    assert all(net.trafo3w.sn_lv_mva == 0.4)
    assert all(net.trafo3w.vn_hv_kv == 15.0)
    assert all(net.trafo3w.vn_mv_kv == 0.9)
    assert all(net.trafo3w.vn_lv_kv == 0.45)
    assert all(net.trafo3w.vk_hv_percent == 1.0)
    assert all(net.trafo3w.vk_mv_percent == 1.0)
    assert all(net.trafo3w.vk_lv_percent == 1.0)
    assert all(net.trafo3w.vkr_hv_percent == 0.3)
    assert all(net.trafo3w.vkr_mv_percent == 0.3)
    assert all(net.trafo3w.vkr_lv_percent == 0.3)
    assert all(net.trafo3w.pfe_kw == 0.2)
    assert all(net.trafo3w.i0_percent == 0.3)
    assert all(net.trafo3w.mag0_rx == 0.4)
    assert all(net.trafo3w.mag0_percent == 30)
    #assert all(net.trafo3w.tap_neutral == 0.0) FIXME add tap_side or remove this
    #assert all(net.trafo3w.tap_pos == 0.0)
    assert all(net.trafo3w.test_kwargs == "dummy_string")

    # setting params as array
    net = create_empty_network()
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
        # tap_neutral=[0.0, 5.0],  FIXME either add tap_side or remove this
        # tap_pos=[1, 2],
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
    assert all(net.trafo3w.vk_hv_percent == 1.0)
    assert all(net.trafo3w.vk_mv_percent == 1.0)
    assert all(net.trafo3w.vk_lv_percent == 1.0)
    assert all(net.trafo3w.vkr_hv_percent == 0.3)
    assert all(net.trafo3w.vkr_mv_percent == 0.3)
    assert all(net.trafo3w.vkr_lv_percent == 0.3)
    assert all(net.trafo3w.pfe_kw == [0.2, 0.1])
    assert all(net.trafo3w.i0_percent == [0.3, 0.2])
    # assert all(net.trafo3w.tap_neutral == [0.0, 5.0]) FIXME either add tap_side or remove
    # assert all(net.trafo3w.tap_pos == [1, 2])
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
            # tap_neutral=0.0,  FIXME either remove this line or add tap_side and tap_pos
            mag0_rx=0.4,
            mag0_percent=30,
            index=[2, 1],
        )
    validate_network(net)
    net = create_empty_network()
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
