# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
import gc
import copy
import geojson
import numpy as np
import pandas as pd

from pandapower.control import SplineCharacteristic, Characteristic
from pandapower.control.util.characteristic import LogSplineCharacteristic
from math import isclose

try:
    import geopandas as gpd
    import shapely.geometry
    GEOPANDAS_INSTALLED = True
except ImportError:
    GEOPANDAS_INSTALLED = False

from pandapower import get_gc_objects_dict
from pandapower.file_io import from_json_string, to_json, create_empty_network
from pandapower.create import create_bus, create_lines, create_line, create_buses, create_shunt
from pandapower.auxiliary import get_indices, pandapowerNet
from pandapower.networks import example_simple, example_multivoltage, mv_oberrhein
from pandapower.timeseries import DFData
from pandapower.control import (
    SplineCharacteristic,
    ContinuousTapControl,
    ConstControl,
    create_trafo_characteristic_object,
)
from pandapower.control.util.auxiliary import (create_shunt_characteristic_object, _create_trafo_characteristics,
                                               create_q_capability_characteristics_object, get_min_max_q_mvar_from_characteristics_object)


class MemoryLeakDemo:
    """
    Dummy class to demonstrate memory leaks
    """

    def __init__(self, net):
        self.net = net
        # it is interesting, that if "self" is just an attribute of net, there are no problems
        # if "self" is saved in a DataFrame, it causes a memory leak
        net['memory_leak_demo'] = pd.DataFrame(data=[self], columns=['object'])


class MemoryLeakDemoDF:
    """
    Dummy class to demonstrate memory leaks
    """

    def __init__(self, df):
        self.df = df
        # if "self" is saved in a DataFrame, it causes a memory leak
        df.loc[0, 'object'] = self


class MemoryLeakDemoDict:
    """
    Dummy class to demonstrate memory leaks
    """

    def __init__(self, d):
        self.d = d
        d['object'] = self


def test_get_indices():
    a = [i + 100 for i in range(10)]
    lookup = {idx: pos for pos, idx in enumerate(a)}
    lookup["before_fuse"] = a

    # First without fused buses no magic here
    # after fuse
    result = get_indices([102, 107], lookup, fused_indices=True)
    assert np.array_equal(result, [2, 7])

    # before fuse
    result = get_indices([2, 7], lookup, fused_indices=False)
    assert np.array_equal(result, [102, 107])

    # Same setup EXCEPT we have fused buses now (bus 102 and 107 are fused)
    lookup[107] = lookup[102]

    # after fuse
    result = get_indices([102, 107], lookup, fused_indices=True)
    assert np.array_equal(result, [2, 2])

    # before fuse
    result = get_indices([2, 7], lookup, fused_indices=False)
    assert np.array_equal(result, [102, 107])


def test_net_deepcopy():
    net = example_simple()
    net.line.at[0, 'geo'] = geojson.dumps(geojson.LineString([(0., 1.), (1., 2.)]))
    net.bus.at[0, 'geo'] = geojson.dumps(geojson.Point((0., 1.)))

    ContinuousTapControl(net, element_index=0, vm_set_pu=1)
    ds = DFData(pd.DataFrame(data=[[0, 1, 2], [3, 4, 5]]))
    ConstControl(net, element='load', variable='p_mw', element_index=[0],
                            profile_name=[0], data_source=ds)

    net1 = copy.deepcopy(net)

    assert not net1.controller.object.at[1].data_source is ds
    assert not net1.controller.object.at[1].data_source.df is ds.df

    if GEOPANDAS_INSTALLED:
        for tab in ('bus', 'line'):
            net[f'{tab}_geodata'] = gpd.GeoDataFrame(net[tab].geo.dropna().apply(geojson.loads).apply(
                lambda x: x["coordinates"] if x is not None else x), geometry=net[tab].geo.dropna().apply(geojson.loads))
        net1 = copy.deepcopy(net)
        assert isinstance(net1.line_geodata, gpd.GeoDataFrame)
        assert isinstance(net1.bus_geodata, gpd.GeoDataFrame)
        assert isinstance(net1.bus_geodata.geometry.iat[0], shapely.geometry.Point)
        assert isinstance(net1.line_geodata.geometry.iat[0], shapely.geometry.LineString)


def test_memory_leaks():
    net = example_simple()

    # first, test to check that there are no memory leaks
    types_dict1 = get_gc_objects_dict()
    num = 3
    for _ in range(num):
        net_copy = copy.deepcopy(net)
        # In each net copy it has only one controller
        ContinuousTapControl(net_copy, element_index=0, vm_set_pu=1)

    gc.collect()

    types_dict2 = get_gc_objects_dict()

    assert types_dict2[pandapowerNet] - types_dict1[pandapowerNet] == 1
    assert types_dict2[ContinuousTapControl] - types_dict1.get(
        ContinuousTapControl, 0) == 1


def test_memory_leaks_demo():
    net = example_simple()
    # first, test to check that there are no memory leaks
    types_dict1 = get_gc_objects_dict()
    # now, demonstrate how a memory leak occurs
    # emulates the earlier behavior before the fix with weakref
    num = 3
    for _ in range(num):
        net_copy = copy.deepcopy(net)
        MemoryLeakDemo(net_copy)

    # demonstrate how the garbage collector doesn't remove the objects even if called explicitly
    gc.collect()
    types_dict2 = get_gc_objects_dict()
    assert types_dict2[pandapowerNet] - types_dict1[pandapowerNet] == num
    assert types_dict2[MemoryLeakDemo] - types_dict1.get(MemoryLeakDemo, 0) == num


def test_memory_leaks_no_copy():
    types_dict0 = get_gc_objects_dict()
    num = 3
    for _ in range(num):
        net = create_empty_network()
        # In each net copy it has only one controller
        ConstControl(net, 'sgen', 'p_mw', 0)

    gc.collect()
    types_dict1 = get_gc_objects_dict()
    assert types_dict1[ConstControl] - types_dict0.get(ConstControl, 0) == 1
    assert types_dict1[pandapowerNet] - types_dict0.get(pandapowerNet, 0) <= 1


def test_memory_leak_no_copy_demo():
    types_dict1 = get_gc_objects_dict()
    # now, demonstrate how a memory leak occurs
    # emulates the earlier behavior before the fix with weakref
    num = 3
    for _ in range(num):
        net = example_simple()
        MemoryLeakDemo(net)

    # demonstrate how the garbage collector doesn't remove the objects even if called explicitly
    gc.collect()
    types_dict2 = get_gc_objects_dict()
    assert types_dict2[pandapowerNet] - \
           types_dict1.get(pandapowerNet, 0) >= num-1
    assert types_dict2[MemoryLeakDemo] - types_dict1.get(MemoryLeakDemo, 0) == num


def test_memory_leak_df():
    types_dict1 = get_gc_objects_dict()
    num = 3
    for _ in range(num):
        df = pd.DataFrame()
        MemoryLeakDemoDF(df)

    gc.collect()
    types_dict2 = get_gc_objects_dict()
    assert types_dict2[MemoryLeakDemoDF] - types_dict1.get(MemoryLeakDemoDF, 0) == num


def test_memory_leak_dict():
    types_dict1 = get_gc_objects_dict()
    num = 3
    for _ in range(num):
        d = dict()
        MemoryLeakDemoDict(d)

    gc.collect()
    types_dict2 = get_gc_objects_dict()
    assert types_dict2[MemoryLeakDemoDict] - types_dict1.get(MemoryLeakDemoDict, 0) <= 1


def test_create_trafo_characteristics():
    net = example_multivoltage()
    net["trafo_characteristic_table"] = pd.DataFrame(
        {'id_characteristic': [0, 0, 0, 0, 0], 'step': [-2, -1, 0, 1, 2], 'voltage_ratio': [1, 1, 1, 1, 1],
         'angle_deg': [0, 0, 0, 0, 0], 'vk_percent': [2, 3, 4, 5, 6],
         'vkr_percent': [1.323, 1.324, 1.325, 1.326, 1.327], 'vk_hv_percent': np.nan, 'vkr_hv_percent': np.nan,
         'vk_mv_percent': np.nan, 'vkr_mv_percent': np.nan, 'vk_lv_percent': np.nan, 'vkr_lv_percent': np.nan})
    net.trafo.at[1, 'id_characteristic_table'] = 0
    net.trafo.at[0, 'tap_dependency_table'] = False
    net.trafo.at[1, 'tap_dependency_table'] = True
    # add spline characteristics for one transformer based on trafo_characteristic_table
    create_trafo_characteristic_object(net)
    assert "trafo_characteristic_spline" in net
    assert "id_characteristic_spline" in net.trafo.columns
    assert len(net.trafo_characteristic_spline) == 1
    assert net.trafo.id_characteristic_spline.dtype == pd.Int64Dtype()
    assert isinstance(net.trafo.id_characteristic_spline.at[1], np.int64)
    assert pd.isna(net.trafo.id_characteristic_spline.at[0])
    assert all(col in net.trafo_characteristic_spline.columns for col in [
        'voltage_ratio_characteristic', 'angle_deg_characteristic',
        'vk_percent_characteristic', 'vkr_percent_characteristic'])
    assert isinstance(net.trafo_characteristic_spline.at[
                          net.trafo.id_characteristic_spline.at[1], 'vk_percent_characteristic'],
                      SplineCharacteristic)
    assert isclose(net.trafo_characteristic_spline.at[
                       net.trafo.id_characteristic_spline.at[1], 'vk_percent_characteristic'](-2).item(),
                   2, rel_tol=1e-9)
    assert pd.isna(net.trafo_characteristic_spline.at[
                       net.trafo.id_characteristic_spline.at[1], 'vkr_hv_percent_characteristic'])

    # create spline characteristics again for two transformers based on the updated trafo_characteristic_table
    new_rows = pd.DataFrame(
        {'id_characteristic': [1, 1, 1, 1, 1], 'step': [-2, -1, 0, 1, 2], 'voltage_ratio': [1, 1, 1, 1, 1],
         'angle_deg': [0, 0, 0, 0, 0], 'vk_percent': [2, 3, 4, 5, 6],
         'vkr_percent': [1.323, 1.324, 1.325, 1.326, 1.327]})
    net["trafo_characteristic_table"] = pd.concat([net["trafo_characteristic_table"], new_rows], ignore_index=True)
    net.trafo.at[0, 'id_characteristic_table'] = 1
    net.trafo.at[0, 'tap_dependency_table'] = True
    create_trafo_characteristic_object(net)
    assert len(net.trafo_characteristic_spline) == 2
    assert net.trafo.at[0, 'id_characteristic_spline'] == 1
    assert isinstance(net.trafo_characteristic_spline.at[
                          net.trafo.id_characteristic_spline.at[0], 'vk_percent_characteristic'],
                      SplineCharacteristic)
    assert isclose(net.trafo_characteristic_spline.at[
               net.trafo.id_characteristic_spline.at[0], 'vk_percent_characteristic'](2).item(), 6, rel_tol=1e-9)
    assert isinstance(net.trafo_characteristic_spline.at[
                          net.trafo.id_characteristic_spline.at[1], 'vk_percent_characteristic'],
                      SplineCharacteristic)
    assert isclose(net.trafo_characteristic_spline.at[
               net.trafo.id_characteristic_spline.at[1], 'vk_percent_characteristic'](-1).item(), 3, rel_tol=1e-9)
    assert pd.isna(net.trafo_characteristic_spline.at[
                       net.trafo.id_characteristic_spline.at[0], 'vkr_hv_percent_characteristic'])
    assert pd.isna(net.trafo_characteristic_spline.at[
                       net.trafo.id_characteristic_spline.at[1], 'vkr_hv_percent_characteristic'])

    # test for 3w trafo
    new_rows = pd.DataFrame(
        {'id_characteristic': [2, 2, 2, 2, 2], 'step': [-8, -4, 0, 4, 8], 'voltage_ratio': [1, 1, 1, 1, 1],
         'angle_deg': [0, 0, 0, 0, 0], 'vk_hv_percent': [8.1, 9.5, 10, 11.1, 12.9],
         'vkr_hv_percent': [1.323, 1.325, 1.329, 1.331, 1.339], 'vk_mv_percent': [8.1, 9.5, 10, 11.1, 12.9],
         'vkr_mv_percent': [1.323, 1.325, 1.329, 1.331, 1.339], 'vk_lv_percent': [8.1, 9.5, 10, 11.1, 12.9],
         'vkr_lv_percent': [1.323, 1.325, 1.329, 1.331, 1.339]})
    net["trafo_characteristic_table"] = pd.concat([net["trafo_characteristic_table"], new_rows], ignore_index=True)
    net.trafo3w.at[0, 'id_characteristic_table'] = 2
    net.trafo3w.at[0, 'tap_dependency_table'] = True
    # create spline characteristics again including a 3-winding transformer
    create_trafo_characteristic_object(net)
    assert len(net.trafo_characteristic_spline) == 3
    assert "id_characteristic_spline" in net.trafo3w.columns
    assert isinstance(net.trafo3w.id_characteristic_spline.at[0], np.int64)
    assert net.trafo_characteristic_spline.loc[net.trafo3w['id_characteristic_table'].at[0]].notna().sum() == 9
    assert isinstance(net.trafo_characteristic_spline.at[
                          net.trafo3w.id_characteristic_spline.at[0], 'vkr_hv_percent_characteristic'],
                      SplineCharacteristic)
    assert isclose(net.trafo_characteristic_spline.at[
               net.trafo3w.id_characteristic_spline.at[0], 'vk_hv_percent_characteristic'](0).item(), 10, rel_tol=1e-9)
    assert isclose(net.trafo_characteristic_spline.at[
               net.trafo3w.id_characteristic_spline.at[0], 'vk_lv_percent_characteristic'](4).item(), 11.1,
                   rel_tol=1e-9)
    assert isclose(net.trafo_characteristic_spline.at[
               net.trafo3w.id_characteristic_spline.at[0], 'vkr_hv_percent_characteristic'](-4).item(), 1.325,
                   rel_tol=1e-9)
    assert isclose(net.trafo_characteristic_spline.at[
               net.trafo3w.id_characteristic_spline.at[0], 'vkr_lv_percent_characteristic'](8).item(), 1.339,
                   rel_tol=1e-9)
    assert pd.isna(net.trafo_characteristic_spline.at[
                       net.trafo3w.id_characteristic_spline.at[0], 'vk_percent_characteristic'])

    # this should be enough testing for adding columns
    # now let's test if it raises errors

    # invalid variable
    with pytest.raises(UserWarning):
        _create_trafo_characteristics(net, "trafo3w", 0, 'vk_percent',
                                                [-8, -4, 0, 4, 8], [8.1, 9.1, 10.1, 11.1, 12.1])

    # invalid shapes
    with pytest.raises(UserWarning):
        _create_trafo_characteristics(net, "trafo3w", 0, 'vk_hv_percent',
                                                [-8, -4, 0, 4, 8], [8.1, 9.1, 10.1, 11.1])

    with pytest.raises(UserWarning):
        _create_trafo_characteristics(net, "trafo3w", [0], 'vk_hv_percent',
                                                [-8, -4, 0, 4, 8], [8.1, 9.1, 10.1, 11.1, 12.1])

    with pytest.raises(UserWarning):
        _create_trafo_characteristics(net, "trafo3w", [0, 1], 'vk_hv_percent',
                                                [[-8, -4, 0, 4, 8]], [[8.1, 9.1, 10.1, 11.1, 12.1]])

    with pytest.raises(UserWarning):
        _create_trafo_characteristics(net, "trafo3w", [0, 1], 'vk_hv_percent',
                                                [[-8, -4, 0, 4, 8], [-8, -4, 0, 4, 8]],
                                                [[8.1, 9.1, 10.1, 11.1, 12.1]])


def test_creation_of_shunt_characteristics():
    net = create_empty_network()
    b = create_buses(net, 2, 110)
    create_shunt(net, bus=b[1], q_mvar=-50, p_mw=0, step=1, max_step=5)
    net["shunt_characteristic_table"] = pd.DataFrame(
        {'id_characteristic': [0, 0, 0, 0, 0], 'step': [1, 2, 3, 4, 5], 'q_mvar': [-25, -55, -75, -120, -125],
         'p_mw': [1, 1.5, 3, 4.5, 5]})
    net.shunt.at[0, "step_dependency_table"] = True
    net.shunt.at[0, "id_characteristic_table"] = 0


    create_shunt_characteristic_object(net)

    assert np.isclose(net.shunt_characteristic_spline.loc[0, "q_mvar_characteristic"](1), -25.)
    assert np.isclose(net.shunt_characteristic_spline.loc[0, "q_mvar_characteristic"](3), -75.)
    assert np.isclose(net.shunt_characteristic_spline.loc[0, "q_mvar_characteristic"](5), -125.)
    assert np.isclose(net.shunt_characteristic_spline.loc[0, "p_mw_characteristic"](1), 1)
    assert np.isclose(net.shunt_characteristic_spline.loc[0, "p_mw_characteristic"](3), 3)
    assert np.isclose(net.shunt_characteristic_spline.loc[0, "p_mw_characteristic"](5), 5)

    # test re-creation of shunt characteristic object (and deletion of old one)
    net["shunt_characteristic_table"] = pd.DataFrame(
        {'id_characteristic': [1, 1, 1, 1, 1], 'step': [1, 2, 3, 4, 5], 'q_mvar': [25, 55, 75, 120, 125],
         'p_mw': [6, 6.5, 7, 8.5, 10]})
    net.shunt.at[0, "id_characteristic_table"] = 1

    create_shunt_characteristic_object(net)

    assert np.isclose(net.shunt_characteristic_spline.loc[0, "q_mvar_characteristic"](1), 25.)
    assert np.isclose(net.shunt_characteristic_spline.loc[0, "q_mvar_characteristic"](3), 75.)
    assert np.isclose(net.shunt_characteristic_spline.loc[0, "q_mvar_characteristic"](5), 125.)
    assert np.isclose(net.shunt_characteristic_spline.loc[0, "p_mw_characteristic"](1), 6)
    assert np.isclose(net.shunt_characteristic_spline.loc[0, "p_mw_characteristic"](3), 7)
    assert np.isclose(net.shunt_characteristic_spline.loc[0, "p_mw_characteristic"](5), 10)


def test_creation_of_q_capability_characteristics():
    net = example_multivoltage()
    net["q_capability_curve_table"] = pd.DataFrame(
        {'id_q_capability_curve': [0, 0, 0, 0, 0], 'p_mw': [0.0, 50.0, 100.0, 125.0, 125.0],
         'q_min_mvar': [-100.0, -75.0, -50.0, -25.0, -10], 'q_max_mvar': [150.0, 125.0, 75, 50.0, 10.0]})
    net.gen.at[0, "id_q_capability_characteristic"] = 0
    net.gen['curve_style'] = "straightLineYValues"

    # Add q_capability_characteristic for one gen based on q_capability_curve_table
    create_q_capability_characteristics_object(net)
    assert "q_capability_characteristic" in net
    assert len(net.q_capability_characteristic) == 1
    assert net.gen.id_q_capability_characteristic.dtype == pd.Int64Dtype()
    assert isinstance(net.gen.id_q_capability_characteristic.at[0], np.int64)
    assert pd.notna(net.gen.id_q_capability_characteristic.at[0])
    assert all(col in net.q_capability_characteristic.columns for col in ["q_max_characteristic",
                                                                                "q_min_characteristic"])
    assert isinstance(net.q_capability_characteristic.loc
                      [net.gen.id_q_capability_characteristic.at[0], 'q_max_characteristic'],
                      Characteristic)
    assert isclose(net.q_capability_characteristic.at
                   [net.gen.id_q_capability_characteristic.at[0], 'q_max_characteristic'](-2).item(), 150,
                   rel_tol=1e-9)
    assert pd.notna(net.q_capability_characteristic.at
                    [net.gen.id_q_capability_characteristic.at[0], 'q_min_characteristic'])

    # Create q_capability_characteristic again for the same gen based on the updated q_capability_curve_table
    new_rows = pd.DataFrame(
        {'id_q_capability_curve': [1, 1, 1, 1, 1], 'p_mw': [0.0, 30.0, 50.0, 70.0, 130],
         'q_min_mvar': [-29.0, -27, -26.0, -25.0, -20.0], 'q_max_mvar': [141.0, 141.0, 137.0, 134.0, 128.0]})
    net["q_capability_curve_table"] = pd.concat([net["q_capability_curve_table"], new_rows], ignore_index=True)
    net.gen.at[0, "id_q_capability_characteristic"] = 1
    create_q_capability_characteristics_object(net)
    assert len(net.q_capability_characteristic) == 2
    assert net.gen.at[0, "id_q_capability_characteristic"] == 1
    assert isinstance(net.q_capability_characteristic.loc
                      [net.gen.id_q_capability_characteristic.at[0], 'q_max_characteristic'],
                      Characteristic)

    assert isinstance(net.q_capability_characteristic.loc
                      [net.gen.id_q_capability_characteristic.at[0], 'q_min_characteristic'],
                      Characteristic)

    assert isclose(net.q_capability_characteristic.at[
                       net.gen.id_q_capability_characteristic.at[0], 'q_max_characteristic'](130).item(), 128.0,
                   rel_tol=1e-9)
    assert isclose(net.q_capability_characteristic.at[
                       net.gen.id_q_capability_characteristic.at[0], 'q_max_characteristic'](0).item(), 141.0,
                   rel_tol=1e-9)
    assert pd.notna(net.q_capability_characteristic.loc
                    [net.gen.id_q_capability_characteristic.at[0], 'q_max_characteristic'])
    assert pd.notna(net.q_capability_characteristic.loc
                    [net.gen.id_q_capability_characteristic.at[0], 'q_min_characteristic'])

def test_get_min_max_q_capability():
    net = example_multivoltage()
    sgen_indices_with_char = [1, 2]
    # create q characteristics table
    p_mw = [-2.0, -1.0, 0.0, 0.5, 2.0, -5.0, 0.5, 4.0]
    q_min_mvar = [-0.5, -0.3, -0.2, -0.3, -0.5, -0.25, -0.35, -0.45]
    q_max_mvar = [0.6, 0.4, 0.25, 0.4, 0.56, 0.25, 0.35, 0.55]
    net["q_capability_curve_table"] = pd.DataFrame(
        {'id_q_capability_curve': [0, 0, 0, 0, 0, 1, 1, 1],
         'p_mw': p_mw,
         'q_min_mvar': q_min_mvar,
         'q_max_mvar': q_max_mvar})

    net.sgen.loc[sgen_indices_with_char, 'id_q_capability_characteristic'] = [0, 1]
    net.sgen.loc[sgen_indices_with_char, 'curve_style'] = "straightLineYValues"
    create_q_capability_characteristics_object(net)

    p_mw_sgen1 = p_mw[:5]
    p_mw_sgen2 = p_mw[5:]
    q_min_mvar_sgen1 = q_min_mvar[:5]
    q_min_mvar_sgen2 = q_min_mvar[5:]
    q_max_mvar_sgen1 = q_max_mvar[:5]
    q_max_mvar_sgen2 = q_max_mvar[5:]

    for i, p1 in enumerate(p_mw_sgen1):
        for j, p2 in enumerate(p_mw_sgen2):
            net.sgen.loc[sgen_indices_with_char, 'p_mw'] = [p1, p2]
            qmin, qmax = get_min_max_q_mvar_from_characteristics_object(net, 'sgen', sgen_indices_with_char)
            assert qmin[1] == q_min_mvar_sgen1[i]
            assert qmin[2] == q_min_mvar_sgen2[j]
            assert qmax[1] == q_max_mvar_sgen1[i]
            assert qmax[2] == q_max_mvar_sgen2[j]

@pytest.mark.parametrize("file_io", (False, True), ids=("Without JSON I/O", "With JSON I/O"))
def test_characteristic(file_io):
    net = create_empty_network()
    c1 = SplineCharacteristic(net, [0, 1, 2], [0, 1, 4], fill_value=(0, 4))
    c2 = SplineCharacteristic(net, [0, 1, 2], [0, 1, 4], interpolator_kind="Pchip", extrapolate=False)
    c3 = SplineCharacteristic(net, [0, 1, 2], [0, 1, 4], interpolator_kind="hello")
    c4 = LogSplineCharacteristic(net, [0,1,2], [0, 1, 4], interpolator_kind="Pchip", extrapolate=False)

    if file_io:
        net_copy = from_json_string(to_json(net))
        c1, c2, c3, c4 = net_copy.characteristic.object.values

    assert np.allclose(c1([-1]), [0], rtol=0, atol=1e-6)
    # assert c1(3) == 4
    # assert c1(1) == 1
    # assert c1(2) == 4
    # assert c1(1.5) == 2.25

    # test that unknown kind causes error:
    with pytest.raises(NotImplementedError):
        c3([0])


def test_log_characteristic_property():
    net = create_empty_network()
    c = LogSplineCharacteristic(net, [10, 1000, 10000], [1000, 0.1, 0.001], interpolator_kind="Pchip", extrapolate=False)
    c._x_vals
    c([2])


def test_geo_accessor_geojson():
    net = create_empty_network()
    b1 = create_bus(net, 10, geodata=(1, 1))
    b2 = create_bus(net, 10, geodata=(2, 2))
    l = create_lines(
        net,
        [b1, b1],
        [b2, b2],
        [1.5, 3],
        std_type="48-AL1/8-ST1A 10.0",
        geodata=[[(1, 1), (2, 2), (3, 3)], [(1, 1), (1, 2)]],
    )
    create_line(net, b1, b2, 1.5, std_type="48-AL1/8-ST1A 10.0")

    assert len(net.line.geo.geojson._coords) == 2
    assert np.array_equal(net.line.geo.geojson._coords.at[l[0]], [[1, 1], [2, 2], [3, 3]])
    assert np.array_equal(net.line.geo.geojson._coords.at[l[1]], [[1, 1], [1, 2]])
    assert np.array_equal(net.bus.geo.geojson._coords.at[b1], [1, 1])
    assert np.array_equal(net.bus.geo.geojson._coords.at[b2], [2, 2])
    assert net.bus.geo.geojson.type.at[b1] == "Point"
    assert net.bus.geo.geojson.type.at[b2] == "Point"
    assert net.line.geo.geojson.type.at[l[0]] == "LineString"
    assert net.line.geo.geojson.type.at[l[1]] == "LineString"
    assert set(net.line.geo.geojson.as_geo_obj.at[l[0]].keys()) == {"coordinates", "type"}
    assert set(net.line.geo.geojson.as_geo_obj.at[l[1]].keys()) == {"coordinates", "type"}
    assert set(net.bus.geo.geojson.as_geo_obj.at[b1].keys()) == {"coordinates", "type"}
    assert set(net.bus.geo.geojson.as_geo_obj.at[b2].keys()) == {"coordinates", "type"}


@pytest.mark.skipif(not GEOPANDAS_INSTALLED, reason="geopandas is not installed")
def test_geo_accessor_geopandas():
    net = mv_oberrhein()
    reference_point = (7.781067, 48.389774)
    radius_m = 2200
    circle_polygon = gpd.GeoSeries([shapely.geometry.Point(reference_point)],
                                   crs=4326).to_crs(epsg=31467).buffer(radius_m).to_crs(epsg=4326).iloc[0]
    assert net.line.geo.geojson.within(circle_polygon).sum() == 11
    assert all(net.line[net.line.geo.geojson.within(circle_polygon)].index == [14, 17, 46, 47, 55, 116,
                                                                               117, 118, 120, 121, 134])

    line = shapely.geometry.LineString([[7.8947079593416, 48.40549007606241],
                                        [7.896048283667894, 48.41060722903666],
                                        [7.896173712216692, 48.41100311474432]])

    assert net.line.geo.geojson.as_shapely_obj.at[0] == line
    assert np.allclose(net.line.geo.geojson.total_bounds, [7.74426069, 48.32845845, 7.93829196, 48.47484423])


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])
