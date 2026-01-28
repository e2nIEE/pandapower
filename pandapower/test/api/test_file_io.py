# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy
import json
import os

import geojson
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from pandapower import pp_dir
from pandapower.auxiliary import pandapowerNet
from pandapower.control import DiscreteTapControl, ConstControl, ContinuousTapControl, Characteristic, \
    SplineCharacteristic
from pandapower.create import create_transformer
from pandapower.convert_format import convert_format
from pandapower.file_io import to_pickle, from_pickle, to_excel, from_excel, from_json, to_json, \
    from_json_string, create_empty_network
from pandapower.io_utils import PPJSONEncoder, PPJSONDecoder
from pandapower.networks import mv_oberrhein, simple_four_bus_system, case9, case14, create_kerber_dorfnetz
from pandapower.run import set_user_pf_options, runpp
from pandapower.sql_io import to_sqlite, from_sqlite
from pandapower.test.helper_functions import assert_net_equal, create_test_network, create_test_network2
from pandapower.timeseries import DFData
from pandapower.toolbox import nets_equal, dataframes_equal
from pandapower.topology.create_graph import create_nxgraph

try:
    import cryptography.fernet # type: ignore

    cryptography_INSTALLED = True
except ImportError:
    cryptography_INSTALLED = False
try:
    import openpyxl

    openpyxl_INSTALLED = True
except ImportError:
    openpyxl_INSTALLED = False
try:
    import xlsxwriter

    xlsxwriter_INSTALLED = True
except ImportError:
    xlsxwriter_INSTALLED = False
try:
    import geopandas as gpd

    GEOPANDAS_INSTALLED = True
except ImportError:
    GEOPANDAS_INSTALLED = False
try:
    import shapely

    SHAPELY_INSTALLED = True
except ImportError:
    SHAPELY_INSTALLED = False


@pytest.fixture(params=[1])
def net_in(request):
    if request.param == 1:
        net = create_test_network()
        net.line.at[0, "geo"] = geojson.dumps(geojson.LineString([(1.1, 2.2), (3.3, 4.4)]))
        net.line.at[1, "geo"] = geojson.dumps(geojson.LineString([(5.5, 5.5), (6.6, 6.6), (7.7, 7.7)]))
        return net

@pytest.fixture()
def net_charactistics():
    return from_json(os.path.join(pp_dir, "test", "test_files", "from_excel_characteristics.json"))


def test_pickle(net_in, tmp_path):
    filename = os.path.abspath(str(tmp_path)) + "testfile.p"
    to_pickle(net_in, filename)
    net_out = from_pickle(filename)
    # pickle sems to changes column types
    assert_net_equal(net_in, net_out)


@pytest.mark.skipif(not xlsxwriter_INSTALLED or not openpyxl_INSTALLED, reason=(
        "xlsxwriter is mandatory to write excel files and openpyxl to read excels, but is not installed."
))
def test_excel(net_in, tmp_path):
    filename = os.path.abspath(str(tmp_path)) + "testfile.xlsx"
    to_excel(net_in, filename)
    net_out = from_excel(filename)
    assert_net_equal(net_in, net_out)

    # test if user_pf_options are equal
    set_user_pf_options(net_in, tolerance_mva=1e3)
    to_excel(net_in, filename)
    net_out = from_excel(filename)
    assert_net_equal(net_in, net_out)
    assert net_out.user_pf_options == net_in.user_pf_options


@pytest.mark.skipif(not xlsxwriter_INSTALLED,
                    reason="xlsxwriter is mandatory to write excel files, but is not installed.")
def test_excel_controllers(net_in, tmp_path):
    filename = os.path.abspath(str(tmp_path)) + "testfile.xlsx"
    DiscreteTapControl(net_in, 0, 0.95, 1.05)
    to_excel(net_in, filename)
    net_out = from_excel(filename)
    assert net_in.controller.object.at[0] == net_out.controller.object.at[0]
    assert_net_equal(net_in, net_out)


@pytest.mark.skipif(not xlsxwriter_INSTALLED,
                    reason="xlsxwriter is mandatory to write excel files, but is not installed.")
def test_excel_characteristics(net_charactistics, tmp_path):
    filename = os.path.abspath(str(tmp_path)) + "testfile.xlsx"
    to_excel(net_charactistics, filename)
    net_out = from_excel(filename)
    pd.testing.assert_frame_equal(net_charactistics['q_capability_characteristic'],
                                  net_out['q_capability_characteristic'], atol=1e-5)
    for minmax in ['q_max_characteristic', 'q_min_characteristic']:
        net_1_ch = net_charactistics['q_capability_characteristic'].loc[0, minmax].to_dict()
        net_2_ch = net_out['q_capability_characteristic'].loc[0, minmax].to_dict()
        for key in net_1_ch:
            if isinstance(net_1_ch[key], list) or isinstance(net_1_ch[key], np.ndarray):
                assert (net_1_ch[key] == net_2_ch[key]).all()
            else:
                assert net_1_ch[key] == net_2_ch[key]


def test_json_basic(net_in, tmp_path):
    # tests the basic json functionality with the encoder/decoder classes
    filename = os.path.abspath(str(tmp_path)) + "testfile.json"
    with open(filename, 'w') as fp:
        json.dump(net_in, fp, cls=PPJSONEncoder)

    with open(filename) as fp:
        net_out = json.load(fp, cls=PPJSONDecoder)
        convert_format(net_out)

    assert_net_equal(net_in, net_out)


def test_json_controller_none():
    try:
        from_json(os.path.join(pp_dir, 'test', 'test_files',
                               'controller_containing_NoneNan.json'), convert=False)
    except:
        raise (UserWarning("empty net with controller containing Nan/None can't be loaded"))


def test_json(net_in, tmp_path):
    filename = os.path.join(os.path.abspath(str(tmp_path)), "testfile.json")

    if GEOPANDAS_INSTALLED and SHAPELY_INSTALLED:
        net_geo = copy.deepcopy(net_in)
        # make GeodataFrame
        from shapely.geometry import shape, Point, LineString
        import geopandas as gpd

        bus_geometry = net_geo.bus["geo"].dropna().apply(geojson.loads).apply(shape)
        net_geo["bus_geodata"] = gpd.GeoDataFrame(geometry=bus_geometry, crs="epsg:4326")
        line_geometry = net_geo.line["geo"].dropna().apply(geojson.loads).apply(shape)
        net_geo["line_geodata"] = gpd.GeoDataFrame(geometry=line_geometry, crs="epsg:4326")

        to_json(net_geo, filename)
        net_out = from_json(filename)
        assert_net_equal(net_geo, net_out)
        assert isinstance(net_out.bus_geodata.geometry.iat[0], Point)
        assert isinstance(net_out.line_geodata.geometry.iat[0], LineString)

    # check if restore_all_dtypes works properly:
    net_in.line['test'] = 123
    net_in.res_line['test'] = 123
    to_json(net_in, filename)
    net_out = from_json(filename)
    assert_net_equal(net_in, net_out)


@pytest.mark.skipif(not cryptography_INSTALLED, reason=("cryptography is mandatory to encrypt "
                                                        "json files, but is not installed."))
def test_encrypted_json(net_in, tmp_path):
    filename = os.path.abspath(str(tmp_path)) + "testfile.json"
    to_json(net_in, filename, encryption_key="verysecret")
    with pytest.raises(UserWarning):
        from_json(filename)
    with pytest.raises(cryptography.fernet.InvalidToken):
        from_json(filename, encryption_key="wrong")
    net_out = from_json(filename, encryption_key="verysecret")
    assert_net_equal(net_in, net_out)


def test_type_casting_json(net_in, tmp_path):
    filename = os.path.abspath(str(tmp_path)) + "testfile.json"
    net_in.sn_kva = 1000
    to_json(net_in, filename)
    net = from_json(filename)
    assert_net_equal(net_in, net)


def test_from_json_add_basic_std_types(tmp_path):
    filename = os.path.abspath(str(tmp_path)) + r"\testfile_std_types.json"
    # load older load network and change std-type
    net = create_test_network2()
    net.std_types["line"]['15-AL1/3-ST1A 0.4']["max_i_ka"] = 111
    num_std_types = sum(len(std) for std in net.std_types.values())

    to_json(net, filename)
    net_updated = from_json(filename, add_basic_std_types=True)

    # check if old std-types didn't change but new ones are added
    assert net.std_types["line"]['15-AL1/3-ST1A 0.4']["max_i_ka"] == 111
    assert sum(len(std) for std in net_updated.std_types.values()) > num_std_types


@pytest.mark.xfail(reason="For std_types, some dtypes are not returned correctly by sql. Therefore,"
                          " a workaround test was created to check everything else.")
def test_sqlite(net_in, tmp_path):
    filename = os.path.abspath(str(tmp_path)) + "testfile.db"
    to_sqlite(net_in, filename)
    net_out = from_sqlite(filename)
    assert_net_equal(net_in, net_out)


def test_sqlite_workaround(net_in, tmp_path):
    filename = os.path.abspath(str(tmp_path)) + "testfile.db"
    to_sqlite(net_in, filename)
    net_out = from_sqlite(filename)
    assert_net_equal(net_in, net_out, exclude_elms=["std_types"])


def test_convert_format():  # TODO what is this thing testing ?
    net = from_pickle(os.path.join(pp_dir, "test", "api", "old_net.p"))
    runpp(net)
    assert net.converged


def test_to_json_dtypes(tmp_path):
    filename = os.path.abspath(str(tmp_path)) + "testfile.json"
    net = create_test_network()
    runpp(net)
    net['res_test'] = pd.DataFrame(columns=['test'], data=[1, 2, 3])
    net['test'] = pd.DataFrame(columns=['test'], data=[1, 2, 3])
    net.line['test'] = 123
    net.res_line['test'] = 123
    net.bus['test'] = 123
    net.res_bus['test'] = 123
    net.res_load['test'] = 123
    to_json(net, filename)
    net1 = from_json(filename)
    assert_net_equal(net, net1)


def test_json_encoding_decoding():
    net = mv_oberrhein()
    net.tuple = (1, "4")
    net.mg = create_nxgraph(net)
    s = {'1', 4}
    t = ('2', 3)
    f = frozenset(['12', 3])
    a = np.array([1., 2.])
    d = {"a": net, "b": f}
    json_string = json.dumps([s, t, f, net, a, d], cls=PPJSONEncoder)
    s1, t1, f1, net1, a1, d1 = json.loads(json_string, cls=PPJSONDecoder)

    assert s == s1
    assert t == t1
    assert f == f1
    assert net.tuple == net1.tuple
    assert np.allclose(a, a1)

    # TODO line_geodata isn't the same since tuples inside DataFrames are converted to lists
    #  (see test_json_tuple_in_dataframe)
    assert nets_equal(net, net1, exclude_elms=["line_geodata", "mg"])
    assert nets_equal(d["a"], d1["a"], exclude_elms=["line_geodata", "mg"])
    assert d["b"] == d1["b"]
    assert_graphs_equal(net.mg, net1.mg)


def test_dataframes_with_integer_columns():
    obj = pd.DataFrame(index=[1, 2, 3], columns=[0, 1])
    json_string = json.dumps(obj, cls=PPJSONEncoder)
    obj_loaded = json.loads(json_string, cls=PPJSONDecoder)
    assert all(obj.columns == obj_loaded.columns)


def assert_graphs_equal(mg1, mg2):
    edge1 = mg1.edges(data=True)
    edge2 = mg2.edges(data=True)
    for (u, v, data), (u1, v1, data1) in zip(sorted(edge1), sorted(edge2)):
        assert u == u1
        assert v == v1
        if "json_id" in data1:
            del data1["json_id"]
        if "json_key" in data1:
            del data1["json_key"]
        assert data == data1


@pytest.mark.xfail
def test_json_tuple_in_pandas():
    s = pd.Series(dtype=object)
    s["test"] = [(1, 2), (3, 4)]
    json_string = json.dumps(s, cls=PPJSONEncoder)
    s1 = json.loads(json_string, cls=PPJSONDecoder)
    assert (type(s["test"][0]) == type(s1["test"][0]))


def test_new_pp_object_io():
    net = mv_oberrhein()
    ds = DFData(pd.DataFrame(data=np.array([[0, 1, 2], [7, 8, 9]])))
    ConstControl(net, 'sgen', 'p_mw', 42, profile_name=0, data_source=ds)
    ContinuousTapControl(net, 142, 1)

    obj = net.controller.object.at[0]
    obj.run = runpp

    s = json.dumps(net, cls=PPJSONEncoder)

    net1 = json.loads(s, cls=PPJSONDecoder)

    obj1 = net1.controller.object.at[0]
    obj2 = net1.controller.object.at[1]

    assert isinstance(obj1, ConstControl)
    assert isinstance(obj2, ContinuousTapControl)
    assert obj1.run is runpp
    assert isinstance(obj1.data_source, DFData)
    assert isinstance(obj1.data_source.df, pd.DataFrame)


def test_convert_format_for_pp_objects(net_in):
    create_transformer(net_in, net_in.bus.index.values[0], net_in.bus.index.values[1],
                       '0.25 MVA 20/0.4 kV', tap_pos=0)
    c1 = ContinuousTapControl(net_in, 0, 1.02)
    c2 = DiscreteTapControl(net_in, 0, 1, 1)
    c1.u_set = 0.98
    c2.u_lower = 0.99
    c2.u_upper = 1.1
    # needed to trigger conversion
    net_in.format_version = "2.1.0"

    net_in.controller = net_in.controller.rename(columns={'object': 'controller'})
    assert 'controller' in net_in.controller.columns

    s = json.dumps(net_in, cls=PPJSONEncoder)
    net1 = from_json_string(s, convert=True)

    assert 'controller' not in net1.controller.columns
    assert 'object' in net1.controller.columns

    obj1 = net1.controller.object.at[0]
    obj2 = net1.controller.object.at[1]

    assert not hasattr(obj1, 'u_set')
    assert not hasattr(obj2, 'u_lower')
    assert not hasattr(obj2, 'u_upper')
    assert obj1.vm_set_pu == 0.98
    assert obj2.vm_lower_pu == 0.99
    assert obj2.vm_upper_pu == 1.1


def test_json_io_same_net(net_in, tmp_path):
    ConstControl(net_in, 'load', 'p_mw', 0)

    s = to_json(net_in)
    net1 = from_json_string(s)
    assert isinstance(net1.controller.object.at[0], ConstControl)

    filename = os.path.abspath(str(tmp_path)) + "testfile.json"
    to_json(net_in, filename)
    net2 = from_json(filename)
    assert isinstance(net2.controller.object.at[0], ConstControl)


def test_json_different_nets():
    net = mv_oberrhein()
    net2 = simple_four_bus_system()
    ContinuousTapControl(net, 114, 1.02)
    net.tuple = (1, "4")
    net.mg = create_nxgraph(net)
    json_string = json.dumps([net, net2], cls=PPJSONEncoder)
    [net_out, net2_out] = json.loads(json_string, cls=PPJSONDecoder)
    assert_net_equal(net_out, net)
    assert_net_equal(net2_out, net2)
    runpp(net_out, run_control=True)
    runpp(net, run_control=True)
    assert_net_equal(net, net_out)


def test_deepcopy_controller():
    net = mv_oberrhein()
    ContinuousTapControl(net, 114, 1.01)
    net2 = copy.deepcopy(net)
    ct1 = net.controller.object.iloc[0]
    ct2 = net2.controller.object.iloc[0]
    assert ct1 is not ct2
    assert ct1 == ct2
    ct2.vm_set_pu = 1.02
    assert ct1 != ct2


def test_elements_to_deserialize(tmp_path):
    net = mv_oberrhein()
    filename = os.path.abspath(str(tmp_path)) + "testfile.json"
    to_json(net, filename)
    net_select = from_json(filename, elements_to_deserialize=['bus', 'load'])
    for key, item in net_select.items():
        if key in ['bus', 'load']:
            assert isinstance(item, pd.DataFrame)
        elif '_empty' in key:
            assert isinstance(item, pd.DataFrame)
        elif '_lookup' in key:
            assert isinstance(item, dict)
        elif key in ['std_types', 'user_pf_options']:
            assert isinstance(item, dict)
        elif '_ppc' in key:
            assert item is None
        elif key == '_is_elements':
            assert item is None
        elif key in ['converged', 'OPF_converged']:
            assert isinstance(item, bool)
        elif key in ['f_hz', 'sn_mva']:
            assert isinstance(item, float)
        else:
            assert isinstance(item, str)
    to_json(net_select, filename)
    net_select = from_json(filename)
    assert net.trafo.equals(net_select.trafo)
    assert_net_equal(net, net_select)


def test_elements_to_deserialize_wo_keep(tmp_path):
    net = mv_oberrhein()
    filename = os.path.abspath(str(tmp_path)) + "testfile.json"
    to_json(net, filename)
    net_select = from_json(filename, elements_to_deserialize=['bus', 'load'],
                           keep_serialized_elements=False)
    for key, item in net_select.items():
        if key in ['bus', 'load']:
            assert isinstance(item, pd.DataFrame)
        elif '_empty' in key:
            assert isinstance(item, pd.DataFrame)
        elif '_lookup' in key:
            assert isinstance(item, dict)
        elif key in ['std_types', 'user_pf_options']:
            assert isinstance(item, dict)
        elif '_ppc' in key:
            assert item is None
        elif key == '_is_elements':
            assert item is None
        elif key in ['converged', 'OPF_converged']:
            assert isinstance(item, bool)
        elif key in ['f_hz', 'sn_mva']:
            assert isinstance(item, float)
        else:
            if isinstance(item, pd.DataFrame):
                assert len(item) == 0
            else:
                assert isinstance(item, str)
    to_json(net_select, filename)
    net_select = from_json(filename)
    assert_net_equal(net, net_select, name_selection=['bus', 'load'])


@pytest.mark.skipif(not GEOPANDAS_INSTALLED, reason="requires the GeoPandas library")
def test_empty_geo_dataframe():
    net = create_empty_network()
    net['bus_geodata'] = pd.DataFrame(columns=['geometry'])
    net['bus_geodata'] = gpd.GeoDataFrame(net['bus_geodata'])
    s = to_json(net)
    net1 = from_json_string(s)
    assert_net_equal(net, net1)


def test_json_io_with_characteristics(net_in):
    c1 = Characteristic.from_points(net_in, [(0, 0), (1, 1)])
    c2 = SplineCharacteristic.from_points(net_in, [(2, 2), (3, 4), (4, 5)])

    net_out = from_json_string(to_json(net_in))
    assert_net_equal(net_in, net_out)
    assert "characteristic" in net_out.keys()
    assert isinstance(net_out.characteristic.object.at[c1.index], Characteristic)
    assert isinstance(net_out.characteristic.object.at[c2.index], SplineCharacteristic)
    assert np.isclose(net_out.characteristic.object.at[c1.index](0.5), c1(0.5), rtol=0, atol=1e-12)
    assert np.isclose(net_out.characteristic.object.at[c2.index](2.5), c2(2.5), rtol=0, atol=1e-12)


def test_replace_elements_json_string(net_in):
    net_orig = copy.deepcopy(net_in)
    ConstControl(net_orig, 'load', 'p_mw', 0)
    json_string = to_json(net_orig)
    net_load = from_json_string(json_string,
                                replace_elements={r'pandapower.control.controller.const_control':
                                                      r'pandapower.test.api.input_files.test_control',
                                                  r'ConstControl': r'MyTestControl'})
    assert net_orig.controller.at[0, 'object'] != net_load.controller.at[0, 'object']
    assert not nets_equal(net_orig, net_load)

    net_load = from_json_string(json_string,
                                replace_elements={r'pandapower.control.controller.const_control':
                                                      r'pandapower.test.api.input_files.test_control'})
    assert net_orig.controller.at[0, 'object'] == net_load.controller.at[0, 'object']
    assert nets_equal(net_orig, net_load)
    runpp(net_load, run_control=True)
    runpp(net_orig, run_control=True)
    assert net_load.controller.loc[0, 'object'].check_word == 'banana'
    assert net_orig.controller.at[0, 'object'] != net_load.controller.at[0, 'object']
    assert not nets_equal(net_orig, net_load)


def test_json_generalized():
    general_net0 = pandapowerNet(pandapowerNet.create_dataframes({
        # structure data
        "df1": {'col1': np.dtype(object),
                'col2': 'f8'},
        "df2": {"col3": 'bool',
                "col4": "i8"}
    }))
    general_net1 = copy.deepcopy(general_net0)
    general_net1.df1.loc[0, ["col1", "col2"]] = ["hey", 1.2]
    general_net1.df2.loc[2, ["col3", "col4"]] = [False, 2]

    for general_in in [general_net0, general_net1]:
        out = from_json_string(to_json(general_in),
                               empty_dict_like_object=pandapowerNet({}))
        assert sorted(out.keys()) == ["df1", "df2"]
        assert nets_equal(out, general_in)


def test_json_simple_index_type():
    s1 = pd.Series([4, 5, 6])
    s2 = pd.Series([4, 5, 6], index=[1, 2, 3])
    s3 = pd.Series([4, 5, 6], index=[1, 2, "3"])
    s4 = pd.Series([4, 5, 6], index=["1", "2", "3"])
    df1 = pd.DataFrame(s1)
    df2 = pd.DataFrame(s2)
    df3 = pd.DataFrame(s3)
    df4 = pd.DataFrame(s4)
    df5, df6, df7, df8 = df1.T, df2.T, df3.T, df4.T
    df9 = pd.DataFrame([[1, 2, 3], [4, 5, 7]], index=[1, "2"], columns=[4, "5", 6])
    json_input = dict(zip("abcdefghijkl", [s1, s2, s3, s4, df1, df2, df3, df4, df5, df6, df7, df8, df9]))
    json_str = to_json(json_input)
    output = from_json_string(json_str, convert=False)
    for key in [*"abcd"]:
        assert_series_equal(json_input[key], output[key], check_dtype=False)
    for key in [*"efghijkl"]:
        assert_frame_equal(json_input[key], output[key], check_dtype=False)


def test_json_index_names():
    net_in = mv_oberrhein()
    net_in.bus.index.name = "bus_index"
    net_in.line.columns.name = "line_column"
    net_in["test_series"] = pd.Series([8], index=pd.Index([2], name="idx_name"))
    json_str = to_json(net_in)
    net_out = from_json_string(json_str)
    assert net_out.bus.index.name == "bus_index"
    assert net_out.line.columns.name == "line_column"
    assert net_out.test_series.index.name == "idx_name"
    assert nets_equal(net_out, net_in)


def test_json_multiindex_and_index_names():
    idx_tuples = tuple(zip([1, 1, 2, 2], ["bar", "baz", "foo", "qux"]))
    col_tuples = tuple(zip(["d", "d", "e"], ["bak", "baq", "fuu"]))
    idx1 = pd.MultiIndex.from_tuples(idx_tuples)
    idx2 = pd.MultiIndex.from_tuples(idx_tuples, names=[5, 6])
    idx3 = pd.MultiIndex.from_tuples(idx_tuples, names=["fifth", "sixth"])
    col1 = pd.MultiIndex.from_tuples(col_tuples)
    col2 = pd.MultiIndex.from_tuples(col_tuples, names=[7, 8])  # ["7", "8"] is not possible since
    # orient="columns" loses info whether index/column is an iteger or a string
    col3 = pd.MultiIndex.from_tuples(col_tuples, names=[7, None])

    for idx, col in zip([idx1, idx2, idx3], [col1, col2, col3]):
        s_mi = pd.Series(range(4), index=idx)
        df_mi = pd.DataFrame(np.arange(4 * 3).reshape((4, 3)), index=idx)
        df_mc = pd.DataFrame(np.arange(4 * 3).reshape((4, 3)), columns=col)
        df_mi_mc = pd.DataFrame(np.arange(4 * 3).reshape((4, 3)), index=idx, columns=col)

        json_input = dict(zip("abcd", [s_mi, df_mi, df_mc, df_mi_mc]))
        json_str = to_json(json_input)
        output = from_json_string(json_str, convert=False)
        assert_series_equal(json_input["a"], output["a"], check_dtype=False)
        assert_frame_equal(json_input["b"], output["b"], check_dtype=False, check_column_type=False)
        assert_frame_equal(json_input["c"], output["c"], check_dtype=False, check_index_type=False)
        assert_frame_equal(json_input["d"], output["d"], check_dtype=False, check_column_type=False,
                           check_index_type=False)


def test_json_dict_of_stuff():
    net1 = case9()
    net2 = case14()
    df = pd.DataFrame([[1, 2, 3], [3, 4, 5]])
    text = "hello world"
    d = {"net1": net1, "net2": net2, "df": df, "text": text}
    s = to_json(d)
    dd = from_json_string(s)
    assert d.keys() == dd.keys()
    assert_net_equal(net1, dd["net1"])
    assert_net_equal(net2, dd["net2"])
    dataframes_equal(df, dd["df"])
    assert text == dd["text"]


def test_json_list_of_stuff():
    net1 = case9()
    net2 = case14()
    df = pd.DataFrame([[1, 2, 3], [3, 4, 5]])
    text = "hello world"
    s = to_json([net1, net2, df, text])
    loaded_list = from_json_string(s)

    assert_net_equal(net1, loaded_list[0])
    assert_net_equal(net2, loaded_list[1])
    dataframes_equal(df, loaded_list[2])
    assert text == loaded_list[3]


def test_multi_index():
    df = pd.DataFrame(columns=["a", "b", "c"], dtype=np.int64)
    df = df.set_index(["a", "b"])
    df2 = from_json_string(to_json(df))
    assert_frame_equal(df, df2)


def test_ignore_unknown_objects():
    net = create_kerber_dorfnetz()
    ContinuousTapControl(net, 0, 1.02)
    json_str = to_json(net)
    net2 = from_json_string(json_str, ignore_unknown_objects=False)

    # in general, reloaded net should be equal to original net
    assert isinstance(net2.controller.object.at[0], ContinuousTapControl)
    assert_net_equal(net, net2)

    # slightly change the class name of the controller so that it cannot be identified
    # by file_io anymore, but can still be loaded as dict if ignore_unknown_objects=True
    json_str2 = json_str.replace("pandapower.control.controller.trafo.ContinuousTapControl",
                                 "pandapower.control.controller.trafo.ContinuousTapControl2")
    with pytest.raises(ModuleNotFoundError):
        from_json_string(json_str2, ignore_unknown_objects=False)
    json_str3 = json_str.replace("\"ContinuousTapControl", "\"ContinuousTapControl2")
    with pytest.raises(AttributeError):
        from_json_string(json_str3, ignore_unknown_objects=False)
    net3 = from_json_string(json_str2, ignore_unknown_objects=True)
    assert isinstance(net3.controller.object.at[0], dict)
    net4 = from_json_string(json_str3, ignore_unknown_objects=True)
    assert isinstance(net4.controller.object.at[0], dict)

    # make sure that the loaded net equals the original net except for the controller
    net3.controller.at[0, "object"] = net.controller.object.at[0]
    net4.controller.at[0, "object"] = net.controller.object.at[0]
    assert_net_equal(net, net3)
    assert_net_equal(net, net4)


def test_omitting_tables_from_json(net_in):
    net = copy.deepcopy(net_in)
    ConstControl(net, 'load', 'p_mw', 0)
    json_string = to_json(net)
    net1 = from_json(json_string, omit_tables=['controller'])
    net2 = from_json(json_string)
    net3 = from_json(json_string, omit_modules=['control.controller'])

    assert(nets_equal(net, net2))
    assert(not nets_equal(net, net1))
    net.controller.drop(0, inplace=True)
    assert(nets_equal(net, net1))
    assert(not nets_equal(net, net3))
    net3.controller.drop(net3.controller.index, inplace=True)
    assert(nets_equal(net, net3))


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
