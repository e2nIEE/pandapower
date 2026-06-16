# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
import geojson
import numpy as np

from pandapower.create import (
    create_bus, create_line_from_parameters, create_line, create_buses, create_lines,
    create_lines_from_parameters, create_bus_dc, create_line_dc, create_lines_dc, create_line_dc_from_parameters,
    create_lines_dc_from_parameters, create_dcline
)
from pandapower.std_types import create_std_type
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network


def test_create_line_conductance():
    net = pandapowerNet(name="test_create_line_conductance")
    create_bus(net, 20)
    create_bus(net, 20)
    create_std_type(
        net,
        {
            "c_nf_per_km": 210,
            "max_i_ka": 0.142,
            "q_mm2": 50,
            "r_ohm_per_km": 0.642,
            "type": "cs",
            "x_ohm_per_km": 0.083,
            "g_us_per_km": 1,
        },
        "test_conductance",
    )

    l = create_line(net, 0, 1, 1.0, "test_conductance", test_kwargs="dummy_string")
    assert net.line.g_us_per_km.at[l] == 1
    assert net.line.test_kwargs.at[l] == "dummy_string"

    validate_network(net)


def test_create_line():
    net = pandapowerNet(name="test_create_line")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)

    # Test basic creation with required parameters
    line_id = create_line(net, b1, b2, length_km=10.0, std_type="48-AL1/8-ST1A 10.0")

    assert line_id == 0
    assert net.line.at[line_id, "from_bus"] == b1
    assert net.line.at[line_id, "to_bus"] == b2
    assert np.isclose(net.line.at[line_id, "length_km"], 10.0)
    assert net.line.at[line_id, "std_type"] == "48-AL1/8-ST1A 10.0"
    assert net.line.at[line_id, "in_service"]
    assert np.isclose(net.line.at[line_id, "df"], 1.0)  # default value
    assert net.line.at[line_id, "parallel"] == 1  # default value

    # Test with all optional parameters
    b3 = create_bus(net, 110)
    b4 = create_bus(net, 110)
    line_id2 = create_line(
        net, b3, b4,
        length_km=5.0,
        std_type="48-AL1/8-ST1A 10.0",
        name="test_line",
        in_service=False,
        df=0.8,
        parallel=2,
        max_loading_percent=100.0,
        geodata=[(1, 1), (2, 2)]
    )

    assert line_id2 == 1
    assert net.line.at[line_id2, "from_bus"] == b3
    assert net.line.at[line_id2, "to_bus"] == b4
    assert np.isclose(net.line.at[line_id2, "length_km"], 5.0)
    assert net.line.at[line_id2, "name"] == "test_line"
    assert not net.line.at[line_id2, "in_service"]
    assert np.isclose(net.line.at[line_id2, "df"], 0.8)
    assert net.line.at[line_id2, "parallel"] == 2
    assert np.isclose(net.line.at[line_id2, "max_loading_percent"], 100.0)

    # Test with custom index
    b5 = create_bus(net, 110)
    b6 = create_bus(net, 110)
    line_id3 = create_line(net, b5, b6, length_km=3.0, std_type="48-AL1/8-ST1A 10.0", index=5)

    assert line_id3 == 5

    # Test with kwargs
    b7 = create_bus(net, 110)
    b8 = create_bus(net, 110)
    line_id4 = create_line(
        net, b7, b8,
        length_km=2.0,
        std_type="48-AL1/8-ST1A 10.0",
        test_custom_attr="custom_value"
    )

    assert net.line.at[line_id4, "test_custom_attr"] == "custom_value"

    validate_network(net)


def test_create_lines():
    # standard
    net = pandapowerNet(name="test_create_lines0")
    b1 = create_bus(net, 10)
    b2 = create_bus(net, 10)
    create_lines(
        net,
        [b1, b1],
        [b2, b2],
        4,
        std_type="48-AL1/8-ST1A 10.0",
        test_kwargs="dummy_string",
    )
    assert len(net.line) == 2
    assert sum(net.line.std_type == "48-AL1/8-ST1A 10.0") == 2
    assert len(set(net.line.r_ohm_per_km)) == 1
    assert all(net.line.test_kwargs == "dummy_string")

    validate_network(net)

    net = pandapowerNet(name="test_create_lines1")
    b1 = create_bus(net, 10)
    b2 = create_bus(net, 10)
    create_lines(
        net,
        [b1, b1],
        [b2, b2],
        4,
        std_type=["48-AL1/8-ST1A 10.0", "NA2XS2Y 1x240 RM/25 6/10 kV"],
    )
    assert len(net.line) == 2
    assert sum(net.line.std_type == "48-AL1/8-ST1A 10.0") == 1
    assert sum(net.line.std_type == "NA2XS2Y 1x240 RM/25 6/10 kV") == 1

    validate_network(net)

    # with geodata
    net = pandapowerNet(name="test_create_lines2")
    b1 = create_bus(net, 10)
    b2 = create_bus(net, 10)
    l = create_lines(
        net,
        [b1, b1],
        [b2, b2],
        [1.5, 3],
        std_type="48-AL1/8-ST1A 10.0",
        geodata=[[(1, 1), (2, 2), (3, 3)], [(1, 1), (1, 2)]],
    )

    assert len(net.line) == 2
    assert net.line.at[l[0], "geo"] == geojson.dumps(geojson.LineString([(1, 1), (2, 2), (3, 3)]), sort_keys=True)
    assert net.line.at[l[1], "geo"] == geojson.dumps(geojson.LineString([(1, 1), (1, 2)]), sort_keys=True)

    validate_network(net)

    # setting params as single value
    net = pandapowerNet(name="test_create_lines3")
    b1 = create_bus(net, 10)
    b2 = create_bus(net, 10)
    l = create_lines(
        net,
        [b1, b1],
        [b2, b2],
        length_km=5,
        df=0.8,
        in_service=False,
        geodata=[(10, 10), (20, 20)],
        parallel=1,
        max_loading_percent=90,
        name="test",
        std_type="48-AL1/8-ST1A 10.0",
    )

    assert len(net.line) == 2
    assert net.line.length_km.at[l[0]] == 5
    assert net.line.length_km.at[l[1]] == 5
    assert net.line.in_service.dtype == bool
    assert not net.line.at[l[0], "in_service"]  # is actually <class 'numpy.bool_'>
    assert not net.line.at[l[1], "in_service"]  # is actually <class 'numpy.bool_'>
    assert net.line.at[l[0], "geo"] == geojson.dumps(geojson.LineString([(10, 10), (20, 20)]), sort_keys=True)
    assert net.line.at[l[1], "geo"] == geojson.dumps(geojson.LineString([(10, 10), (20, 20)]), sort_keys=True)
    assert net.line.at[l[0], "name"] == "test"
    assert net.line.at[l[1], "name"] == "test"
    assert net.line.at[l[0], "max_loading_percent"] == 90
    assert net.line.at[l[1], "max_loading_percent"] == 90
    assert net.line.at[l[0], "parallel"] == 1
    assert net.line.at[l[1], "parallel"] == 1

    validate_network(net)

    # setting params as array
    net = pandapowerNet(name="test_create_lines4")
    b1 = create_bus(net, 10)
    b2 = create_bus(net, 10)
    l = create_lines(
        net,
        [b1, b1],
        [b2, b2],
        length_km=[1, 5],
        df=[0.8, 0.7],
        in_service=[True, False],
        geodata=[[(10, 10), (20, 20)], [(100, 10), (200, 20)]],
        parallel=[2, 1],
        max_loading_percent=[80, 90],
        name=["test1", "test2"],
        std_type="48-AL1/8-ST1A 10.0",
    )

    assert len(net.line) == 2
    assert net.line.at[l[0], "length_km"] == 1
    assert net.line.at[l[1], "length_km"] == 5
    assert net.line.in_service.dtype == bool
    assert net.line.at[l[0], "in_service"]  # is actually <class 'numpy.bool_'>
    assert not net.line.at[l[1], "in_service"]  # is actually <class 'numpy.bool_'>
    assert net.line.at[l[0], "geo"] == geojson.dumps(geojson.LineString([(10, 10), (20, 20)]), sort_keys=True)
    assert net.line.at[l[1], "geo"] == geojson.dumps(geojson.LineString([(100, 10), (200, 20)]), sort_keys=True)
    assert net.line.at[l[0], "name"] == "test1"
    assert net.line.at[l[1], "name"] == "test2"
    assert net.line.at[l[0], "max_loading_percent"] == 80
    assert net.line.at[l[1], "max_loading_percent"] == 90
    assert net.line.at[l[0], "parallel"] == 2
    assert net.line.at[l[1], "parallel"] == 1

    validate_network(net)


def test_create_line_form_parameters():
    net = pandapowerNet(name="test_create_line_form_parameters")
    b1 = create_bus(net, 110)
    b2 = create_bus(net, 110)

    # Test basic creation with required parameters
    line_id = create_line_from_parameters(
        net,
        b1, b2,
        length_km=10.0,
        r_ohm_per_km=0.1,
        x_ohm_per_km=0.05,
        c_nf_per_km=10.0,
        max_i_ka=0.5
    )

    assert line_id == 0
    assert net.line.at[line_id, "from_bus"] == b1
    assert net.line.at[line_id, "to_bus"] == b2
    assert np.allclose(net.line.at[line_id, "length_km"], 10.0)
    assert np.allclose(net.line.at[line_id, "r_ohm_per_km"], 0.1)
    assert np.allclose(net.line.at[line_id, "x_ohm_per_km"], 0.05)
    assert np.allclose(net.line.at[line_id, "c_nf_per_km"], 10.0)
    assert np.allclose(net.line.at[line_id, "max_i_ka"], 0.5)
    assert net.line.at[line_id, "in_service"]
    assert np.allclose(net.line.at[line_id, "df"], 1.0)
    assert net.line.at[line_id, "parallel"] == 1

    # Test with all optional parameters

    # tdpf kwargs:
    tdpf_args = {
        "endtemp_degree": 100.0,
        "tdpf": True,
        "wind_speed_m_per_s": 2.,
        "wind_angle_degree": 3.,
        "conductor_outer_diameter_m": 4.,
        "air_temperature_degree_celsius": 5.,
        "reference_temperature_degree_celsius": 6.,
        "solar_radiation_w_per_sq_m": 7.,
        "solar_absorptivity": 8.,
        "emissivity": 9.,
        "r_theta_kelvin_per_mw": 10.,
        "mc_joule_per_m_k": 11.,
    }
    b3 = create_bus(net, 110)
    b4 = create_bus(net, 110)
    line_id2 = create_line_from_parameters(
        net, b3, b4,
        length_km=5.0,
        r_ohm_per_km=0.2,
        x_ohm_per_km=0.1,
        c_nf_per_km=20.0,
        max_i_ka=0.3,
        name="test_line_params",
        in_service=False,
        df=0.8,
        parallel=2,
        type="cs",
        g_us_per_km=0.0,
        max_loading_percent=100.0,
        alpha=0.004,
        temperature_degree_celsius=80,
        r0_ohm_per_km=0.15,
        x0_ohm_per_km=0.08,
        c0_nf_per_km=5.0,
        g0_us_per_km=0.0,
        **tdpf_args
    )

    validate_network(net)

    assert line_id2 == 1
    assert net.line.at[line_id2, "from_bus"] == b3
    assert net.line.at[line_id2, "to_bus"] == b4
    assert np.isclose(net.line.at[line_id2, "r_ohm_per_km"], 0.2)
    assert np.isclose(net.line.at[line_id2, "x_ohm_per_km"], 0.1)
    assert np.isclose(net.line.at[line_id2, "c_nf_per_km"], 20.0)
    assert np.isclose(net.line.at[line_id2, "max_i_ka"], 0.3)
    assert net.line.at[line_id2, "name"] == "test_line_params"
    assert not net.line.at[line_id2, "in_service"]
    assert np.isclose(net.line.at[line_id2, "df"], 0.8)
    assert net.line.at[line_id2, "parallel"] == 2
    assert net.line.at[line_id2, "type"] == "cs"
    assert np.isclose(net.line.at[line_id2, "g_us_per_km"], 0.0)
    assert np.isclose(net.line.at[line_id2, "max_loading_percent"], 100.0)
    assert np.isclose(net.line.at[line_id2, "alpha"], 0.004)
    assert np.isclose(net.line.at[line_id2, "temperature_degree_celsius"], 80.0)
    assert np.isclose(net.line.at[line_id2, "r0_ohm_per_km"], 0.15)
    assert np.isclose(net.line.at[line_id2, "x0_ohm_per_km"], 0.08)
    assert np.isclose(net.line.at[line_id2, "c0_nf_per_km"], 5.0)
    assert np.isclose(net.line.at[line_id2, "g0_us_per_km"], 0.0)

    # Test with geodata
    b5 = create_bus(net, 110)
    b6 = create_bus(net, 110)
    line_id3 = create_line_from_parameters(
        net, b5, b6,
        length_km=3.0,
        r_ohm_per_km=0.3,
        x_ohm_per_km=0.15,
        c_nf_per_km=5.0,
        max_i_ka=0.2,
        geodata=[(1, 1), (2, 2), (3, 3)]
    )

    assert line_id3 == 2
    assert "geo" in net.line.columns

    # Test with custom kwargs
    b7 = create_bus(net, 110)
    b8 = create_bus(net, 110)
    line_id4 = create_line_from_parameters(
        net, b7, b8,
        length_km=2.0,
        r_ohm_per_km=0.4,
        x_ohm_per_km=0.2,
        c_nf_per_km=2.0,
        max_i_ka=0.1,
        custom_attr="test_value"
    )

    assert net.line.at[line_id4, "custom_attr"] == "test_value"

    validate_network(net)


def test_create_lines_from_parameters():
    # standard
    net = pandapowerNet(name="test_create_lines_from_parameters0")
    b1 = create_bus(net, 10)
    b2 = create_bus(net, 10)
    l = create_lines_from_parameters(
        net,
        [b1, b1],
        [b2, b2],
        length_km=[10.0, 5.0],
        x_ohm_per_km=[1.0, 1.0],
        r_ohm_per_km=[0.2, 0.2],
        c_nf_per_km=[0, 0],
        max_i_ka=[100, 100],
        test_kwargs=["dummy_string", "dummy_string"],
    )
    assert len(net.line) == 2
    assert len(net.line.x_ohm_per_km) == 2
    assert len(net.line.r_ohm_per_km) == 2
    assert len(net.line.c_nf_per_km) == 2
    assert len(net.line.max_i_ka) == 2
    assert len(net.line.df) == 2
    assert net.line.test_kwargs.at[l[0]] == "dummy_string"

    # with geodata
    net = pandapowerNet(name="test_create_lines_from_parameters1")
    b1 = create_bus(net, 10)
    b2 = create_bus(net, 10)
    l = create_lines_from_parameters(
        net,
        [b1, b1],
        [b2, b2],
        length_km=[10.0, 5.0],
        x_ohm_per_km=[1.0, 1.0],
        r_ohm_per_km=[0.2, 0.2],
        c_nf_per_km=[0, 0],
        max_i_ka=[100, 100],
        geodata=[[(1, 1), (2, 2), (3, 3)], [(1, 1), (1, 2)]],
    )

    assert len(net.line) == 2
    assert net.line.at[l[0], "geo"] == geojson.dumps(geojson.LineString([(1, 1), (2, 2), (3, 3)]), sort_keys=True)
    assert net.line.at[l[1], "geo"] == geojson.dumps(geojson.LineString([(1, 1), (1, 2)]), sort_keys=True)

    validate_network(net)

    # setting params as single value
    net = pandapowerNet(name="test_create_lines_from_parameters2")
    b1 = create_bus(net, 10)
    b2 = create_bus(net, 10)
    l = create_lines_from_parameters(
        net,
        [b1, b1],
        [b2, b2],
        length_km=5,
        x_ohm_per_km=1,
        r_ohm_per_km=0.2,
        c_nf_per_km=0,
        max_i_ka=100,
        df=0.8,
        in_service=False,
        geodata=[(10., 10.), (20., 20.)],
        parallel=1,
        max_loading_percent=90,
        name="test",
        r0_ohm_per_km=0.1,
        g0_us_per_km=0.0,
        c0_nf_per_km=0.0,
        temperature_degree_celsius=20,
        alpha=0.04,
        test_kwargs="dummy_string",
    )

    assert len(net.line) == 2
    assert all(net.line["length_km"].values == 5)
    assert all(net.line["x_ohm_per_km"].values == 1)
    assert np.allclose(net.line["r_ohm_per_km"], 0.2)
    assert np.allclose(net.line["r0_ohm_per_km"], 0.1)
    assert all(net.line["g0_us_per_km"].values == 0)
    assert all(net.line["c0_nf_per_km"].values == 0)
    assert net.line.in_service.dtype == np.dtype(bool)
    assert not net.line.at[l[0], "in_service"]  # is actually <class 'numpy.bool_'>
    assert not net.line.at[l[1], "in_service"]  # is actually <class 'numpy.bool_'>
    assert net.line.at[l[0], "geo"] == geojson.dumps(geojson.LineString([(10., 10.), (20., 20.)]), sort_keys=True)
    assert net.line.at[l[1], "geo"] == geojson.dumps(geojson.LineString([(10., 10.), (20., 20.)]), sort_keys=True)
    assert all(net.line["name"].values == "test")
    assert all(net.line["max_loading_percent"].values == 90)
    assert all(net.line["parallel"].values == 1)
    assert np.allclose(net.line["temperature_degree_celsius"], 20.0)
    assert np.allclose(net.line["alpha"], 0.04)
    assert all(net.line.test_kwargs == "dummy_string")

    validate_network(net)

    # setting params as array
    net = pandapowerNet(name="test_create_lines_from_parameters3")
    b1 = create_bus(net, 10)
    b2 = create_bus(net, 10)
    l = create_lines_from_parameters(
        net,
        [b1, b1],
        [b2, b2],
        length_km=[1, 5],
        r_ohm_per_km=[1, 2],
        x_ohm_per_km=[0.3, 0.5],
        c_nf_per_km=[0.0, 0.1],
        r0_ohm_per_km=[0.1, 0.15],
        x0_ohm_per_km=[0.2, 0.25],
        g0_us_per_km=[0.0, 0.0],
        c0_nf_per_km=[0.0, 0.0],
        df=[0.8, 0.7],
        in_service=[True, False],
        geodata=[[(10, 10), (20, 20)], [(100, 10), (200, 20)]],
        parallel=[2, 1],
        max_loading_percent=[80, 90],
        name=["test1", "test2"],
        max_i_ka=[100, 200],
    )

    assert len(net.line) == 2
    assert net.line.at[l[0], "length_km"] == 1
    assert net.line.at[l[1], "length_km"] == 5
    assert net.line.at[l[0], "r_ohm_per_km"] == 1
    assert net.line.at[l[1], "r_ohm_per_km"] == 2
    assert np.isclose(net.line.at[l[0], "x_ohm_per_km"], 0.3)
    assert np.isclose(net.line.at[l[1], "x_ohm_per_km"], 0.5)
    assert np.isclose(net.line.at[l[0], "c_nf_per_km"], 0.0)
    assert np.isclose(net.line.at[l[1], "c_nf_per_km"], 0.1)
    assert np.isclose(net.line.at[l[0], "r0_ohm_per_km"], 0.1)
    assert np.isclose(net.line.at[l[1], "r0_ohm_per_km"], 0.15)
    assert np.isclose(net.line.at[l[0], "x0_ohm_per_km"], 0.2)
    assert np.isclose(net.line.at[l[1], "x0_ohm_per_km"], 0.25)
    assert all(net.line["g0_us_per_km"].values == 0)
    assert all(net.line["c0_nf_per_km"].values == 0)
    assert net.line.in_service.dtype == np.dtype(bool)
    assert net.line.at[l[0], "in_service"]
    assert not net.line.at[l[1], "in_service"]
    assert net.line.at[l[0], "name"] == "test1"
    assert net.line.at[l[1], "name"] == "test2"
    assert net.line.at[l[0], "geo"] == geojson.dumps(geojson.LineString([(10, 10), (20, 20)]), sort_keys=True)
    assert net.line.at[l[1], "geo"] == geojson.dumps(geojson.LineString([(100, 10), (200, 20)]), sort_keys=True)
    assert net.line.at[l[0], "max_loading_percent"] == 80
    assert net.line.at[l[1], "max_loading_percent"] == 90
    assert net.line.at[l[0], "parallel"] == 2
    assert net.line.at[l[1], "parallel"] == 1
    assert net.line.at[l[0], "max_i_ka"] == 100
    assert net.line.at[l[1], "max_i_ka"] == 200

    validate_network(net)


def test_create_lines_raise_errorexcept():
    # standard
    net = pandapowerNet(name="test_create_lines_raise_errorexcept")
    b1 = create_bus(net, 10)
    b2 = create_bus(net, 10)
    create_lines_from_parameters(
        net,
        [b1, b1],
        [b2, b2],
        length_km=[10.0, 5.0],
        x_ohm_per_km=[1.0, 1.0],
        r_ohm_per_km=[0.2, 0.2],
        c_nf_per_km=[0, 0],
        max_i_ka=[100, 100],
    )

    with pytest.raises(UserWarning, match="Lines trying to attach .*"):
        create_lines_from_parameters(
            net,
            [b1, 2],
            [2, b2],
            length_km=[10.0, 5.0],
            x_ohm_per_km=[1.0, 1.0],
            r_ohm_per_km=[0.2, 0.2],
            c_nf_per_km=[0, 0],
            max_i_ka=[100, 100],
        )
    with pytest.raises(UserWarning, match="Lines with indexes .*"):
        create_lines_from_parameters(
            net,
            [b1, b1],
            [b2, b2],
            index=[0, 1],
            length_km=[10.0, 5.0],
            x_ohm_per_km=[1.0, 1.0],
            r_ohm_per_km=[0.2, 0.2],
            c_nf_per_km=[0, 0],
            max_i_ka=[100, 100],
        )

    with pytest.raises(UserWarning, match="Passed indexes"):
        create_lines_from_parameters(
            net,
            [b1, b1],
            [b2, b2],
            index=[2, 2],
            length_km=[10.0, 5.0],
            x_ohm_per_km=[1.0, 1.0],
            r_ohm_per_km=[0.2, 0.2],
            c_nf_per_km=[0, 0],
            max_i_ka=[100, 100],
        )

    validate_network(net)


def test_create_lines_optional_columns():
    #
    net = pandapowerNet(name="test_create_lines_optional_columns")
    create_buses(net, 5, 110)
    create_line(net, 0, 1, 10, "48-AL1/8-ST1A 10.0")
    create_line_from_parameters(net, 3, 4, 10, 1, 1, 1, 100)
    create_lines(net, [0, 1], [1, 0], 10, "48-AL1/8-ST1A 10.0")
    create_lines_from_parameters(net, [3, 4], [4, 3], [10, 11], 1, 1, 1, 100)
    assert "max_loading_percent" not in net.line.columns

    v = np.nan
    create_line(net, 0, 1, 10, "48-AL1/8-ST1A 10.0", max_loading_percent=v)
    create_line_from_parameters(net, 3, 4, 10, 1, 1, 1, 100, max_loading_percent=v)
    # np.nan is not None:
    # create_lines(net, [0, 1], [1, 0], 10, "48-AL1/8-ST1A 10.0", max_loading_percent=v)
    # create_lines(net, [0, 1], [1, 0], 10, "48-AL1/8-ST1A 10.0", max_loading_percent=[v, v])  # would be added
    # create_lines_from_parameters(net, [3, 4], [4, 3], [10, 11], 1, 1, 1, 100, max_loading_percent=v)
    # create_lines_from_parameters(net, [3, 4], [4, 3], [10, 11], 1, 1, 1, 100, max_loading_percent=[v, v])
    assert "max_loading_percent" not in net.line.columns

    validate_network(net)


def test_create_line_alpha_temperature():
    net = pandapowerNet(name="test_create_line_alpha_temperature")
    create_buses(net, 5, 110)

    l1 = create_line(net, 0, 1, 10, "48-AL1/8-ST1A 10.0")
    l2 = create_line(
        net,
        1,
        2,
        10,
        "48-AL1/8-ST1A 10.0",
        alpha=4.03e-3,
        temperature_degree_celsius=80,
    )
    l3 = create_line(net, 2, 3, 10, "48-AL1/8-ST1A 10.0")
    l4 = create_line_from_parameters(net, 3, 4, 10, 1, 1, 1, 100)
    l5 = create_line_from_parameters(net, 3, 4, 10, 1, 1, 1, 100, alpha=4.03e-3)

    assert "alpha" in net.line.columns
    assert np.allclose(net.line.loc[[l2, l3, l5], "alpha"], 4.03e-3)
    assert all(net.line.loc[[l1, l4], "alpha"].isnull())
    assert net.line.loc[l2, "temperature_degree_celsius"] == 80
    assert all(net.line.loc[[l1, l3, l4, l5], "temperature_degree_celsius"].isnull())

    # make sure optional columns are not created if None or np.nan:
    create_line(net, 2, 3, 10, "48-AL1/8-ST1A 10.0", wind_speed_m_per_s=None)
    create_lines(net, [2], [3], 10, "48-AL1/8-ST1A 10.0", wind_speed_m_per_s=None)
    create_line_from_parameters(net, 3, 4, 10, 1, 1, 1, 100, wind_speed_m_per_s=None)
    create_line_from_parameters(net, 3, 4, 10, 1, 1, 1, 100, alpha=4.03e-3, wind_speed_m_per_s=np.nan)
    assert "wind_speed_m_per_s" not in net.line.columns

    validate_network(net)


def test_create_line_dc():
    net = pandapowerNet(name="test_create_line_dc")
    b1 = create_bus_dc(net, 110)
    b2 = create_bus_dc(net, 110)

    # Test basic creation with required parameters
    line_id = create_line_dc(net, b1, b2, length_km=10.0, std_type="95-CU")

    assert line_id == 0
    assert net.line_dc.at[line_id, "from_bus_dc"] == b1
    assert net.line_dc.at[line_id, "to_bus_dc"] == b2
    assert np.isclose(net.line_dc.at[line_id, "length_km"], 10.0)
    assert net.line_dc.at[line_id, "std_type"] == "95-CU"
    assert net.line_dc.at[line_id, "in_service"]
    assert np.isclose(net.line_dc.at[line_id, "df"], 1.0)  # default value
    assert net.line_dc.at[line_id, "parallel"] == 1  # default value

    # Test with all optional parameters
    b3 = create_bus(net, 110, bus_type="dc")
    b4 = create_bus(net, 110, bus_type="dc")
    line_id2 = create_line_dc(
        net, b3, b4,
        length_km=5.0,
        std_type="95-CU",
        name="test_line_dc",
        in_service=False,
        df=0.8,
        parallel=2,
        max_loading_percent=100.0,
        geodata=[(1, 1), (2, 2)]
    )

    assert line_id2 == 1
    assert net.line_dc.at[line_id2, "from_bus_dc"] == b3
    assert net.line_dc.at[line_id2, "to_bus_dc"] == b4
    assert np.isclose(net.line_dc.at[line_id2, "length_km"], 5.0)
    assert net.line_dc.at[line_id2, "name"] == "test_line_dc"
    assert not net.line_dc.at[line_id2, "in_service"]
    assert np.isclose(net.line_dc.at[line_id2, "df"], 0.8)
    assert net.line_dc.at[line_id2, "parallel"] == 2
    assert np.isclose(net.line_dc.at[line_id2, "max_loading_percent"], 100.0)

    validate_network(net)


def test_create_lines_dc():
    net = pandapowerNet(name="test_create_lines_dc0")
    b1 = create_bus_dc(net, 110)
    b2 = create_bus_dc(net, 110)
    b3 = create_bus_dc(net, 110)
    b4 = create_bus_dc(net, 110)

    # Test basic creation
    line_ids = create_lines_dc(
        net,
        from_buses_dc=[b1, b3],
        to_buses_dc=[b2, b4],
        length_km=10.0,
        std_type="95-CU"
    )

    assert len(line_ids) == 2
    assert line_ids[0] == 0
    assert line_ids[1] == 1
    assert net.line_dc.at[0, "from_bus_dc"] == b1
    assert net.line_dc.at[0, "to_bus_dc"] == b2
    assert net.line_dc.at[1, "from_bus_dc"] == b3
    assert net.line_dc.at[1, "to_bus_dc"] == b4

    # Test with different lengths
    net2 = pandapowerNet(name="test_create_lines_dc1")
    b1 = create_bus_dc(net2, 110)
    b2 = create_bus_dc(net2, 110)
    b3 = create_bus_dc(net2, 110)
    b4 = create_bus_dc(net2, 110)

    create_lines_dc(
        net2,
        from_buses_dc=[b1, b3],
        to_buses_dc=[b2, b4],
        length_km=[5.0, 15.0],
        std_type="95-CU",
        name=["line_dc_1", "line_dc_2"],
        in_service=[True, False],
        df=0.9
    )

    assert np.isclose(net2.line_dc.at[0, "length_km"], 5.0)
    assert np.isclose(net2.line_dc.at[1, "length_km"], 15.0)
    assert net2.line_dc.at[0, "name"] == "line_dc_1"
    assert net2.line_dc.at[1, "name"] == "line_dc_2"
    assert net2.line_dc.at[0, "in_service"]
    assert not net2.line_dc.at[1, "in_service"]

    validate_network(net)
    validate_network(net2)


def test_create_line_dc_from_parameters():
    net = pandapowerNet(name="test_create_line_dc_from_parameters")
    b1 = create_bus_dc(net, 110)
    b2 = create_bus_dc(net, 110)

    # Test basic creation with required parameters
    line_id = create_line_dc_from_parameters(
        net,
        b1, b2,
        length_km=10.0,
        r_ohm_per_km=0.1,
        max_i_ka=0.5
    )

    assert line_id == 0
    assert net.line_dc.at[line_id, "from_bus_dc"] == b1
    assert net.line_dc.at[line_id, "to_bus_dc"] == b2
    assert np.isclose(net.line_dc.at[line_id, "length_km"], 10.0)
    assert np.isclose(net.line_dc.at[line_id, "r_ohm_per_km"], 0.1)
    assert np.isclose(net.line_dc.at[line_id, "max_i_ka"], 0.5)
    assert net.line_dc.at[line_id, "in_service"]
    assert np.isclose(net.line_dc.at[line_id, "df"], 1.0)
    assert net.line_dc.at[line_id, "parallel"] == 1

    # Test with all optional parameters
    b3 = create_bus_dc(net, 110)
    b4 = create_bus_dc(net, 110)
    line_id2 = create_line_dc_from_parameters(
        net, b3, b4,
        length_km=5.0,
        r_ohm_per_km=0.2,
        max_i_ka=0.3,
        name="test_line_dc_params",
        in_service=False,
        df=0.8,
        parallel=2,
        type="ol",
        max_loading_percent=100.0,
        g_us_per_km=0.0
    )

    assert line_id2 == 1
    assert net.line_dc.at[line_id2, "from_bus_dc"] == b3
    assert net.line_dc.at[line_id2, "to_bus_dc"] == b4
    assert np.isclose(net.line_dc.at[line_id2, "r_ohm_per_km"], 0.2)
    assert np.isclose(net.line_dc.at[line_id2, "max_i_ka"], 0.3)
    assert net.line_dc.at[line_id2, "name"] == "test_line_dc_params"
    assert not net.line_dc.at[line_id2, "in_service"]
    assert np.isclose(net.line_dc.at[line_id2, "df"], 0.8)
    assert net.line_dc.at[line_id2, "parallel"] == 2
    assert net.line_dc.at[line_id2, "type"] == "ol"

    validate_network(net)


def test_create_lines_dc_from_parameters():
    net = pandapowerNet(name="test_create_lines_dc_from_parameters0")
    b1 = create_bus_dc(net, 110)
    b2 = create_bus_dc(net, 110)
    b3 = create_bus_dc(net, 110)
    b4 = create_bus_dc(net, 110)

    # Test basic creation
    line_ids = create_lines_dc_from_parameters(
        net,
        from_buses_dc=[b1, b3],
        to_buses_dc=[b2, b4],
        length_km=10.0,
        r_ohm_per_km=0.1,
        max_i_ka=0.5
    )

    assert len(line_ids) == 2
    assert line_ids[0] == 0
    assert line_ids[1] == 1
    assert net.line_dc.at[0, "from_bus_dc"] == b1
    assert net.line_dc.at[0, "to_bus_dc"] == b2
    assert np.isclose(net.line_dc.at[0, "r_ohm_per_km"], 0.1)
    assert net.line_dc.at[1, "from_bus_dc"] == b3
    assert net.line_dc.at[1, "to_bus_dc"] == b4

    # Test with array parameters
    net2 = pandapowerNet(name="test_create_lines_dc_from_parameters1")
    b1 = create_bus_dc(net2, 110)
    b2 = create_bus_dc(net2, 110)
    b3 = create_bus_dc(net2, 110)
    b4 = create_bus_dc(net2, 110)

    create_lines_dc_from_parameters(
        net2,
        from_buses_dc=[b1, b3],
        to_buses_dc=[b2, b4],
        length_km=[5.0, 15.0],
        r_ohm_per_km=[0.1, 0.2],
        max_i_ka=[0.3, 0.4],
        name=["line_dc_1", "line_dc_2"],
        in_service=[True, False],
        df=[0.9, 0.95],
        parallel=[1, 2]
    )

    assert np.isclose(net2.line_dc.at[0, "length_km"], 5.0)
    assert np.isclose(net2.line_dc.at[1, "length_km"], 15.0)
    assert np.isclose(net2.line_dc.at[0, "r_ohm_per_km"], 0.1)
    assert np.isclose(net2.line_dc.at[1, "r_ohm_per_km"], 0.2)
    assert np.isclose(net2.line_dc.at[0, "max_i_ka"], 0.3)
    assert np.isclose(net2.line_dc.at[1, "max_i_ka"], 0.4)
    assert net2.line_dc.at[0, "name"] == "line_dc_1"
    assert net2.line_dc.at[1, "name"] == "line_dc_2"
    assert net2.line_dc.at[0, "in_service"]
    assert not net2.line_dc.at[1, "in_service"]
    assert net2.line_dc.at[0, "parallel"] == 1
    assert net2.line_dc.at[1, "parallel"] == 2

    validate_network(net)
    validate_network(net2)


def test_create_dcline():
    net = pandapowerNet(name="test_create_dcline")
    b1 = create_bus(net, 380)
    b2 = create_bus(net, 110)

    # Test basic creation with required parameters
    dcline_id = create_dcline(
        net,
        from_bus=b1,
        to_bus=b2,
        p_mw=1000,
        loss_percent=2.0,
        loss_mw=20,
        vm_from_pu=1.0,
        vm_to_pu=1.0
    )

    assert dcline_id == 0
    assert net.dcline.at[dcline_id, "from_bus"] == b1
    assert net.dcline.at[dcline_id, "to_bus"] == b2
    assert net.dcline.at[dcline_id, "p_mw"] == 1000
    assert np.isclose(net.dcline.at[dcline_id, "loss_percent"], 2.0)
    assert net.dcline.at[dcline_id, "loss_mw"] == 20
    assert np.isclose(net.dcline.at[dcline_id, "vm_from_pu"], 1.0)
    assert np.isclose(net.dcline.at[dcline_id, "vm_to_pu"], 1.0)
    assert net.dcline.at[dcline_id, "in_service"]

    # Test with all optional parameters
    b3 = create_bus(net, 380)
    b4 = create_bus(net, 110)
    dcline_id2 = create_dcline(
        net,
        from_bus=b3,
        to_bus=b4,
        p_mw=2000,
        loss_percent=1.5,
        loss_mw=30,
        vm_from_pu=1.02,
        vm_to_pu=0.98,
        name="test_dcline",
        in_service=False,
        max_p_mw=3000,
        min_p_mw=500,
        min_q_from_mvar=-100,
        max_q_from_mvar=100,
        min_q_to_mvar=-50,
        max_q_to_mvar=50
    )

    assert dcline_id2 == 1
    assert net.dcline.at[dcline_id2, "from_bus"] == b3
    assert net.dcline.at[dcline_id2, "to_bus"] == b4
    assert net.dcline.at[dcline_id2, "p_mw"] == 2000
    assert np.isclose(net.dcline.at[dcline_id2, "loss_percent"], 1.5)
    assert net.dcline.at[dcline_id2, "loss_mw"] == 30
    assert np.isclose(net.dcline.at[dcline_id2, "vm_from_pu"], 1.02)
    assert np.isclose(net.dcline.at[dcline_id2, "vm_to_pu"], 0.98)
    assert net.dcline.at[dcline_id2, "name"] == "test_dcline"
    assert not net.dcline.at[dcline_id2, "in_service"]
    assert net.dcline.at[dcline_id2, "max_p_mw"] == 3000
    assert net.dcline.at[dcline_id2, "min_p_mw"] == 500
    assert net.dcline.at[dcline_id2, "min_q_from_mvar"] == -100
    assert net.dcline.at[dcline_id2, "max_q_from_mvar"] == 100
    assert net.dcline.at[dcline_id2, "min_q_to_mvar"] == -50
    assert net.dcline.at[dcline_id2, "max_q_to_mvar"] == 50

    validate_network(net)
