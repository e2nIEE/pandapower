# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
import geojson
import numpy as np

from pandapower.create import (
    create_empty_network, create_bus, create_line_from_parameters, create_line, create_buses, create_lines,
    create_lines_from_parameters,
)
from pandapower.std_types import create_std_type
from pandapower.network_schema.tools.validation.network_validation import validate_network


def test_create_line_conductance():
    net = create_empty_network()
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


def test_create_line(): raise NotImplementedError()


def test_create_lines():
    # standard
    net = create_empty_network()
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

    net = create_empty_network()
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
    net = create_empty_network()
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
    net = create_empty_network()
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
    net = create_empty_network()
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


def test_create_line_form_parameters(): raise NotImplementedError()


def test_create_lines_from_parameters():
    # standard
    net = create_empty_network()
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
    net = create_empty_network()
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
    net = create_empty_network()
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
        geodata=[(10, 10), (20, 20)],
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
    assert all(net.line["r_ohm_per_km"].values == 0.2)
    assert all(net.line["r0_ohm_per_km"].values == 0.1)
    assert all(net.line["g0_us_per_km"].values == 0)
    assert all(net.line["c0_nf_per_km"].values == 0)
    assert net.line.in_service.dtype == np.dtype(bool)
    assert not net.line.at[l[0], "in_service"]  # is actually <class 'numpy.bool_'>
    assert not net.line.at[l[1], "in_service"]  # is actually <class 'numpy.bool_'>
    assert net.line.at[l[0], "geo"] == geojson.dumps(geojson.LineString([(10, 10), (20, 20)]), sort_keys=True)
    assert net.line.at[l[1], "geo"] == geojson.dumps(geojson.LineString([(10, 10), (20, 20)]), sort_keys=True)
    assert all(net.line["name"].values == "test")
    assert all(net.line["max_loading_percent"].values == 90)
    assert all(net.line["parallel"].values == 1)
    assert all(net.line["temperature_degree_celsius"].values == 20.0)
    assert all(net.line["alpha"].values == 0.04)
    assert all(net.line.test_kwargs == "dummy_string")

    validate_network(net)

    # setting params as array
    net = create_empty_network()
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
    assert net.line.at[l[0], "x_ohm_per_km"] == 0.3
    assert net.line.at[l[1], "x_ohm_per_km"] == 0.5
    assert net.line.at[l[0], "c_nf_per_km"] == 0.0
    assert net.line.at[l[1], "c_nf_per_km"] == 0.1
    assert net.line.at[l[0], "r0_ohm_per_km"] == 0.1
    assert net.line.at[l[1], "r0_ohm_per_km"] == 0.15
    assert net.line.at[l[0], "x0_ohm_per_km"] == 0.2
    assert net.line.at[l[1], "x0_ohm_per_km"] == 0.25
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
    net = create_empty_network()
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
    net = create_empty_network()
    create_buses(net, 5, 110)
    create_line(net, 0, 1, 10, "48-AL1/8-ST1A 10.0")
    create_line_from_parameters(net, 3, 4, 10, 1, 1, 1, 100)
    create_lines(net, [0, 1], [1, 0], 10, "48-AL1/8-ST1A 10.0")
    create_lines_from_parameters(net, [3, 4], [4, 3], [10, 11], 1, 1, 1, 100)
    assert "max_loading_percent" not in net.line.columns

    v = None
    create_line(net, 0, 1, 10, "48-AL1/8-ST1A 10.0", max_loading_percent=v)
    create_line_from_parameters(net, 3, 4, 10, 1, 1, 1, 100, max_loading_percent=v)
    create_lines(net, [0, 1], [1, 0], 10, "48-AL1/8-ST1A 10.0", max_loading_percent=v)
    # create_lines(net, [0, 1], [1, 0], 10, "48-AL1/8-ST1A 10.0", max_loading_percent=[v, v])  # would be added
    create_lines_from_parameters(net, [3, 4], [4, 3], [10, 11], 1, 1, 1, 100, max_loading_percent=v)
    # create_lines_from_parameters(net, [3, 4], [4, 3], [10, 11], 1, 1, 1, 100, max_loading_percent=[v, v])  # would be added
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
    net = create_empty_network()
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
    assert all(net.line.loc[[l2, l3, l5], "alpha"] == 4.03e-3)
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


def test_create_line_dc(): raise NotImplementedError()


def test_create_lines_dc(): raise NotImplementedError()


def test_create_line_dc_from_parameters(): raise NotImplementedError()


def test_create_lines_dc_from_parameters(): raise NotImplementedError()


def test_create_dcline(): raise NotImplementedError()
