# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from copy import deepcopy
import numpy as np
import pandas as pd
import pytest

import pandapower as pp

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)


def test_convenience_create_functions():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 110.0)
    b2 = pp.create_bus(net, 110.0)
    b3 = pp.create_bus(net, 20)
    pp.create_ext_grid(net, b1)
    pp.create_line_from_parameters(
        net,
        b1,
        b2,
        length_km=20.0,
        r_ohm_per_km=0.0487,
        x_ohm_per_km=0.1382301,
        c_nf_per_km=160.0,
        max_i_ka=0.664,
    )

    l0 = pp.create_load_from_cosphi(
        net, b2, 10, 0.95, "underexcited", name="load", test_kwargs="dummy_string"
    )
    pp.runpp(net, init="flat")

    assert net.load.p_mw.at[l0] == 9.5
    assert net.load.q_mvar.at[l0] > 0
    assert np.sqrt(net.load.p_mw.at[l0] ** 2 + net.load.q_mvar.at[l0] ** 2) == 10
    assert np.isclose(net.res_bus.vm_pu.at[b2], 0.99990833838)
    assert net.load.name.at[l0] == "load"
    assert net.load.test_kwargs.at[l0] == "dummy_string"

    sh0 = pp.create_shunt_as_capacitor(
        net, b2, 10, loss_factor=0.01, name="shunt", test_kwargs="dummy_string"
    )
    pp.runpp(net, init="flat")
    assert np.isclose(net.res_shunt.q_mvar.at[sh0], -10.043934174)
    assert np.isclose(net.res_shunt.p_mw.at[sh0], 0.10043933665)
    assert np.isclose(net.res_bus.vm_pu.at[b2], 1.0021942964)
    assert net.shunt.name.at[sh0] == "shunt"
    assert net.shunt.test_kwargs.at[sh0] == "dummy_string"

    sg0 = pp.create_sgen_from_cosphi(
        net, b2, 5, 0.95, "overexcited", name="sgen", test_kwargs="dummy_string"
    )
    pp.runpp(net, init="flat")
    assert np.sqrt(net.sgen.p_mw.at[sg0] ** 2 + net.sgen.q_mvar.at[sg0] ** 2) == 5
    assert net.sgen.p_mw.at[sg0] == 4.75
    assert net.sgen.q_mvar.at[sg0] > 0
    assert np.isclose(net.res_bus.vm_pu.at[b2], 1.0029376578)
    assert net.sgen.name.at[sg0] == "sgen"
    assert net.sgen.test_kwargs.at[sg0] == "dummy_string"

    tol = 1e-6
    base_z = 110 ** 2 / 100
    sind = pp.create_series_reactor_as_impedance(
        net, b1, b2, r_ohm=100, x_ohm=200, sn_mva=100, test_kwargs="dummy_string"
    )
    assert net.impedance.at[sind, "rft_pu"] - 100 / base_z < tol
    assert net.impedance.at[sind, "xft_pu"] - 200 / base_z < tol
    assert net.impedance.test_kwargs.at[sind] == "dummy_string"

    tid = pp.create_transformer_from_parameters(
        net,
        hv_bus=b2,
        lv_bus=b3,
        sn_mva=0.1,
        vn_hv_kv=110,
        vn_lv_kv=20,
        vkr_percent=5,
        vk_percent=20,
        pfe_kw=1,
        i0_percent=1,
        test_kwargs="dummy_string",
    )
    pp.create_load(net, b3, 0.1)
    assert net.trafo.at[tid, "df"] == 1
    pp.runpp(net)
    tr_l = net.res_trafo.at[tid, "loading_percent"]
    net.trafo.at[tid, "df"] = 2
    pp.runpp(net)
    tr_l_2 = net.res_trafo.at[tid, "loading_percent"]
    assert tr_l == tr_l_2 * 2
    net.trafo.at[tid, "df"] = 0
    with pytest.raises(UserWarning):
        pp.runpp(net)
    assert net.trafo.test_kwargs.at[tid] == "dummy_string"


def test_nonexistent_bus():
    from functools import partial

    net = pp.create_empty_network()
    create_functions = [
        partial(pp.create_load, net=net, p_mw=0, q_mvar=0, bus=0, index=0),
        partial(pp.create_sgen, net=net, p_mw=0, q_mvar=0, bus=0, index=0),
        partial(
            pp.create_dcline,
            net,
            from_bus=0,
            to_bus=1,
            p_mw=0.1,
            loss_percent=0,
            loss_mw=0.01,
            vm_from_pu=1.0,
            vm_to_pu=1.0,
            index=0,
        ),
        partial(pp.create_gen, net=net, p_mw=0, bus=0, index=0),
        partial(pp.create_ward, net, 0, 0, 0, 0, 0, index=0),
        partial(pp.create_xward, net, 0, 0, 0, 0, 0, 1, 1, 1, index=0),
        partial(pp.create_shunt, net=net, q_mvar=0, bus=0, index=0),
        partial(pp.create_ext_grid, net=net, bus=1, index=0),
        partial(
            pp.create_line,
            net=net,
            from_bus=0,
            to_bus=1,
            length_km=1.0,
            std_type="NAYY 4x50 SE",
            index=0,
        ),
        partial(
            pp.create_line_from_parameters,
            net=net,
            from_bus=0,
            to_bus=1,
            length_km=1.0,
            r_ohm_per_km=0.1,
            x_ohm_per_km=0.1,
            max_i_ka=0.4,
            c_nf_per_km=10,
            index=1,
        ),
        partial(
            pp.create_transformer,
            net=net,
            hv_bus=0,
            lv_bus=1,
            std_type="63 MVA 110/20 kV",
            index=0,
        ),
        partial(
            pp.create_transformer3w,
            net=net,
            hv_bus=0,
            lv_bus=1,
            mv_bus=2,
            std_type="63/25/38 MVA 110/20/10 kV",
            index=0,
        ),
        partial(
            pp.create_transformer3w_from_parameters,
            net=net,
            hv_bus=0,
            lv_bus=1,
            mv_bus=2,
            i0_percent=0.89,
            pfe_kw=3.5,
            vn_hv_kv=110,
            vn_lv_kv=10,
            vn_mv_kv=20,
            sn_hv_mva=63,
            sn_lv_mva=38,
            sn_mv_mva=25,
            vk_hv_percent=10.4,
            vk_lv_percent=10.4,
            vk_mv_percent=10.4,
            vkr_hv_percent=0.28,
            vkr_lv_percent=0.35,
            vkr_mv_percent=0.32,
            index=1,
        ),
        partial(
            pp.create_transformer_from_parameters,
            net=net,
            hv_bus=0,
            lv_bus=1,
            sn_mva=60,
            vn_hv_kv=20.0,
            vn_lv_kv=0.4,
            vk_percent=10,
            vkr_percent=0.1,
            pfe_kw=0,
            i0_percent=0,
            index=1,
        ),
        partial(
            pp.create_impedance,
            net=net,
            from_bus=0,
            to_bus=1,
            rft_pu=0.1,
            xft_pu=0.1,
            sn_mva=0.6,
            index=0,
        ),
        partial(pp.create_switch, net, bus=0, element=1, et="b", index=0),
    ]
    for func in create_functions:
        with pytest.raises(
                Exception
        ):  # exception has to be raised since bus doesn't exist
            func()
    pp.create_bus(net, 0.4)
    pp.create_bus(net, 0.4)
    pp.create_bus(net, 0.4)
    for func in create_functions:
        func()  # buses exist, element can be created
        with pytest.raises(
                Exception
        ):  # exception is raised because index already exists
            func()


def test_tap_phase_shifter_default():
    expected_default = False
    net = pp.create_empty_network()
    pp.create_bus(net, 110)
    pp.create_bus(net, 20)
    data = pp.load_std_type(net, "25 MVA 110/20 kV", "trafo")
    if "tap_phase_shifter" in data:
        del data["tap_phase_shifter"]
    pp.create_std_type(net, data, "without_tap_shifter_info", "trafo")
    pp.create_transformer_from_parameters(net, 0, 1, 25e3, 110, 20, 0.4, 12, 20, 0.07)
    pp.create_transformer(net, 0, 1, "without_tap_shifter_info")
    assert (net.trafo.tap_phase_shifter == expected_default).all()


def test_create_line_conductance():
    net = pp.create_empty_network()
    pp.create_bus(net, 20)
    pp.create_bus(net, 20)
    pp.create_std_type(
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

    l = pp.create_line(net, 0, 1, 1.0, "test_conductance", test_kwargs="dummy_string")
    assert net.line.g_us_per_km.at[l] == 1
    assert net.line.test_kwargs.at[l] == "dummy_string"


def test_create_buses():
    net = pp.create_empty_network()
    # standard
    b1 = pp.create_buses(net, 3, 110, test_kwargs="dummy_string")
    # with geodata
    b2 = pp.create_buses(net, 3, 110, geodata=(10, 20))
    # with geodata as array
    geodata = np.array([[10, 20], [20, 30], [30, 40]])
    b3 = pp.create_buses(net, 3, 110, geodata=geodata)

    assert len(net.bus) == 9
    assert len(net.bus_geodata) == 6
    assert net.bus.test_kwargs.at[b1[0]] == "dummy_string"

    for i in b2:
        assert net.bus_geodata.at[i, "x"] == 10
        assert net.bus_geodata.at[i, "y"] == 20
    assert (net.bus_geodata.loc[b3, ["x", "y"]].values == geodata).all()

    # no way of creating buses with not matching shape
    with pytest.raises(ValueError):
        pp.create_buses(net, 2, 110, geodata=geodata)


def test_create_lines():
    # standard
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 10)
    b2 = pp.create_bus(net, 10)
    l = pp.create_lines(
        net,
        [b1, b1],
        [b2, b2],
        4,
        std_type="48-AL1/8-ST1A 10.0",
        test_kwargs="dummy_string",
    )
    assert len(net.line) == 2
    assert len(net.line_geodata) == 0
    assert sum(net.line.std_type == "48-AL1/8-ST1A 10.0") == 2
    assert len(set(net.line.r_ohm_per_km)) == 1
    assert all(net.line.test_kwargs == "dummy_string")

    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 10)
    b2 = pp.create_bus(net, 10)
    l = pp.create_lines(
        net,
        [b1, b1],
        [b2, b2],
        4,
        std_type=["48-AL1/8-ST1A 10.0", "NA2XS2Y 1x240 RM/25 6/10 kV"],
    )
    assert len(net.line) == 2
    assert len(net.line_geodata) == 0
    assert sum(net.line.std_type == "48-AL1/8-ST1A 10.0") == 1
    assert sum(net.line.std_type == "NA2XS2Y 1x240 RM/25 6/10 kV") == 1

    # with geodata
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 10)
    b2 = pp.create_bus(net, 10)
    l = pp.create_lines(
        net,
        [b1, b1],
        [b2, b2],
        [1.5, 3],
        std_type="48-AL1/8-ST1A 10.0",
        geodata=[[(1, 1), (2, 2), (3, 3)], [(1, 1), (1, 2)]],
    )

    assert len(net.line) == 2
    assert len(net.line_geodata) == 2
    assert net.line_geodata.at[l[0], "coords"] == [(1, 1), (2, 2), (3, 3)]
    assert net.line_geodata.at[l[1], "coords"] == [(1, 1), (1, 2)]

    # setting params as single value
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 10)
    b2 = pp.create_bus(net, 10)
    l = pp.create_lines(
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
    assert len(net.line_geodata) == 2
    assert net.line.length_km.at[l[0]] == 5
    assert net.line.length_km.at[l[1]] == 5
    assert net.line.in_service.dtype == bool
    assert not net.line.at[l[0], "in_service"]  # is actually <class 'numpy.bool_'>
    assert not net.line.at[l[1], "in_service"]  # is actually <class 'numpy.bool_'>
    assert net.line_geodata.at[l[0], "coords"] == [(10, 10), (20, 20)]
    assert net.line_geodata.at[l[1], "coords"] == [(10, 10), (20, 20)]
    assert net.line.at[l[0], "name"] == "test"
    assert net.line.at[l[1], "name"] == "test"
    assert net.line.at[l[0], "max_loading_percent"] == 90
    assert net.line.at[l[1], "max_loading_percent"] == 90
    assert net.line.at[l[0], "parallel"] == 1
    assert net.line.at[l[1], "parallel"] == 1

    # setting params as array
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 10)
    b2 = pp.create_bus(net, 10)
    l = pp.create_lines(
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
    assert len(net.line_geodata) == 2
    assert net.line.at[l[0], "length_km"] == 1
    assert net.line.at[l[1], "length_km"] == 5
    assert net.line.in_service.dtype == bool
    assert net.line.at[l[0], "in_service"]  # is actually <class 'numpy.bool_'>
    assert not net.line.at[l[1], "in_service"]  # is actually <class 'numpy.bool_'>
    assert net.line_geodata.at[l[0], "coords"] == [(10, 10), (20, 20)]
    assert net.line_geodata.at[l[1], "coords"] == [(100, 10), (200, 20)]
    assert net.line.at[l[0], "name"] == "test1"
    assert net.line.at[l[1], "name"] == "test2"
    assert net.line.at[l[0], "max_loading_percent"] == 80
    assert net.line.at[l[1], "max_loading_percent"] == 90
    assert net.line.at[l[0], "parallel"] == 2
    assert net.line.at[l[1], "parallel"] == 1


def test_create_lines_from_parameters():
    # standard
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 10)
    b2 = pp.create_bus(net, 10)
    l = pp.create_lines_from_parameters(
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
    assert len(net.line_geodata) == 0
    assert len(net.line.x_ohm_per_km) == 2
    assert len(net.line.r_ohm_per_km) == 2
    assert len(net.line.c_nf_per_km) == 2
    assert len(net.line.max_i_ka) == 2
    assert len(net.line.df) == 2
    assert net.line.test_kwargs.at[l[0]] == "dummy_string"

    # with geodata
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 10)
    b2 = pp.create_bus(net, 10)
    l = pp.create_lines_from_parameters(
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
    assert len(net.line_geodata) == 2
    assert net.line_geodata.at[l[0], "coords"] == [(1, 1), (2, 2), (3, 3)]
    assert net.line_geodata.at[l[1], "coords"] == [(1, 1), (1, 2)]

    # setting params as single value
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 10)
    b2 = pp.create_bus(net, 10)
    l = pp.create_lines_from_parameters(
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
    assert len(net.line_geodata) == 2
    assert all(net.line["length_km"].values == 5)
    assert all(net.line["x_ohm_per_km"].values == 1)
    assert all(net.line["r_ohm_per_km"].values == 0.2)
    assert all(net.line["r0_ohm_per_km"].values == 0.1)
    assert all(net.line["g0_us_per_km"].values == 0)
    assert all(net.line["c0_nf_per_km"].values == 0)
    assert net.line.in_service.dtype == bool
    assert not net.line.at[l[0], "in_service"]  # is actually <class 'numpy.bool_'>
    assert not net.line.at[l[1], "in_service"]  # is actually <class 'numpy.bool_'>
    assert net.line_geodata.at[l[0], "coords"] == [(10, 10), (20, 20)]
    assert net.line_geodata.at[l[1], "coords"] == [(10, 10), (20, 20)]
    assert all(net.line["name"].values == "test")
    assert all(net.line["max_loading_percent"].values == 90)
    assert all(net.line["parallel"].values == 1)
    assert all(net.line["temperature_degree_celsius"].values == 20.0)
    assert all(net.line["alpha"].values == 0.04)
    assert all(net.line.test_kwargs == "dummy_string")

    # setting params as array
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 10)
    b2 = pp.create_bus(net, 10)
    l = pp.create_lines_from_parameters(
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
    assert len(net.line_geodata) == 2
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
    assert net.line.in_service.dtype == bool
    assert net.line.at[l[0], "in_service"]  # is actually <class 'numpy.bool_'>
    assert not net.line.at[l[1], "in_service"]  # is actually <class 'numpy.bool_'>
    assert net.line_geodata.at[l[0], "coords"] == [(10, 10), (20, 20)]
    assert net.line_geodata.at[l[1], "coords"] == [(100, 10), (200, 20)]
    assert net.line.at[l[0], "name"] == "test1"
    assert net.line.at[l[1], "name"] == "test2"
    assert net.line.at[l[0], "max_loading_percent"] == 80
    assert net.line.at[l[1], "max_loading_percent"] == 90
    assert net.line.at[l[0], "parallel"] == 2
    assert net.line.at[l[1], "parallel"] == 1
    assert net.line.at[l[0], "max_i_ka"] == 100
    assert net.line.at[l[1], "max_i_ka"] == 200


def test_create_lines_raise_errorexcept():
    # standard
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 10)
    b2 = pp.create_bus(net, 10)
    pp.create_lines_from_parameters(
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
        pp.create_lines_from_parameters(
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
        pp.create_lines_from_parameters(
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
        pp.create_lines_from_parameters(
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


def test_create_lines_optional_columns():
    #
    net = pp.create_empty_network()
    pp.create_buses(net, 5, 110)
    pp.create_line(net, 0, 1, 10, "48-AL1/8-ST1A 10.0")
    pp.create_line_from_parameters(net, 3, 4, 10, 1, 1, 1, 100)
    pp.create_lines(net, [0, 1], [1, 0], 10, "48-AL1/8-ST1A 10.0")
    pp.create_lines_from_parameters(net, [3, 4], [4, 3], [10, 11], 1, 1, 1, 100)
    assert "max_loading_percent" not in net.line.columns

    v = None
    pp.create_line(net, 0, 1, 10, "48-AL1/8-ST1A 10.0", max_loading_percent=v)
    pp.create_line_from_parameters(net, 3, 4, 10, 1, 1, 1, 100, max_loading_percent=v)
    pp.create_lines(net, [0, 1], [1, 0], 10, "48-AL1/8-ST1A 10.0", max_loading_percent=v)
    # pp.create_lines(net, [0, 1], [1, 0], 10, "48-AL1/8-ST1A 10.0", max_loading_percent=[v, v])  # would be added
    pp.create_lines_from_parameters(net, [3, 4], [4, 3], [10, 11], 1, 1, 1, 100, max_loading_percent=v)
    # pp.create_lines_from_parameters(net, [3, 4], [4, 3], [10, 11], 1, 1, 1, 100, max_loading_percent=[v, v])  # would be added
    assert "max_loading_percent" not in net.line.columns

    v = np.nan
    pp.create_line(net, 0, 1, 10, "48-AL1/8-ST1A 10.0", max_loading_percent=v)
    pp.create_line_from_parameters(net, 3, 4, 10, 1, 1, 1, 100, max_loading_percent=v)
    # np.nan is not None:
    # pp.create_lines(net, [0, 1], [1, 0], 10, "48-AL1/8-ST1A 10.0", max_loading_percent=v)
    # pp.create_lines(net, [0, 1], [1, 0], 10, "48-AL1/8-ST1A 10.0", max_loading_percent=[v, v])  # would be added
    # pp.create_lines_from_parameters(net, [3, 4], [4, 3], [10, 11], 1, 1, 1, 100, max_loading_percent=v)
    # pp.create_lines_from_parameters(net, [3, 4], [4, 3], [10, 11], 1, 1, 1, 100, max_loading_percent=[v, v])
    assert "max_loading_percent" not in net.line.columns


def test_create_line_alpha_temperature():
    net = pp.create_empty_network()
    b = pp.create_buses(net, 5, 110)

    l1 = pp.create_line(net, 0, 1, 10, "48-AL1/8-ST1A 10.0")
    l2 = pp.create_line(
        net,
        1,
        2,
        10,
        "48-AL1/8-ST1A 10.0",
        alpha=4.03e-3,
        temperature_degree_celsius=80,
    )
    l3 = pp.create_line(net, 2, 3, 10, "48-AL1/8-ST1A 10.0")
    l4 = pp.create_line_from_parameters(net, 3, 4, 10, 1, 1, 1, 100)
    l5 = pp.create_line_from_parameters(net, 3, 4, 10, 1, 1, 1, 100, alpha=4.03e-3)

    assert "alpha" in net.line.columns
    assert all(net.line.loc[[l2, l3, l5], "alpha"] == 4.03e-3)
    assert all(net.line.loc[[l1, l4], "alpha"].isnull())
    assert net.line.loc[l2, "temperature_degree_celsius"] == 80
    assert all(net.line.loc[[l1, l3, l4, l5], "temperature_degree_celsius"].isnull())

    # make sure optional columns are not created if None or np.nan:
    pp.create_line(net, 2, 3, 10, "48-AL1/8-ST1A 10.0", wind_speed_m_per_s=None)
    pp.create_lines(net, [2], [3], 10, "48-AL1/8-ST1A 10.0", wind_speed_m_per_s=None)
    pp.create_line_from_parameters(net, 3, 4, 10, 1, 1, 1, 100, wind_speed_m_per_s=None)
    pp.create_line_from_parameters(net, 3, 4, 10, 1, 1, 1, 100, alpha=4.03e-3, wind_speed_m_per_s=np.nan)
    assert "wind_speed_m_per_s" not in net.line.columns


def test_create_transformers_from_parameters():
    # standard
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 15)
    b2 = pp.create_bus(net, 0.4)
    index = pp.create_transformers_from_parameters(
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
        pp.create_transformers_from_parameters(
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

    # setting params as single value
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 15)
    b2 = pp.create_bus(net, 0.4)
    pp.create_transformers_from_parameters(
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
        mag0_percent=0.3,
        tap_neutral=0.0,
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
    assert all(net.trafo.mag0_percent == 0.3)
    assert all(net.trafo.tap_neutral == 0.0)
    assert all(net.trafo.tap_pos == 0.0)
    assert all(net.trafo.vector_group.values == "Dyn")
    assert all(net.trafo.max_loading_percent == 80.0)
    assert all(net.trafo.si0_hv_partial == 0.1)
    assert all(net.trafo.test_kwargs == "dummy_string")

    # setting params as array
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 10)
    b2 = pp.create_bus(net, 10)
    t = pp.create_transformers_from_parameters(
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
        mag0_percent=[0.3, 0.3],
        tap_neutral=[0.0, 1.0],
        tap_pos=[-1, 4],
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
    assert all(net.trafo.mag0_percent == 0.3)
    assert all(net.trafo.test_kwargs == "dummy_string")
    assert net.trafo.tap_neutral.at[t[0]] == 0
    assert net.trafo.tap_neutral.at[t[1]] == 1
    assert net.trafo.tap_pos.at[t[0]] == -1
    assert net.trafo.tap_pos.at[t[1]] == 4


def test_create_transformers_raise_errorexcept():
    # standard
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 10)
    b2 = pp.create_bus(net, 10)
    pp.create_transformers_from_parameters(
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
        pp.create_transformers_from_parameters(
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
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 10)
    b2 = pp.create_bus(net, 10)
    pp.create_transformers_from_parameters(
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
        pp.create_transformers_from_parameters(
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
        pp.create_transformers_from_parameters(
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


def test_trafo_2_tap_changers():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 20)
    pp.create_transformer(net, b1, b2, "40 MVA 110/20 kV")

    tap2_data = {"tap2_side": "hv",
                 "tap2_neutral": 0,
                 "tap2_max": 10,
                 "tap2_min": -10,
                 "tap2_step_percent": 1,
                 "tap2_step_degree": 0,
                 "tap2_phase_shifter": False}

    for c in tap2_data.keys():
        assert c not in net.trafo.columns

    std_type = pp.load_std_type(net, "40 MVA 110/20 kV", "trafo")

    pp.create_std_type(net, {**std_type, **tap2_data}, "test_trafo_type", "trafo")

    t = pp.create_transformer(net, b1, b2, "test_trafo_type")

    for c in tap2_data.keys():
        assert c in net.trafo.columns
        assert net.trafo.at[t, c] == tap2_data[c]


def test_trafo_2_tap_changers_parameters():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 20)

    std_type = pp.load_std_type(net, "40 MVA 110/20 kV", "trafo")
    tap2_data = {"tap2_side": "hv",
                 "tap2_neutral": 0,
                 "tap2_max": 10,
                 "tap2_min": -10,
                 "tap2_step_percent": 1,
                 "tap2_step_degree": 0,
                 "tap2_phase_shifter": False}

    pp.create_transformer_from_parameters(net, b1, b2, **std_type)

    for c in tap2_data.keys():
        assert c not in net.trafo.columns

    t = pp.create_transformer_from_parameters(net, b1, b2, **std_type, **tap2_data)

    for c in tap2_data.keys():
        assert c in net.trafo.columns
        assert net.trafo.at[t, c] == tap2_data[c]


def test_trafos_2_tap_changers_parameters():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 20)

    std_type = pp.load_std_type(net, "40 MVA 110/20 kV", "trafo")
    tap2_data = {"tap2_side": "hv",
                 "tap2_neutral": 0,
                 "tap2_max": 10,
                 "tap2_min": -10,
                 "tap2_step_percent": 1,
                 "tap2_step_degree": 0,
                 "tap2_phase_shifter": False}

    std_type_p = {k: np.array([v, v]) if not isinstance(v, str) else v for k, v in std_type.items()}

    pp.create_transformers_from_parameters(net, [b1, b1], [b2, b2], **std_type_p)

    for c in tap2_data.keys():
        assert c not in net.trafo.columns

    t = pp.create_transformer_from_parameters(net, b1, b2, **std_type, **tap2_data)

    for c in tap2_data.keys():
        assert c in net.trafo.columns
        assert net.trafo.at[t, c] == tap2_data[c]


def test_create_transformers3w_from_parameters():
    # setting params as single value
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 15)
    b2 = pp.create_bus(net, 0.4)
    b3 = pp.create_bus(net, 0.9)
    pp.create_transformers3w_from_parameters(
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
        tap_neutral=0.0,
        mag0_rx=0.4,
        mag0_percent=0.3,
        test_kwargs="dummy_string",
    )
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
    assert all(net.trafo3w.mag0_percent == 0.3)
    assert all(net.trafo3w.tap_neutral == 0.0)
    assert all(net.trafo3w.tap_pos == 0.0)
    assert all(net.trafo3w.test_kwargs == "dummy_string")

    # setting params as array
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 10)
    b2 = pp.create_bus(net, 0.4)
    b3 = pp.create_bus(net, 0.9)
    pp.create_transformers3w_from_parameters(
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
        tap_neutral=[0.0, 5.0],
        tap_pos=[1, 2],
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
    assert all(net.trafo3w.tap_neutral == [0.0, 5.0])
    assert all(net.trafo3w.tap_pos == [1, 2])
    assert all(net.trafo3w.in_service == [True, False])
    assert all(net.trafo3w.test_kwargs == ["foo", "bar"])


def test_create_transformers3w_raise_errorexcept():
    # standard
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 15)
    b2 = pp.create_bus(net, 0.4)
    b3 = pp.create_bus(net, 0.9)
    pp.create_transformers3w_from_parameters(
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
        tap_neutral=0.0,
        mag0_rx=0.4,
        mag0_percent=0.3,
    )

    with pytest.raises(
            UserWarning,
            match=r"Three winding transformers with indexes \[1\] already exist.",
    ):
        pp.create_transformers3w_from_parameters(
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
            tap_neutral=0.0,
            mag0_rx=0.4,
            mag0_percent=0.3,
            index=[2, 1],
        )
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 15)
    b2 = pp.create_bus(net, 0.4)
    b3 = pp.create_bus(net, 0.9)
    with pytest.raises(
            UserWarning, match=r"Transformers trying to attach to non existing buses \{6\}"
    ):
        pp.create_transformers3w_from_parameters(
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
            mag0_percent=0.3,
            index=[0, 1],
        )
    with pytest.raises(
            UserWarning, match=r"Transformers trying to attach to non existing buses \{3\}"
    ):
        pp.create_transformers3w_from_parameters(
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
            mag0_percent=0.3,
        )
    with pytest.raises(
            UserWarning,
            match=r"Transformers trying to attach to non existing buses \{3, 4\}",
    ):
        pp.create_transformers3w_from_parameters(
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
            mag0_percent=0.3,
        )


def test_create_switches():
    net = pp.create_empty_network()
    # standard
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 15)
    b4 = pp.create_bus(net, 15)
    l1 = pp.create_line(net, b1, b2, length_km=1, std_type="48-AL1/8-ST1A 10.0")
    t1 = pp.create_transformer(net, b2, b3, std_type="160 MVA 380/110 kV")

    sw = pp.create_switches(
        net,
        buses=[b1, b2, b3],
        elements=[l1, t1, b4],
        et=["l", "t", "b"],
        z_ohm=0.0,
        test_kwargs="aaa",
    )

    assert net.switch.bus.at[0] == b1
    assert net.switch.bus.at[1] == b2
    assert net.switch.bus.at[2] == b3
    assert net.switch.element.at[sw[0]] == l1
    assert net.switch.element.at[sw[1]] == t1
    assert net.switch.element.at[sw[2]] == b4
    assert net.switch.et.at[0] == "l"
    assert net.switch.et.at[1] == "t"
    assert net.switch.et.at[2] == "b"
    assert net.switch.z_ohm.at[0] == 0
    assert net.switch.z_ohm.at[1] == 0
    assert net.switch.z_ohm.at[2] == 0
    assert net.switch.test_kwargs.at[0] == "aaa"
    assert net.switch.test_kwargs.at[1] == "aaa"
    assert net.switch.test_kwargs.at[2] == "aaa"


def test_create_switches_raise_errorexcept():
    net = pp.create_empty_network()
    # standard
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 15)
    b4 = pp.create_bus(net, 15)
    b5 = pp.create_bus(net, 0.9)
    b6 = pp.create_bus(net, 0.4)
    l1 = pp.create_line(net, b1, b2, length_km=1, std_type="48-AL1/8-ST1A 10.0")
    t1 = pp.create_transformer(net, b2, b3, std_type="160 MVA 380/110 kV")
    t3w1 = pp.create_transformer3w_from_parameters(
        net,
        hv_bus=b4,
        mv_bus=b5,
        lv_bus=b6,
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
    )
    sw = pp.create_switch(net, bus=b1, element=l1, et="l", z_ohm=0.0)
    with pytest.raises(
            UserWarning, match=r"Switches with indexes \[0\] already exist."
    ):
        pp.create_switches(
            net,
            buses=[b1, b2, b3],
            elements=[l1, t1, b4],
            et=["l", "t", "b"],
            z_ohm=0.0,
            index=[sw, 1, 2],
        )
    with pytest.raises(
            UserWarning, match=r"Cannot attach to buses \{6\}, they do not exist"
    ):
        pp.create_switches(
            net, buses=[6, b2, b3], elements=[l1, t1, b4], et=["l", "t", "b"], z_ohm=0.0
        )
    with pytest.raises(UserWarning, match="Line 1 does not exist"):
        pp.create_switches(
            net, buses=[b1, b2, b3], elements=[1, t1, b4], et=["l", "t", "b"], z_ohm=0.0
        )
    with pytest.raises(UserWarning, match="Line %s not connected to bus %s" % (l1, b3)):
        pp.create_switches(
            net,
            buses=[b3, b2, b3],
            elements=[l1, t1, b4],
            et=["l", "t", "b"],
            z_ohm=0.0,
        )
    with pytest.raises(UserWarning, match="Trafo 1 does not exist"):
        pp.create_switches(
            net, buses=[b1, b2, b3], elements=[l1, 1, b4], et=["l", "t", "b"], z_ohm=0.0
        )
    with pytest.raises(
            UserWarning, match="Trafo %s not connected to bus %s" % (t1, b1)
    ):
        pp.create_switches(
            net,
            buses=[b1, b1, b3],
            elements=[l1, t1, b4],
            et=["l", "t", "b"],
            z_ohm=0.0,
        )
    with pytest.raises(
            UserWarning, match=r"Cannot attach to bus 6, bus does not exist"
    ):
        pp.create_switches(
            net, buses=[b1, b2, b3], elements=[l1, t1, 6], et=["l", "t", "b"], z_ohm=0.0
        )
    with pytest.raises(UserWarning, match="Trafo3w 1 does not exist"):
        pp.create_switches(
            net,
            buses=[b1, b2, b3],
            elements=[l1, t1, 1],
            et=["l", "t", "t3"],
            z_ohm=0.0,
        )
    with pytest.raises(
            UserWarning, match="Trafo3w %s not connected to bus %s" % (t3w1, b3)
    ):
        pp.create_switches(
            net,
            buses=[b1, b2, b3],
            elements=[l1, t1, t3w1],
            et=["l", "t", "t3"],
            z_ohm=0.0,
        )


def test_create_loads():
    net = pp.create_empty_network()
    # standard
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 110)
    pp.create_loads(
        net,
        buses=[b1, b2, b3],
        p_mw=[0, 0, 1],
        q_mvar=0.0,
        controllable=[True, False, False],
        max_p_mw=0.2,
        min_p_mw=[0, 0.1, 0],
        max_q_mvar=0.2,
        min_q_mvar=[0, 0.1, 0],
        test_kwargs=["dummy_string_1", "dummy_string_2", "dummy_string_3"],
    )

    assert net.load.bus.at[0] == b1
    assert net.load.bus.at[1] == b2
    assert net.load.bus.at[2] == b3
    assert net.load.p_mw.at[0] == 0
    assert net.load.p_mw.at[1] == 0
    assert net.load.p_mw.at[2] == 1
    assert net.load.q_mvar.at[0] == 0
    assert net.load.q_mvar.at[1] == 0
    assert net.load.q_mvar.at[2] == 0
    assert net.load.controllable.dtype == bool
    assert net.load.controllable.at[0]
    assert not net.load.controllable.at[1]
    assert not net.load.controllable.at[2]
    assert all(net.load.max_p_mw.values == 0.2)
    assert all(net.load.min_p_mw.values == [0, 0.1, 0])
    assert all(net.load.max_q_mvar.values == 0.2)
    assert all(net.load.min_q_mvar.values == [0, 0.1, 0])
    assert all(
        net.load.test_kwargs.values
        == ["dummy_string_1", "dummy_string_2", "dummy_string_3"]
    )


def test_create_loads_raise_errorexcept():
    net = pp.create_empty_network()
    # standard
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 110)

    with pytest.raises(
            UserWarning, match=r"Cannot attach to buses \{3, 4, 5\}, they do not exist"
    ):
        pp.create_loads(
            net,
            buses=[3, 4, 5],
            p_mw=[0, 0, 1],
            q_mvar=0.0,
            controllable=[True, False, False],
            max_p_mw=0.2,
            min_p_mw=[0, 0.1, 0],
            max_q_mvar=0.2,
            min_q_mvar=[0, 0.1, 0],
        )
    l = pp.create_loads(
        net,
        buses=[b1, b2, b3],
        p_mw=[0, 0, 1],
        q_mvar=0.0,
        controllable=[True, False, False],
        max_p_mw=0.2,
        min_p_mw=[0, 0.1, 0],
        max_q_mvar=0.2,
        min_q_mvar=[0, 0.1, 0],
    )
    with pytest.raises(
            UserWarning, match=r"Loads with indexes \[0 1 2\] already exist"
    ):
        pp.create_loads(
            net,
            buses=[b1, b2, b3],
            p_mw=[0, 0, 1],
            q_mvar=0.0,
            controllable=[True, False, False],
            max_p_mw=0.2,
            min_p_mw=[0, 0.1, 0],
            max_q_mvar=0.2,
            min_q_mvar=[0, 0.1, 0],
            index=l,
        )


def test_create_storages():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 110)
    net_bulk = deepcopy(net)

    pp.create_storage(net, b1, 0, 3, 0.5, controllable=True, max_p_mw=0.2, min_p_mw=0,
                      max_q_mvar=0.2, min_q_mvar=0, test_kwargs="dummy_string_1")
    pp.create_storage(net, b2, 0, 5, 0.5, controllable=False, max_p_mw=0.2, min_p_mw=0.1,
                      max_q_mvar=0.2, min_q_mvar=0.1, test_kwargs="dummy_string_2")
    pp.create_storage(net, b3, 1, 7, 0.5, max_p_mw=0.2, min_p_mw=0,
                      max_q_mvar=0.2, min_q_mvar=0, test_kwargs="dummy_string_3")

    pp.create_storages(
        net_bulk,
        buses=[b1, b2, b3],
        p_mw=[0, 0, 1],
        max_e_mwh=[3, 5, 7],
        q_mvar=0.5,
        controllable=[True, False, False],
        max_p_mw=0.2,
        min_p_mw=[0, 0.1, 0],
        max_q_mvar=0.2,
        min_q_mvar=[0, 0.1, 0],
        test_kwargs=["dummy_string_1", "dummy_string_2", "dummy_string_3"],
    )

    assert net.storage.bus.at[0] == b1
    assert net.storage.bus.at[1] == b2
    assert net.storage.bus.at[2] == b3
    assert net.storage.p_mw.at[0] == 0
    assert net.storage.p_mw.at[1] == 0
    assert net.storage.p_mw.at[2] == 1
    assert net.storage.max_e_mwh.at[0] == 3
    assert net.storage.max_e_mwh.at[1] == 5
    assert net.storage.max_e_mwh.at[2] == 7
    assert net.storage.q_mvar.at[0] == 0.5
    assert net.storage.q_mvar.at[1] == 0.5
    assert net.storage.q_mvar.at[2] == 0.5
    assert net.storage.controllable.dtype == bool
    assert net.storage.controllable.at[0]
    assert not net.storage.controllable.at[1]
    assert not net.storage.controllable.at[2]
    assert all(net.storage.max_p_mw.values == 0.2)
    assert all(net.storage.min_p_mw.values == [0, 0.1, 0])
    assert all(net.storage.max_q_mvar.values == 0.2)
    assert all(net.storage.min_q_mvar.values == [0, 0.1, 0])
    assert all(
        net.storage.test_kwargs.values
        == ["dummy_string_1", "dummy_string_2", "dummy_string_3"]
    )
    assert pp.nets_equal(net, net_bulk)


def test_create_wards():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 110)
    net_bulk = deepcopy(net)
    vals = np.c_[[b1, b2, b3], np.reshape(np.arange(12), (3, 4)), ["asd", None, "123"], [True, False, False]]

    pp.create_ward(net, *vals[0, :])
    pp.create_ward(net, *vals[1, :])
    pp.create_ward(net, *vals[2, :])

    pp.create_wards(net_bulk, vals[:, 0], vals[:, 1], vals[:, 2], vals[:, 3], vals[:, 4],
                    vals[:, 5], vals[:, 6])

    assert net.ward.bus.at[0] == b1
    assert net.ward.bus.at[1] == b2
    assert net.ward.bus.at[2] == b3
    assert net.ward.ps_mw.at[0] == 0
    assert net.ward.ps_mw.at[1] == 4
    assert net.ward.ps_mw.at[2] == 8
    assert net.ward.qs_mvar.at[0] == 1
    assert net.ward.qs_mvar.at[1] == 5
    assert net.ward.qs_mvar.at[2] == 9
    assert net.ward.pz_mw.at[0] == 2
    assert net.ward.pz_mw.at[1] == 6
    assert net.ward.pz_mw.at[2] == 10
    assert net.ward.qz_mvar.at[0] == 3
    assert net.ward.qz_mvar.at[1] == 7
    assert net.ward.qz_mvar.at[2] == 11
    assert net.ward.name.at[0] == "asd"
    assert net.ward.name.at[1] == None
    assert net.ward.name.at[2] == "123"
    assert net.ward.in_service.at[0]
    assert not net.ward.in_service.at[1]
    assert not net.ward.in_service.at[2]
    assert pp.nets_equal(net, net_bulk)


def test_create_sgens():
    net = pp.create_empty_network()
    # standard
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 110)
    pp.create_sgens(
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
    assert all(net.sgen.max_p_mw.values == 0.2)
    assert all(net.sgen.min_p_mw.values == [0, 0.1, 0])
    assert all(net.sgen.max_q_mvar.values == 0.2)
    assert all(net.sgen.min_q_mvar.values == [0, 0.1, 0])
    assert all(net.sgen.k.values == 1.3)
    assert all(net.sgen.rx.values == 0.4)
    assert all(net.sgen.current_source)
    assert all(net.sgen.test_kwargs == "dummy_string")


def test_create_sgens_raise_errorexcept():
    net = pp.create_empty_network()
    # standard
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 110)

    with pytest.raises(
            UserWarning, match=r"Cannot attach to buses \{3, 4, 5\}, they do not exist"
    ):
        pp.create_sgens(
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
    sg = pp.create_sgens(
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
        pp.create_sgens(
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


def test_create_gens():
    net = pp.create_empty_network()
    # standard
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 110)
    pp.create_gens(
        net,
        buses=[b1, b2, b3],
        p_mw=[0, 0, 1],
        vm_pu=1.0,
        controllable=[True, False, False],
        max_p_mw=0.2,
        min_p_mw=[0, 0.1, 0],
        max_q_mvar=0.2,
        min_q_mvar=[0, 0.1, 0],
        min_vm_pu=0.85,
        max_vm_pu=1.15,
        vn_kv=0.4,
        xdss_pu=0.1,
        rdss_pu=0.1,
        cos_phi=1.0,
        test_kwargs="dummy_string",
    )
    assert net.gen.bus.at[0] == b1
    assert net.gen.bus.at[1] == b2
    assert net.gen.bus.at[2] == b3
    assert net.gen.p_mw.at[0] == 0
    assert net.gen.p_mw.at[1] == 0
    assert net.gen.p_mw.at[2] == 1
    assert net.gen.controllable.dtype == bool
    assert net.gen.controllable.at[0]
    assert not net.gen.controllable.at[1]
    assert not net.gen.controllable.at[2]
    assert all(net.gen.max_p_mw.values == 0.2)
    assert all(net.gen.min_p_mw.values == [0, 0.1, 0])
    assert all(net.gen.max_q_mvar.values == 0.2)
    assert all(net.gen.min_q_mvar.values == [0, 0.1, 0])
    assert all(net.gen.min_vm_pu.values == 0.85)
    assert all(net.gen.max_vm_pu.values == 1.15)
    assert all(net.gen.vn_kv.values == 0.4)
    assert all(net.gen.xdss_pu.values == 0.1)
    assert all(net.gen.rdss_pu.values == 0.1)
    assert all(net.gen.cos_phi.values == 1.0)
    assert all(net.gen.test_kwargs == "dummy_string")


def test_create_gens_raise_errorexcept():
    net = pp.create_empty_network()
    # standard
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 110)

    with pytest.raises(
            UserWarning, match=r"Cannot attach to buses \{3, 4, 5\}, they do not exist"
    ):
        pp.create_gens(
            net,
            buses=[3, 4, 5],
            p_mw=[0, 0, 1],
            vm_pu=1.0,
            controllable=[True, False, False],
            max_p_mw=0.2,
            min_p_mw=[0, 0.1, 0],
            max_q_mvar=0.2,
            min_q_mvar=[0, 0.1, 0],
            min_vm_pu=0.85,
            max_vm_pu=1.15,
            vn_kv=0.4,
            xdss_pu=0.1,
            rdss_pu=0.1,
            cos_phi=1.0,
        )
    g = pp.create_gens(
        net,
        buses=[b1, b2, b3],
        p_mw=[0, 0, 1],
        vm_pu=1.0,
        controllable=[True, False, False],
        max_p_mw=0.2,
        min_p_mw=[0, 0.1, 0],
        max_q_mvar=0.2,
        min_q_mvar=[0, 0.1, 0],
        min_vm_pu=0.85,
        max_vm_pu=1.15,
        vn_kv=0.4,
        xdss_pu=0.1,
        rdss_pu=0.1,
        cos_phi=1.0,
    )

    with pytest.raises(UserWarning, match=r"Gens with indexes \[0 1 2\] already exist"):
        pp.create_gens(
            net,
            buses=[b1, b2, b3],
            p_mw=[0, 0, 1],
            vm_pu=1.0,
            controllable=[True, False, False],
            max_p_mw=0.2,
            min_p_mw=[0, 0.1, 0],
            max_q_mvar=0.2,
            min_q_mvar=[0, 0.1, 0],
            min_vm_pu=0.85,
            max_vm_pu=1.15,
            vn_kv=0.4,
            xdss_pu=0.1,
            rdss_pu=0.1,
            cos_phi=1.0,
            index=g,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
