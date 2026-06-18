# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from functools import partial

import pytest
import geojson

from pandapower.create import (
    create_bus, create_ext_grid, create_line_from_parameters, create_transformer_from_parameters,
    create_load, create_sgen, create_dcline, create_gen, create_ward, create_xward, create_shunt, create_line,
    create_transformer, create_transformer3w, create_transformer3w_from_parameters, create_impedance, create_switch,
    create_buses, create_bus_dc, create_buses_dc
)
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network


def test_nonexistent_bus():
    net = pandapowerNet(name="test_nonexistent_bus")
    create_functions = [
        partial(create_load, net=net, p_mw=0, q_mvar=0, bus=0, index=0),
        partial(create_sgen, net=net, p_mw=0, q_mvar=0, bus=0, index=0),
        partial(
            create_dcline,
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
        partial(create_gen, net=net, p_mw=0, bus=0, index=0),
        partial(create_ward, net, 0, 0, 0, 0, 0, index=0),
        partial(create_xward, net, 0, 0, 0, 0, 0, 1, 1, 1, index=0),
        partial(create_shunt, net=net, q_mvar=0, bus=0, index=0),
        partial(create_ext_grid, net=net, bus=1, index=0),
        partial(
            create_line,
            net=net,
            from_bus=0,
            to_bus=1,
            length_km=1.0,
            std_type="NAYY 4x50 SE",
            index=0,
        ),
        partial(
            create_line_from_parameters,
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
            create_transformer,
            net=net,
            hv_bus=0,
            lv_bus=1,
            std_type="63 MVA 110/20 kV",
            index=0,
        ),
        partial(
            create_transformer3w,
            net=net,
            hv_bus=0,
            lv_bus=1,
            mv_bus=2,
            std_type="63/25/38 MVA 110/20/10 kV",
            index=0,
        ),
        partial(
            create_transformer3w_from_parameters,
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
            create_transformer_from_parameters,
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
            create_impedance,
            net=net,
            from_bus=0,
            to_bus=1,
            rft_pu=0.1,
            xft_pu=0.1,
            sn_mva=0.6,
            index=0,
        ),
        partial(create_switch, net, bus=0, element=1, et="b", index=0),
    ]
    for func in create_functions:
        with pytest.raises(
                Exception
        ):  # exception has to be raised since bus doesn't exist
            func()
    create_bus(net, 0.4)
    create_bus(net, 0.4)
    create_bus(net, 0.4)
    for func in create_functions:
        func()  # buses exist, element can be created
        with pytest.raises(
                Exception
        ):  # exception is raised because index already exists
            func()

    validate_network(net)


def test_create_bus():
    net = pandapowerNet(name="test_create_bus")
    # default
    b1 = create_bus(net, 110, test_kwargs="dummy_string")
    # with geodata
    b2 = create_bus(net, 110, geodata=(10, 20))

    assert len(net.bus) == 2
    assert net.bus.test_kwargs.at[b1] == "dummy_string"
    assert net.bus.at[b2, "geo"] == geojson.dumps(geojson.Point((10, 20)), sort_keys=True)

    validate_network(net)


def test_create_buses():
    net = pandapowerNet(name="test_create_buses")
    # standard
    b1 = create_buses(net, 3, 110, test_kwargs="dummy_string")
    # with geodata
    b2 = create_buses(net, 3, 110, geodata=(10, 20))
    # with geodata as array
    geodata = [(10, 20), (20, 30), (30, 40)]
    b3 = create_buses(net, 3, 110, geodata=geodata)

    assert len(net.bus) == 9
    assert net.bus.test_kwargs.at[b1[0]] == "dummy_string"

    for i in b2:
        assert net.bus.at[i, "geo"] == geojson.dumps(geojson.Point((10, 20)), sort_keys=True)
    for i, ind in enumerate(b3):
        assert net.bus.at[ind, "geo"] == geojson.dumps(geojson.Point(geodata[i]), sort_keys=True)

    validate_network(net)


def test_create_bus_dc():
    net = pandapowerNet(name="test_create_bus_dc")
    # default
    b1 = create_bus_dc(net, 110, test_kwargs="dummy_string")
    # with geodata
    b2 = create_bus_dc(net, 110, geodata=(10, 20))

    assert len(net.bus_dc) == 2
    assert net.bus_dc.test_kwargs.at[b1] == "dummy_string"
    assert net.bus_dc.at[b2, "geo"] == geojson.dumps(geojson.Point((10, 20)), sort_keys=True)

    validate_network(net)


def test_create_buses_dc():
    net = pandapowerNet(name="test_create_buses_dc")
    # standard
    b1 = create_buses_dc(net, 3, 110, test_kwargs="dummy_string")
    # with geodata
    b2 = create_buses_dc(net, 3, 110, geodata=(10, 20))
    # with geodata as array
    geodata = [(10, 20), (20, 30), (30, 40)]
    b3 = create_buses_dc(net, 3, 110, geodata=geodata)

    assert len(net.bus_dc) == 9
    assert net.bus_dc.test_kwargs.at[b1[0]] == "dummy_string"

    for i in b2:
        assert net.bus_dc.at[i, "geo"] == geojson.dumps(geojson.Point((10, 20)), sort_keys=True)
    for i, ind in enumerate(b3):
        assert net.bus_dc.at[ind, "geo"] == geojson.dumps(geojson.Point(geodata[i]), sort_keys=True)

    validate_network(net)
