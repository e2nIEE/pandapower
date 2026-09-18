# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pytest

from pandapower.create import (
    create_pwl_cost, create_pwl_costs, create_poly_costs, create_bus,
    create_gen, create_sgen, create_ext_grid, create_load, create_storage
)
from pandapower.network import pandapowerNet
from pandapower.network_schema.tools.validation.network_validation import validate_network

elements = [
    ("gen", create_gen, {"p_mw": 1.1}, {"test_kwargs": "dummy_string"}),
    ("sgen", create_sgen, {"p_mw": 1.1}, {"power_type": "q"}),
    ("ext_grid", create_ext_grid, {}, {}),
    ("load", create_load, {"p_mw": 1.1}, {}),
    #("dcline", create_dcline, {}, {}), # how to add second bus in this?
    ("storage", create_storage, {"p_mw": 1.1, "max_e_mwh": 27.2}, {})
]

@pytest.mark.parametrize("element, create_func, create_kwargs, kwargs", elements)
def test_create_pwl_cost(element, create_func, create_kwargs, kwargs):
    net = pandapowerNet(name="test_create_pwl_cost")
    b1 = create_bus(net, 110)
    points = [[0., 0.3, 10.2], [0.3, 0.6, 20.4], [0.6, 1., 30.6], [1., 1.3, 40.8]]
    # create element
    e = create_func(net, b1, **create_kwargs)
    pwl = create_pwl_cost(net, e, element, points, **kwargs)

    assert len(net.pwl_cost) == 1
    for k, v in kwargs.items():
        assert net.pwl_cost.at[pwl, k] == v

    validate_network(net)


def test_create_pwl_costs():
    net = pandapowerNet(name="test_create_pwl_costs")
    b1 = create_bus(net, 110)
    points = [[0., 0.3, 10.2], [0.3, 0.6, 20.4], [0.6, 1., 30.6], [1., 1.3, 40.8]]
    elms = []
    ets = []
    elm_count = len(elements)
    for et, c_func, c_kwargs, _ in elements:
        ets.append(et)
        elms.append(c_func(net, b1, **c_kwargs))
    create_pwl_costs(net, elms, ets, [points]*elm_count)

    assert len(net.pwl_cost) == elm_count

    validate_network(net)


@pytest.mark.parametrize("element, create_func, create_kwargs, kwargs", elements)
def test_create_poly_cost(element, create_func, create_kwargs, kwargs):
    net = pandapowerNet(name="test_create_poly_cost")
    b1 = create_bus(net, 110)
    points = [[0., 0.3, 10.2], [0.3, 0.6, 20.4], [0.6, 1., 30.6], [1., 1.3, 40.8]]
    # create element
    e = create_func(net, b1, **create_kwargs)
    pwl = create_pwl_cost(net, e, element, points, **kwargs)

    assert len(net.pwl_cost) == 1
    for k, v in kwargs.items():
        assert net.pwl_cost.at[pwl, k] == v

    validate_network(net)


def test_create_poly_costs():
    net = pandapowerNet(name="test_create_poly_costs")
    b1 = create_bus(net, 110)
    elms = []
    ets = []
    elm_count = len(elements)
    for et, c_func, c_kwargs, _ in elements:
        ets.append(et)
        elms.append(c_func(net, b1, **c_kwargs))

    # Create polynomial costs for all elements
    cp1_values = [1.0] * elm_count  # linear costs per MW
    create_poly_costs(net, elms, ets, cp1_eur_per_mw=cp1_values)

    assert len(net.poly_cost) == elm_count
    for i, elm in enumerate(elms):
        assert net.poly_cost.at[i, "element"] == elm
        assert np.isclose(net.poly_cost.at[i, "cp1_eur_per_mw"], 1.0)

    validate_network(net)
