# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest

from pandapower.networks import simple_mv_open_ring_net
from pandapower import create_empty_network, add_temperature_coefficient
from pandapower.create import (
    create_bus,
    create_buses,
    create_transformer3w,
    create_line,
    create_line_from_parameters
)
from pandapower.std_types import (
    create_std_type,
    create_std_types,
    load_std_type,
    change_std_type,
    find_std_type_alternative,
    find_std_type_by_parameter,
    delete_std_type,
    rename_std_type,
    copy_std_types,
    std_type_exists,
    parameter_from_std_type
)


def test_create_and_load_std_type_line():
    net = create_empty_network()
    c = 40
    r = 0.01
    x = 0.02
    i = 0.2
    name = "test_line"

    typdata = {}
    with pytest.raises(UserWarning):
        create_std_type(net, name=name, data=typdata, element="line")

    typdata = {"c_nf_per_km": c}
    with pytest.raises(UserWarning):
        create_std_type(net, name=name, data=typdata, element="line")

    typdata = {"c_nf_per_km": c, "r_ohm_per_km": r}
    with pytest.raises(UserWarning):
        create_std_type(net, name=name, data=typdata, element="line")

    typdata = {"c_nf_per_km": c, "r_ohm_per_km": r, "x_ohm_per_km": x}
    with pytest.raises(UserWarning):
        create_std_type(net, name=name, data=typdata, element="line")

    typdata = {"c_nf_per_km": c, "r_ohm_per_km": r, "x_ohm_per_km": x, "max_i_ka": i}
    create_std_type(net, name=name, data=typdata, element="line")
    assert net.std_types["line"][name] == typdata

    loaded_type = load_std_type(net, name)
    assert loaded_type == typdata


def test_create_std_types_line():
    net = create_empty_network()
    c = 40
    r = 0.01
    x = 0.02
    i = 0.2

    typdata = {"c_nf_per_km": c, "r_ohm_per_km": r, "x_ohm_per_km": x, "max_i_ka": i}

    typdatas = {"typ1": typdata, "typ2": typdata}
    create_std_types(net, data=typdatas, element="line")
    assert net.std_types["line"]["typ1"] == typdata
    assert net.std_types["line"]["typ1"] == typdata


def test_create_std_types_from_net_line():
    net1 = create_empty_network()
    net2 = create_empty_network()

    c = 40
    r = 0.01
    x = 0.02
    i = 0.2

    typdata = {"c_nf_per_km": c, "r_ohm_per_km": r, "x_ohm_per_km": x, "max_i_ka": i,
               "additional": 8}
    create_std_type(net1, typdata, "test_copy")
    copy_std_types(net2, net1, element="line")
    assert std_type_exists(net2, "test_copy")


def test_create_and_load_std_type_trafo():
    net = create_empty_network()
    sn_mva = 40
    vn_hv_kv = 110
    vn_lv_kv =  20
    vk_percent = 5.
    vkr_percent = 2.
    pfe_kw=50
    i0_percent = 0.1
    shift_degree = 30
    name = "test_trafo"

    typdata = {"sn_mva": sn_mva}
    with pytest.raises(UserWarning):
        create_std_type(net, name=name, data=typdata, element="trafo")

    typdata = {"sn_mva": sn_mva, "vn_hv_kv": vn_hv_kv}
    with pytest.raises(UserWarning):
        create_std_type(net, name=name, data=typdata, element="trafo")

    typdata = {"sn_mva": sn_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv}
    with pytest.raises(UserWarning):
        create_std_type(net, name=name, data=typdata, element="trafo")

    typdata = {"sn_mva": sn_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_percent": vk_percent}
    with pytest.raises(UserWarning):
        create_std_type(net, name=name, data=typdata, element="trafo")

    typdata = {"sn_mva": sn_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_percent": vk_percent,
               "vkr_percent": vkr_percent}
    with pytest.raises(UserWarning):
        create_std_type(net, name=name, data=typdata, element="trafo")

    typdata = {"sn_mva": sn_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_percent": vk_percent,
               "vkr_percent": vkr_percent, "pfe_kw": pfe_kw}
    with pytest.raises(UserWarning):
        create_std_type(net, name=name, data=typdata, element="trafo")

    typdata = {"sn_mva": sn_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_percent": vk_percent,
               "vkr_percent": vkr_percent, "pfe_kw": pfe_kw, "i0_percent": i0_percent}
    with pytest.raises(UserWarning):
        create_std_type(net, name=name, data=typdata, element="trafo")
    typdata = {"sn_mva": sn_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_percent": vk_percent,
               "vkr_percent": vkr_percent, "pfe_kw": pfe_kw, "i0_percent": i0_percent,
               "shift_degree": shift_degree}
    create_std_type(net, name=name, data=typdata, element="trafo")
    assert net.std_types["trafo"][name] == typdata

    loaded_type = load_std_type(net, name, element="trafo")
    assert loaded_type == typdata


def test_create_and_load_std_type_trafo3w():
    net = create_empty_network()
    sn_hv_mva = 40; sn_mv_mva = 20; sn_lv_mva = 20
    vn_hv_kv = 110; vn_mv_kv = 50; vn_lv_kv = 20
    vk_hv_percent = 5.; vk_mv_percent = 5.; vk_lv_percent = 5.
    vkr_hv_percent = 2.; vkr_mv_percent = 2.; vkr_lv_percent = 2.
    pfe_kw=50
    i0_percent = 0.1
    shift_mv_degree = 30; shift_lv_degree = 30
    name = "test_trafo3w"

    typdata = {"sn_hv_mva": sn_hv_mva}
    with pytest.raises(UserWarning):
        create_std_type(net, name=name, data=typdata, element="trafo3w")

    typdata = {"sn_mv_mva": sn_mv_mva, "vn_hv_kv": vn_hv_kv}
    with pytest.raises(UserWarning):
        create_std_type(net, name=name, data=typdata, element="trafo3w")

    typdata = {"sn_lv_mva": sn_lv_mva, "vn_mv_kv": vn_mv_kv, "vn_lv_kv": vn_lv_kv}
    with pytest.raises(UserWarning):
        create_std_type(net, name=name, data=typdata, element="trafo3w")

    typdata = {"sn_mv_mva": sn_mv_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_hv_percent": vk_hv_percent}
    with pytest.raises(UserWarning):
        create_std_type(net, name=name, data=typdata, element="trafo3w")

    typdata = {"sn_hv_mva": sn_hv_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_mv_percent": vk_mv_percent,
               "vkr_hv_percent": vkr_hv_percent}
    with pytest.raises(UserWarning):
        create_std_type(net, name=name, data=typdata, element="trafo3w")

    typdata = {"sn_hv_mva": sn_hv_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_lv_percent": vk_lv_percent,
               "vkr_mv_percent": vkr_mv_percent, "pfe_kw": pfe_kw}
    with pytest.raises(UserWarning):
        create_std_type(net, name=name, data=typdata, element="trafo3w")

    typdata = {"sn_hv_mva": sn_hv_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_hv_percent": vk_hv_percent,
               "vkr_lv_percent": vkr_lv_percent, "pfe_kw": pfe_kw, "i0_percent": i0_percent}
    with pytest.raises(UserWarning):
        create_std_type(net, name=name, data=typdata, element="trafo3w")
    typdata = {"sn_hv_mva": sn_hv_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_hv_percent": vk_hv_percent,
               "vkr_hv_percent": vkr_hv_percent, "pfe_kw": pfe_kw, "i0_percent": i0_percent,
               "shift_mv_degree": shift_mv_degree}
    with pytest.raises(UserWarning):
        create_std_type(net, name=name, data=typdata, element="trafo3w")
    typdata = {"vn_hv_kv": vn_hv_kv, "vn_mv_kv": vn_mv_kv, "vn_lv_kv": vn_lv_kv, "sn_hv_mva": sn_hv_mva,
          "sn_mv_mva": sn_mv_mva, "sn_lv_mva": sn_lv_mva, "vk_hv_percent": vk_hv_percent, "vk_mv_percent": vk_mv_percent,
          "vk_lv_percent": vk_lv_percent, "vkr_hv_percent": vkr_hv_percent, "vkr_mv_percent": vkr_mv_percent,
          "vkr_lv_percent": vkr_lv_percent, "pfe_kw": pfe_kw, "i0_percent": i0_percent,
          "shift_mv_degree":shift_mv_degree, "shift_lv_degree": shift_lv_degree}
    create_std_type(net, name=name, data=typdata, element="trafo3w")
    assert net.std_types["trafo3w"][name] == typdata

    loaded_type = load_std_type(net, name, element="trafo3w")
    assert loaded_type == typdata


def test_create_std_types_trafo():
    net = create_empty_network()
    sn_mva = 40
    vn_hv_kv = 110
    vn_lv_kv =  20
    vk_percent = 5.
    vkr_percent = 2.
    pfe_kw=50
    i0_percent = 0.1
    shift_degree = 30

    typdata = {"sn_mva": sn_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_percent": vk_percent,
               "vkr_percent": vkr_percent, "pfe_kw": pfe_kw, "i0_percent": i0_percent,
               "shift_degree": shift_degree}
    typdatas = {"typ1": typdata, "typ2": typdata}
    create_std_types(net, data=typdatas, element="trafo")
    assert net.std_types["trafo"]["typ1"] == typdata
    assert net.std_types["trafo"]["typ2"] == typdata


def test_create_std_types_trafo3w():
    net = create_empty_network()
    sn_hv_mva = 40; sn_mv_mva = 20; sn_lv_mva = 20
    vn_hv_kv = 110; vn_mv_kv = 50; vn_lv_kv = 20
    vk_hv_percent = 5.; vk_mv_percent = 5.; vk_lv_percent = 5.
    vkr_hv_percent = 2.; vkr_mv_percent = 2.; vkr_lv_percent = 2.
    pfe_kw=50
    i0_percent = 0.1
    shift_mv_degree = 30; shift_lv_degree = 30

    typdata = {"vn_hv_kv": vn_hv_kv, "vn_mv_kv": vn_mv_kv, "vn_lv_kv": vn_lv_kv, "sn_hv_mva": sn_hv_mva,
          "sn_mv_mva": sn_mv_mva, "sn_lv_mva": sn_lv_mva, "vk_hv_percent": vk_hv_percent, "vk_mv_percent": vk_mv_percent,
          "vk_lv_percent": vk_lv_percent, "vkr_hv_percent": vkr_hv_percent, "vkr_mv_percent": vkr_mv_percent,
          "vkr_lv_percent": vkr_lv_percent, "pfe_kw": pfe_kw, "i0_percent": i0_percent,
          "shift_mv_degree":shift_mv_degree, "shift_lv_degree": shift_lv_degree}

    typdatas = {"typ1": typdata, "typ2": typdata}
    create_std_types(net, data=typdatas, element="trafo3w")
    assert net.std_types["trafo3w"]["typ1"] == typdata
    assert net.std_types["trafo3w"]["typ2"] == typdata


def test_find_line_type():
    net = create_empty_network()
    c = 40000
    r = 1.5
    x = 2.0
    i = 10
    name = "test_line1"
    typdata = {"c_nf_per_km": c, "r_ohm_per_km": r, "x_ohm_per_km": x, "max_i_ka": i}
    create_std_type(net, data=typdata, name=name, element="line")

    fitting_type = find_std_type_by_parameter(net, typdata)
    assert len(fitting_type) == 1
    assert fitting_type[0] == name

    fitting_type = find_std_type_by_parameter(net, {"r_ohm_per_km":r+0.05}, epsilon=.06)
    assert len(fitting_type) == 1
    assert fitting_type[0] == name


def test_find_std_alternative():
    net = create_empty_network()
    c = 210
    r = 0.642
    x = 0.083
    i = 0.142
    vr = "LV"
    ## {'NAYY 4x50 SE': {'c_nf_per_km': 210, 'r_ohm_per_km': 0.642, 'x_ohm_per_km': 0.083, 'max_i_ka': 0.142, 'voltage_rating': 'LV'}
    # Assuming we are looking for the cable NAYY 4X50 SE with a maximum ampacity of 0.142 A
    name ='NAYY 4x50 SE'
    typdata = {"c_nf_per_km": c, "r_ohm_per_km": r, "x_ohm_per_km": x, "max_i_ka": i, "voltage_rating": vr}
    fitting_type = find_std_type_alternative(net, {"r_ohm_per_km":r+0.05}, voltage_rating ="LV", epsilon=0.06)
    assert len(fitting_type) == 1
    assert fitting_type[0] == name

    fitting_type = find_std_type_alternative(net, {"r_ohm_per_km":r+0.07}, voltage_rating ="LV", epsilon=0.06)
    assert len(fitting_type) == 0

    fitting_type = find_std_type_alternative(net, {"r_ohm_per_km":r+0.07}, voltage_rating ="MV", epsilon=0.06)
    assert len(fitting_type) == 0


def test_change_type_line():
    net = create_empty_network()
    r1 = 0.01
    x1 = 0.02
    c1 = 40
    i1 = 0.2
    name1 = "test_line1"
    typ1 = {"c_nf_per_km": c1, "r_ohm_per_km": r1, "x_ohm_per_km": x1, "max_i_ka": i1}
    create_std_type(net, data=typ1, name=name1, element="line")

    r2 = 0.02
    x2 = 0.04
    c2 = 20
    i2 = 0.4
    name2 = "test_line2"
    typ2 = {"c_nf_per_km": c2, "r_ohm_per_km": r2, "x_ohm_per_km": x2, "max_i_ka": i2}
    create_std_type(net, data=typ2, name=name2, element="line")

    b1 = create_bus(net, vn_kv=0.4)
    b2 = create_bus(net, vn_kv=0.4)
    lid = create_line(net, b1, b2, 1., std_type=name1)
    assert net.line.r_ohm_per_km.at[lid] == r1
    assert net.line.x_ohm_per_km.at[lid] == x1
    assert net.line.c_nf_per_km.at[lid] == c1
    assert net.line.max_i_ka.at[lid] == i1
    assert net.line.std_type.at[lid] == name1

    change_std_type(net, lid, name2)

    assert net.line.r_ohm_per_km.at[lid] == r2
    assert net.line.x_ohm_per_km.at[lid] == x2
    assert net.line.c_nf_per_km.at[lid] == c2
    assert net.line.max_i_ka.at[lid] == i2
    assert net.line.std_type.at[lid] == name2


def test_parameter_from_std_type_line():
    net = create_empty_network()
    r1 = 0.01
    x1 = 0.02
    c1 = 40
    i1 = 0.2
    name1 = "test_line1"
    typ1 = {"c_nf_per_km": c1, "r_ohm_per_km": r1, "x_ohm_per_km": x1, "max_i_ka": i1}
    create_std_type(net, data=typ1, name=name1, element="line")

    r2 = 0.02
    x2 = 0.04
    c2 = 20
    i2 = 0.4
    endtemp2 = 40

    endtemp_fill = 20

    name2 = "test_line2"
    typ2 = {"c_nf_per_km": c2, "r_ohm_per_km": r2, "x_ohm_per_km": x2, "max_i_ka": i2,
            "endtemp_degree": endtemp2}
    create_std_type(net, data=typ2, name=name2, element="line")

    b1 = create_bus(net, vn_kv=0.4)
    b2 = create_bus(net, vn_kv=0.4)
    lid1 = create_line(net, b1, b2, 1., std_type=name1)
    lid2 = create_line(net, b1, b2, 1., std_type=name2)
    lid3 = create_line_from_parameters(net, b1, b2, 1., r_ohm_per_km=0.03, x_ohm_per_km=0.04,
                                          c_nf_per_km=20, max_i_ka=0.3)

    parameter_from_std_type(net, "endtemp_degree", fill=endtemp_fill)
    assert net.line.endtemp_degree.at[lid1] == endtemp_fill #type1 one has not specified an endtemp
    assert net.line.endtemp_degree.at[lid2] == endtemp2 #type2 has specified endtemp
    assert net.line.endtemp_degree.at[lid3] == endtemp_fill #line3 has no standard type

    net.line.at[lid3, "endtemp_degree"] = 10
    parameter_from_std_type(net, "endtemp_degree", fill=endtemp_fill)
    assert net.line.endtemp_degree.at[lid3] == 10 #check that existing values arent overwritten


def test_add_temperature_coefficient():
    net = simple_mv_open_ring_net()
    add_temperature_coefficient(net)
    assert "alpha" in net.line.columns
    assert all(net.line.alpha == 4.03e-3)


def test_delete_std_type():
    net = create_empty_network()
    trafo3w_types = set(net.std_types["trafo3w"].keys())
    existing_trafo3w_std_type = sorted(trafo3w_types)[0]
    delete_std_type(net, existing_trafo3w_std_type, "trafo3w")
    assert trafo3w_types == set(net.std_types["trafo3w"].keys()) | {existing_trafo3w_std_type}


def test_rename_std_type():
    net = create_empty_network()
    existing_line_std_type = sorted(net.std_types["line"].keys())[0]
    existing_line_std_type2 = sorted(net.std_types["line"].keys())[1]
    existing_trafo3w_std_type = sorted(net.std_types["trafo3w"].keys())[0]
    tr3w_std_type_params = load_std_type(net, existing_trafo3w_std_type, "trafo3w")

    vn_kvs = [tr3w_std_type_params["vn_hv_kv"], tr3w_std_type_params["vn_hv_kv"],
              tr3w_std_type_params["vn_mv_kv"], tr3w_std_type_params["vn_lv_kv"]]
    create_buses(net, 4, vn_kvs)
    create_line(net, 0, 1, 1.2, existing_line_std_type)
    create_line(net, 0, 1, 1.2, existing_line_std_type2)
    create_transformer3w(net, 0, 2, 3, existing_trafo3w_std_type)

    rename_std_type(net, existing_line_std_type, "new_line_std_type")
    rename_std_type(net, existing_trafo3w_std_type, "new_trafo3w_std_type", "trafo3w")

    assert existing_line_std_type not in net.std_types["line"].keys()
    assert "new_line_std_type" in net.std_types["line"].keys()
    assert (net.line.std_type == existing_line_std_type).sum() == 0
    assert (net.line.std_type == "new_line_std_type").sum() == 1

    assert existing_line_std_type not in net.std_types["trafo3w"].keys()
    assert "new_trafo3w_std_type" in net.std_types["trafo3w"].keys()
    assert (net.trafo3w.std_type == existing_line_std_type).sum() == 0
    assert (net.trafo3w.std_type == "new_trafo3w_std_type").sum() == 1

    try:
        rename_std_type(net, "abcdefghijklmnop", "new_line_std_type2")
        assert False, "an error is expected"
    except UserWarning:
        pass
    try:
        rename_std_type(net, existing_line_std_type2, "new_line_std_type")
        assert False, "an error is expected"
    except UserWarning:
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
