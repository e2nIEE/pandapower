# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest
from copy import deepcopy
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as nw


def typed_list(iterable, dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return [int(it) for it in iterable]
    elif pd.api.types.is_numeric_dtype(dtype):
        return [float(it) for it in iterable]
    else:
        return [str(it) for it in iterable]


def typed_set(iterable, dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return {int(it) for it in iterable}
    elif pd.api.types.is_numeric_dtype(dtype):
        return {float(it) for it in iterable}
    else:
        return {str(it) for it in iterable}


def test_group_create():
    nets = list()
    reference_columns = [None, "name"]
    types = [int, str]
    for reference_column, type in zip(reference_columns, types):
        net = nw.case24_ieee_rts()
        for elm in pp.pp_elements():
            net[elm]["name"] = np.arange(net[elm].shape[0]).astype(str)
        idx0 = pp.create_group_from_dict(net, {
            "gen": typed_list([0, 1], type),
            "sgen": typed_list([2, 3], type),
            "xward": typed_list([0], type)}, name='1st Group',
            reference_column=reference_column)
        name_gr2 = 'Group of transformers'
        idx1 = pp.create_group(net, "trafo", typed_list(net.trafo.index[:3], type),
                              name=name_gr2, index=3, reference_column=reference_column)

        # --- test definition of groups
        assert idx0 == 0
        assert idx1 == 3
        assert len(net.group.loc[idx0].set_index("element_type").at["gen"]) == \
            len(net.group.loc[idx0].set_index("element_type").at["sgen"]) == 2
        assert len(net.group.loc[idx0].set_index("element_type").at["trafo"]) == 3
        assert net.group.name.loc[idx1].values[0] == name_gr2

        nets.append(net)
    return nets, types


def test_update_elements_dict():
    for net, type in zip(*test_group_create()):

        # 1) update_elements_dict()
        assert set(net.group.loc[0].element_type.tolist()) == {"gen", "sgen"}
        net.group = pd.concat([net.group, pd.DataFrame()])

        gr2.elements_dict["trafo"] = gr2.elements_dict["trafo"].union([8])
        gr2.elements_dict["impedance"] = typed_list([8], type)
        gr2.elements_dict["line"] = []
        gr2.elements_dict["gen"] = typed_list([998, 999], type)
        assert len(gr2.elements_dict["trafo"]) == 4
        assert "gen" in gr2.elements_dict.keys()
        gr2.update_elements_dict(net, verbose=False)
        assert len(gr2.elements_dict["trafo"]) == 3
        assert "impedance" not in gr2.elements_dict.keys()
        assert "line" not in gr2.elements_dict.keys()
        assert "gen" not in gr2.elements_dict.keys()


def test_drop_and_return():
    for net, type in zip(*test_group_create()):

        # 2) drop_elements_and_group & 3) return_group_as_net
        for keep_everything_else in [False, True]:

            net2 = deepcopy(net)
            pp.drop_group_and_elements(net2, 0)

            net3 = pp.return_group_as_net(
                net, 0, keep_everything_else=keep_everything_else)
            if keep_everything_else:
                assert len(set(net3.group.index)) == 2
            else:
                assert len(set(net3.group.index)) == 1

            assert net.gen.shape[0] == 10  # unchanged
            assert net2.gen.shape[0] == 8
            assert set(net3.gen.index) == {0, 1}
            for elm in pp.pp_elements():
                assert net2[elm].shape[0] <= net[elm].shape[0]
                assert set(net2[elm].index) | set(net3[elm].index) == set(net[elm].index)
                assert net3[elm].shape[0] >= 0


def test_set_out_of_service():
    for net, type in zip(*test_group_create()):

        # 4) set_out_of_service
        assert net.trafo.in_service.all()
        pp.set_group_out_of_service(net, 3)
        assert (net.trafo.in_service == [False]*3 + [True]*2).all()
        pp.set_group_in_service(net, 3)
        assert net.trafo.in_service.all()


def test_append_to_group():
    for net, type in zip(*test_group_create()):

        # 5) group_element_lists() and 6) append_to_group()
        et0, elm0, rc0 = pp.group_element_lists(net, 0)
        assert len(et0) == len(elm0) == len(rc0)
        pp.append_to_group(net, 3, et0, elm0, rc0)
        assert set(net.group.loc[3].element_type.tolist()) == {"gen", "sgen", "trafo"}
        pp.append_to_group(net, 3,
            ["xward", "line"], [typed_list([1, 2], type), typed_list([2], type)])  # TODO: hier wird wahrscheinlich nicht getestet, ob xward existiert
        assert set(net.group.loc[3].element_type.tolist()) == {"gen", "sgen", "trafo", "line"}


def test_drop_and_compare():
    for net, type in zip(*test_group_create()):

        # 7) drop_from_group() & 8) compare_elements_dict()
        et3, elm3, rc3 = pp.group_element_lists(net, 3)
        pp.drop_from_group(net, "xward", typed_set({1, 17}, type), index=3)
        pp.drop_from_group(net, "line", 2 if type is int else "2", index=3)
        idx_new = pp.create_group(net, et3, elm3, rc3, name="for testing")
        assert pp.compare_group_elements(net, 3, idx_new)
        assert not gr2.compare_elements_dict({
            'trafo': typed_list([0, 1, 2], type), 'gen': typed_list([1], type),
            'sgen': typed_list([2, 3], type)})


def test_get_index():
    for net, type in zip(*test_group_create()):

        # 9) get_index()
        assert (net.group.object.at[0].get_index(net, "gen") == pd.Index([0, 1], dtype=int)).all()
        assert (net.group.object.at[0].get_index(net, "sgen") == pd.Index([2, 3], dtype=int)).all()
        assert (net.group.object.at[0].get_index(net, "dcline") == pd.Index([], dtype=int)).all()


def test_res_power():
    for net, type in zip(*test_group_create()):
        gr2.append_to_group(net.group.object.at[0].elements_dict)

        # 10) res_p_mw() and res_q_mvar()
        pp.runpp(net)
        p_val = net.res_trafo.pl_mw.loc[[0, 1, 2]].sum() - net.res_gen.p_mw.loc[[0, 1]].sum() - \
            net.res_sgen.p_mw.loc[[2, 3]].sum()
        assert np.isclose(gr2.res_p_mw(net), p_val)


def test_compare_groups():
    for net, type in zip(*test_group_create()):

        # 11) compare groups
        gr2_copy = deepcopy(gr2)
        assert gr2 == gr2_copy


def test_group_io():
    net = nw.case24_ieee_rts()
    gr1 = pp.Group(net, {"gen": [0, 1], "sgen": [2, 3], "load": [0]}, name='1st Group', index=2)
    gr2 = pp.Group(net, {"trafo": net.trafo.index}, name='Group of transformers')
    s1 = pp.to_json(gr1)
    s2 = pp.to_json(gr2)
    gr11 = pp.from_json_string(s1)
    gr22 = pp.from_json_string(s2)
    assert gr1 == gr11
    assert gr2 == gr22


if __name__ == "__main__":
    if 0:
        pytest.main(['-x', "test_group.py"])
    else:
        test_group_create()
        test_update_elements_dict()
        test_drop_and_return()
        test_set_out_of_service()
        test_append_to_group()
        test_drop_and_compare()
        test_get_index()
        test_res_power()
        test_compare_groups()
        test_group_io()
        pass
