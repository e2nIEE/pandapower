# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from this import d
import pytest
from copy import deepcopy
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pandapower as pp
import pandapower.networks as nw


def typed_list(iterable, dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return [int(it) for it in iterable]
    elif pd.api.types.is_numeric_dtype(dtype):
        return [float(it) for it in iterable]
    else:
        return [str(it) for it in iterable]


def nets_to_test_group():
    nets = list()
    reference_columns = [None, "name"]
    types = [int, str]
    idxs = list()
    for reference_column, type_ in zip(reference_columns, types):
        net = nw.case24_ieee_rts()
        for elm in pp.pp_elements():
            net[elm]["name"] = np.arange(net[elm].shape[0]).astype(str)
        idx0 = pp.create_group_from_dict(net, {
            "gen": typed_list([0, 1], type_),
            "sgen": typed_list([2, 3], type_)}, name='1st Group',
            reference_column=reference_column)
        idx1 = pp.create_group(
            net, "trafo", [typed_list(net.trafo.loc[:2].index, type_)],
            name='Group of transformers', index=3, reference_columns=reference_column)
        nets.append(net)
        idxs.append([idx0, idx1])
    return nets, types, reference_columns, idxs


def test_group_create():
    for net, type_, rc, idxs in zip(*nets_to_test_group()):

        # --- test definition of groups
        assert idxs[0] == 0
        assert idxs[1] == 3
        assert len(net.group.loc[[idxs[0]]].set_index("element_type").at["gen", "element"]) == \
            len(net.group.loc[[idxs[0]]].set_index("element_type").at["sgen", "element"]) == 2
        assert len(net.group.loc[[idxs[1]]].set_index("element_type").at["trafo", "element"]) == 3
        assert net.group.name.loc[[idxs[1]]].values[0] == 'Group of transformers'

        try:
            # no xward in net
            pp.create_group_from_dict(net, {
                "xward": typed_list([0], type_)}, reference_column=rc)
            assert False
        except UserWarning:
            pass

        try:
            # no sgen 100 in net
            pp.create_group_from_dict(net, {
                "sgen": typed_list([3, 100], type_)}, reference_column=rc)
            assert False
        except UserWarning:
            pass


def test_group_element_index():
    for net, type_, rc, idxs in zip(*nets_to_test_group()):

        # ! group_element_index()
        assert (pp.group_element_index(net, 0, "gen") == pd.Index([0, 1], dtype=int)).all()
        assert (pp.group_element_index(net, 0, "sgen") == pd.Index([2, 3], dtype=int)).all()
        assert (pp.group_element_index(net, 0, "dcline") == pd.Index([], dtype=int)).all()


def test_groups_equal():
    for net, type_, rc, idxs in zip(*nets_to_test_group()):

        idx_new = pp.create_group(net, "trafo", [typed_list(net.trafo.loc[:2].index, type_)],
                              name='Group of transformers', reference_columns=rc)

        # ! compare groups
        assert pp.groups_equal(net, 3, idx_new)


def test_set_group_reference_column():
    for net, type_, rc, idxs in zip(*nets_to_test_group()):

        idx_new = pp.create_group(net, "trafo", [typed_list(net.trafo.loc[:2].index, type_)],
                              name='Group of transformers', reference_columns=rc)
        assert pp.groups_equal(net, 3, idx_new)  # ensure that we have an equal group idx_new to
        # compare with after using set_group_reference_column() for and back

        pp.set_group_reference_column(net, 3, "origin_id")
        assert net.group.reference_column.at[3] == "origin_id"
        assert net.group.element.at[3] == net.trafo.origin_id.loc[:2].tolist()

        pp.set_group_reference_column(net, 3, rc)
        assert pp.groups_equal(net, 3, idx_new)


def test_compare_group_elements():
    for net, type_, rc, idxs in zip(*nets_to_test_group()):

        ok = pp.create_group(net, "trafo", [net.trafo.loc[:2].index], name='xxx')
        fail1 = pp.create_group(net, ["trafo", "bus"], [net.trafo.loc[:2].index, [0]], name='xxx')
        fail2 = pp.create_group(net, ["trafo"], [net.trafo.index[1:3]], name='xxx')

        # ! compare_group_elements
        assert pp.compare_group_elements(net, 3, ok)
        assert not pp.compare_group_elements(net, 3, fail1)
        assert not pp.compare_group_elements(net, 3, fail2)

        pp.set_group_reference_column(net, ok, "name")
        assert pp.compare_group_elements(net, 3, ok)


def test_ensure_lists_in_group_element_column():
    net = nets_to_test_group()[0][0]

    no_nans = [1, 1]
    vals = [[np.nan, pd.Index([2, 3]), {0, 1, 2}],
            [(2, 3, 4), None, 3]]
    for no_nan, val in zip(no_nans, vals):
        for drop in [True, False]:
            if drop and no_nan == 0:
                continue  # don't need to check dropping if there is nothing to drop

            netc = deepcopy(net)

            # manipulate element entries
            netc.group["element"] = val

            pp.ensure_lists_in_group_element_column(netc, drop_empty_lines=drop)

            expected_rows = net.group.shape[0] - no_nan if drop else net.group.shape[0]
            assert expected_rows == netc.group.shape[0]
            for i in range(netc.group.shape[0]):
                assert isinstance(netc.group.element.iat[i], list)



def test_remove_not_existing_group_members():
    for net, type_, rc, idxs in zip(*nets_to_test_group()):

        # ! remove_not_existing_group_members()
        assert set(net.group.loc[0].element_type.tolist()) == {"gen", "sgen"}

        # manipulate group table with false data
        net.group.element.iat[-1] = net.group.element.iat[-1] + [8]  # tafo 8 doesn't exist
        net.group = pd.concat([net.group, pd.DataFrame({
            "name": [net.group.name.iat[-1]]*3,
            "element_type": ["impedance", "line", "gen"],
            "element": [typed_list([8], type_),  # impedances don't exist
                        [],  # empty list
                        typed_list([998, 999], type_)],  # gen 998, 999 don't exist
            "reference_column": [rc]*3,
        }, index=[idxs[1]]*3)])

        # ensure that maipulations are done as expected
        assert len(net.group.at[idxs[1], "element"]) == 4
        assert "gen" in net.group.element_type.loc[[idxs[1]]].values

        # run remove_not_existing_group_members()
        pp.remove_not_existing_group_members(net, verbose=False)

        assert len(net.group.at[idxs[1], "element"]) == 3
        assert "impedance" not in net.group.element_type.loc[[idxs[1]]].values
        assert "line" not in net.group.element_type.loc[[idxs[1]]].values
        assert "gen" not in net.group.element_type.loc[[idxs[1]]].values


def test_drop_element():
    net = nw.case24_ieee_rts()
    gr1 = pp.create_group_from_dict(net, {
        "bus": [0, 1, 2], "gen": [0, 1], "sgen": [2, 3], "line": [0, 1], "trafo": [0, 1]},
                                    name='1st Group', index=2)

    pp.drop_lines(net, [0])
    assert net.group.loc[[gr1]].set_index("element_type").at["line", "element"] == [1]
    pp.drop_lines(net, [1])
    assert "line" not in net.group.element_type.values

    pp.drop_trafos(net, [1])
    assert net.group.loc[[gr1]].set_index("element_type").at["trafo", "element"] == [0]

    pp.drop_buses(net, [0], drop_elements=False)
    assert net.group.loc[[gr1]].set_index("element_type").at["bus", "element"] == [1, 2]

    pp.drop_buses(net, [1])  # not only bus 0 is dropped but also connected elements
    assert net.group.loc[[gr1]].set_index("element_type").at["bus", "element"] == [2]
    assert net.group.loc[[gr1]].set_index("element_type").at["gen", "element"] == [0]
    assert net.group.loc[[gr1]].set_index("element_type").at["sgen", "element"] == [2]

    pp.drop_elements_simple(net, "sgen", [2])
    assert "sgen" not in net.group.element_type.values


def test_drop_and_return():
    for net, type_, rc, idxs in zip(*nets_to_test_group()):

        # ! drop_elements_and_group & ! return_group_as_net
        for keep_everything_else in [False, True]:

            net2 = deepcopy(net)
            pp.drop_group_and_elements(net2, 0)

            net3 = pp.return_group_as_net(
                net, 0, keep_everything_else=keep_everything_else, verbose=False)
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
    for net, type_, rc, idxs in zip(*nets_to_test_group()):

        # ! set_out_of_service
        assert net.trafo.in_service.all()
        pp.set_group_out_of_service(net, 3)
        assert (net.trafo.in_service == [False]*3 + [True]*2).all()
        pp.set_group_in_service(net, 3)
        assert net.trafo.in_service.all()


def test_append_to_group():
    for net, type_, rc, idxs in zip(*nets_to_test_group()):

        # ! group_element_lists() and ! append_to_group()
        et0, elm0, rc0 = pp.group_element_lists(net, 0)
        assert len(et0) == len(elm0) == len(rc0)
        pp.append_to_group(net, idxs[1], et0, elm0, rc0)
        assert set(net.group.loc[[idxs[1]]].element_type.tolist()) == {"gen", "sgen", "trafo"}

        try:
            # no xward in net
            pp.append_to_group(net, idxs[1], ["xward"], [typed_list([0], type_)],
                               reference_columns=rc)
            assert False
        except UserWarning:
            pass

        pp.append_to_group(net, idxs[1], ["trafo", "line"],
                           [typed_list([3], type_), typed_list([2], type_)], reference_columns=rc)
        assert set(net.group.loc[[idxs[1]]].element_type.tolist()) == {
            "gen", "sgen", "trafo", "line"}
        assert len(net.group.loc[[idxs[1]]].set_index("element_type").at["trafo", "element"]) == 4


def test_drop_and_compare():
    for net, type_, rc, idxs in zip(*nets_to_test_group()):

        # ! drop_from_group() & ! compare_elements_dict()

        # copy group 3
        et3, elm3, rc3 = pp.group_element_lists(net, 3)
        copy_idx = pp.create_group(net, et3, elm3, reference_columns=rc3, name="copy of group 3")

        # drop elements which are not in group 3
        pp.drop_from_group(net, 3, "xward", [1, 17])
        pp.drop_from_group(net, 3, "line", 2)

        # check that group3 is still the same as the copy
        assert pp.compare_group_elements(net, 3, copy_idx)

        # drop some members
        pp.drop_from_group(net, 3, "trafo", 1)
        assert pp.group_element_lists(net, 3)[0] == ["trafo"]
        assert pp.group_element_lists(net, 3)[1] == [typed_list([0, 2], type_)]
        assert pp.group_element_lists(net, 3)[2] == [None if type_ is int else "name"]


def test_res_power():
    for net, type_, rc, idxs in zip(*nets_to_test_group()):
        et0, elm0, rc0 = pp.group_element_lists(net, 0)
        pp.append_to_group(net, 3, et0, elm0, reference_columns=rc0)

        # ! res_p_mw() and res_q_mvar()
        pp.runpp(net)
        p_val = net.res_trafo.pl_mw.loc[[0, 1, 2]].sum() - net.res_gen.p_mw.loc[[0, 1]].sum() - \
            net.res_sgen.p_mw.loc[[2, 3]].sum()
        assert np.isclose(pp.group_res_p_mw(net, 3), p_val)


def test_group_io():
    net = nw.case24_ieee_rts()
    gr1 = pp.create_group_from_dict(net, {"gen": [0, 1], "sgen": [2, 3], "load": [0]},
                                    name='1st Group', index=2)
    gr2 = pp.create_group_from_dict(net, {"trafo": net.trafo.index}, name='Group of transformers')
    json_str = pp.to_json(net)
    net2 = pp.from_json_string(json_str)
    pp.runpp(net)
    pp.runpp(net2)
    assert pp.group_res_p_mw(net, gr1) == pp.group_res_p_mw(net2, gr1)
    assert pp.group_res_p_mw(net, gr2) == pp.group_res_p_mw(net2, gr2)
    pdt.assert_frame_equal(net.group.loc[[gr1]], net2.group.loc[[gr1]])
    pdt.assert_frame_equal(net.group.loc[[gr2]], net2.group.loc[[gr2]])


def test_count_group_elements():
    for net, type_, rc, idxs in zip(*nets_to_test_group()):
        pdt.assert_series_equal(
            pp.count_group_elements(net, idxs[0]),
            pd.Series({"gen": 2, "sgen": 2}, dtype=int))
        pdt.assert_series_equal(
            pp.count_group_elements(net, idxs[1]),
            pd.Series({"trafo": 3}, dtype=int))


def test_isin():
    for net, type_, rc, idxs in zip(*nets_to_test_group()):
        assert np.all(np.array([False, True, True, False]) == \
            pp.isin_group(net, "sgen", [0, 2, 3, 4]))
        assert pp.isin_group(net, "gen", 0)
        assert not pp.isin_group(net, "gen", 0, index=idxs[1])
        assert not pp.isin_group(net, "gen", 6)


if __name__ == "__main__":
    if 0:
        pytest.main(['-x', "test_group.py"])
    else:
        test_group_create()
        test_group_element_index()
        test_groups_equal()
        test_set_group_reference_column()
        test_compare_group_elements()
        test_ensure_lists_in_group_element_column()
        test_remove_not_existing_group_members()
        test_drop_element()
        test_drop_and_return()
        test_set_out_of_service()
        test_append_to_group()
        test_drop_and_compare()
        test_res_power()
        test_group_io()
        test_count_group_elements()
        test_isin()
        pass
