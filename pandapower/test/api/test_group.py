import pytest
from copy import deepcopy
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as nw
from pandapower.groups import Group_to_json_string, Group_from_json_string

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


def test_groups():
    for element_column, elm_type in zip([None, "name"], [int, str]):
        net = nw.case24_ieee_rts()
        for elm in pp.pp_elements():
            net[elm]["name"] = np.arange(net[elm].shape[0]).astype(str)
        pp.Group(net, {"gen": typed_list([0, 1], elm_type), "sgen": typed_list([2, 3], elm_type),
                    "xward": typed_list([0], elm_type)}, name='1st Group',
                    element_column=element_column)
        name_gr2 = 'Group of transformers'
        gr2 = pp.Group(net, {"trafo": typed_list(net.trafo.index[:3], elm_type)}, name=name_gr2,
                       index=3, element_column=element_column)

        # --- test definition of groups
        assert "group" in net.keys()
        assert len(net.group.object.at[0].elements_dict["gen"]) == \
            len(net.group.object.at[0].elements_dict["sgen"]) == 2
        assert len(net.group.object.at[3].elements_dict["trafo"]) == 3
        assert net.group.name.at[3] == name_gr2

        # --- test functions
        # 1) update_elements_dict()
        assert set(net.group.object.at[0].elements_dict.keys()) == {"gen", "sgen"}
        gr2.elements_dict["trafo"] = gr2.elements_dict["trafo"].union([8])
        gr2.elements_dict["impedance"] = typed_list([8], elm_type)
        gr2.elements_dict["line"] = []
        gr2.elements_dict["gen"] = typed_list([998, 999], elm_type)
        assert len(gr2.elements_dict["trafo"]) == 4
        assert "gen" in gr2.elements_dict.keys()
        gr2.update_elements_dict(net, verbose=False)
        assert len(gr2.elements_dict["trafo"]) == 3
        assert "impedance" not in gr2.elements_dict.keys()
        assert "line" not in gr2.elements_dict.keys()
        assert "gen" not in gr2.elements_dict.keys()

        # 2) drop_elements_and_group & 3) return_group_as_net
        for keep_everything_else in [False, True]:

            net2 = deepcopy(net)
            net2.group.object.at[0].drop_elements_and_group(net2)

            net3 = net.group.object.at[0].return_group_as_net(
                net, keep_everything_else=keep_everything_else)
            if keep_everything_else:
                assert net3.group.shape[0] == 2
            else:
                assert net3.group.shape[0] == 1

            assert net.gen.shape[0] == 10  # unchanged
            assert net2.gen.shape[0] == 8
            assert set(net3.gen.index) == {0, 1}
            for elm in pp.pp_elements():
                assert net2[elm].shape[0] <= net[elm].shape[0]
                assert set(net2[elm].index) | set(net3[elm].index) == set(net[elm].index)
                assert net3[elm].shape[0] >= 0

        # 4) set_out_of_service
        assert net.trafo.in_service.all()
        gr2.set_out_of_service(net)
        assert (net.trafo.in_service == [False]*3 + [True]*2).all()
        gr2.set_in_service(net)
        assert net.trafo.in_service.all()

        # 5) append_to_group()
        gr2.append_to_group(net.group.object.at[0].elements_dict)
        assert set(gr2.elements_dict.keys()) == {"gen", "sgen", "trafo"}
        elms_b4_append2 = deepcopy(gr2.elements_dict)
        gr2.append_to_group({"xward": typed_list([1, 2], elm_type),
                             "line": typed_list([2], elm_type)}, net)
        assert set(gr2.elements_dict.keys()) == {"gen", "sgen", "trafo", "line"}

        # 6) drop_from_group() & 7) compare_elements_dict()
        gr2.drop_from_group({"xward": typed_set({1, 17}, elm_type), "bus": None, "line": 2 if
                             elm_type is int else "2"})
        assert gr2.compare_elements_dict(elms_b4_append2)
        assert not gr2.compare_elements_dict({
            'trafo': typed_list([0, 1, 2], elm_type), 'gen': typed_list([1], elm_type),
            'sgen': typed_list([2, 3], elm_type)})

        # 7) get_index()
        assert (gr2.get_index(net, "gen") == pd.Index([0, 1], dtype=int)).all()
        assert (gr2.get_index(net, "sgen") == pd.Index([2, 3], dtype=int)).all()
        assert (gr2.get_index(net, "dcline") == pd.Index([], dtype=int)).all()

        # 8) res_p_mw() and res_q_mvar()
        pp.runpp(net)
        p_val = net.res_trafo.pl_mw.loc[[0, 1, 2]].sum() - net.res_gen.p_mw.loc[[0, 1]].sum() - \
            net.res_sgen.p_mw.loc[[2, 3]].sum()
        assert np.isclose(gr2.res_p_mw(net), p_val)

        # 9) compare groups
        gr2_copy = deepcopy(gr2)
        assert gr2 == gr2_copy

        # 10) group to and from json
        json_str = Group_to_json_string(gr2)
        gr3 = Group_from_json_string(json_str)
        assert gr2 == gr3

if __name__ == "__main__":
    if 0:
        pytest.main(['-x', "test_ampl_excel.py"])
    else:
        test_groups()
        pass
