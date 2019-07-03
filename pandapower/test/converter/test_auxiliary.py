import pytest
import numpy as np
import pandas as pd
from copy import deepcopy
from packaging import version
import pandapower as pp

import pandapower.converter as cv

__author__ = 'smeinecke'


def test_ensure_iterability():
    assert len(cv.ensure_iterability(2))
    assert len(cv.ensure_iterability("df")) == 1
    assert len(cv.ensure_iterability(2, 4)) == 4
    assert len(cv.ensure_iterability("df", 5)) == 5
    a = [2, 3.1, "df"]
    assert cv.ensure_iterability(a) == a
    a = np.array(a)
    assert all(cv.ensure_iterability(a) == a)
    assert all(cv.ensure_iterability(a, 3) == a)
    try:
        cv.ensure_iterability(a, 5)
        bool_ = False
    except ValueError:
        bool_ = True
    assert bool_


def test_idx_in_2nd_array():
    arr1 = np.array([1, 6, 4.6, 3.4, 6, 1, "Hallo", "hallo", "Hallo"])
    arr2 = np.array([8, 4, 1, 2, 5, 5.6, 4.6, "Hallo", "hallo", 6, 3.4])

    expected_res = np.array([2, 9, 6, 10, 9, 2, 7, 8, 7])
    res = cv.idx_in_2nd_array(arr1, arr2)
    assert all(res == expected_res)

    arr2[-1] = 4.7
    expected_res[3] = 1
    res = cv.idx_in_2nd_array(arr1, arr2, match=False)
    assert all(res == expected_res)

    try:
        cv.idx_in_2nd_array(arr1, arr2, match=True)
        except_ = False
    except ValueError:
        except_ = True
    assert except_


def test_column_indices():
    df = pd.DataFrame([[0, 3, 2, 4, 5, 6]], columns=["b", "q", "a", "g", "f", "c"])
    query_cols = ["b", "g", "f", "c", "q", "a", "q", "a", "f"]
    col_idx = cv.column_indices(df, query_cols)
    assert all(pp.compare_arrays(col_idx, np.array([0, 3, 4, 5, 1, 2, 1, 2, 4])))


@pytest.mark.xfail(reason="to do python 3.4")
def test_get_unique_duplicated_dict():
    # --- with numeric data
    A = pd.DataFrame([[8.0, 2, 1, 2.0, 3],
                      [4.0, 3, 2, 5.0, 2],
                      [6.0, 4, 2, 3.0, 4],
                      [8.0, 3, 1, 2.0, 3],
                      [np.nan, 7, 5, np.nan, 2],
                      [6.0, 4, 2, 3.0, 4],
                      [8.0, 3, 1, 2.0, 3],
                      [np.nan, 7, 5, np.nan, 2],
                      [8.0, 2, 1, 2.0, 3],
                      [8.0, 2, 1, 2.0, 3]])
    dict_ = cv.get_unique_duplicated_dict(A)
    assert dict_ == {0: [8, 9], 1: [], 2: [5], 3: [6], 4: [7]}
    dict_ = cv.get_unique_duplicated_dict(A, subset=[0, 2, 3, 4])
    assert dict_ == {0: [3, 6, 8, 9], 1: [], 2: [5], 4: [7]}

    # with strings
    A.loc[[2, 5], 2] = "fd"
    A.loc[9, 4] = "kl"
    dict_ = cv.get_unique_duplicated_dict(A)
    assert dict_ == {0: [8], 1: [], 2: [5], 3: [6], 4: [7], 9: []}
    dict_ = cv.get_unique_duplicated_dict(A, subset=[0, 2, 3, 4])
    assert dict_ == {0: [3, 6, 8], 1: [], 2: [5], 4: [7], 9: []}


def test_reindex_dict_dataframes():
    df0 = pd.DataFrame([[0], [1]])
    df1 = pd.DataFrame([[0], [1]], index=[2, 4])
    df2 = pd.DataFrame([[0], [1]], index=[1, 0])
    dict_ = {1: df1, 2: df2}
    expected = {1: df0, 2: df0}
    cv.reindex_dict_dataframes(dict_)
    for k in dict_.keys():
        assert pp.dataframes_equal(dict_[k], expected[k])


def test_avoid_duplicates_in_column():
    net = pp.create_empty_network()
    pp.create_buses(net, 3, 10, name="test")
    expected = net.bus.name.values + [" (%i)" % i for i in range(net.bus.shape[0])]
    cv.avoid_duplicates_in_column(net, "bus", 'name')
    assert all(net.bus.name.values == expected)
    expected = net.bus.type.values + [" (%i)" % i for i in range(net.bus.shape[0])]
    cv.avoid_duplicates_in_column(net, "bus", 'type')
    assert all(net.bus.type.values == expected)


def test_append_str_by_underline_count():
    # simple
    input1 = pd.Series(["a%i" % i for i in range(8)])
    out1, out2 = cv.append_str_by_underline_count(input1)
    assert (out1 == input1 + "_1").all()
    assert not len(out2 ^ set(out1))

    # with counting_start
    cs = 2
    out1, out2 = cv.append_str_by_underline_count(input1, counting_start=cs)
    assert (out1 == input1 + "_"+str(cs)).all()
    assert not len(out2 ^ set(out1))

    # with append_only_duplicates
    out1, out2 = cv.append_str_by_underline_count(input1, append_only_duplicates=True)
    assert (out1 == input1).all()
    assert not len(out2)

    # let's consider duplicates in input
    input1.loc[3] = "a6"
    input1.loc[[1, 5]] = "a0"

    # with append_only_duplicates
    to_add = pd.Series([""]*len(input1))
    to_add.loc[[1, 6]] = "_1"
    to_add.loc[5] = "_2"
    out1, out2 = cv.append_str_by_underline_count(input1, append_only_duplicates=True)
    assert (out1 == input1+to_add).all()
    assert not len(out2 ^ {'a0_1', 'a0_2', 'a6_1'})

    # with reserved_strings
    reserved = {"a0_1", "a0_3"}
    out1, out2 = cv.append_str_by_underline_count(input1, reserved_strings=reserved)
    expected_out1 = pd.Series(["a0_2", "a0_4", "a2_1", "a6_1", "a4_1", "a0_5", "a6_2", "a7_1"])
    assert (out1 == expected_out1).all()
    assert not len(out2 ^ (set(expected_out1) | reserved))

    # with append_only_duplicates and reserved_strings
    out1, out2 = cv.append_str_by_underline_count(input1, append_only_duplicates=True,
                                                  reserved_strings=reserved)
    assert (out1 == pd.Series(["a0", "a0_2", "a2", "a6", "a4", "a0_4", "a6_1", "a7"])).all()
    assert not len(out2 ^ {'a0_1', 'a0_2', 'a0_3', 'a0_4', 'a6_1'})


def test_merge_dataframes():
    df1 = pd.DataFrame([["01.01.2016 00:00:00", 5, 1, "str"],
                        ["03.01.2016 01:00:00", 4, 2, "hallo"],
                        ["04.01.2016 10:00:00", 3, 3, 5]],
                       columns=["time", "B", "A", "C"])

    df2 = pd.DataFrame([["01.02.2016 00:00:00", -1, 3.2, 2],
                        ["01.01.2016 00:00:00",  5, 4,   2.1],
                        ["02.01.2016 00:30:15",  8, 7,   3],
                        ["02.02.2016 13:45:00",  3, 1,   4]],
                       columns=["time", "A", "B", "D"])

    df3 = pd.DataFrame([["01.01.2016 00:00:00", 9, 6, 8.1, 3]],
                       columns=["time", "A", "B", "D", "C"])

    # ordered index and column, df1 with precedence, time as index
    return1 = cv.merge_dataframes([df1, df2], column_to_sort="time", keep="first",
                                  index_time_str="%d.%m.%Y %H:%M:%S")
    res1 = pd.DataFrame([["01.01.2016 00:00:00",  1, 5,   "str",   2.1],
                         ["02.01.2016 00:30:15",  8, 7,   None,    3],
                         ["03.01.2016 01:00:00",  2, 4,   "hallo", None],
                         ["04.01.2016 10:00:00",  3, 3,   5,       None],
                         ["01.02.2016 00:00:00", -1, 3.2, None,    2],
                         ["02.02.2016 13:45:00",  3, 1,   None,    4]],
                        columns=["time", "A", "B", "C", "D"])
    assert pp.dataframes_equal(return1, res1)

    # ordered index and column, df2 with precedence, time as index
    return2 = cv.merge_dataframes([df1, df2], column_to_sort="time", keep="last",
                                  index_time_str="%d.%m.%Y %H:%M:%S")
    res2 = deepcopy(res1)
    res2.loc[0, ["A", "B"]] = df2.loc[1, ["A", "B"]]
    assert pp.dataframes_equal(return2, res2)

    # --- changed input
    new_df1_idx = [1, 3, 4]
    new_df2_idx = [11, 1, 2, 12]
    unsorted_index = new_df1_idx + [11, 2, 12]
    unsorted_columns = list(df1.columns) + ["D"]
    df1.index = new_df1_idx
    df2.index = new_df2_idx

    # ordered index and column, df1 with precedence, no extra index
    return5 = cv.merge_dataframes([df1, df2], keep="first")
    res5 = deepcopy(res1)
    if version.parse(pd.__version__) >= version.parse("0.21.0"):
        res5 = res5.reindex(columns=["A", "B", "C", "D", "time"])
    else:
        res5 = res5.reindex_axis(["A", "B", "C", "D", "time"], axis=1)
    res5.index = [1, 2, 3, 4, 11, 12]
    assert pp.dataframes_equal(return5, res5)

    # ordered index and column, df2 with precedence, no extra index
    return6 = cv.merge_dataframes([df1, df2], keep="last")
    res6 = deepcopy(res5)
    res6.loc[1, ["A", "B"]] = df2.loc[1, ["A", "B"]]
    assert pp.dataframes_equal(return6, res6)

    # beware idx order, df1 with precedence, no extra index
    return7 = cv.merge_dataframes([df1, df2], keep="first", sort=False)
    try:
        res7 = deepcopy(res5).reindex(unsorted_index, columns=unsorted_columns)
    except TypeError:  # legacy for pandas <0.21
        res7 = deepcopy(res5).reindex_axis(unsorted_index)
        res7 = res7.reindex_axis(unsorted_columns, axis=1)
    assert pp.dataframes_equal(return7, res7)

    # beware idx order, df1 with precedence, no extra index
    return8 = cv.merge_dataframes([df1, df2], keep="last", sort=False)
    try:
        res8 = deepcopy(res6).reindex(unsorted_index, columns=unsorted_columns)
    except TypeError:  # legacy for pandas <0.21
        res8 = deepcopy(res6).reindex_axis(unsorted_index)
        res8 = res8.reindex_axis(unsorted_columns, axis=1)
    assert pp.dataframes_equal(return8, res8)

    # merge 3 dfs while keeping first duplicates
    return9 = cv.merge_dataframes([df1, df2, df3], keep="first", column_to_sort="time",
                                  index_time_str="%d.%m.%Y %H:%M:%S")
    assert pp.dataframes_equal(return9, res1)

    # merge 3 dfs while keeping last duplicates
    return10 = cv.merge_dataframes([df1, df2, df3], keep="last", column_to_sort="time",
                                   index_time_str="%d.%m.%Y %H:%M:%S")
    res10 = deepcopy(res1)
    df3_col_except_time = df3.columns.difference(["time"])
    res10.loc[0, df3_col_except_time] = df3.loc[0, df3_col_except_time].values
    assert pp.dataframes_equal(return10, res10)

    # merge 3 dfs while keeping all duplicates
    return11 = cv.merge_dataframes([df1, df2, df3], keep="all")
    assert return11.shape == (len(df1)+len(df2)+len(df3),
                              len(df1.columns.union(df2.columns.union(df3.columns))))


if __name__ == "__main__":
    pytest.main(["test_auxiliary.py", "-xs"])