import numpy as np
import pandas as pd
import datetime as dt
from packaging import version
from pandapower import compare_arrays

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

__author__ = 'smeinecke'


def ensure_iterability(var, len_=None):
    """ This function ensures iterability of a variable (and optional length). """
    if hasattr(var, "__iter__") and not isinstance(var, str):
        if isinstance(len_, int) and len(var) != len_:
            raise ValueError("Length of variable differs from %i." % len_)
    else:
        len_ = len_ or 1
        var = [var]*len_
    return var


def find_idx_by_name(df, column, name):
    idx = df.index[df[column] == name]
    if len(idx) == 0:
        raise UserWarning("In column '%s', there is no element named %s" % (column, name))
    if len(idx) > 1:
        raise UserWarning("In column '%s', multiple elements are named %s" % (column, name))
    return idx[0]


def idx_in_2nd_array(arr1, arr2, match=True):
    """ This function returns an array of indices of arr1 matching arr2.
        arr1 may include duplicates. If an item of arr1 misses in arr2, 'match' decides whether
        the idx of the nearest value is returned (False) or an error is raised (True).
    """
    if match:
        missings = list(set(arr1) - set(arr2))
        if len(missings):
            raise ValueError("These values misses in arr2: " + str(missings))
    arr1_, uni_inverse = np.unique(arr1, return_inverse=True)
    sort_lookup = np.argsort(arr2)
    arr2_ = np.sort(arr2)
    idx = np.searchsorted(arr2_, arr1_)
    res = sort_lookup[idx][uni_inverse]
    return res


def column_indices(df, query_cols):
    """ returns an numpy array with the indices of the columns requested by 'query_cols'.
        Works propperly for string column names. """
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols, query_cols, sorter=sidx)]


def merge_dataframes(dfs, keep="first", sort_index=True, sort_column=True, column_to_sort=None,
                     index_time_str=None, **kwargs):
    """
    This is a wrapper function of pandas.concat(dfs, axis=0) to merge DataFrames.

    INPUT:
        **dfs** (DataFrames) - a sequence or mapping of DataFrames

    OPTIONAL:
        **keep** (str, "first") - Flag to decide which data are kept in case of duplicated
            indices - first, last or all duplicated data.

        **sort_index** (bool, True) - If True, the indices of the returning DataFrame will be
            sorted. If False, the indices and columns will be in order of the original DataFrames.

        **sort_column** (bool, True) - If True, the indices of the returning DataFrame will be
            sorted. If False, the indices and columns will be in order of the original DataFrames.

        **column_to_sort** (-, None) - If given, 'column_to_sort' must be a column name occuring in
            both DataFrames. The returning DataFrame will be sorted by this column. The input
            indices get lost.

        **index_time_str** (str, None) - If given, the indices or the 'column_to_sort' if given will
            be sorted in datetime order.

        ****kwargs** - Keyword arguments for pandas.concat() except axis, such as sort, join,
            join_axes, ignore_index, keys. 'sort' can overwrite 'sort_index' and 'sort_column'.
    """
    if "axis" in kwargs:
        if kwargs["axis"] != 0:
            logger.warning("'axis' is always assumed as zero.")
        kwargs.pop("axis")
    if "sort" in kwargs:
        if not kwargs["sort"] == sort_index == sort_column:
            sort_index = kwargs["sort"]
            sort_column = kwargs["sort"]
            if not sort_index or not sort_column:
                logger.warning("'sort' overwrites 'sort_index' and 'sort_column'.")
        kwargs.pop("sort")

    # --- set index_column as index
    if column_to_sort is not None:
        if any([column_to_sort not in df.columns for df in dfs]):
            raise KeyError("column_to_sort '%s' must be a column of " % column_to_sort +
                           "both dataframes, df1 and df2")
        if not sort_index:
            logger.warning("Since 'column_to_sort' is given, the returning DataFrame will be" +
                           "sorted by this column as well as the columns, although 'sort' " +
                           "was given as False.")
            sort_index = True
        dfs = [df.set_index(column_to_sort) for df in dfs]

    # --- concat
    df = pd.concat(dfs, axis=0, **kwargs)

    # --- unsorted index and columns
    output_index = df.index.drop_duplicates()

    # --- drop rows with duplicated indices
    if keep == "first":
        df = df.groupby(df.index).first()
    elif keep == "last":
        df = df.groupby(df.index).last()
    elif keep != "all":
        raise ValueError("This value %s is unknown to 'keep'" % keep)

    # --- sorted index and reindex columns
    if sort_index:
        if index_time_str:
            dates = [dt.datetime.strptime(ts, index_time_str) for ts in df.index]
            dates.sort()
            output_index = [dt.datetime.strftime(ts, index_time_str) for ts in dates]
            if keep == "all":
                logger.warning("If 'index_time_str' is not None, keep cannot be 'all' but are " +
                               "assumed as 'first'.")
        else:
            output_index = sorted(df.index)

    # --- reindex as required
    if keep != "all":
        if version.parse(pd.__version__) >= version.parse("0.21.0"):
            df = df.reindex(output_index)
        else:
            df = df.reindex_axis(output_index)
    if sort_column:
        if version.parse(pd.__version__) >= version.parse("0.21.0"):
            df = df.reindex(columns=sorted(df.columns))
        else:
            df = df.reindex_axis(sorted(df.columns), axis=1)

    # --- get back column_to_sort as column from index
    if column_to_sort is not None:
        df.reset_index(inplace=True)

    return df


def get_unique_duplicated_dict(df, subset=None, only_dupl_entries=False):
    """ Returns a dict which keys are the indices of unique row of the dataframe 'df'. The values
        of the dict are the indices which are duplicated to each key index.
        This is a wrapper function of _get_unique_duplicated_dict() to consider only_dupl_entries.
    """
    is_dupl = df.duplicated(subset=subset, keep=False)
    uniq_dupl_dict = _get_unique_duplicated_dict(df[is_dupl], subset)
    if not only_dupl_entries:
        others = df.index[~is_dupl]
        uniq_empties = {o: [] for o in others}
        # uniq_dupl_dict = {**uniq_dupl_dict, **uniq_empties}  # python 3.5+
        for k, v in uniq_empties.items():
            uniq_dupl_dict[k] = v
    return uniq_dupl_dict


def _get_unique_duplicated_dict(df, subset=None):
    """ Returns a dict which keys are the indices of unique row of the dataframe 'df'. The values
        of the dict are the indices which are duplicated to each key index. """
    subset = subset or df.columns
    dupl = df.index[df.duplicated(subset=subset)]
    uniq = df.index[~df.duplicated(subset=subset)]
    uniq_dupl_dict = {}
    for uni in uniq:
        do_dupl_fit = compare_arrays(
            np.repeat(df.loc[uni, subset].values.reshape(1, -1), len(dupl), axis=0),
            df.loc[dupl, subset].values).all(axis=1)
        uniq_dupl_dict[uni] = list(dupl[do_dupl_fit])
    return uniq_dupl_dict


def reindex_dict_dataframes(dataframes_dict):
    """ Set new continous index starting at zero for every DataFrame in the dict. """
    for key in dataframes_dict.keys():
        if isinstance(dataframes_dict[key], pd.DataFrame) and key != "StudyCases":
            dataframes_dict[key].index = list(range(dataframes_dict[key].shape[0]))


def ensure_full_column_data_existence(dict_, tablename, column):
    """
    Ensures that the column of a dict's DataFrame is fully filled with information. If there are
    missing data, it will be filled up by name tablename+index
    """
    missing_data = dict_[tablename].index[dict_[tablename][column].isnull()]
    # fill missing data by tablename+index, e.g. "Bus 2"
    dict_[tablename][column].loc[missing_data] = [tablename + ' %s' % n for n in (
            missing_data.values + 1)]
    return dict_[tablename]


def avoid_duplicates_in_column(dict_, tablename, column):
    """ Avoids duplicates in given column (as type string) of a dict's DataFrame """
    query = dict_[tablename][column].duplicated(keep=False)
    for double in dict_[tablename][column].loc[query].unique():
        idx = dict_[tablename][column].index[dict_[tablename][column] == double]
        dict_[tablename][column].loc[idx] = [double + " (%i)" % i for i in range(len(idx))]
    if sum(dict_[tablename][column].duplicated()):
        raise ValueError("The renaming by 'double + int' was not appropriate to remove all " +
                         "duplicates.")


def append_str_by_underline_count(str_series, append_only_duplicates=False, counting_start=1,
                                  reserved_strings=None):
    """
    Returns a Series of appended strings and a set of all strings which were appended or are set as
    reserved by input.

    INPUT:
        **str_series** (Series with string values) - strings to be appended by "_" + a number

    OPTIONAL:
        **append_only_duplicates** (bool, False) - If True, all strings will be appended. If False,
         only duplicated strings will be appended.

        **counting_start** (int, 1) - Integer to start appending with

        **reserved_strings** (iterable, None) - strings which are not allowed in str_series and must
            be appended.

    OUTPUT:
        **appended_strings** (Series with string values) - appended strings

        **reserved_strings** (set) - all reserved_strings from input and all strings which were
            appended
    """
    # --- initalizations
    # ensure only unique values in reserved_strings:
    reserved_strings = pd.Series(sorted(set(reserved_strings))) if reserved_strings is not None \
        else pd.Series()
    count = counting_start

    # --- do first append
    # concatenate reserved_strings and str_series (which should be appended by "_%i")
    # must be in this order (first reserved_strings) to append only the str_series (keep='first')
    if not append_only_duplicates:
        series = str_series + "_%i" % count
        series = pd.concat([reserved_strings, series], ignore_index=True)
        all_dupl = pd.Series([True]*len(series))
    else:
        series = pd.concat([reserved_strings, str_series], ignore_index=True)
        all_dupl = pd.Series([True]*len(reserved_strings)+[False]*len(str_series))
        dupl = series.duplicated()
        all_dupl |= dupl
        series.loc[dupl] += "_%i" % count
    dupl = series.duplicated()
    all_dupl |= dupl

    # --- append as much as necessary -> while loop
    while sum(dupl):
        series.loc[dupl] = series[dupl].str.replace("_%i" % count, "_%i" % (count+1))
        dupl = series.duplicated()
        all_dupl |= dupl
        count += 1

    # --- output adaptations
    appended_strings = series.iloc[len(reserved_strings):]
    appended_strings.index = str_series.index
    reserved_strings = set(series[all_dupl])
    return appended_strings, reserved_strings
