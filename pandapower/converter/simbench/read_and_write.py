import numpy as np
import pandas as pd
import os

try:
    import pplog as logging
except ImportError:
    import logging

from pandapower.converter.simbench.auxiliary import merge_dataframes
from pandapower.converter.simbench.format_information import get_columns, csv_tablenames, get_dtypes

logger = logging.getLogger(__name__)

__author__ = 'smeinecke'


def _init_csv_table(tablename):
    """ This function returns an initial, empty DataFrame with appropriate column names. """
    return pd.DataFrame([], columns=get_columns(tablename))


def _init_csv_tables(which_tables):
    """
    This function returns the initial dict of empty DataFrames with appropriate column names.
    """
    tablenames = []
    for which in which_tables:
        tablenames += csv_tablenames(which)
    csv_tables = dict()
    for i in tablenames:
        csv_tables[i] = _init_csv_table(i)
    return csv_tables


def _correct_float_to_object_dtype(df, tablename):
    """ This function corrects a DataFrame (df) belonging to 'tablename' float dtypes to object type
        if get_dtypes() expects those columns to be in object type. """
    dtypes = pd.DataFrame(df.dtypes)
    is_float_col = dtypes.index[dtypes[0] == float]
    eq_len = len(get_columns(tablename)) == len(get_dtypes(tablename))
    if not eq_len:
        raise ValueError("For table '%s', there are %i columns but %i dtypes" % (
            tablename, len(get_columns(tablename)), len(get_dtypes(tablename))))
    should_be_object_col = np.array(get_columns(tablename))[
        np.array(get_dtypes(tablename)) == object]
    col_to_new_type = set(is_float_col) & set(should_be_object_col)
    for col in col_to_new_type:
        df[col] = df[col].astype(object)
    return df


def read_csv_data(path, sep, tablename=None, nrows=None):
    """
    This function reads the csv files, given by tablename or all, and returns a dict of DataFrames
    for each element type.
    If no tablenames are given, 'Node' and 'Load' tables are mandatory to read. Other element tables
    are integrated as empty DataFrames, in case of reading error.

    INPUT:
        **path** (str) - path to folder with csv data files

        **sep** (str) - csv seperator, e.g. ',' or ';'

    OPTIONAL:
        **tablename** (str or list of str, None) - name(s) of csv table(s) to be read. If tablename
            is None, all csv_tablenames are read.

        **nrows** (int, None) - number of rows to be read for load, sgen and storage profiles. If
            None, all rows will be read.
    """
    csv_tables = dict()
    if isinstance(tablename, str):
        return_dataframe = True
        tablename = [tablename]
    elif tablename is None:
        return_dataframe = False
        tablename = csv_tablenames(['elements', 'profiles', 'types', 'cases', 'res_elements'])
    else:
        return_dataframe = False
    for i in tablename:
        nrows = None if "Profile" not in i else nrows
        try:
            csv_tables[i] = pd.read_csv(os.path.join(path, "%s.csv" % i), sep=sep, nrows=nrows,
                                        index_col=False)
            _correct_float_to_object_dtype(csv_tables[i], i)  # possible but not necessary
        except (FileNotFoundError, OSError):
            if i in ['Node', 'Load']:
                logger.error(i + ".csv cannot be read from csv file. " +
                             "Possibly the path %s does not exist." % path)
            csv_tables[i] = _init_csv_table(i)
            csv_tables[i] = csv_tables[i][get_columns(i)]
    if not return_dataframe:
        return csv_tables
    else:
        return csv_tables[tablename[0]]


def write2csv(path, data, mode="w", sep=";", float_format='%g', keys=None, keep="last",
              must_store=None, nrows=None):
    """ Writes 'data' to csv files.

    INPUT:
        **path** (str) - path to folder with csv data files

        **data** (dict) - dict of DataFrames containing the grid data which should be written to csv
            files

    OPTIONAL:
        **mode** (str, "w") - writing mode. "w" for writing, "a" for appending and "append_unique"
            for append only unique data are common to this function.

        **sep** (str, ";") - seperator of csv files

        **float_format** (str, "%g") - format of how to write floats into csv files

        **keys** (list, None) - list of keys, which should be considered for writing. If None, all
            keys of data will be considered.

        **keep** (str, "last") - Flag to set, which duplicated named data will be kept. Only
            relevant in case of mode == "append_unique"

        **must_store** (list, None) - list of element tables that always will be stored, if they are
            in 'data', even if they are empty. If 'must_store' is None,
            'Node', 'Load'] is assumed.

        **nrows** (int, None) - number of rows to be write to csv for Load, RES and Storage
            profiles. If None, all rows will be written.
    """
    if mode not in ['append_unique', 'a', 'w']:
        mode = "w"
        logger.warning("'mode' must be in ['append_unique', 'a', 'w']. 'w' " +
                       "is assumed.")
    # element tables that always will be stored if they are in 'data' - even if they are empty:
    must_store = ['Node', 'Load'] if must_store is None else must_store
    keys = data.keys() if keys is None else keys
    for i, d in data.items():
        # write all must_store and element tables with content
        if (d.shape[0] > 0) | (i in must_store) and i in keys:
            this_path = os.path.join(path, "%s.csv" % i)
            file_misses = not os.path.exists(this_path)
            # only use "append_unique" if the file exists:
            mod = 'a' if mode == "append_unique" and (file_misses or "Profile" in i) else mode

            if "Profile" not in i:
                # append only unique named elements to existing csv
                index = (i == "StudyCases") & ("Study Case" not in d.columns)
                if mod == "append_unique":
                    d = pd.concat([read_csv_data(path, sep, i), d], ignore_index=True)
                    dupl_cols = ["id"] if "id" in d.columns else ["node"]
                    dupl_cols += [col for col in ["voltLvl", "subnet"] if col in d.columns]
                    duplicates = d.loc[d.duplicated(dupl_cols, keep=keep)]
                    if len(duplicates) and "Type" not in i:
                        logger.info("Writing to table '%s', these duplicated names are " % i +
                                    "dropped: " + str(['%s' % name for name in duplicates.id]))
                    d.drop(duplicates.index, inplace=True)
                    d.replace("", "NULL").fillna("NULL").to_csv(this_path, sep, index=index,
                                                                float_format=float_format)

                else:
                    d.replace("", "NULL").fillna("NULL").to_csv(
                        this_path, sep, mode=mod, index=index, float_format=float_format,
                        header=(file_misses or mod == "w"))

            # --- writing Profiles
            else:  # always merge "Profiles" via dropping duplicates
                if mod == "a" and not file_misses:
                    d_prof = merge_dataframes(
                        [read_csv_data(path, sep, i), d], column_to_sort="time",
                        index_time_str="%d.%m.%Y %H:%M")
                else:
                    d_prof = d
                if d_prof.shape[1] > 1:  # only print if data (not only time column) are available
                    d_prof = d_prof if nrows is None or d_prof.shape[0] <= nrows else \
                        d_prof.loc[:nrows-1]
                    d_prof.replace("", "NULL").fillna("NULL").to_csv(
                        this_path, sep, mode="w", index=False, float_format=float_format)
