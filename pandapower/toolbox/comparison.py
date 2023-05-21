import networkx as nx
import numpy as np
import pandas as pd
from pandas import testing as pdt
from deepdiff import DeepDiff

from pandapower.auxiliary import pandapowerNet

try:
    from networkx.utils.misc import graphs_equal
    GRAPHS_EQUAL_POSSIBLE = True
except ImportError:
    GRAPHS_EQUAL_POSSIBLE = False

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def dataframes_equal(df1, df2, ignore_index_order=True, **kwargs):
    """
    Returns a boolean whether the given two dataframes are equal or not.
    """
    if "tol" in kwargs:
        if "atol" in kwargs:
            raise ValueError("'atol' and 'tol' are given to dataframes_equal(). Don't use 'tol' "
                             "anymore.")
        logger.warning("in dataframes_equal() parameter 'tol' is deprecated. Use 'atol' instead.")
        kwargs["atol"] = kwargs.pop("tol")

    if ignore_index_order:
        df1 = df1.sort_index().sort_index(axis=1)
        df2 = df2.sort_index().sort_index(axis=1)

    # --- pandas implementation
    try:
        pdt.assert_frame_equal(df1, df2, **kwargs)
        return True
    except AssertionError:
        return False

    # --- alternative (old) implementation
    # if df1.shape == df2.shape:
    #     if df1.shape[0]:
    #         # we use numpy.allclose to grant a tolerance on numerical values
    #         numerical_equal = np.allclose(df1.select_dtypes(include=[np.number]),
    #                                       df2.select_dtypes(include=[np.number]),
    #                                       atol=tol, equal_nan=True)
    #     else:
    #         numerical_equal = True
    #     # ... use pandas .equals for the rest, which also evaluates NaNs to be equal
    #     rest_equal = df1.select_dtypes(exclude=[np.number]).equals(
    #         df2.select_dtypes(exclude=[np.number]))

    #     return numerical_equal & rest_equal
    # else:
    #     return False


def compare_arrays(x, y):
    """
    Returns an array of bools whether array x is equal to array y. Strings are allowed in x
    or y. NaN values are assumed as equal.
    """
    if x.shape == y.shape:
        # (x != x) is like np.isnan(x) - but works also for strings
        return np.equal(x, y) | ((x != x) & (y != y))
    else:
        raise ValueError("x and y need to have the same shape.")


def nets_equal(net1, net2, check_only_results=False, check_without_results=False, exclude_elms=None,
               name_selection=None, **kwargs):
    """
    Returns a boolean whether the two given pandapower networks are equal.

    pandapower net keys starting with "_" are ignored. Same for the key "et" (elapsed time).

    If the element tables contain JSONSerializableClass objects, they will also be compared:
    attributes are compared but not the addresses of the objects.

    INPUT:
        **net1** (pandapowerNet)

        **net2** (pandapowerNet)

    OPTIONAL:
        **check_only_results** (bool, False) - if True, only result tables (starting with ``res_``)
        are compared

        **check_without_results** (bool, False) - if True, result tables (starting with ``res_``)
        are ignored for comparison

        **exclude_elms** (list, None) - list of element tables which should be ignored in the
        comparison

        **name_selection** (list, None) - list of element tables which should be compared

        **kwargs** - key word arguments for dataframes_equal()
    """
    if not (isinstance(net1, pandapowerNet) and isinstance(net2, pandapowerNet)):
        logger.warning("At least one net is not of type pandapowerNet.")
        return False
    not_equal, not_checked_keys = nets_equal_keys(
        net1, net2, check_only_results, check_without_results, exclude_elms, name_selection,
        **kwargs)
    if len(not_checked_keys) > 0:
        logger.warning("These keys were ignored by the comparison of the networks: %s" % (', '.join(
            not_checked_keys)))

    if len(not_equal) > 0:
        logger.warning("Networks do not match in DataFrame(s): %s" % (', '.join(not_equal)))
        return False
    else:
        return True


def nets_equal_keys(net1, net2, check_only_results, check_without_results, exclude_elms,
                     name_selection, **kwargs):
    """ Returns a lists of keys which are 1) not equal and 2) not checked.
    Used within nets_equal(). """
    if check_without_results and check_only_results:
        raise UserWarning("Please provide only one of the options to check without results or to "
                          "exclude results in comparison.")

    exclude_elms = [] if exclude_elms is None else list(exclude_elms)
    exclude_elms += ["res_" + ex for ex in exclude_elms]
    not_equal = []

    # for two networks make sure both have the same keys
    if name_selection is not None:
        net1_keys = net2_keys = name_selection
    elif check_only_results:
        net1_keys = [key for key in net1.keys() if key.startswith("res_")
                     and key not in exclude_elms]
        net2_keys = [key for key in net2.keys() if key.startswith("res_")
                     and key not in exclude_elms]
    else:
        net1_keys = [key for key in net1.keys() if not (
            key.startswith("_") or key in exclude_elms or key == "et"
            or key.startswith("res_") and check_without_results)]
        net2_keys = [key for key in net2.keys() if not (
            key.startswith("_") or key in exclude_elms or key == "et"
            or key.startswith("res_") and check_without_results)]
    keys_to_check = set(net1_keys) & set(net2_keys)
    key_difference = set(net1_keys) ^ set(net2_keys)
    not_checked_keys = list()

    if len(key_difference) > 0:
        logger.warning(f"Networks entries mismatch at: {key_difference}")
        return key_difference, set()

    # ... and then iter through the keys, checking for equality for each table
    for key in list(keys_to_check):

        if isinstance(net1[key], pd.DataFrame):
            if not isinstance(net2[key], pd.DataFrame) or not dataframes_equal(
                    net1[key], net2[key], **kwargs):
                not_equal.append(key)

        elif isinstance(net1[key], np.ndarray):
            if not isinstance(net2[key], np.ndarray):
                not_equal.append(key)
            else:
                if not np.array_equal(net1[key], net2[key], equal_nan=True):
                    not_equal.append(key)

        elif isinstance(net1[key], int) or isinstance(net1[key], float) or \
                isinstance(net1[key], complex):
            if not np.isclose(net1[key], net2[key]):
                not_equal.append(key)

        elif isinstance(net1[key], nx.Graph):
            if GRAPHS_EQUAL_POSSIBLE:
                if not graphs_equal(net1[key], net2[key]):
                    not_equal.append(key)
            else:
                # Maybe there is a better way, but at least this could be checked
                if net1[key].nodes != net2[key].nodes or net1[key].edges != net2[key].edges:
                    not_equal.append(key)
        elif isinstance(net1[key], dict):
            diff = DeepDiff(net1[key], net2[key], math_epsilon=1e-20, ignore_numeric_type_changes=True)
            if len(diff) > 0:
                not_equal.append(key)

        else:
            try:
                is_eq = net1[key] == net2[key]
                if not is_eq:
                    not_equal.append(key)
            except:
                not_checked_keys.append(key)
    return not_equal, not_checked_keys
