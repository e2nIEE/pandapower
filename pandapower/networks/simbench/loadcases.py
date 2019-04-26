"""
This script is a function toolbox for analysis of MV networks, especially for SimBench networks.
"""
from pandapower.converter import get_voltlvl
from pandapower.converter.simbench.csv_pp_converter import _is_pp_type

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

__author__ = "smeinecke"


def filter_loadcases_df(data, factors):
    """ Since SimBench defines different slack voltages and scaling factors of loads for different
        voltage levels, this function filters unused study case data for the given net.
    """
    if _is_pp_type(data):
        vn_kvs = data.bus.vn_kv.value_counts()
    else:
        vn_kvs = data["Node"].vmR.value_counts()
    lv_vn_kv = vn_kvs.loc[vn_kvs > 2].index.min()  # minimum vn_kv which occurs at more than 2 buses
    lv_level = get_voltlvl(lv_vn_kv)
    factors = factors.loc[factors.voltLvl == lv_level].set_index("Study Case").drop(["voltLvl"], axis=1)
    return factors


def filter_loadcases(data, factors=None):
    """ Since SimBench defines different slack voltages and scaling factors of loads for different
        voltage levels, this function filters unused study case data for the given net.

    INPUT:
        **data** (dict or pandapowerNet) - grid data in pandapower format or SimBench csv format

    OPTIONAL:
        **factors** (DataFrame, None) - factors of Study Cases. If None, the factors are taken from
            data.
    """
    if factors is None:
        factors = data.loadcases if _is_pp_type(data) else data["StudyCases"]
    if factors.shape[0] == 0 or factors.shape[1] == 0:
        logger.warning("factors are empty -> no loadcases are filtered.")
    else:
        factors = filter_loadcases_df(data, factors)
        if _is_pp_type(data):
            data.loadcases = factors
        else:
            data["StudyCases"] = factors


if __name__ == "__main__":
    pass
