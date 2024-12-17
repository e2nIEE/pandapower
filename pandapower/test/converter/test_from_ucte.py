# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from copy import deepcopy
import os
import pytest
import numpy as np
import pandas as pd

import pandapower as pp
import pandapower.converter as pc

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def _testfiles_folder():
    return os.path.join(pp.pp_dir, 'test', 'converter', "testfiles")


def _results_from_powerfactory():
    pf_res = {f"res_{et}": pd.read_csv(
        os.path.join(_testfiles_folder(), f"test_ucte_res_{et}.csv"),
        sep=";", index_col=0) for et in ["bus", "line", "trafo", "trafo3w"]}
    return pf_res


def country_code_mapping(test_case=None):
    mapping = {
        "test_ucte_line_trafo_load": "_DE",
        "test_ucte_line": xy,
        "test_ucte_load_sgen": xy,
        "test_ucte_load_sgen_split": xy,
        "test_ucte_ext_grid": xy,
        "test_ucte_trafo": xy,
        "test_ucte_single_load_single_eg": xy,
        "test_ucte_ward": xy,
        "test_ucte_ward_split": xy,
        "test_ucte_xward": xy,
        "test_ucte_xward_combination": xy,
        "test_ucte_gen": xy,
        "test_ucte_ext_grid_gen_switch": xy,
        "test_ucte_enforce_qlims": xy,
        "test_ucte_trafo3w": xy,
    }
    if test_case is None
        return mapping
    else:
        return mapping[test_case]


# @pytest.mark.parametrize("test_case", [
@pytest.mark.parametrize("ucte_file_name", [
    "test_ucte3", # LU, DK, HK, IT, RS
    "test_ucte_AL",
    "test_ucte_DE",
    "test_ucte_ES",
    "test_ucte_HR",
    "test_ucte_HU",
    "test_ucte_NL"
])
def _test_ucte_file(ucte_file_name):
    # ucte_file_name = "test_ucte" + country_code_mapping(test_case)
    ucte_file = os.path.join(_testfiles_folder(), f"{ucte_file_name}.uct")

    # --- convert UCTE data
    ucte_parser = pc.ucte_parser.UCTEParser(ucte_file)
    ucte_parser.parse_file()
    ucte_dict = ucte_parser.get_data()

    ucte_converter = pc.ucte_converter.UCTE2pandapower()
    net = ucte_converter.convert(ucte_dict=ucte_dict)

    # --- run power flow
    pp.runpp(net)

    # --- compare results
    res_target = _results_from_powerfactory()
    failed = list()
    atol_dict = {"res_bus": {"vm_pu": 1e-4, "va_degree": 5e-3},
                 "res_line": {"p_from_mw": 5e-2, "q_from_mvar": 2e-1},
                 "res_trafo": {"p_hv_mw": 5e-2, "q_hv_mvar": 1e-1},
                 "res_trafo3w": {"p_hv_mw": 5e-2, "q_hv_mvar": 1e-1},
                #  "res_line": {"p_from_mw": 1e-3, "q_from_mvar": 1e-2},
                #  "res_trafo": {"p_hv_mw": 1e-3, "q_hv_mvar": 1e-2},
                #  "res_trafo3w": {"p_hv_mw": 1e-3, "q_hv_mvar": 1e-2},
                 }
    if ucte_file_name == "test_ucte_file_NL":
        atol_dict["res_line"]["q_from_mvar"] = 0.8  # xwards are converted as
        # PV gens towards uct format -> lower tolerance (compared to powerfactory results cannot be
        # expected)

    for res_et, df_target in res_target.items():
        et = res_et[4:]
        name_col = "name" if et != "bus" else "add_name"
        missing_names = pd.Index(net[et][name_col]).difference(df_target.index)
        if len(missing_names):
            logger.error(f"{res_et=} comparison fails since same element names of the PowerFactory "
                         f"results are missing in the pandapower net: {missing_names}")
        df_after_conversion = net[res_et][df_target.columns].set_axis(
            pd.Index(net[et][name_col], name="name"))
        if et == "line" and "Allgemeine I" in df_after_conversion.index:
            df_after_conversion = df_after_conversion.drop("Allgemeine I")
        same_shape = df_after_conversion.shape == df_target.loc[df_after_conversion.index].shape
        df_str = (f"df_after_conversion:\n{df_after_conversion}\n\ndf_target:\n"
                  f"{df_target.loc[df_after_conversion.index]}")
        if not same_shape:
            logger.error(f"{res_et=} comparison fails due to different shape.\n{df_str}")
        all_close = all([np.allclose(
            df_after_conversion[col].values,
            df_target.loc[df_after_conversion.index, col].values, atol=atol) for col, atol in
            atol_dict[res_et].items()])
        if not all_close:
            logger.error(f"{res_et=} comparison fails due to different values.\n{df_str}")
            failed.append(res_et)
    if test_is_failed := len(failed):
        logger.error(f"The powerflow result comparisons of these elements failed: {failed}.")
    assert not test_is_failed


if __name__ == '__main__':
    if 0:
        pytest.main([__file__, "-s"])
    else:

        ucte_file = os.path.join(_testfiles_folder(), "test_ucte.uct")

        ucte_parser = pc.ucte_parser.UCTEParser(ucte_file)
        ucte_parser.parse_file()
        ucte_dict = ucte_parser.get_data()

        ucte_converter = pc.ucte_converter.UCTE2pandapower()
        net = ucte_converter.convert(ucte_dict=ucte_dict)

        print(net)
        print()