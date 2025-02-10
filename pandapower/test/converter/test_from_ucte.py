# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

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
        sep=";", index_col=0) for et in ["bus", "line", "trafo"]}
    return pf_res


def _country_code_mapping(test_case=None):
    mapping = {
        "test_ucte_impedance": "AL",
        "test_ucte_line_trafo_load": "DE",
        "test_ucte_line": "DK",
        "test_ucte_bus_switch": "ES",
        "test_ucte_shunt": "HR",
        "test_ucte_ward": "HU",
        "test_ucte_load_sgen": "IT",
        "test_ucte_gen": "LU",
        "test_ucte_xward": "NL",
        "test_ucte_trafo": "RS",
        "test_ucte_trafo3w": "FR",
    }
    if test_case is None:
        return mapping
    else:
        return mapping[test_case]


@pytest.mark.parametrize("test_case", [
    "test_ucte_impedance",
    "test_ucte_line_trafo_load",
    "test_ucte_line",
    "test_ucte_bus_switch",
    "test_ucte_shunt",
    "test_ucte_ward",
    "test_ucte_load_sgen",
    "test_ucte_gen",
    "test_ucte_xward",
    "test_ucte_trafo",
    "test_ucte_trafo3w"
])
def test_from_ucte(test_case):
    """Tets the UCTE converter by

    1. checking the number of elements per element type in the grid
    2. comparing the power flow results with expected results

    The expected power flow results and the UCTE data files are derived from the
    pandapower/test/converter/testfiles/test_ucte.pfd file and power factory version 2024 service
    pack 1. The test case are derived from the test cases to
    check the power flow results without any convter, cf
    pandapower/test/test_files/test_results.pfd and pandapower/test/loadflow/test_runpp.py.

    Parameters
    ----------
    test_case : str
        description on which kind of test case is tested for th UCTE conversion
    """
    country_code = _country_code_mapping(test_case)
    ucte_file_name = f"test_ucte_{country_code}"
    ucte_file = os.path.join(_testfiles_folder(), f"{ucte_file_name}.uct")

    # --- convert UCTE data -------------------------------------------------------------------
    net = pc.from_ucte(ucte_file)

    assert isinstance(net, pp.pandapowerNet)
    assert len(net.bus)

    # --- run power flow ----------------------------------------------------------------------
    pp.runpp(net)
    assert net.converged

    # --- check expected element counts -------------------------------------------------------
    exp_elm_count_df = pd.read_csv(os.path.join(
        _testfiles_folder(), "ucte_expected_element_counts.csv"), sep=";", index_col=0)
    exp_elm_count = exp_elm_count_df.loc[country_code]
    exp_elm_count = exp_elm_count.loc[exp_elm_count > 0]
    assert dict(pp.count_elements(net)) == dict(exp_elm_count)

    # --- compare results ---------------------------------------------------------------------
    res_target = _results_from_powerfactory()
    failed = list()
    atol_dict = {
        "res_bus": {"vm_pu": 1e-4, "va_degree": 7e-3},
        "res_line": {"p_from_mw": 5e-2, "q_from_mvar": 2e-1},
        "res_trafo": {"p_hv_mw": 5e-2, "q_hv_mvar": 1e-1},
    }
    if test_case == "test_ucte_xward":
        atol_dict["res_line"]["q_from_mvar"] = 0.8  # xwards are converted as
        # PV gens towards uct format -> lower tolerance (compared to powerfactory results cannot be
        # expected)

    # --- for loop per result table
    for res_et, df_target in res_target.items():
        et = res_et[4:]
        name_col = "name" # if et != "bus" else "add_name"
        missing_names = pd.Index(net[et][name_col]).difference(df_target.index)
        if len(missing_names):
            logger.error(f"{res_et=} comparison fails since same element names of the PowerFactory "
                         f"results are missing in the pandapower net: {missing_names}")
        df_after_conversion = net[res_et][df_target.columns].set_axis(
            pd.Index(net[et][name_col], name="name"))

        # --- prepare comparison
        if test_case == "test_ucte_trafo3w" and et == "bus":
            df_after_conversion = df_after_conversion.drop("tr3_star_FR")
        if test_case == "test_ucte_trafo3w" and et == "trafo":
            df_after_conversion = df_after_conversion.loc[
                (df_after_conversion.index.values != "trafo3w_FR") |
                ~df_after_conversion.index.duplicated()]
        if et == "line" and "Allgemeine I" in df_after_conversion.index:
            df_after_conversion = df_after_conversion.drop("Allgemeine I")

        # --- compare the shape of the results to be compared
        same_shape = df_after_conversion.shape == df_target.loc[df_after_conversion.index].shape
        df_str = (f"df_after_conversion:\n{df_after_conversion}\n\ndf_target:\n"
                  f"{df_target.loc[df_after_conversion.index]}")
        if not same_shape:
            logger.error(f"{res_et=} comparison fails due to different shape.\n{df_str}")

        # --- compare the results itself
        all_close = all([np.allclose(
            df_after_conversion[col].values,
            df_target.loc[df_after_conversion.index, col].values, atol=atol) for col, atol in
            atol_dict[res_et].items()])
        if not all_close:
            logger.error(f"{res_et=} comparison fails due to different values.\n{df_str}")
            failed.append(res_et)

    # --- overall test evaluation
    if test_is_failed := len(failed):
        logger.error(f"The powerflow result comparisons of these elements failed: {failed}.")
    assert not test_is_failed


if __name__ == '__main__':
    if 1:
        pytest.main([__file__, "-s"])
    else:

        ucte_file = os.path.join(_testfiles_folder(), "test_ucte_DE.uct")
        net = pc.from_ucte(ucte_file)

        print(net)
        print()