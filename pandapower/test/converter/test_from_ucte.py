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
except:
    import logging

logger = logging.getLogger(__name__)


def _testfiles_folder():
    return os.path.join(pp.pp_dir, 'test', 'converter', "testfiles")


def _results_from_powerfactory():
    csv_files = {f"res_{et}": os.path.join(_testfiles_folder(), f"test_ucte_res_{et}.csv") for et in [
        "bus", "line", "trafo", "trafo3w"]}
    pf_res = {et: pd.read_csv(file, sep=";", index_col=0) for et, file in csv_files.items()}
    return pf_res


def _test_ucte_file(ucte_file=None):
    if ucte_file is None:
        ucte_file = os.path.join(_testfiles_folder(), "test_ucte.uct")

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
    for res_et, df_target in res_target.items():
        et = res_et[4:]
        name_col = "name" if et != "bus" else "add_name"
        missing_names = pd.Index(net[et][name_col]).difference(df_target.index)
        if len(missing_names):
            logger.error(f"{res_et=} comparison fails since same element names of the PowerFactory "
                         f"results are missing in the pandapower net: {missing_names}")
        df_after_conversion = net[res_et][df_target.columns].set_axis(
            pd.Index(net[et][name_col], name="name"))
        all_close = np.allclose(df_after_conversion.values,
                                df_target.loc[df_after_conversion.index].values, atol=1e-5)
        df_str = f"df_after_conversion:\n{df_after_conversion}\n\ndf_target:\n{df_target}"
        same_shape = df_after_conversion.shape == df_str.shape
        if not same_shape:
            logger.error(f"{res_et=} comparison fails due to different shape.\n{df_str}")
        if not all_close:
            logger.error(f"{res_et=} comparison fails due to different values.\n{df_str}")
        assert all_close  # TODO
        failed.append(res_et)
    if len(failed):  # TODO
        logger.error(f"This res_et failed: {failed}.")
        assert True

    ### remarks
    # _3 (alle möglichen unkritischen Elemente): ??
    # AL (Line+impedance): Größenordnung passt, Ergebnisse nicht - AUßERDEM: Funktioniert nicht mit impedance zwischen Spannungsebenen (wird dann als Trafo exportiert)
    # ES (Line+Ward/xWard/sgen/load + bus-bus-schalter): P passt, Q so gut wie
    # FR (2 Lines zwischen 2 ExtGrid): passt gar nicht
    # HR (Line+Shunt): passt perfekt
    # HU (Line+Ward): P passt, Q so gut wie
    # NL 2x(Line+xWard): P passt, Q so gut wie


def test_ucte_file3():
    _test_ucte_file(os.path.join(_testfiles_folder(), "test_ucte3.uct"))

def test_ucte_file_AL():
    _test_ucte_file(os.path.join(_testfiles_folder(), "test_ucte_AL.uct"))

def test_ucte_file_ES():
    _test_ucte_file(os.path.join(_testfiles_folder(), "test_ucte_ES.uct"))

def test_ucte_file_FR():
    _test_ucte_file(os.path.join(_testfiles_folder(), "test_ucte_FR.uct"))

def test_ucte_file_HR():
    _test_ucte_file(os.path.join(_testfiles_folder(), "test_ucte_HR.uct"))

def test_ucte_file_HU():
    _test_ucte_file(os.path.join(_testfiles_folder(), "test_ucte_HU.uct"))

def test_ucte_file_NL():
    _test_ucte_file(os.path.join(_testfiles_folder(), "test_ucte_NL.uct"))


if __name__ == '__main__':
    if 0:
        pytest.main([__file__, "-s"])
    elif 1:
        test_ucte_file3()
    else:

        ucte_file = os.path.join(_testfiles_folder(), "test_ucte.uct")

        ucte_parser = pc.ucte_parser.UCTEParser(ucte_file)
        ucte_parser.parse_file()
        ucte_dict = ucte_parser.get_data()

        ucte_converter = pc.ucte_converter.UCTE2pandapower()
        net = ucte_converter.convert(ucte_dict=ucte_dict)

        print(net)
        print()