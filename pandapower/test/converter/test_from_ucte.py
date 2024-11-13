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


def testfiles_folder():
    return os.path.join(pp.pp_dir, 'test', 'converter', "testfiles")


def _results_from_powerfactory():
    csv_files = {f"res_{et}": os.path.join(testfiles_folder(), f"test_ucte_res_{et}.csv") for et in [
        "bus", "line", "trafo", "trafo3w"]}
    pf_res = {et: pd.read_csv(file, sep=";", index_col=0) for et, file in csv_files.items()}
    return pf_res


def test_ucte_files():
    ucte_file = os.path.join(testfiles_folder(), "test_ucte.uct")

    # --- convert UCTE data
    ucte_parser = pc.ucte_parser.UCTEParser(ucte_file)
    ucte_parser.parse_file()
    ucte_dict = ucte_parser.get_data()

    ucte_converter = pc.ucte_converter.UCTE2pandapower()
    net = ucte_converter.convert(ucte_dict=ucte_dict)

    # --- run power flow
    pp.runpp(net)

    # --- compare results
    bus_name = pd.Series(net.bus.name, index=net.bus.name)
    res_target = _results_from_powerfactory()
    for et, df_target in res_target.items():
        missing_names = df_target.index.difference(pd.Index(net[df_target]["name"]))
        if len(missing_names):
            logger.error(f"{et=} comparison fails since same element names of the PowerFactory "
                         f"results are missing in the pandapower net: {missing_names}")
        all_close = np.allclose(df_after_conversion.values, df_target.values)
        df_after_conversion = net[df_target].loc[bus_name.loc[df_target.index], df_target.columns]
        df_str = f"df_after_conversion:\n{df_after_conversion}\n\ndf_target:\n{df_target}"
        same_shape = df_after_conversion.shape == df_str.shape
        if not same_shape:
            logger.error(f"{et=} comparison fails due to different shape.\n{df_str}")
        all_close = np.allclose(df_after_conversion.values, df_target.values)
        if not all_close:
            logger.error(f"{et=} comparison fails due to different values.\n{df_str}")
        assert all_close



if __name__ == '__main__':
    # pytest.main([__file__, "-xs"])

    ucte_file = os.path.join(testfiles_folder(), "test_ucte.uct")

    ucte_parser = pc.ucte_parser.UCTEParser(ucte_file)
    ucte_parser.parse_file()
    ucte_dict = ucte_parser.get_data()

    ucte_converter = pc.ucte_converter.UCTE2pandapower()
    net = ucte_converter.convert(ucte_dict=ucte_dict)

    print(net)
    print()