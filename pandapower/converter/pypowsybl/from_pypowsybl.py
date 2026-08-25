# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import logging
import time

import pandas as pd

from pandapower.auxiliary import pandapowerNet
from pandapower.converter.pypowsybl.pypowsybl_converter import PyPowSyBlConverter

logger = logging.getLogger("pypowsybl.from_pypowsybl")


def from_pypowsybl(
    pypowsybl_file: str,
    log_static_comparison: bool = False,
    log_loadflow_comparison: bool = False,
    return_loadflow_table: bool = False,
    default_shift_degree: float = 0.0,
    default_length_km: float = 1.0,
) -> pandapowerNet | tuple[pandapowerNet, pd.DataFrame]:
    """Convert net data stored as a powsybl XIIDM file to a pandapower net.

    Args:
        pypowsybl_file: Path to the powsybl ``.xiidm`` file containing the grid data.
        log_static_comparison: Whether static transfer comparison tables are logged.
        log_loadflow_comparison: Whether powsybl and pandapower AC load-flow results
            are compared and logged.
        return_loadflow_table: Whether the load-flow comparison table is returned
            together with the pandapower net.
        default_shift_degree: Fallback phase-shift angle in degrees.
        default_length_km: Fallback line length in kilometres.

    Returns:
        The converted pandapower net. If ``return_loadflow_table`` is ``True``,
        returns a tuple containing the pandapower net and the load-flow comparison
        table.

    Examples:
        >>> from pandapower.converter.pypowsybl.from_pypowsybl import from_pypowsybl
        >>> net = from_pypowsybl("network.xiidm")
    """
    
    time_start_converting = time.time()

    pypowsybl_converter = PyPowSyBlConverter()

    conversion_result = pypowsybl_converter._powsybl_to_pandapower(
        filename=pypowsybl_file,
        log_static_comparison=log_static_comparison,
        log_loadflow_comparison=log_loadflow_comparison,
        return_loadflow_table=return_loadflow_table,
        default_shift_degree=default_shift_degree,
        default_length_km=default_length_km,
    )

    if len(conversion_result) == 4:
        pp_net, _, _, loadflow_table = conversion_result
    else:
        pp_net, _, _ = conversion_result

    converting_time = time.time() - time_start_converting

    logger.info("Needed time for converting from pypowsybl: %s", converting_time)

    if return_loadflow_table:
        return pp_net, loadflow_table

    return pp_net