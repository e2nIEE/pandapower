# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import logging
import os
import time
from typing import Union, List, Type, Dict
from pandapower.converter.ucte.ucte_converter import UCTE2pandapower
from pandapower.converter.ucte.ucte_parser import UCTEParser

logger = logging.getLogger('ucte.from_ucte')


def from_ucte_dict(ucte_parser: UCTEParser):
    """Creates a pandapower net from an UCTE data structure.

    Parameters
    ----------
    ucte_parser : UCTEParser
        The UCTEParser with parsed UCTE data.

    Returns
    -------
    pandapowerNet
        net
    """

    ucte_converter = UCTE2pandapower()
    net = ucte_converter.convert(ucte_parser.get_data())

    return net


def from_ucte(ucte_file: str):
    """Converts net data stored as an UCTE file to a pandapower net.

    Parameters
    ----------
    ucte_file : str
        path to the ucte file which includes all the data of the grid (EHV or HV or both)

    Returns
    -------
    pandapowerNet
        net

    Example
    -------
    >>> import os
    >>> import pandapower as pp
    >>> ucte_file = os.path.join(pp.pp_dir, "test", "converter", "testfiles", "test_ucte_DK.uct")
    >>> net = pp.converter.from_ucte(ucte_file)
    """

    # Note:
    # the converter functionality from_ucte() and internal functions are structured similar to
    # the cim converter

    time_start_parsing = time.time()

    ucte_parser = UCTEParser(ucte_file)
    ucte_parser.parse_file()

    time_start_converting = time.time()

    pp_net = from_ucte_dict(ucte_parser)

    time_end_converting = time.time()

    logger.info("Needed time for parsing from ucte: %s" % (time_start_converting - time_start_parsing))
    logger.info("Needed time for converting from ucte: %s" % (time_end_converting - time_start_converting))
    logger.info("Total Time (from_ucte()): %s" % (time_end_converting - time_start_parsing))

    return pp_net


if __name__ == "__main__":
    import os
    import pandapower as pp

    ### loading the line test as example
    ucte_file = os.path.join(pp.pp_dir, "test", "converter", "testfiles", "test_ucte_DK.uct")
    net = pp.converter.from_ucte(ucte_file)
