# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import logging
import time
from pandapower.converter.ucte.ucte_converter import UCTE2pandapower
from pandapower.converter.ucte.ucte_parser import UCTEParser
from pandapower.auxiliary import pandapowerNet

logger = logging.getLogger('ucte.from_ucte')


def from_ucte_dict(ucte_parser: UCTEParser, slack_as_gen: bool = True) -> pandapowerNet:
    """
    Creates a pandapower net from an UCTE data structure.

    :param UCTEParser ucte_parser: The UCTEParser with parsed UCTE data.
    :param bool slack_as_gen: decides whether slack elements are converted as gen or ext_grid elements, default True

    :return: A pandapower net.
    :rtype: pandapowerNet

    """
    ucte_converter = UCTE2pandapower(slack_as_gen=slack_as_gen)
    net = ucte_converter.convert(ucte_parser.get_data())
    return net


def from_ucte(ucte_file: str, slack_as_gen: bool = True) -> pandapowerNet:
    """
    Converts net data stored as an UCTE file to a pandapower net.

    :param str ucte_file: path to the ucte file which includes all the data of the grid (EHV or HV or both)
    :param bool slack_as_gen: decides whether slack elements are converted as gen or ext_grid elements.

    :return: A pandapower net
    :rtype: pandapowerNet

    :example:
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

    pp_net = from_ucte_dict(ucte_parser, slack_as_gen=slack_as_gen)

    time_end_converting = time.time()

    logger.info("Needed time for parsing from ucte: %s" % (time_start_converting - time_start_parsing))
    logger.info("Needed time for converting from ucte: %s" % (time_end_converting - time_start_converting))
    logger.info("Total Time (from_ucte()): %s" % (time_end_converting - time_start_parsing))

    return pp_net


if __name__ == "__main__":
    import os
    import pandapower as pp

    # loading the line test as example
    ucte_file = os.path.join(pp.pp_dir, "test", "converter", "testfiles", "test_ucte_DK.uct")
    net = pp.converter.from_ucte(ucte_file)
