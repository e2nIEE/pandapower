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


@pytest.mark.skip(reason="test writing in progress")
def test_ucte_files():
    pass


if __name__ == '__main__':
    # pytest.main([__file__, "-xs"])

    folder = os.path.join(pp.pp_dir, 'test', 'converter', "testfiles")
    file = os.path.join(folder, "test_ucte.uct")

    ucte_parser = pc.ucte_parser.UCTEParser(file)
    ucte_parser.parse_file()
    ucte_dict = ucte_parser.get_data()

    ucte_converter = pc.ucte_converter.UCTE2pandapower()
    net = ucte_converter.convert(ucte_dict=ucte_dict)

    print(net)
    print()