# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np

import pandapower as pp
import pandapower.networks
import pandapower.contingency

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def test_contingency():
    net = pp.networks.case9()

    element_limits = pp.contingency.get_element_limits(net)
    nminus1_cases = {"line": {"index": net.line.index.values}}
    res = pp.contingency.run_contingency(net, nminus1_cases)
    pp.contingency.report_contingency_results(element_limits, res)

    pp.contingency.check_elements_within_limits(element_limits, res, True)

    net.line["max_loading_percent"] = np.nan
    net.line.loc[0:5, 'max_loading_percent'] = 70

    element_limits = pp.contingency.get_element_limits(net)
    nminus1_cases = {"line": {"index": net.line.index.values}}
    res = pp.contingency.run_contingency(net, nminus1_cases)
    pp.contingency.report_contingency_results(element_limits, res)

