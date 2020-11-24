# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
from copy import deepcopy
from pandas import DataFrame
from numpy import array

from pandapower.control.util.auxiliary import get_controller_index
from pandapower.control.controller.trafo_control import TrafoController

try:
    import pplog
except:
    import logging as pplog

logger = pplog.getLogger(__name__)


def control_diagnostic(net, respect_in_service=True):
    """
    Diagnostic function to find obvious mistakes in control data
    """
    # --- find and log same type controllers connected to same elements
    indices = list(net.controller.index)
    for idx in indices:
        current_controller = net.controller.object.loc[idx]
        parameters = deepcopy(current_controller.matching_params) if "matching_params" in \
                                                                     current_controller.__dict__ else {}
        if respect_in_service:
            if not net.controller.in_service.at[idx]:
                continue
            parameters["in_service"] = True
        same_type_existing_ctrl = get_controller_index(net, ctrl_type=type(current_controller),
                                                       parameters=parameters)
        if len(same_type_existing_ctrl) > 1:
            logger.info("Same type and same matching parameters controllers " + str([
                '%i' % i for i in same_type_existing_ctrl]) +
                        " could affect convergence.")
            for val in same_type_existing_ctrl:
                indices.remove(val)

    # --- find trafo controller of the same trafo
    trafo_ctrl = net.controller.object.apply(lambda x: isinstance(x, TrafoController))
    trafo_ctrl_idx = net.controller.loc[trafo_ctrl].index.values
    trafo_type = net.controller.loc[trafo_ctrl].object.apply(lambda x: x.trafotype)
    trafo_id = net.controller.loc[trafo_ctrl].object.apply(lambda x: x.tid)
    in_service = net.controller.loc[trafo_ctrl_idx, 'in_service']
    temp = DataFrame(index=trafo_ctrl_idx, columns=['trafotype', 'tid', 'in_service'],
                     data=array([trafo_type.values, trafo_id.values, in_service.values]).T)
    if respect_in_service:
        cols = ['trafotype', 'tid', 'in_service']
    else:
        cols = ['trafotype', 'tid']
    d = temp.loc[temp.duplicated(subset=cols, keep=False)]
    if len(d) > 0:
        for t in d.tid.unique():
            logger.info("Trafo Controllers %s at the transformer %s probably could affect convergence."
                        % (str(['%i' % i for i in d.loc[d.tid == t].index.values]), t))
