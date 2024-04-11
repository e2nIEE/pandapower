# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
from copy import deepcopy
import numpy as np
import pandas as pd

from pandapower.control.util.auxiliary import get_controller_index
from pandapower.control.controller.trafo_control import TrafoController

try:
    import pandaplan.core.pplog as pplog
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
    trafo_ctrl = []
    for idx in net.controller.index:
        current_controller = net.controller.object.loc[idx]
        if issubclass(type(current_controller), TrafoController):
            trafo_ctrl += [idx]
    for idx in trafo_ctrl:
        current_controller = net.controller.object.loc[idx]
        parameters = {"tid": current_controller.tid, "trafotype": current_controller.trafotype}
        if respect_in_service:
            if not net.controller.in_service.at[idx]:
                continue
            parameters["in_service"] = True
        trafo_ctrl_at_same_trafo = get_controller_index(net, parameters=parameters, idx=trafo_ctrl)
        if len(trafo_ctrl_at_same_trafo) > 1:
            logger.info("Trafo Controllers %s at the %s transformer %s probably could affect convergence." %
                        (str(['%i' % i for i in trafo_ctrl_at_same_trafo]), parameters['trafotype'], parameters["tid"]))
            for val in trafo_ctrl_at_same_trafo:
                trafo_ctrl.remove(val)


def trafo_characteristics_diagnostic(net):
    logger.info("Checking transformer characteristics")
    cols2w = ["vk_percent_characteristic", "vkr_percent_characteristic"]
    cols3w = [f"vk{r}_{side}_percent_characteristic" for side in ["hv", "mv", "lv"] for r in ["", "r"]]
    for trafo_table, cols in zip(["trafo", "trafo3w"], [cols2w, cols3w]):
        if len(net[trafo_table]) == 0 or \
                'tap_dependent_impedance' not in net[trafo_table] or \
                not net[trafo_table]['tap_dependent_impedance'].any():
            logger.info("No %s with tap-dependent impedance found." % trafo_table)
            continue
        # check if there are any missing characteristics
        tap_dependent_impedance = net[trafo_table]['tap_dependent_impedance'].fillna(False).values
        logger.info(f"{trafo_table}: found {sum(tap_dependent_impedance)} transformer(s) with tap-dependent impedance")
        if len(np.intersect1d(net[trafo_table].columns, cols)) == 0:
            logger.warning("No columns defined for transformer tap characteristics in %s. "
                           "Power flow calculation will raise an error." % trafo_table)
        elif net[trafo_table].loc[tap_dependent_impedance, np.intersect1d(cols, net[trafo_table].columns)].isnull().all(axis=1).any():
            logger.warning(f"Some transformers in {trafo_table} table have tap_dependent_impedance set to True, "
                           f"but no defined characteristics. Power flow calculation will raise an error.")
        for col in cols:
            if col not in net[trafo_table]:
                logger.info("%s: %s is missing" % (trafo_table, col))
                continue
            elif net[trafo_table].loc[tap_dependent_impedance, col].isnull().any():
                logger.info("%s: %s is missing for some transformers" % (trafo_table, col))
            elif len(set(net[trafo_table].loc[tap_dependent_impedance, col]) - set(net.characteristic.index)) > 0:
                logger.info("%s: %s contains invalid characteristics indices" % (trafo_table, col))
            else:
                logger.debug(f"{trafo_table}: {col} has {len(net[trafo_table][col].dropna())} characteristics")

            # chack if any characteristics have value at the neutral point that deviates from the transformer parameter
            variable = col.replace("_characteristic", "")
            for tid in net[trafo_table].index[tap_dependent_impedance]:
                tap_neutral = net[trafo_table].tap_neutral.fillna(0).at[tid]
                s_id = net[trafo_table][col].at[tid]
                if pd.isnull(s_id):
                    continue
                s = net.characteristic.object.at[s_id]
                s_val = s(tap_neutral)
                var_val = net[trafo_table].at[tid, variable]
                if not np.isclose(s_val, var_val, rtol=0, atol=1e-6):
                    logger.warning(f"The characteristic value of {s_val} at the neutral tap position {tap_neutral} "
                                   f"does not match the value {var_val} of {variable} for the {trafo_table} with index {tid} "
                                   f"(deviation of {s_val-var_val})")

