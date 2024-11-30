# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
from copy import deepcopy

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
        parameters = deepcopy(current_controller.matching_params) if \
            "matching_params" in current_controller.__dict__ else {}
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
        parameters = {"element_index": current_controller.element_index,
                      "element": current_controller.element}
        if respect_in_service:
            if not net.controller.in_service.at[idx]:
                continue
            parameters["in_service"] = True
        trafo_ctrl_at_same_trafo = get_controller_index(net, parameters=parameters, idx=trafo_ctrl)
        if len(trafo_ctrl_at_same_trafo) > 1:
            logger.info(
                "Trafo Controllers %s at the %s transformer %s probably could affect convergence." %
                (str(['%i' % i for i in trafo_ctrl_at_same_trafo]), parameters['element'],
                 parameters["element_index"]))
            for val in trafo_ctrl_at_same_trafo:
                trafo_ctrl.remove(val)


def trafo_characteristic_table_diagnostic(net):
    logger.info("Checking transformer characteristic table")
    if "trafo_characteristic_table" not in net:
        logger.info("No transformer characteristic table found")
        return False
    cols2w = ["id_characteristic", "step", "voltage_ratio", "angle_deg", "vk_percent", "vkr_percent"]
    cols3w = ["id_characteristic", "step", "voltage_ratio", "angle_deg", "vk_hv_percent", "vkr_hv_percent",
              "vk_mv_percent", "vkr_mv_percent", "vk_lv_percent", "vkr_lv_percent"]
    for trafo_table, cols in zip(["trafo", "trafo3w"], [cols2w, cols3w]):
        if len(net[trafo_table]) == 0 or \
                not all(col in net[trafo_table] for col in ['id_characteristic_table', 'tap_dependency_table']) or \
                (not net[trafo_table]['id_characteristic_table'].notna().any() and
                 not net[trafo_table]['tap_dependency_table'].any()):
            logger.info("No %s with tap-dependent characteristics found." % trafo_table)
            continue
        # check if both tap_dependency_table & id_characteristic_table columns are populated
        mismatch = net[trafo_table][
            (net[trafo_table]['tap_dependency_table'] & net[trafo_table]['id_characteristic_table'].isna()) |
            (~net[trafo_table]['tap_dependency_table'] & net[trafo_table]['id_characteristic_table'].notna())
            ].shape[0]
        if mismatch != 0:
            raise UserWarning(
                f"{trafo_table}: found {mismatch} transformer(s) with not both "
                f"tap_dependency_table and id_characteristic_table parameters populated. "
                f"Power flow calculation will raise an error.")
        # check if all relevant columns are populated in the trafo_characteristic_table
        temp = net[trafo_table].dropna(subset=["id_characteristic_table"])[
            ["tap_dependency_table", "id_characteristic_table"]]
        merged_df = temp.merge(net["trafo_characteristic_table"], left_on="id_characteristic_table",
                               right_on="id_characteristic", how="inner")
        unpopulated = merged_df.loc[~merged_df[cols].notna().all(axis=1)]
        if not unpopulated.empty:
            raise UserWarning(f"There are some transformers in the {trafo_table} table with not all characteristics "
                              f"populated in the trafo_characteristic_table.")
        # check tap_dependency_table & id_characteristic_table column types
        if net[trafo_table]['tap_dependency_table'].dtype != 'bool':
            raise UserWarning(f"The tap_dependency_table column in the {trafo_table} table is not of bool type.")
        if net[trafo_table]['id_characteristic_table'].dtype != 'Int64':
            raise UserWarning(f"The id_characteristic_table column in the {trafo_table} table is not of Int64 type.")
        # check if all id_characteristic_table values are present in id_characteristic column
        # of trafo_characteristic_table
        if not net[trafo_table]['id_characteristic_table'].isin(
                net["trafo_characteristic_table"]['id_characteristic']).all():
            raise UserWarning(f"Not all id_characteristic_table values in the {trafo_table} table are present"
                              f"in id_characteristic column of trafo_characteristic_table. "
                              f"Power flow calculation will raise an error.")
    return True
