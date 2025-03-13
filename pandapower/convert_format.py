# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pandas as pd

from packaging.version import Version

from pandapower._version import __version__, __format_version__
from pandapower.create import create_empty_network, create_poly_cost
from pandapower.results import reset_results
from pandapower.control import TrafoController
import pandapower.plotting.geo as geo

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def convert_format(net, elements_to_deserialize=None):
    """
    Converts old nets to new format to ensure consistency. The converted net is returned.
    """
    from pandapower.toolbox import set_data_type_of_columns_to_default
    if not isinstance(net.version, str) or not hasattr(net, 'format_version') or \
            Version(net.format_version) > Version(net.version):
        net.format_version = net.version
    if isinstance(net.format_version, str) and Version(net.format_version) >= Version(__format_version__):
        return net
    _add_nominal_power(net)
    _add_missing_tables(net)
    _rename_columns(net, elements_to_deserialize)
    _add_missing_columns(net, elements_to_deserialize)
    _create_seperate_cost_tables(net, elements_to_deserialize)
    if Version(str(net.format_version)) < Version("3.0.0"):
        _convert_geo_data(net, elements_to_deserialize)
        _convert_group_element_index(net)
        _convert_trafo_controller_parameter_names(net)
        convert_trafo_pst_logic(net)
    if Version(str(net.format_version)) < Version("2.4.0"):
        _convert_bus_pq_meas_to_load_reference(net, elements_to_deserialize)
    if Version(str(net.format_version)) < Version("2.0.0"):
        _convert_to_generation_system(net, elements_to_deserialize)
        _convert_costs(net)
        _convert_to_mw(net)
        _update_trafo_parameter_names(net, elements_to_deserialize)
        reset_results(net)
    if Version(str(net.format_version)) < Version("1.6"):
        set_data_type_of_columns_to_default(net)
    _convert_objects(net, elements_to_deserialize)
    _update_characteristics(net, elements_to_deserialize)
    correct_dtypes(net, error=False)
    _add_missing_std_type_tables(net)
    net.format_version = __format_version__
    net.version = __version__
    _restore_index_names(net)
    return net


def _convert_geo_data(net, elements_to_deserialize=None):
    if ((_check_elements_to_deserialize('bus_geodata', elements_to_deserialize)
         and _check_elements_to_deserialize('bus', elements_to_deserialize))
        or (_check_elements_to_deserialize('line_geodata', elements_to_deserialize)
            and _check_elements_to_deserialize('line', elements_to_deserialize))):
        if hasattr(net, 'bus_geodata') or hasattr(net, 'line_geodata'):
            if Version(str(net.format_version)) < Version("1.6"):
                net.bus_geodata = pd.DataFrame.from_dict(net.bus_geodata)
                net.line_geodata = pd.DataFrame.from_dict(net.line_geodata)
            geo.convert_geodata_to_geojson(net)


def _restore_index_names(net):
    """Restores dataframes index names stored as dictionary. With newer pp to_json() this
    information is stored to the dataframe its self.
    """
    if "index_names" in net.keys():
        if not isinstance(net["index_names"], dict):
            raise ValueError("To restore the index names of the dataframes, a dict including this "
                             f"information is expected, not {type(net['index_names'])}")
        for key, index_name in net["index_names"].items():
            if key in net.keys():
                net[key].index.name = index_name
        del net["index_names"]


def correct_dtypes(net, error):
    """
    Corrects all dtypes of pp element tables if possible. If not and error is True, an Error is
    raised.
    """
    empty_net = create_empty_network()
    empty_net.trafo['tap_changer_type'] = empty_net.trafo['tap_changer_type']
    empty_net.trafo3w['tap_changer_type'] = empty_net.trafo3w['tap_changer_type']
    not_corrected = list()
    failed = dict()
    for key, table in empty_net.items():
        if isinstance(table, pd.DataFrame):
            if key in net.keys() and isinstance(net[key], pd.DataFrame):
                cols = table.columns.intersection(net[key].columns)
                diff_cols = cols[~(table.dtypes.loc[cols] == net[key].dtypes.loc[cols])]
                for col in diff_cols:
                    try:
                        net[key][col] = net[key][col].astype(table[col].dtype)
                    except ValueError:
                        if key not in failed.keys():
                            failed[key] = [col]
                        else:
                            failed[key].append(col)
            else:
                not_corrected.append(key)
    if not_corrected:
        logger.warning("These keys were not corrected since they miss or are no dataframes: " + str(
            not_corrected))
    if failed:
        msg = "These dtypes could not be corrected: " + str(failed)
        if error:
            raise ValueError(msg)
        else:
            logger.info(msg)


def _convert_group_element_index(net):
    if isinstance(net.group, pd.DataFrame) and "element" in net.group.columns:
        if "element_index" in net.group.columns:
            logger.warning("element cannot be renamed by element_index because columns exist already.")
        net.group = net.group.rename(columns={"element": "element_index"})


def _convert_trafo_controller_parameter_names(net):
    if not isinstance(net.controller, pd.DataFrame):
        return
    for ctrl_idx in net.controller.index:
        controller = net.controller.at[ctrl_idx, "object"]
        if issubclass(type(controller), TrafoController):

            if "tid" in controller.__dict__.keys():
                controller.__dict__["element_index"] = controller.__dict__.pop("tid")
            elif "transformer_index" in controller.__dict__.keys():
                controller.__dict__["element_index"] = controller.__dict__.pop("transformer_index")

            if "trafotable" in controller.__dict__.keys():
                controller.__dict__["element"] = controller.__dict__.pop("trafotable")
                if "trafotype" in controller.__dict__.keys():
                    del controller.__dict__["trafotype"]
            elif "trafotype" in controller.__dict__.keys():
                controller.__dict__["element"] = controller.__dict__.pop("trafotype")

            if "controlled_bus" in controller.__dict__.keys():
                controller.__dict__["trafobus"] = controller.__dict__.pop("controlled_bus")


def _convert_bus_pq_meas_to_load_reference(net, elements_to_deserialize):
    if _check_elements_to_deserialize('measurement', elements_to_deserialize):
        bus_pq_meas_mask = net.measurement.measurement_type.isin(["p", "q"]) & \
                           (net.measurement.element_type == "bus")
        net.measurement.loc[bus_pq_meas_mask, "value"] *= -1


def _convert_to_generation_system(net, elements_to_deserialize):
    update_elements = []
    if _check_elements_to_deserialize('sgen', elements_to_deserialize):
        net.sgen.p_kw *= -1
        net.sgen.q_kvar *= -1
        update_elements += ['sgen']
    if _check_elements_to_deserialize('gen', elements_to_deserialize):
        net.gen.p_kw *= -1
        update_elements += ['gen']
    if _check_elements_to_deserialize('ext_grid', elements_to_deserialize):
        update_elements += ['ext_grid']
    for element in update_elements:
        for suffix in ["p_kw", "q_kvar"]:
            constraints = {}
            if "min_%s" % suffix in net[element]:
                constraints["max_%s" % suffix] = net[element]["min_%s" % suffix] * -1
                del net[element]["min_%s" % suffix]
            if "max_%s" % suffix in net[element]:
                constraints["min_%s" % suffix] = net[element]["max_%s" % suffix] * -1
                del net[element]["max_%s" % suffix]
            for column, values in constraints.items():
                net[element][column] = values
    pq_measurements = net.measurement[net.measurement.measurement_type.isin(["p", "q"])].index
    net.measurement.loc[pq_measurements, ["value", "std_dev"]] *= 1e-3


def _convert_costs(net):
    if "polynomial_cost" in net:
        for cost in net.polynomial_cost.itertuples():
            values = cost.c[0]
            if len(values) == 2:
                cp0 = values[1]
                cp1 = values[0]
                cp2 = 0
            elif len(values) == 3:
                cp0 = values[2]
                cp1 = values[1]
                cp2 = values[0]
            create_poly_cost(net, et=cost.element_type, element=cost.element, cp0_eur=cp0,
                             cp1_eur_per_mw=cp1 * 1e3, cp2_eur_per_mw2=cp2 * 1e6)
        del net.polynomial_cost
    if "piecewise_linear_cost" in net:
        if len(net.piecewise_linear_cost) > 0:
            raise NotImplementedError
        del net.piecewise_linear_cost


def _add_nominal_power(net):
    if "sn_kva" in net:
        net.sn_mva = net.pop("sn_kva") * 1e-3

    # Reset sn_mva only if sn_mva not available
    if "sn_mva" not in net:
        net.sn_mva = 1.0


def _add_missing_tables(net):
    net_new = create_empty_network()
    for key in net_new.keys():
        if key.startswith("_empty_res"):
            net[key] = net_new[key]
        elif key not in net.keys():
            net[key] = net_new[key]


def _create_seperate_cost_tables(net, elements_to_deserialize):
    if _check_elements_to_deserialize('gen', elements_to_deserialize) and "cost_per_kw" in net.gen:
        for index, cost in net.gen.cost_per_kw.items():
            if not np.isnan(cost):
                create_poly_cost(net, index, "gen", cp1_eur_per_mw=cost * 1e3)

    if _check_elements_to_deserialize('sgen', elements_to_deserialize) and \
            "cost_per_kw" in net.sgen:
        for index, cost in net.sgen.cost_per_kw.items():
            if not np.isnan(cost):
                create_poly_cost(net, index, "sgen", cp1_eur_per_kw=cost)

    if _check_elements_to_deserialize('ext_grid', elements_to_deserialize) and \
            "cost_per_kw" in net.ext_grid:
        for index, cost in net.ext_grid.cost_per_kw.items():
            if not np.isnan(cost):
                create_poly_cost(net, index, "ext_grid", cp1_eur_per_kw=cost)

    if _check_elements_to_deserialize('gen', elements_to_deserialize) and \
            "cost_per_kvar" in net.gen:
        for index, cost in net.gen.cost_per_kvar.items():
            if not np.isnan(cost):
                create_poly_cost(net, index, "ext_grid", cp1_eur_per_mw=0,
                                 cq1_eur_per_mvar=cost * 1e3)

    if _check_elements_to_deserialize('sgen', elements_to_deserialize) and \
            "cost_per_kvar" in net.sgen:
        for index, cost in net.sgen.cost_per_kvar.items():
            if not np.isnan(cost):
                create_poly_cost(net, index, "sgen", cp1_eur_per_mw=0,
                                 cq1_eur_per_mvar=cost * 1e3)

    if _check_elements_to_deserialize('ext_grid', elements_to_deserialize) and \
            "cost_per_kvar" in net.ext_grid:
        for index, cost in net.ext_grid.cost_per_kvar.items():
            if not np.isnan(cost):
                create_poly_cost(net, index, "ext_grid", cp1_eur_per_mw=0,
                                 cq1_eur_per_mvar=cost * 1e3)


def _rename_columns(net, elements_to_deserialize):
    if _check_elements_to_deserialize('line', elements_to_deserialize):
        net.line = net.line.rename(columns={'imax_ka': 'max_i_ka'})
    if _check_elements_to_deserialize('gen', elements_to_deserialize):
        net.gen = net.gen.rename(columns={"qmin_mvar": "min_q_mvar", "qmax_mvar": "max_q_mvar"})
    for typ, data in net.std_types["line"].items():
        if "imax_ka" in data:
            net.std_types["line"][typ]["max_i_ka"] = net.std_types["line"][typ].pop("imax_ka")
    _update_trafo_parameter_names(net, elements_to_deserialize)
    # initialize measurement dataframe
    if _check_elements_to_deserialize('measurement', elements_to_deserialize):
        if "measurement" in net and "type" in net.measurement and "measurement":
            if net.measurement.empty:
                net["measurement"] = create_empty_network()["measurement"]
            else:
                net.measurement["side"] = None
                bus_measurements = net.measurement.element_type == "bus"
                net.measurement.loc[bus_measurements, "element"] = \
                    net.measurement.loc[bus_measurements, "bus"].values
                net.measurement.loc[~bus_measurements, "side"] = \
                    net.measurement.loc[~bus_measurements, "bus"].values
                net.measurement = net.measurement.rename(columns={'type': 'measurement_type'})
                net.measurement = net.measurement.drop(["bus"], axis=1)
    if _check_elements_to_deserialize('controller', elements_to_deserialize):
        if "controller" in net:
            net["controller"] = net["controller"].rename(columns={"controller": "object"})
    if "options" in net:
        if "recycle" in net["options"]:
            if "Ybus" in net["options"]["recycle"]:
                if net["options"]["recycle"]["Ybus"]:
                    net["options"]["recycle"]["trafo"] = False
                del net["options"]["recycle"]["Ybus"]
            else:
                net["options"]["recycle"]["trafo"] = True
            if "ppc" in net["options"]["recycle"]:
                if net["options"]["recycle"]["ppc"]:
                    net["options"]["recycle"]["bus_pq"] = False
                del net["options"]["recycle"]["ppc"]
            else:
                net["options"]["recycle"]["bus_pq"] = True


def _add_missing_columns(net, elements_to_deserialize):
    update_elements = []
    if _check_elements_to_deserialize('trafo', elements_to_deserialize):
        update_elements += ['trafo']
    if _check_elements_to_deserialize('line', elements_to_deserialize):
        update_elements += ['line']
    for element in update_elements:
        if "df" not in net[element]:
            net[element]["df"] = 1.0

    if _check_elements_to_deserialize('bus', elements_to_deserialize) \
            and _check_elements_to_deserialize('bus_geodata', elements_to_deserialize) \
            and "geo" not in net.bus:
        net.bus["geo"] = np.nan
    if _check_elements_to_deserialize('trafo3w', elements_to_deserialize) and \
            "tap_at_star_point" not in net.trafo3w:
        net.trafo3w["tap_at_star_point"] = False
    if _check_elements_to_deserialize('trafo3w', elements_to_deserialize) and \
            "tap_step_degree" not in net.trafo3w:
        net.trafo3w["tap_step_degree"] = 0
    if _check_elements_to_deserialize('load', elements_to_deserialize) and \
            "const_z_percent" not in net.load or "const_i_percent" not in net.load:
        net.load["const_z_percent"] = np.zeros(net.load.shape[0])
        net.load["const_i_percent"] = np.zeros(net.load.shape[0])

    if _check_elements_to_deserialize('shunt', elements_to_deserialize) and \
            "vn_kv" not in net["shunt"]:
        net.shunt["vn_kv"] = net.bus.vn_kv.loc[net.shunt.bus.values].values
    if _check_elements_to_deserialize('shunt', elements_to_deserialize) and \
            "step" not in net["shunt"]:
        net.shunt["step"] = 1
    if _check_elements_to_deserialize('shunt', elements_to_deserialize) and \
            "max_step" not in net["shunt"]:
        net.shunt["max_step"] = 1
    if _check_elements_to_deserialize('trafo3w', elements_to_deserialize) and \
            "std_type" not in net.trafo3w:
        net.trafo3w["std_type"] = None

    if _check_elements_to_deserialize('sgen', elements_to_deserialize) and \
            "current_source" not in net.sgen:
        net.sgen["current_source"] = net.sgen["type"].apply(
            func=lambda x: False if x == "motor" else True)

    if _check_elements_to_deserialize('line', elements_to_deserialize):
        if "g_us_per_km" not in net.line:
            net.line["g_us_per_km"] = 0.
        if _check_elements_to_deserialize('line_geodata', elements_to_deserialize) and "geo" not in net.line:
            net.line["geo"] = np.nan

    if _check_elements_to_deserialize('gen', elements_to_deserialize) and \
            "slack" not in net.gen:
        net.gen["slack"] = False

    if _check_elements_to_deserialize('trafo', elements_to_deserialize) and \
            "tap_changer_type" not in net.trafo:
        net.trafo["tap_changer_type"] = None

    # asymmetric impedance
    if _check_elements_to_deserialize('impedance', elements_to_deserialize):
        if "r_pu" in net.impedance:
            net.impedance["rft_pu"] = net.impedance["rtf_pu"] = net.impedance["r_pu"]
            net.impedance["xft_pu"] = net.impedance["xtf_pu"] = net.impedance["x_pu"]
        if "gf_pu" not in net.impedance:
            net.impedance["gf_pu"] = 0.
            net.impedance["gt_pu"] = 0.
            net.impedance["bf_pu"] = 0.
            net.impedance["bt_pu"] = 0.

    # Update the switch table with 'z_ohm'
    if _check_elements_to_deserialize('switch', elements_to_deserialize) and \
            'z_ohm' not in net.switch:
        net.switch['z_ohm'] = 0

    # Update the switch table with 'in_ka'
    if _check_elements_to_deserialize('switch', elements_to_deserialize) and \
            'in_ka' not in net.switch:
        net.switch['in_ka'] = np.nan

    # Update the switch table with 'in_ka'
    if _check_elements_to_deserialize('res_switch', elements_to_deserialize) and \
            'p_from_mw' not in net.res_switch:
        net.res_switch['p_from_mw'] = np.nan
        net.res_switch['q_from_mvar'] = np.nan
        net.res_switch['p_to_mw'] = np.nan
        net.res_switch['q_to_mvar'] = np.nan

    # Update the switch table with 'in_ka'
    if _check_elements_to_deserialize('res_switch_est', elements_to_deserialize) and \
            'p_from_mw' not in net.res_switch_est:
        net.res_switch_est['p_from_mw'] = np.nan
        net.res_switch_est['q_from_mvar'] = np.nan
        net.res_switch_est['p_to_mw'] = np.nan
        net.res_switch_est['q_to_mvar'] = np.nan

    if _check_elements_to_deserialize('measurement', elements_to_deserialize) and \
            "name" not in net.measurement:
        net.measurement.insert(0, "name", None)

    if _check_elements_to_deserialize('controller', elements_to_deserialize) and \
            "initial_run" not in net.controller:
        net.controller.insert(4, 'initial_run', False)
        for _, ctrl in net.controller.iterrows():
            if hasattr(ctrl['object'], 'initial_run'):
                net.controller.at[ctrl.name, 'initial_run'] = ctrl['object'].initial_run
            else:
                net.controller.at[ctrl.name, 'initial_run'] = ctrl['object'].initial_powerflow

    # distributed slack
    if _check_elements_to_deserialize('ext_grid', elements_to_deserialize) and \
            "slack_weight" not in net.ext_grid:
        net.ext_grid['slack_weight'] = 1.0

    if _check_elements_to_deserialize('gen', elements_to_deserialize) and \
            "slack_weight" not in net.gen:
        net.gen['slack_weight'] = 0.0

    if _check_elements_to_deserialize('xward', elements_to_deserialize) and \
            "slack_weight" not in net.xward:
        net.xward['slack_weight'] = 0.0


def _update_trafo_type_parameter_names(net):
    for element in ('trafo', 'trafo3w'):
        for type in net.std_types[element].keys():
            keys = {col: _update_column(col) for col in net.std_types[element][type].keys() if
                    col.startswith("tp") or col.startswith("vsc")}
            for old_key, new_key in keys.items():
                net.std_types[element][type][new_key] = net.std_types[element][type].pop(old_key)


def _update_trafo_parameter_names(net, elements_to_deserialize):
    update_data = []
    if _check_elements_to_deserialize('trafo', elements_to_deserialize):
        update_data += ['trafo']
    if _check_elements_to_deserialize('trafo3w', elements_to_deserialize):
        update_data += ['trafo3w']
    for element in update_data:
        replace_cols = {col: _update_column(col) for col in net[element].columns if
                        col.startswith("tp") or col.startswith("vsc")}
        net[element] = net[element].rename(columns=replace_cols)
    _update_trafo_type_parameter_names(net)


def _update_column(column):
    column = column.replace("tp_", "tap_")
    column = column.replace("_st_", "_step_")
    column = column.replace("_mid", "_neutral")
    column = column.replace("vsc", "vk")
    return column


def _set_data_type_of_columns(net):
    new_net = create_empty_network()
    for key, item in net.items():
        if isinstance(item, pd.DataFrame):
            for col in item.columns:
                if col == "tap_pos":
                    continue
                if key in new_net and col in new_net[key].columns:
                    if set(item.columns) == set(new_net[key]):
                        net[key] = net[key].reindex(new_net[key].columns, axis=1)
                    net[key][col] = net[key][col].astype(new_net[key][col].dtype,
                                                         errors="ignore")


def _convert_to_mw(net):
    replace = [("kw", "mw"), ("kvar", "mvar"), ("kva", "mva")]
    for element in net.keys():
        if isinstance(net[element], pd.DataFrame):
            for old, new in replace:
                diff = {column: column.replace(old, new) for column in net[element].columns if
                        old in column and column != "pfe_kw"}
                net[element] = net[element].rename(columns=diff)
                if len(net[element]) == 0:
                    continue
                for old, new in diff.items():
                    net[element][new] *= 1e-3

    for element, std_types in net.std_types.items():
        for std_type, parameters in std_types.items():
            for parameter in set(parameters.keys()):
                for old, new in replace:
                    if old in parameter and parameter != "pfe_kw":
                        parameters[parameter.replace(old, new)] = parameters[parameter] * 1e-3
                        del parameters[parameter]


def _update_object_attributes(obj):
    """
    Rename attributes of a given object. A new attribute is added and the old one is removed.
    """
    to_rename = {"u_set": "vm_set_pu",
                 "u_lower": "vm_lower_pu",
                 "u_upper": "vm_upper_pu"}

    for key, val in to_rename.items():
        if key in obj.__dict__:
            obj.__dict__[val] = obj.__dict__.pop(key)

    if "vm_lower_pu" in obj.__dict__ and "hunting_limit" not in obj.__dict__:
        obj.__dict__["hunting_limit"] = None


def _convert_objects(net, elements_to_deserialize):
    """
    The function updates attribute names in pandapower objects. For now, it affects TrafoController.
    Should be expanded for other objects if necessary.
    """
    _check_elements_to_deserialize('controller', elements_to_deserialize)
    if _check_elements_to_deserialize('controller', elements_to_deserialize) and \
            "controller" in net.keys():
        for obj in net["controller"].object.values:
            _update_object_attributes(obj)


def _check_elements_to_deserialize(element, elements_to_deserialize):
    if elements_to_deserialize is None:
        return True
    else:
        return element in elements_to_deserialize


def _add_missing_std_type_tables(net):
    type_names = ("fuse", "line_dc")
    for tn in type_names:
        if tn not in net.std_types:
            net.std_types[tn] = {}


def _update_characteristics(net, elements_to_deserialize):
    # new interpolator type has been added to SplineCharacteristic - "pchip", and the attributes have been refactored
    if not _check_elements_to_deserialize("characteristic", elements_to_deserialize) or \
            "characteristic" not in net or net.characteristic.empty:
        return
    for c in net.characteristic.object.values:
        # meta check for old SplineCharacteristic (cannot import it here to use isinstance):
        if not (hasattr(c, "kind") and hasattr(c, "fill_value")):
            continue
        c.interpolator_kind = "interp1d"
        c.kwargs = {"kind": c.__dict__.pop("kind"), "bounds_error": False, "fill_value": c.__dict__.pop("fill_value")}


def convert_trafo_pst_logic(net):
    """
    Converts trafo and trafo3w phase shifter logic to version 3.0 or later
    """
    for trafotable in ["trafo", "trafo3w"]:
        if trafotable in net and isinstance(net[trafotable], pd.DataFrame):
            if net[trafotable].index.size > 0:
                for t in ("", "2"):
                    # drop old tap_phase_shifter flag
                    if f"tap{t}_phase_shifter" in net[trafotable]:
                        net[trafotable] = net[trafotable].drop(columns=f"tap{t}_phase_shifter")
                    if (f"tap{t}_step_degree" in net[trafotable]) or (f"tap{t}_step_percent" in net[trafotable]):
                        # no phase shifters - check if both tap_step_percent & tap_step_degree are 0 or nan
                        mask_na = (((net[trafotable][f"tap{t}_step_degree"].isna()) |
                                   (net[trafotable][f"tap{t}_step_degree"] == 0)) &
                                   ((net[trafotable][f"tap{t}_step_percent"].isna()) |
                                    (net[trafotable][f"tap{t}_step_percent"] == 0)))
                        net[trafotable].loc[mask_na, f"tap{t}_changer_type"] = None
                        # ratio/asymmetrical phase shifters
                        mask_ratio_asym = ((net[trafotable][f"tap{t}_step_degree"] != 90) &
                                           ((net[trafotable][f"tap{t}_step_percent"].notna()) &
                                            (net[trafotable][f"tap{t}_step_percent"] != 0)))
                        net[trafotable].loc[mask_ratio_asym, f"tap{t}_changer_type"] = "Ratio"
                        # symmetrical phase shifters
                        mask_sym = ((net[trafotable][f"tap{t}_step_degree"] == 90) &
                                    ((net[trafotable][f"tap{t}_step_percent"].notna()) &
                                    (net[trafotable][f"tap{t}_step_percent"] != 0)))
                        net[trafotable].loc[mask_sym, f"tap{t}_changer_type"] = "Symmetrical"
                        # ideal phase shifters
                        mask_ideal = (((net[trafotable][f"tap{t}_step_degree"].notna()) &
                                      (net[trafotable][f"tap{t}_step_degree"] != 0)) &
                                      ((net[trafotable][f"tap{t}_step_percent"].isna()) |
                                       (net[trafotable][f"tap{t}_step_percent"] == 0)))
                        net[trafotable].loc[mask_ideal, f"tap{t}_changer_type"] = "Ideal"
