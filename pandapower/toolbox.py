# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandas as pd
import copy
import numbers
from collections import defaultdict
try:
    import pplog as logging
except:
    import logging

logger = logging.getLogger(__name__)

from pandapower.auxiliary import get_indices, PandapowerNet
from pandapower.create import create_empty_network

# --- Information


def lf_info(net, numv=1, numi=2):
    """
    Prints some basic information of the results in a net.
    (max/min voltage, max trafo load, max line load)
    """
    logger.info("Max voltage")
    for _, r in net.res_bus.sort_values("vm_pu", ascending=False).iloc[:numv].iterrows():
        logger.info("  %s at busidx %s (%s)", r.vm_pu, r.name, net.bus.name.at[r.name])
    logger.info("Min voltage")
    for _, r in net.res_bus.sort_values("vm_pu").iloc[:numv].iterrows():
        logger.info("  %s at busidx %s (%s)", r.vm_pu, r.name, net.bus.name.at[r.name])
    logger.info("Max loading trafo")
    if net.res_trafo is not None:
        for _, r in net.res_trafo.sort_values("loading_percent", ascending=False).iloc[
                :numi].iterrows():
            logger.info("  %s loading at trafo %s (%s)", r.loading_percent, r.name,
                        net.trafo.name.at[r.name])
    logger.info("Max loading line")
    for _, r in net.res_line.sort_values("loading_percent", ascending=False).iloc[:numi].iterrows():
        logger.info("  %s loading at line %s (%s)", r.loading_percent, r.name,
                    net.line.name.at[r.name])


def switch_info(net, sidx):
    """
    Prints what buses and elements are connected by a certain switch.
    """
    switch_type = net.switch.at[sidx, "et"]
    bidx = net.switch.at[sidx, "bus"]
    bus_name = net.bus.at[bidx, "name"]
    eidx = net.switch.at[sidx, "element"]
    if switch_type == "b":
        bus2_name = net.bus.at[eidx, "name"]
        logger.info("Switch %u connects bus %u (%s) with bus %u (%s)" % (sidx, bidx, bus_name, eidx,
                                                                         bus2_name))
    elif switch_type == "l":
        line_name = net.line.at[eidx, "name"]
        logger.info("Switch %u connects bus %u (%s) with line %u (%s)" % (sidx, bidx, bus_name, eidx,
                                                                          line_name))
    elif switch_type == "t":
        trafo_name = net.trafo.at[eidx, "name"]
        logger.info("Switch %u connects bus %u (%s) with trafo %u (%s)" % (sidx, bidx, bus_name, eidx,
                                                                           trafo_name))


def overloaded_lines(net, max_load=100):
    """
    Returns the results for all lines with loading_percent > max_load or None, if
    there are none.
    """
    if net.converged:
        return net["res_line"].index[net["res_line"]["loading_percent"] > max_load]
    else:
        raise UserWarning("The last loadflow terminated erratically, results are invalid!")


def violated_buses(net, u_min, u_max):
    """
    Returns the results for all buses with if vm_pu is not within u_max and u_min or None, if
    there are none.
    """
    if net.converged:
        return net["bus"].index[(net["res_bus"]["vm_pu"] < u_min) |
                                (net["res_bus"]["vm_pu"] > u_max)]
    else:
        raise UserWarning("The last loadflow terminated erratically, results are invalid!")


def equal_nets(x, y, check_only_results=False, tol=1.e-14):
    """
    Compares two networks. The networks are considered equal
    if they share the same keys and values, except of the
    'et' (elapsed time) entry which differs depending on
    runtime conditions and entries stating with '_'
    """
    eq = True
    eq_count = 0
    if isinstance(x, dict) or isinstance(x, PandapowerNet) and \
            isinstance(y, dict) or isinstance(y, PandapowerNet):

        # for two networks make sure both have the same keys ...
        if len(set(x.keys()) - set(y.keys())) + len(set(y.keys()) - set(x.keys())) > 0:
            logger.info("Networks entries mismatch:", list(x.keys()), " - VS. - ", list(y.keys()))
            return False

        # ... and then iter through the keys, checking for equality for each table
        for k in [k for k in x.keys() if(k != 'et' and not k.startswith("_"))]:
            if check_only_results and not k.startswith("res_"):
                continue  # skip anything thats not a result table

            eq = equal_nets(x[k], y[k], check_only_results, tol)

            # counts the mismatches and creates a list to print
            if not eq:
                if eq_count == 0:
                    logger_str = "Mismatch(es) at table(s): %s" % k
                else:
                    logger_str = logger_str + ", %s" % k
                eq_count += 1

        if eq_count:
            logger.info(logger_str)
            return False
    else:
        if isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
            # for two DataFrames eval if all entries are equal
            if len(x) != len(y):
                logger.info("The number of elements of differ.")
                return False

            if len(x.dtypes) != len(y.dtypes):
                logger.info("The number of columns of differ.")
                return False

            # we use numpy.allclose to grant a tolerance
            numerical_equal = np.allclose(x.select_dtypes(include=[np.number]),
                                          y.select_dtypes(include=[np.number]),
                                          atol=tol, equal_nan=True)

            # we use pandas equals for the rest, which also evaluates NaNs to be equal
            rest_equal = x.select_dtypes(exclude=[np.number]).equals(
                y.select_dtypes(exclude=[np.number]))

            eq &= numerical_equal & rest_equal
        else:
            # else items are assumed to be single values, compare them
            if isinstance(x, numbers.Number):
                eq &= np.isclose(x, y, atol=tol)
            else:
                eq &= x == y

            # Note: if we deal with HDFStores we might encounter a problem here:
            # empty DataFrames are stored as None to HDFStore so we might be
            # comparing None to an empty DataFrame
    return eq


# --- Simulation setup and preparations

def convert_format(net):
    """
    Converts old nets to new format to ensure consistency. Converted net is returned
    """
    from pandapower.std_types import add_basic_std_types, create_std_type, parameter_from_std_type
    from pandapower.run import reset_results
    if "std_types" not in net:
        net.std_types = {"line": {}, "trafo": {}, "trafo3w": {}}
        add_basic_std_types(net)

        import os
        import json
        path, file = os.path.split(os.path.realpath(__file__))
        linedb = os.path.join(path, "linetypes.json")
        if os.path.isfile(linedb):
            with open(linedb, 'r') as f:
                lt = json.load(f)
        else:
            lt = {}
        for std_type in net.line.std_type.unique():
            if std_type in lt:
                if "shift_degree" not in lt[std_type]:
                    lt[std_type]["shift_degree"] = 0
                create_std_type(net, lt[std_type], std_type, element="line")
        trafodb = os.path.join(path, "trafotypes.json")
        if os.path.isfile(trafodb):
            with open(trafodb, 'r') as f:
                tt = json.load(f)
        else:
            tt = {}
        for std_type in net.trafo.std_type.unique():
            if std_type in tt:
                create_std_type(
                    net, tt[std_type], std_type, element="trafo")

    net.trafo.tp_side.replace(1, "hv", inplace=True)
    net.trafo.tp_side.replace(2, "lv", inplace=True)
    net.trafo.tp_side = net.trafo.tp_side.where(pd.notnull(net.trafo.tp_side), None)
    net.trafo3w.tp_side.replace(1, "hv", inplace=True)
    net.trafo3w.tp_side.replace(2, "mv", inplace=True)
    net.trafo3w.tp_side.replace(3, "lv", inplace=True)
    net.trafo3w.tp_side = net.trafo3w.tp_side.where(pd.notnull(net.trafo3w.tp_side), None)

    net["bus"] = net["bus"].rename(
        columns={'voltage_level': 'vn_kv', 'bus_type': 'type', "un_kv": "vn_kv"})
    net["bus"]["type"].replace("s", "b", inplace=True)
    net["bus"]["type"].replace("k", "n", inplace=True)
    net["line"] = net["line"].rename(columns={'vf': 'df', 'line_type': 'type'})
    net["ext_grid"] = net["ext_grid"].rename(columns={"angle_degree": "va_degree",
                                                      "ua_degree": "va_degree", "sk_max_mva": "s_sc_max_mva", "sk_min_mva": "s_sc_min_mva"})
    net["line"]["type"].replace("f", "ol", inplace=True)
    net["line"]["type"].replace("k", "cs", inplace=True)
    net["trafo"] = net["trafo"].rename(columns={'trafotype': 'std_type', "type": "std_type",
                                                "un1_kv": "vn_hv_kv", "un2_kv": "vn_lv_kv",
                                                'vfe_kw': 'pfe_kw', "unh_kv": "vn_hv_kv",
                                                "unl_kv": "vn_lv_kv", 'trafotype': 'std_type',
                                                "type": "std_type", 'vfe_kw': 'pfe_kw',
                                                "uk_percent": "vsc_percent",
                                                "ur_percent": "vscr_percent",
                                                "vnh_kv": "vn_hv_kv", "vnl_kv": "vn_lv_kv"})
    net["trafo3w"] = net["trafo3w"].rename(columns={"unh_kv": "vn_hv_kv", "unm_kv": "vn_mv_kv",
                                                    "unl_kv": "vn_lv_kv",
                                                    "ukh_percent": "vsc_hv_percent",
                                                    "ukm_percent": "vsc_mv_percent",
                                                    "ukl_percent": "vsc_lv_percent",
                                                    "urh_percent": "vscr_hv_percent",
                                                    "urm_percent": "vscr_mv_percent",
                                                    "url_percent": "vscr_lv_percent",
                                                    "vnh_kv": "vn_hv_kv", "vnm_kv": "vn_mv_kv",
                                                    "vnl_kv": "vn_lv_kv", "snh_kv": "sn_hv_kv",
                                                    "snm_kv": "sn_mv_kv", "snl_kv": "sn_lv_kv"})
    net["switch"]["type"].replace("LS", "CB", inplace=True)
    net["switch"]["type"].replace("LTS", "LBS", inplace=True)
    net["switch"]["type"].replace("TS", "DS", inplace=True)
    if "name" not in net.switch.columns:
        net.switch["name"] = None
    net["switch"] = net["switch"].rename(columns={'element_type': 'et'})
    net["ext_grid"] = net["ext_grid"].rename(columns={'voltage': 'vm_pu', "u_pu": "vm_pu",
                                                      "sk_max": "sk_max_mva", "ua_degree": "va_degree"})
    if "in_service" not in net["ext_grid"].columns:
        net["ext_grid"]["in_service"] = 1
    if "shift_mv_degree" not in net["trafo3w"].columns:
        net["trafo3w"]["shift_mv_degree"] = 0
    if "shift_lv_degree" not in net["trafo3w"].columns:
        net["trafo3w"]["shift_lv_degree"] = 0
    parameter_from_std_type(net, "shift_degree", element="trafo", fill=0)
    if "gen" not in net:
        net["gen"] = pd.DataFrame(np.zeros(0, dtype=[("name", np.dtype(object)),
                                                     ("bus", "u4"),
                                                     ("p_kw", "f8"),
                                                     ("vm_pu", "f8"),
                                                     ("sn_kva", "f8"),
                                                     ("scaling", "f8"),
                                                     ("in_service", "i8"),
                                                     ("min_q_kvar", "f8"),
                                                     ("max_q_kvar", "f8"),
                                                     ("type", np.dtype(object))]))

    if "impedance" not in net:
        net["impedance"] = pd.DataFrame(np.zeros(0, dtype=[("name", np.dtype(object)),
                                                           ("from_bus", "u4"),
                                                           ("to_bus", "u4"),
                                                           ("r_pu", "f8"),
                                                           ("x_pu", "f8"),
                                                           ("sn_kva", "f8"),
                                                           ("in_service", 'bool')]))
    if "ward" not in net:
        net["ward"] = pd.DataFrame(np.zeros(0, dtype=[("name", np.dtype(object)),
                                                      ("bus", "u4"),
                                                      ("ps_kw", "u4"),
                                                      ("qs_kvar", "f8"),
                                                      ("pz_kw", "f8"),
                                                      ("qz_kvar", "f8"),
                                                      ("in_service", "f8")]))
    if "xward" not in net:
        net["xward"] = pd.DataFrame(np.zeros(0, dtype=[("name", np.dtype(object)),
                                                       ("bus", "u4"),
                                                       ("ps_kw", "u4"),
                                                       ("qs_kvar", "f8"),
                                                       ("pz_kw", "f8"),
                                                       ("qz_kvar", "f8"),
                                                       ("r_ohm", "f8"),
                                                       ("x_ohm", "f8"),
                                                       ("vm_pu", "f8"),
                                                       ("in_service", "f8")]))
    if "shunt" not in net:
        net["shunt"] = pd.DataFrame(np.zeros(0, dtype=[("bus", "u4"),
                                                       ("name", np.dtype(object)),
                                                       ("p_kw", "f8"),
                                                       ("q_kvar", "f8"),
                                                       ("scaling", "f8"),
                                                       ("in_service", "i8")]))

    if "parallel" not in net.line:
        net.line["parallel"] = 1
    if "_empty_res_bus" not in net:
        net2 = create_empty_network()
        for key, item in net2.items():
            if key.startswith("_empty"):
                net[key] = copy.copy(item)
        reset_results(net)

    for attribute in ['tp_st_percent', 'tp_pos', 'tp_mid', 'tp_min', 'tp_max']:
        if net.trafo[attribute].dtype == 'O':
            net.trafo[attribute] = pd.to_numeric(net.trafo[attribute])
    net["gen"] = net["gen"].rename(columns={"u_pu": "vm_pu"})
    for element, old, new in [("trafo", "unh_kv", "vn_hv_kv"),
                              ("trafo", "unl_kv", "vn_lv_kv"),
                              ("trafo", "uk_percent", "vsc_percent"),
                              ("trafo", "ur_percent", "vscr_percent"),
                              ("trafo3w", "unh_kv", "vn_hv_kv"),
                              ("trafo3w", "unm_kv", "vn_mv_kv"),
                              ("trafo3w", "unl_kv", "vn_lv_kv")]:
        for std_type, parameters in net.std_types[element].items():
            if old in parameters:
                net.std_types[element][std_type][new] = net.std_types[element][std_type].pop(old)
    net.version = 1.0
    if "f_hz" not in net:
        net["f_hz"] = 50.

    if "type" not in net.load.columns:
        net.load["type"] = None
    if "zone" not in net.bus:
        net.bus["zone"] = None
    for element in ["line", "trafo", "bus", "load", "sgen", "ext_grid"]:
        net[element].in_service = net[element].in_service.astype(bool)
    net.switch.closed = net.switch.closed.astype(bool)
    return net


def add_zones_to_elements(net, elements=["line", "trafo", "ext_grid", "switch"]):
    """
    Adds zones to elements, inferring them from the zones of buses they are
    connected to.
    """
    for element in elements:
        if element == "sgen":
            net["sgen"]["zone"] = net["bus"]["zone"].loc[net["sgen"]["bus"]].values
        elif element == "load":
            net["load"]["zone"] = net["bus"]["zone"].loc[net["load"]["bus"]].values
        elif element == "ext_grid":
            net["ext_grid"]["zone"] = net["bus"]["zone"].loc[net["ext_grid"]["bus"]].values
        elif element == "switch":
            net["switch"]["zone"] = net["bus"]["zone"].loc[net["switch"]["bus"]].values
        elif element == "line":
            net["line"]["zone"] = net["bus"]["zone"].loc[net["line"]["from_bus"]].values
            crossing = sum(net["bus"]["zone"].loc[net["line"]["from_bus"]].values !=
                           net["bus"]["zone"].loc[net["line"]["to_bus"]].values)
            if crossing > 0:
                logger.warn("There have been %i lines with different zones at from- and to-bus"
                            % crossing)
        elif element == "trafo":
            net["trafo"]["zone"] = net["bus"]["zone"].loc[net["trafo"]["hv_bus"]].values
            crossing = sum(net["bus"]["zone"].loc[net["trafo"]["hv_bus"]].values !=
                           net["bus"]["zone"].loc[net["trafo"]["lv_bus"]].values)
            if crossing > 0:
                logger.warn("There have been %i trafos with different zones at lv_bus and hv_bus"
                            % crossing)
        elif element == "impedance":
            net["impedance"]["zone"] = net["bus"]["zone"].loc[net["impedance"]["from_bus"]].values
            crossing = sum(net["bus"]["zone"].loc[net["impedance"]["from_bus"]].values !=
                           net["bus"]["zone"].loc[net["impedance"]["to_bus"]].values)
            if crossing > 0:
                logger.warn("There have been %i impedances with different zones at from_bus and "
                            "to_bus" % crossing)
        elif element == "shunt":
            net["shunt"]["zone"] = net["bus"]["zone"].loc[net["shunt"]["bus"]].values
        elif element == "ward":
            net["ward"]["zone"] = net["bus"]["zone"].loc[net["ward"]["bus"]].values
        elif element == "xward":
            net["xward"]["zone"] = net["bus"]["zone"].loc[net["xward"]["bus"]].values
        else:
            raise UserWarning("Unkown element %s" % element)


def create_continuous_bus_index(net):
    """
    Creates a continuous bus index starting at zero and replaces all
    references of old indices by the new ones.
    """
    new_bus_idxs = np.arange(len(net.bus))
    bus_lookup = dict(zip(net["bus"].index.values, new_bus_idxs))
    net.bus.index = new_bus_idxs

    for element, value in [("line", "from_bus"), ("line", "to_bus"), ("trafo", "hv_bus"),
                           ("trafo", "lv_bus"), ("sgen", "bus"), ("load", "bus"),
                           ("switch", "bus"), ("ward", "bus"), ("xward", "bus"),
                           ("impedance", "from_bus"), ("impedance", "to_bus"),
                           ("shunt", "bus"), ("ext_grid", "bus")]:
        net[element][value] = get_indices(net[element][value], bus_lookup)
    return net


def set_scaling_by_type(net, scalings, scale_load=True, scale_sgen=True):
    """
    Sets scaling of loads and/or sgens according to a dictionary
    mapping type to a scaling factor. Note that the type-string is case
    sensitive.
    E.g. scaling = {"pv": 0.8, "bhkw": 0.6}

    :param net:
    :param scaling: A dictionary containing a mapping from element type to
    scaling factor.
    :return:
    """
    if not isinstance(scalings, dict):
        raise UserWarning("The parameter scaling has to be a dictionary, "
                          "see docstring")

    def scaleit(what):
        et = net[what]
        et["scaling"] = [scale[t] or s for t, s in zip(et.type.values, et.scaling.values)]

    scale = defaultdict(lambda: None, scalings)
    if scale_load:
        scaleit("load")
    if scale_sgen:
        scaleit("sgen")


# --- Modify topology

def close_switch_at_line_with_two_open_switches(net):
    """
    Finds lines that have opened switches at both ends and closes one of them.
    Function is usually used when optimizing section points to
    prevent the algorithm from ignoring isolated lines.
    """
    nl = net.switch[(net.switch.et == 'l') & (net.switch.closed == 0)]
    for i, switch in nl.groupby("element"):
        if len(switch.index) > 1:  # find all lines that have open switches at both ends
            # and close on of them
            net.switch.at[switch.index[0], "closed"] = 1


def drop_inactive_elements(net):
    """
    Drops any elements not in service AND any elements connected to inactive
    buses.
    """
    # removes inactive lines and its switches and geodata
    inactive_lines = net["line"][net["line"]["in_service"] == False].index
    drop_lines(net, inactive_lines)

    # removes inactive buses safely by deleting all connected elements and
    # its geodata as well
    disconnected_buses = (set(net["bus"].index)
                          - set(net["line"]["from_bus"])
                          - set(net["line"]["to_bus"])
                          - set(net["impedance"]["from_bus"])
                          - set(net["impedance"]["to_bus"])
                          - set(net["trafo"]["hv_bus"])
                          - set(net["trafo"]["lv_bus"])
                          - set(net["trafo3w"]["hv_bus"])
                          - set(net["trafo3w"]["mv_bus"])
                          - set(net["trafo3w"]["lv_bus"])
                          - set(net["switch"]["bus"])
                          - set(net["switch"][net["switch"]["et"] == "b"]["element"]))
    inactive_buses = (disconnected_buses | set(net["bus"].query("in_service==0").index))
    drop_buses(net, inactive_buses, also_drop_elements=True)

    # finally drops any remaining elements out of service
    for element in net.keys():
        # by iterating over all tables
        try:
            try:
                # and try to drop any elements OOS
                net[element] = net[element].query("in_service==1")
            except pd.computation.ops.UndefinedVariableError:
                # except for tables that have no "in_service"
                continue
        except AttributeError:
            # and except if net[element] is no DataFrame, and has no attribute "query"
            continue


def drop_buses(net, buses, also_drop_elements=True):
    """
    Drops buses and by default safely drops all elements connected to them as well.
    """
    # Note: lines and trafos are deleted seperately since switches may be
    # deleted along with them

    # drop any lines connected to the bus
    droplines = net["line"][(net["line"]["from_bus"].isin(buses)) | (
        net["line"]["to_bus"].isin(buses))].index
    if not also_drop_elements and len(droplines) > 0:
        raise UserWarning("Can not drop buses, "
                          "there are still lines connected")
    elif len(droplines) > 0:
        drop_lines(net, droplines)

    # drop any trafo connected to the bus
    droptrafos = net["trafo"][(net["trafo"]["hv_bus"].isin(buses)) | (
        net["trafo"]["lv_bus"].isin(buses))].index
    if not also_drop_elements and len(droptrafos) > 0:
        raise UserWarning("Can not drop buses, "
                          "there are still trafos connected")
    elif len(droptrafos) > 0:
        drop_trafos(net, droptrafos)

    # drop any other elements connected to the bus
    for element, value in [("sgen", "bus"), ("load", "bus"), ("switch", "bus"),
                           ("ext_grid", "bus"), ("shunt", "bus"),
                           ("impedance", "from_bus"), ("impedance", "to_bus"),
                           ("ward", "bus"), ("xward", "bus")]:
        i = net[element][net[element][value].isin(buses)].index
        if not also_drop_elements and i:
            raise UserWarning("Can not drop buses, "
                              "there are still elements connected")
        net[element].drop(i, inplace=True)

    # drop busbus switches
    i = net["switch"].index[
        (net["switch"]["element"].isin(buses)) & (net["switch"]["et"] == "b")]
    net["switch"].drop(i, inplace=True)

    # finally drop buses and their geodata
    net["bus"].drop(buses, inplace=True)
    net["bus_geodata"].drop(set(buses) & set(
        net["bus_geodata"].index), inplace=True)


def drop_trafos(net, trafos):
    """
    Deletes all trafos and in the given list of indices and removes
    any switches connected to it.
    """
    # drop any switches
    i = net["switch"].index[
        (net["switch"]["element"].isin(trafos)) & (net["switch"]["et"] == "t")]
    net["switch"].drop(i, inplace=True)

    # drop the lines+geodata
    net["trafo"].drop(trafos, inplace=True)


def drop_lines(net, lines):
    """
    Deletes all lines and their geodata in the given list of indices and removes
    any switches connected to it.
    """
    # drop any switches
    i = net["switch"].index[
        (net["switch"]["element"].isin(lines)) & (net["switch"]["et"] == "l")]
    net["switch"].drop(i, inplace=True)

    # drop the lines+geodata
    net["line"].drop(lines, inplace=True)
    net["line_geodata"].drop(set(lines) & set(
        net["line_geodata"].index), inplace=True)


def fuse_buses(net, b1, b2, drop=True):
    """
    Reroutes any connections to buses in b2 to the given bus b1. Additionally drops the buses b2,
    if drop=True (default).
    """
    try:
        b2.__iter__
    except:
        b2 = [b2]

    for element, value in [("line", "from_bus"), ("line", "to_bus"), ("impedance", "from_bus"),
                           ("impedance", "to_bus"), ("trafo", "hv_bus"), ("trafo", "lv_bus"),
                           ("sgen", "bus"), ("load", "bus"),
                           ("switch", "bus"), ("ext_grid", "bus"),
                           ("ward", "bus"), ("xward", "bus"),
                           ("shunt", "bus")]:
        i = net[element][net[element][value].isin(b2)].index
        net[element].loc[i, value] = b1

    i = net["switch"][(net["switch"]["et"] == 'b') & (
        net["switch"]["element"].isin(b2))].index
    net["switch"].loc[i, "element"] = b1
    net["switch"].drop(net["switch"][(net["switch"]["bus"] == net["switch"]["element"]) &
                                     (net["switch"]["et"] == "b")].index, inplace=True)
    if drop:
        net["bus"].drop(b2, inplace=True)
    return net


def set_element_status(net, buses, in_service):
    """
    Sets buses and all elements connected to them out of service.
    """
    net.bus.loc[buses, "in_service"] = in_service

    lines = net.line[(net.line.from_bus.isin(buses)) |
                     (net.line.to_bus.isin(buses))].index
    net.line.loc[lines, "in_service"] = in_service

    trafo = net.trafo[(net.trafo.hv_bus.isin(buses)) |
                      (net.trafo.lv_bus.isin(buses))].index
    net.trafo.loc[trafo, "in_service"] = in_service

    impedances = net.impedance[(net.impedance.from_bus.isin(buses)) |
                               (net.impedance.to_bus.isin(buses))].index
    net.impedance.loc[impedances, "in_service"] = in_service

    loads = net.load[net.load.bus.isin(buses)].index
    net.load.loc[loads, "in_service"] = in_service

    sgens = net.sgen[net.sgen.bus.isin(buses)].index
    net.sgen.loc[sgens, "in_service"] = in_service

    wards = net.ward[net.ward.bus.isin(buses)].index
    net.ward.loc[wards, "in_service"] = in_service

    xwards = net.xward[net.xward.bus.isin(buses)].index
    net.xward.loc[xwards, "in_service"] = in_service

    shunts = net.shunt[net.shunt.bus.isin(buses)].index
    net.shunt.loc[shunts, "in_service"] = in_service


def select_subnet(net, buses, include_switch_buses=False, include_results=False,
                  keep_everything_else=False):
    """
    Selects a subnet by a list of bus indices and returns a net with all elements
    connected to them.
    """
    buses = set(buses)
    if include_switch_buses:
        # we add both buses of a connected line, the one selected is not switch.bus

        # for all line switches
        for _, s in net["switch"].query("et=='l'").iterrows():
            # get from/to-bus of the connected line
            fb = net["line"]["from_bus"].at[s["element"]]
            tb = net["line"]["to_bus"].at[s["element"]]
            # if one bus of the line is selected and its not the switch-bus, add the other bus
            if fb in buses and s["bus"] != fb:
                buses.add(tb)
            if tb in buses and s["bus"] != tb:
                buses.add(fb)

    p2 = create_empty_network()

    p2.bus = net.bus.loc[buses]
    p2.ext_grid = net.ext_grid[net.ext_grid.bus.isin(buses)]
    p2.load = net.load[net.load.bus.isin(buses)]
    p2.sgen = net.sgen[net.sgen.bus.isin(buses)]
    p2.gen = net.gen[net.gen.bus.isin(buses)]
    p2.shunt = net.shunt[net.shunt.bus.isin(buses)]
    p2.ward = net.ward[net.ward.bus.isin(buses)]
    p2.xward = net.xward[net.xward.bus.isin(buses)]

    p2.line = net.line[(net.line.from_bus.isin(buses)) & (net.line.to_bus.isin(buses))]
    p2.trafo = net.trafo[(net.trafo.hv_bus.isin(buses)) & (net.trafo.lv_bus.isin(buses))]
    p2.impedance = net.impedance[(net.impedance.from_bus.isin(buses)) &
                                 (net.impedance.to_bus.isin(buses))]

    if include_results:
        for table in net.keys():
            if net[table] is None:
                continue
            elif table == "res_bus":
                p2[table] = net[table].loc[buses]
            elif table.startswith("res_"):
                p2[table] = net[table].loc[p2[table.split("res_")[1]].index]
    if "bus_geodata" in net:
        p2["bus_geodata"] = net["bus_geodata"].loc[net["bus_geodata"].index.isin(buses)]
    if "line_geodata" in net:
        lines = p2.line.index
        p2["line_geodata"] = net["line_geodata"].loc[net["line_geodata"].index.isin(lines)]

    # switches
    si = [i for i, s in net["switch"].iterrows()
          if s["bus"] in buses and
          ((s["et"] == "b" and s["element"] in p2["bus"].index) or
           (s["et"] == "l" and s["element"] in p2["line"].index) or
           (s["et"] == "t" and s["element"] in p2["trafo"].index))]
    p2["switch"] = net["switch"].loc[si]
    # return a PandapowerNet
    if keep_everything_else:
        newnet = copy.deepcopy(net)
        newnet.update(p2)
        return PandapowerNet(newnet)
    return PandapowerNet(p2)


# --- item/element selections

def get_element_index(net, element, name, exact_match=True, regex=False):
    """
    Returns the element identified by the element string and a name.

    INPUT:
      **net** - pandapower network

      **element** - line indices of lines that are considered. If None, all lines in the network are \
                  considered.

      **name** - if True, switches are also considered as connected elements.

    OUTPUT:

      **connections** - pandapower Series with number of connected elements for each bus. If a \
                      selection of lines is given, only buses connected to these lines are in \
                      the returned series.
    """
    if exact_match:
        idx = net[element][net[element]["name"] == name].index
        if len(idx) == 0:
            raise UserWarning("There is no %s with name %s" % (element, name))
        if len(idx) > 1:
            raise UserWarning("Duplicate %s names for %s" % (element, name))
    else:
        return net[element][net[element]["name"].str.contains(name, regex=regex)].index
    return idx[0]


def next_bus(net, bus, element_id, et='line', **kwargs):
    """
    Returns the index of the second bus an element is connected to, given a
    first one. E.g. the from_bus given the to_bus of a line.
    """
    # for legacy compliance
    if "element_type" in kwargs:
        et = kwargs["element_type"]

    if et == 'line':
        bc = ["from_bus", "to_bus"]
    elif et == 'trafo':
        bc = ["hv_bus", "lv_bus"]
    elif et == "switch":
        bc = ["bus", "element"]
    else:
        raise Exception("unknown element type")
    nb = list(net[et].loc[element_id, bc].values)
    nb.remove(bus)
    return nb[0]


def get_connected_elements(net, element, buses, respect_switches=True, respect_in_service=False):
    """
     Returns elements connected to a given bus.

     INPUT:

        **net** (PandapowerNet)

        **element** (string, name of the element table)

        **buses** (single integer or iterable of ints)

     OPTIONAL:

        **respect_switches** (boolean, True)    - True: open switches will be respected
                                                  False: open switches will be ignored
        **respect_in_service** (boolean, False) - True: in_service status of connected lines will be
                                                        respected
                                                  False: in_service status will be ignored
     RETURN:

        **cl** (set) - Returns connected lines.

    """

    if not hasattr(buses, "__iter__"):
        buses = [buses]

    if element in ["line", "l"]:
        element = "l"
        element_table = net.line
        connected_elements = set(net.line.index[net.line.from_bus.isin(buses)
                                                | net.line.to_bus.isin(buses)])

    elif element in ["trafo"]:
        element = "t"
        element_table = net.trafo
        connected_elements = set(net["trafo"].index[(net.trafo.hv_bus.isin(buses))
                                                    | (net.trafo.lv_bus.isin(buses))])
    elif element in ["trafo3w", "t3w"]:
        element = "t3w"
        element_table = net.trafo3w
        connected_elements = set(net["trafo3w"].index[(net.trafo3w.hv_bus.isin(buses))
                                                      | (net.trafo3w.mv_bus.isin(buses))
                                                      | (net.trafo3w.lv_bus.isin(buses))])
    elif element == "impedance":
        element_table = net.impedance
        connected_elements = set(net["impedance"].index[(net.impedance.from_bus.isin(buses))
                                                        | (net.impedance.to_bus.isin(buses))])
    elif element == "load":
        element_table = net.load
        connected_elements = set(element_table.index[(element_table.bus.isin(buses))])
    elif element == "sgen":
        element_table = net.sgen
        connected_elements = set(element_table.index[(element_table.bus.isin(buses))])
    elif element == "ward":
        element_table = net.ward
        connected_elements = set(element_table.index[(element_table.bus.isin(buses))])
    elif element == "shunt":
        element_table = net.shunt
        connected_elements = set(element_table.index[(element_table.bus.isin(buses))])
    elif element == "xward":
        element_table = net.xward
        connected_elements = set(element_table.index[(element_table.bus.isin(buses))])
    elif element == "ext_grid":
        element_table = net.ext_grid
        connected_elements = set(element_table.index[(element_table.bus.isin(buses))])
    elif element == "gen":
        element_table = net.gen
        connected_elements = set(element_table.index[(element_table.bus.isin(buses))])
    else:
        raise UserWarning("Unknown element! ", element)

    if respect_switches and element in ["l", "t", "t3w"]:
        open_switches = get_connected_switches(net, buses, consider=element, status="open")
        if open_switches:
            open_and_connected = net.switch.loc[net.switch.index.isin(open_switches)
                                                & net.switch.element.isin(connected_elements)].index
            connected_elements -= set(net.switch.element[open_and_connected])

    if respect_in_service:
        connected_elements -= set(element_table[element_table.in_service == False].index)

    return connected_elements


def get_connected_buses(net, buses, consider=("l", "s", "t"), respect_switches=True, respect_in_service=False):
    """
     Returns buses connected to given buses. The source buses will NOT be returned.

     INPUT:

        **net** (PandapowerNet)

        **buses** (single integer or iterable of ints)

     OPTIONAL:

        **respect_switches** (boolean, True)        - True: open switches will be respected
                                                      False: open switches will be ignored
        **respect_in_service** (boolean, False)     - True: in_service status of connected buses
                                                            will be respected
                                                            False: in_service status will be
                                                            ignored
        **consider** (iterable, ("l", "s", "t"))    - Determines, which types of connections will
                                                      be will be considered.
                                                      l: lines
                                                      s: switches
                                                      t: trafos
     RETURN:

        **cl** (set) - Returns connected buses.

    """
    if not hasattr(buses, "__iter__"):
        buses = [buses]

    cb = set()
    if "l" in consider:
        cl = get_connected_elements(net, "line", buses, respect_switches, respect_in_service)
        cb |= set(net.line[net.line.index.isin(cl)].from_bus)
        cb |= set(net.line[net.line.index.isin(cl)].to_bus)

    if "s" in consider:
        cs = get_connected_switches(net, buses, consider='b',
                                    status="closed" if respect_switches else "all")
        cb |= set(net.switch[net.switch.index.isin(cs)].element)
        cb |= set(net.switch[net.switch.index.isin(cs)].bus)

    if "t" in consider:
        ct = get_connected_elements(net, "trafo", buses, respect_switches, respect_in_service)
        cb |= set(net.trafo[net.trafo.index.isin(ct)].lv_bus)
        cb |= set(net.trafo[net.trafo.index.isin(cl)].hv_bus)

    if respect_in_service:
        cb -= set(net.bus[~net.bus.in_service].index)

    return cb - set(buses)


def get_connected_buses_at_element(net, element, et, respect_in_service=False):
    """
     Returns buses connected to a given line, switch or trafo. In case of a bus switch, two buses
     will be returned, else one.

     INPUT:

        **net** (PandapowerNet)

        **element** (integer)

        **et** (string)                             - Type of the source element:
                                                      l: line
                                                      s: switch
                                                      t: trafo

     OPTIONAL:

        **respect_in_service** (boolean, False)     - True: in_service status of connected buses
                                                            will be respected
                                                      False: in_service status will be ignored
     RETURN:

        **cl** (set) - Returns connected switches.

    """

    cb = set()
    if et == 'l':
        cb.add(net.line.from_bus.at[element])
        cb.add(net.line.to_bus.at[element])

    elif et == 's':
        cb.add(net.switch.bus.at[element])
        if net.switch.et.at[element] == 'b':
            cb.add(net.switch.element.at[element])
    elif et == 't':
        cb.add(net.trafo.hv_bus.at[element])
        cb.add(net.trafo.lv_bus.at[element])

    if respect_in_service:
        cb -= set(net.bus[~net.bus.in_service].index)

    return cb


def get_connected_switches(net, buses, consider=('b', 'l', 't'), status="all"):
    """
    Returns switches connected to given buses.

    INPUT:

        **net** (PandapowerNet)

        **buses** (single integer or iterable of ints)

    OPTIONAL:

        **respect_switches** (boolean, True)        - True: open switches will be respected
                                                     False: open switches will be ignored

        **respect_in_service** (boolean, False)     - True: in_service status of connected
                                                            buses will be respected

                                                      False: in_service status will be ignored
        **consider** (iterable, ("l", "s", "t"))    - Determines, which types of connections
                                                      will be will be considered.
                                                      l: lines
                                                      s: switches
                                                      t: trafos

        **status** (string, ("all", "closed", "open"))    - Determines, which switches will
                                                            be considered
    RETURN:

       **cl** (set) - Returns connected buses.

    """

    if not hasattr(buses, "__iter__"):
        buses = [buses]

    if status == "closed":
        switch_selection = net.switch.closed == True
    elif status == "open":
        switch_selection = net.switch.closed == False
    elif status == "all":
        switch_selection = np.full(len(net.switch), True, dtype=bool)
    else:
        logger.warn("Unknown switch status \"%s\" selected! "
                    "Selecting all switches by default." % status)

    cs = set()
    if 'b' in consider:
        cs |= set(net['switch'].index[(net['switch']['bus'].isin(buses)
                                       | net['switch']['element'].isin(buses))
                                      & (net['switch']['et'] == 'b')
                                      & switch_selection])
    if 'l' in consider:
        cs |= set(net['switch'].index[(net['switch']['bus'].isin(buses))
                                      & (net['switch']['et'] == 'l')
                                      & switch_selection])

    if 't' in consider:
        cs |= set(net['switch'].index[net['switch']['bus'].isin(buses)
                                      & (net['switch']['et'] == 't')
                                      & switch_selection])

    return cs
