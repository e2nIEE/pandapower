# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy
from collections import defaultdict

import numpy as np
import pandas as pd

from pandapower.auxiliary import get_indices, pandapowerNet, _preserve_dtypes
from pandapower.create import create_empty_network, create_piecewise_linear_cost, create_switch
from pandapower.topology import unsupplied_buses
from pandapower.run import runpp
from pandapower import __version__
from pandapower.opf.validate_opf_input import _check_necessary_opf_parameters

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


# --- Information
def lf_info(net, numv=1, numi=2):  # pragma: no cover
    """
    Prints some basic information of the results in a net
    (max/min voltage, max trafo load, max line load).

    OPTIONAL:

        **numv** (integer, 1) - maximal number of printed maximal respectively minimal voltages

        **numi** (integer, 2) - maximal number of printed maximal loading at trafos or lines
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


def _check_plc_full_range(net, element_type):  # pragma: no cover
    """ This is an auxiliary function for check_opf_data to check full range of piecewise linear
    cost function """
    plc = net.piecewise_linear_cost
    plc_el_p = plc.loc[(plc.element_type == element_type) & (plc.type == 'p')]
    plc_el_q = plc.loc[(plc.element_type == element_type) & (plc.type == 'q')]
    p_idx = []
    q_idx = []
    if element_type != 'dcline':
        if plc_el_p.shape[0]:
            p_idx = net[element_type].loc[
                (net[element_type].index.isin(plc_el_p.element_type)) &
                ((net[element_type].min_p_kw < plc_el_p.p[plc_el_p.index.values[0]].min()) |
                 (net[element_type].max_p_kw > plc_el_p.p[plc_el_p.index.values[0]].max()))].index
        if plc_el_q.shape[0]:
            q_idx = net[element_type].loc[
                (net[element_type].index.isin(plc_el_q.element_type)) &
                ((net[element_type].min_p_kw < plc_el_q.p[plc_el_q.index.values[0]].min()) |
                 (net[element_type].max_p_kw > plc_el_q.p[plc_el_q.index.values[0]].max()))].index
    else:  # element_type == 'dcline'
        if plc_el_p.shape[0]:
            p_idx = net[element_type].loc[
                (net[element_type].index.isin(plc_el_p.element_type)) &
                ((net[element_type].max_p_kw > plc_el_p.p[plc_el_p.index.values[0]].max()))].index
        if plc_el_q.shape[0]:
            q_idx = net[element_type].loc[
                (net[element_type].index.isin(plc_el_q.element_type)) &
                ((net[element_type].min_q_to_kvar < plc_el_q.p[plc_el_q.index.values[0]].min()) |
                 (net[element_type].min_q_from_kvar < plc_el_q.p[plc_el_q.index.values[0]].min()) |
                 (net[element_type].max_q_to_kvar > plc_el_q.p[plc_el_q.index.values[0]].max()) |
                 (net[element_type].max_q_from_kvar > plc_el_q.p[plc_el_q.index.values[0]].max()))
                ].index
    if len(p_idx):
        logger.warn("At" + element_type + str(p_idx.values) +
                    "the piecewise linear costs do not cover full active power range. " +
                    "In OPF the costs will be extrapolated.")
    if len(q_idx):
        logger.warn("At" + element_type + str(q_idx.values) +
                    "the piecewise linear costs do not cover full reactive power range." +
                    "In OPF the costs will be extrapolated.")


def check_opf_data(net):  # pragma: no cover
    """
    This function checks net data ability for opf calculations via runopp.

    INPUT:
        **net** (pandapowerNet) - The pandapower network in which is checked for runopp
    """
    _check_necessary_opf_parameters(net, logger)

    # --- Determine duplicated cost data
    all_costs = net.piecewise_linear_cost[['type', 'element', 'element_type']].append(
        net.polynomial_cost[['type', 'element', 'element_type']]).reset_index(drop=True)
    duplicates = all_costs.loc[all_costs.duplicated()]
    if duplicates.shape[0]:
        raise ValueError("There are elements with multiply costs.\nelement_types: %s\n"
                         "element: %s\ntypes: %s" % (duplicates.element_type.values,
                                                     duplicates.element.values,
                                                     duplicates.type.values))

    # --- check full range of piecewise linear cost functions
    _check_plc_full_range(net, 'ext_grid')
    _check_plc_full_range(net, 'dcline')
    for element_type in ['gen', 'sgen', 'load']:
        if hasattr(net[element_type], "controllable"):
            if (net[element_type].controllable.any()):
                _check_plc_full_range(net, element_type)


def _opf_controllables(elm_df, to_log, control_elm, control_elm_name, all_costs):  # pragma: no cover
    """ This is an auxiliary function for opf_task to add controllables data to to_log """
    if len(elm_df):
        to_log += '\n' + "  " + control_elm_name
        elm_p_cost_idx = set(all_costs.loc[(all_costs.element_type == control_elm) &
                                           (all_costs.type == 'p')].element)
        elm_q_cost_idx = set(all_costs.loc[(all_costs.element_type == control_elm) &
                                           (all_costs.type == 'q')].element)
        with_pq_cost = elm_df.loc[elm_p_cost_idx & elm_q_cost_idx].index
        with_p_cost = elm_df.loc[elm_p_cost_idx - elm_q_cost_idx].index
        with_q_cost = elm_df.loc[elm_q_cost_idx - elm_p_cost_idx].index
        without_cost = elm_df.loc[set(elm_df.index) - (elm_p_cost_idx | elm_q_cost_idx)].index
        if len(with_pq_cost) and len(with_pq_cost) < len(elm_df):
            to_log += '\n' + '    ' + control_elm_name + ' ' +  \
                ', '.join(map(str, elm_df.loc[with_pq_cost].index)) + " with p and q costs"
        elif len(with_pq_cost):
            to_log += '\n' + '    all %i ' % len(elm_df) + control_elm_name + " with p and q costs"
        if len(with_p_cost) and len(with_p_cost) < len(elm_df):
            to_log += '\n' + '    ' + control_elm_name + ' ' + \
                ', '.join(map(str, elm_df.loc[with_p_cost].index)) + " with p costs"
        elif len(with_p_cost):
            to_log += '\n' + '    all %i ' % len(elm_df) + control_elm_name + " with p costs"
        if len(with_q_cost) and len(with_q_cost) < len(elm_df):
            to_log += '\n' + '    ' + control_elm_name + ' ' + \
                ', '.join(map(str, elm_df.loc[with_q_cost].index)) + " with q costs"
        elif len(with_q_cost):
            to_log += '\n' + '    all %i ' % len(elm_df) + control_elm_name + " with q costs"
        if len(without_cost) and len(without_cost) < len(elm_df):
            to_log += '\n' + '    ' + control_elm_name + ' ' + \
                ', '.join(map(str, elm_df.loc[without_cost].index)) + " without costs"
        elif len(without_cost):
            to_log += '\n' + '    all %i ' % len(elm_df) + control_elm_name + " without costs"
    return to_log


def opf_task(net):  # pragma: no cover
    """
    Prints some basic inforamtion of the optimal powerflow task.
    """
    check_opf_data(net)

    plc = net.piecewise_linear_cost
    pol = net.polynomial_cost

    # --- store cost data to all_costs
    all_costs = net.piecewise_linear_cost[['type', 'element', 'element_type']].append(
        net.polynomial_cost[['type', 'element', 'element_type']]).reset_index(drop=True)
    all_costs['str'] = None
    for i, j in all_costs.iterrows():
        costs = plc.loc[(plc.element == j.element) & (plc.element_type == j.element_type) &
                        (plc.type == j.type)]
        if len(costs):
            all_costs.str.at[i] = "p: " + str(costs.p.values[0]) + ", f: " + str(costs.f.values[0])
        else:
            costs = pol.loc[(pol.element == j.element) & (pol.element_type == j.element_type) &
                            (pol.type == j.type)]
            all_costs.str.at[i] = "c: " + str(costs.c.values[0])

    # --- examine logger info

    # --- controllables & costs
    to_log = '\n' + "Cotrollables & Costs:"
    # dcline always is assumed as controllable
    to_log = _opf_controllables(net.ext_grid, to_log, 'ext_grid', 'Ext_Grid', all_costs)
    # check controllables in gen, sgen and load
    control_elms = ['gen', 'sgen', 'load']
    control_elm_names = ['Gen', 'SGen', 'Load']
    for j, control_elm in enumerate(control_elms):
        # only for net[control_elm] with len > 0, check_data has checked 'controllable' in columns
        if len(net[control_elm]):
            to_log = _opf_controllables(net[control_elm].loc[net[control_elm].controllable],
                                        to_log, control_elm, control_elm_names[j], all_costs)
    if len(net.dcline):  # dcline always is assumed as controllable
        to_log = _opf_controllables(net.dcline, to_log, 'dcline', 'DC Line', all_costs)
    to_log += '\n' + "Constraints:"
    constr_exist = False  # stores if there are any constraints

    # --- variables constraints
    variables = ['ext_grid', 'gen', 'sgen', 'load']
    variable_names = ['Ext_Grid', 'Gen', 'SGen', 'Load']
    variable_long_names = ['External Grid', 'Generator', 'Static Generator', 'Load']
    for j, variable in enumerate(variables):
        constr_col = pd.Series(['min_p_kw', 'max_p_kw', 'min_q_kvar', 'max_q_kvar'])
        constr_col_exist = constr_col[constr_col.isin(net[variable].columns)]
        constr = net[variable][constr_col_exist]
        if (constr.shape[1] > 0) & (constr.shape[0] > 0):
            constr_exist = True
            if variable != 'ext_grid':
                constr = constr.loc[net[variable].loc[net[variable].controllable].index]
            to_log += '\n' + "  " + variable_long_names[j] + " Constraints"
            for i in constr_col[~constr_col.isin(net[variable].columns)]:
                constr[i] = np.nan
            if (constr.min_p_kw >= constr.max_p_kw).any():
                logger.warn("The value of min_p_kw must be less than max_p_kw for all " +
                            variable_names[j] + ". " + "Please observe the pandapower " +
                            "signing system.")
            if (constr.min_q_kvar >= constr.max_q_kvar).any():
                logger.warn("The value of min_q_kvar must be less than max_q_kvar for all " +
                            variable_names[j] + ". Please observe the pandapower signing system.")
            if constr.duplicated()[1:].all():  # all with the same constraints
                to_log += '\n' + "    at all " + variable_names[j] + \
                          " [min_p_kw, max_p_kw, min_q_kvar, max_q_kvar] is " + \
                          "[%s, %s, %s, %s]" % (
                              constr.min_p_kw.values[0], constr.max_p_kw.values[0],
                              constr.min_q_kvar.values[0], constr.max_q_kvar.values[0])
            else:  # different constraints exist
                unique_rows = ~constr.duplicated()
                duplicated_rows = constr.duplicated()
                for i in constr[unique_rows].index:
                    same_data = list([i])
                    for i2 in constr[duplicated_rows].index:
                        if (constr.iloc[i] == constr.iloc[i2]).all():
                            same_data.append(i2)
                    to_log += '\n' + '    at ' + variable_names[j] + ' ' + \
                              ', '.join(map(str, same_data)) + \
                              ' [min_p_kw, max_p_kw, min_q_kvar, max_q_kvar] is ' + \
                              '[%s, %s, %s, %s]' % (constr.min_p_kw[i], constr.max_p_kw[i],
                                                    constr.min_q_kvar[i], constr.max_q_kvar[i])
    # --- DC Line constraints
    constr_col = pd.Series(['max_p_kw', 'min_q_from_kvar', 'max_q_from_kvar', 'min_q_to_kvar',
                            'max_q_to_kvar'])
    constr_col_exist = constr_col[constr_col.isin(net['dcline'].columns)]
    constr = net['dcline'][constr_col_exist].dropna(how='all')
    if (constr.shape[1] > 0) & (constr.shape[0] > 0):
        constr_exist = True
        to_log += '\n' + "  DC Line Constraints"
        for i in constr_col[~constr_col.isin(net['dcline'].columns)]:
            constr[i] = np.nan
        if (constr.min_q_from_kvar >= constr.max_q_from_kvar).any():
            logger.warning("The value of min_q_from_kvar must be less than max_q_from_kvar for " +
                           "all DC Line. Please observe the pandapower signing system.")
        if (constr.min_q_to_kvar >= constr.max_q_to_kvar).any():
            logger.warning("The value of min_q_to_kvar must be less than min_q_to_kvar for " +
                           "all DC Line. Please observe the pandapower signing system.")
        if constr.duplicated()[1:].all():  # all with the same constraints
            to_log += '\n' + "    at all DC Line [max_p_kw, min_q_from_kvar, max_q_from_kvar, " + \
                      "min_q_to_kvar, max_q_to_kvar] is [%s, %s, %s, %s, %s]" % \
                      (constr.max_p_kw.values[0], constr.min_q_from_kvar.values[0],
                       constr.max_q_from_kvar.values[0], constr.min_q_to_kvar.values[0],
                       constr.max_q_to_kvar.values[0])
        else:  # different constraints exist
            unique_rows = ~constr.duplicated()
            duplicated_rows = constr.duplicated()
            for i in constr[unique_rows].index:
                same_data = list([i])
                for i2 in constr[duplicated_rows].index:
                    if (constr.iloc[i] == constr.iloc[i2]).all():
                        same_data.append(i2)
                to_log += '\n' + '    at DC Line ' + ', '.join(map(str, same_data)) + \
                          ' [max_p_kw, min_q_from_kvar, max_q_from_kvar, min_q_to_kvar, ' + \
                          'max_q_to_kvar] is [%s, %s, %s, %s, %s]' % (
                              constr.max_p_kw.values[0], constr.min_q_from_kvar.values[0],
                              constr.max_q_from_kvar.values[0],
                              constr.min_q_to_kvar.values[0], constr.max_q_to_kvar.values[0])
    # --- Voltage constraints
    if pd.Series(['min_vm_pu', 'max_vm_pu']).isin(net.bus.columns).any():
        c_bus = net.bus[['min_vm_pu', 'max_vm_pu']].dropna(how='all')
        if c_bus.shape[0] > 0:
            constr_exist = True
            to_log += '\n' + "  Voltage Constraints"
            if (net.bus.min_vm_pu >= net.bus.max_vm_pu).any():
                logger.warn("The value of min_vm_pu must be less than max_vm_pu.")
            if c_bus.duplicated()[1:].all():  # all with the same constraints
                to_log += '\n' + '    at all Nodes [min_vm_pu, max_vm_pu] is [%s, %s]' % \
                                 (c_bus.min_vm_pu[0], c_bus.max_vm_pu[0])
            else:  # different constraints exist
                unique_rows = ~c_bus.duplicated()
                duplicated_rows = c_bus.duplicated()
                for i in c_bus[unique_rows].index:
                    same_data_nodes = list([i])
                    for i2 in c_bus[duplicated_rows].index:
                        if (c_bus.iloc[i] == c_bus.iloc[i2]).all():
                            same_data_nodes.append(i2)
                    to_log += '\n' + '    at Nodes ' + ', '.join(map(str, same_data_nodes)) + \
                              ' [min_vm_pu, max_vm_pu] is [%s, %s]' % (c_bus.min_vm_pu[i],
                                                                       c_bus.max_vm_pu[i])
    # --- Branch constraints
    branches = ['trafo', 'line']
    branch_names = ['Trafo', 'Line']
    for j, branch in enumerate(branches):
        if "max_loading_percent" in net[branch].columns:
            constr = net[branch]['max_loading_percent'].dropna()
            if constr.shape[0] > 0:
                constr_exist = True
                to_log += '\n' + "  " + branch_names[j] + " Constraint"
                if constr.duplicated()[1:].all():  # all with the same constraints
                    to_log += '\n' + '    at all ' + branch_names[j] + \
                              ' max_loading_percent is %s' % (constr[0])
                else:  # different constraints exist
                    unique_rows = ~c_bus.duplicated()
                    duplicated_rows = c_bus.duplicated()
                    for i in constr[unique_rows].index:
                        same_data = list([i])
                        for i2 in constr[duplicated_rows].index:
                            if (constr.iloc[i] == constr.iloc[i2]).all():
                                same_data.append(i2)
                        to_log += '\n' + "    at " + branch_names[j] + " " + \
                                  ', '.join(map(str, same_data)) + \
                                  " max_loading_percent is %s" % (constr[j])
    if not constr_exist:
        to_log += '\n' + "  There are no constraints."
    # --- do logger info
    logger.info(to_log)


def switch_info(net, sidx):  # pragma: no cover
    """
    Prints what buses and elements are connected by a certain switch.
    """
    switch_type = net.switch.at[sidx, "et"]
    bidx = net.switch.at[sidx, "bus"]
    bus_name = net.bus.at[bidx, "name"]
    eidx = net.switch.at[sidx, "element"]
    if switch_type == "b":
        bus2_name = net.bus.at[eidx, "name"]
        logger.info("Switch %u connects bus %u (%s) with bus %u (%s)" % (sidx, bidx, bus_name,
                                                                         eidx, bus2_name))
    elif switch_type == "l":
        line_name = net.line.at[eidx, "name"]
        logger.info("Switch %u connects bus %u (%s) with line %u (%s)" % (sidx, bidx, bus_name,
                                                                          eidx, line_name))
    elif switch_type == "t":
        trafo_name = net.trafo.at[eidx, "name"]
        logger.info("Switch %u connects bus %u (%s) with trafo %u (%s)" % (sidx, bidx, bus_name,
                                                                           eidx, trafo_name))


def overloaded_lines(net, max_load=100):
    """
    Returns the results for all lines with loading_percent > max_load or None, if
    there are none.
    """
    if net.converged:
        return net["res_line"].index[net["res_line"]["loading_percent"] > max_load]
    else:
        raise UserWarning("The last loadflow terminated erratically, results are invalid!")


def violated_buses(net, min_vm_pu, max_vm_pu):
    """
    Returns all bus indices where vm_pu is not within min_vm_pu and max_vm_pu or returns None, if
    there are none of those buses.
    """
    if net.converged:
        return net["bus"].index[(net["res_bus"]["vm_pu"] < min_vm_pu) |
                                (net["res_bus"]["vm_pu"] > max_vm_pu)]
    else:
        raise UserWarning("The last loadflow terminated erratically, results are invalid!")


def nets_equal(x, y, check_only_results=False, tol=1.e-14):
    """
    Compares the DataFrames of two networks. The networks are considered equal
    if they share the same keys and values, except of the
    'et' (elapsed time) entry which differs depending on
    runtime conditions and entries stating with '_'.
    """
    eq = True
    not_equal = []

    if isinstance(x, pandapowerNet) and isinstance(y, pandapowerNet):
        # for two networks make sure both have the same keys that do not start with "_"...
        x_keys = [key for key in x.keys() if not key.startswith("_")]
        y_keys = [key for key in y.keys() if not key.startswith("_")]
        key_union = set(x_keys) | set(y_keys)
        key_difference = set(x_keys) ^ set(y_keys)

        if len(key_difference) > 0:
            logger.info("Networks entries mismatch at: %s" % key_difference)
            if not check_only_results:
                return False

        # ... and then iter through the keys, checking for equality for each table
        for df_name in list(key_union):
            # skip 'et' (elapsed time) and entries starting with '_' (internal vars)
            if (df_name != 'et' and not df_name.startswith("_")):
                if check_only_results and not df_name.startswith("res_"):
                    continue  # skip anything that is not a result table

                if isinstance(x[df_name], pd.DataFrame) and isinstance(y[df_name], pd.DataFrame):
                    frames_equal = dataframes_equal(x[df_name], y[df_name], tol)
                    eq &= frames_equal

                    if not frames_equal:
                        not_equal.append(df_name)

    if len(not_equal) > 0:
        logger.info("Networks do not match in DataFrame(s): %s" % (', '.join(not_equal)))

    return eq


def dataframes_equal(x_df, y_df, tol=1.e-14):
    # eval if two DataFrames are equal, with regard to a tolerance
    if x_df.shape == y_df.shape:
        # we use numpy.allclose to grant a tolerance on numerical values
        numerical_equal = np.allclose(x_df.select_dtypes(include=[np.number]),
                                      y_df.select_dtypes(include=[np.number]),
                                      atol=tol, equal_nan=True)

        # ... use pandas .equals for the rest, which also evaluates NaNs to be equal
        rest_equal = x_df.select_dtypes(exclude=[np.number]).equals(
            y_df.select_dtypes(exclude=[np.number]))

        return numerical_equal & rest_equal
    else:
        return False


# --- Simulation setup and preparations
def convert_format(net):
    """
    Converts old nets to new format to ensure consistency. The converted net is returned.
    """
    _pre_release_changes(net)
    if net.name is None:
        net.name = ""
    if "sn_kva" not in net:
        net.sn_kva = 1e3
    if "OPF_converged" not in net:
        net["OPF_converged"] = False
    net.line.rename(columns={'imax_ka': 'max_i_ka'}, inplace=True)
    for typ, data in net.std_types["line"].items():
        if "imax_ka" in data:
            net.std_types["line"][typ]["max_i_ka"] = net.std_types["line"][typ].pop("imax_ka")
    # unsymmetric impedance
    if "r_pu" in net.impedance:
        net.impedance["rft_pu"] = net.impedance["rtf_pu"] = net.impedance["r_pu"]
        net.impedance["xft_pu"] = net.impedance["xtf_pu"] = net.impedance["x_pu"]
    # initialize measurement dataframe
    if "measurement" in net and "element_type" not in net.measurement:
        if net.measurement.empty:
            del net["measurement"]
        else:
            logger.warn("The measurement structure seems outdated. Please adjust it "
                        "according to the documentation.")
    if "measurement" in net and "name" not in net.measurement:
        net.measurement.insert(0, "name", None)
    if "measurement" not in net:
        net["measurement"] = pd.DataFrame(np.zeros(0, dtype=[("name", np.dtype(object)),
                                                             ("type", np.dtype(object)),
                                                             ("element_type", np.dtype(object)),
                                                             ("value", "f8"),
                                                             ("std_dev", "f8"),
                                                             ("bus", "u4"),
                                                             ("element", np.dtype(object))]))
    if "dcline" not in net:
        net["dcline"] = pd.DataFrame(np.zeros(0, dtype=[("name", np.dtype(object)),
                                                        ("from_bus", "u4"),
                                                        ("to_bus", "u4"),
                                                        ("p_kw", "f8"),
                                                        ("loss_percent", 'f8'),
                                                        ("loss_kw", 'f8'),
                                                        ("vm_from_pu", "f8"),
                                                        ("vm_to_pu", "f8"),
                                                        ("max_p_kw", "f8"),
                                                        ("min_q_from_kvar", "f8"),
                                                        ("min_q_to_kvar", "f8"),
                                                        ("max_q_from_kvar", "f8"),
                                                        ("max_q_to_kvar", "f8"),
                                                        ("cost_per_kw", 'f8'),
                                                        ("in_service", 'bool')]))
    if "_empty_res_dcline" not in net:
        net["_empty_res_dcline"] = pd.DataFrame(np.zeros(0, dtype=[("p_from_kw", "f8"),
                                                                   ("q_from_kvar", "f8"),
                                                                   ("p_to_kw", "f8"),
                                                                   ("q_to_kvar", "f8"),
                                                                   ("pl_kw", "f8"),
                                                                   ("vm_from_pu", "f8"),
                                                                   ("va_from_degree", "f8"),
                                                                   ("vm_to_pu", "f8"),
                                                                   ("va_to_degree", "f8")]))
    if "_empty_res_storage" not in net:
        net["_empty_res_storage"] = pd.DataFrame(np.zeros(0, dtype=[("p_kw", "f8"),
                                                                   ("q_kvar", "f8"),
                                                                   ("soc_percent", "f8")]))

    if len(net["_empty_res_line"]) < 10:
        net["_empty_res_line"] = pd.DataFrame(np.zeros(0, dtype=[("p_from_kw", "f8"),
                                                                 ("q_from_kvar", "f8"),
                                                                 ("p_to_kw", "f8"),
                                                                 ("q_to_kvar", "f8"),
                                                                 ("pl_kw", "f8"),
                                                                 ("ql_kvar", "f8"),
                                                                 ("i_from_ka", "f8"),
                                                                 ("i_to_ka", "f8"),
                                                                 ("i_ka", "f8"),
                                                                 ("loading_percent", "f8")]))
    if "storage" not in net:
        net["storage"] = pd.DataFrame(np.zeros(0, dtype=[("name", np.dtype(object)),
                                                         ("bus", "i8"),
                                                         ("p_kw", "f8"),
                                                         ("q_kvar", "f8"),
                                                         ("sn_kva", "f8"),
                                                         ("soc_percent", "f8"),
                                                         ("min_e_kwh", "f8"),
                                                         ("max_e_kwh", "f8"),
                                                         ("scaling", "f8"),
                                                         ("in_service", 'bool'),
                                                         ("type", np.dtype(object))]))
    if "version" not in net or net.version < 1.1:
        if "min_p_kw" in net.gen and "max_p_kw" in net.gen:
            if np.any(net.gen.min_p_kw > net.gen.max_p_kw):
                pmin = copy.copy(net.gen.min_p_kw.values)
                pmax = copy.copy(net.gen.max_p_kw.values)
                net.gen["min_p_kw"] = pmax
                net.gen["max_p_kw"] = pmin
    if "piecewise_linear_cost" not in net:
        net["piecewise_linear_cost"] = pd.DataFrame(np.zeros(0, dtype=[("type", np.dtype(object)),
                                                                       ("element",
                                                                        np.dtype(object)),
                                                                       ("element_type",
                                                                        np.dtype(object)),
                                                                       ("p", np.dtype(object)),
                                                                       ("f", np.dtype(object))]))

    if "polynomial_cost" not in net:
        net["polynomial_cost"] = pd.DataFrame(np.zeros(0, dtype=[("type", np.dtype(object)),
                                                                 ("element", np.dtype(object)),
                                                                 ("element_type", np.dtype(object)),
                                                                 ("c", np.dtype(object))]))

    if "cost_per_kw" in net.gen:
        if not "piecewise_linear_cost" in net:
            for index, cost in net.gen.cost_per_kw.iteritems():
                if not np.isnan(cost):
                    p = net.gen.min_p_kw.at[index]
                    create_piecewise_linear_cost(net, index, "gen", np.array([[p, cost * p], [0, 0]]))

    if "cost_per_kw" in net.sgen:
        if "min_p_kw" not in net.sgen:
            net.sgen["min_p_kw"] = net.sgen.p_kw
        if "max_p_kw" not in net.sgen:
            net.sgen["max_p_kw"] = 0

        if not "piecewise_linear_cost" in net:
            for index, cost in net.sgen.cost_per_kw.iteritems():
                if not np.isnan(cost):
                    p = net.sgen.min_p_kw.at[index]
                    create_piecewise_linear_cost(net, index, "sgen", np.array([[p, cost * p], [0, 0]]))

    if "cost_per_kw" in net.ext_grid:
        if "min_p_kw" not in net.ext_grid:
            net.ext_grid["min_p_kw"] = -1e9
        if "max_p_kw" not in net.ext_grid:
            net.ext_grid["max_p_kw"] = 0
        if not "piecewise_linear_cost" in net:
            for index, cost in net.ext_grid.cost_per_kw.iteritems():
                if not np.isnan(cost):
                    p = net.ext_grid.min_p_kw.at[index]
                    create_piecewise_linear_cost(net, index, "ext_grid",
                                                 np.array([[p, cost * p], [0, 0]]))

    if "cost_per_kvar" in net.gen:

        if not "piecewise_linear_cost" in net:
            for index, cost in net.gen.cost_per_kvar.iteritems():
                if not np.isnan(cost):
                    qmin = net.gen.min_q_kvar.at[index]
                    qmax = net.gen.max_q_kvar.at[index]
                    create_piecewise_linear_cost(net, index, "gen",
                                                 np.array([[qmin, cost * qmin], [0, 0],
                                                           [qmax, cost * qmax]]), type="q")

    if "cost_per_kvar" in net.sgen:

        if not "piecewise_linear_cost" in net:
            for index, cost in net.sgen.cost_per_kvar.iteritems():
                if not np.isnan(cost):
                    qmin = net.sgen.min_q_kvar.at[index]
                    qmax = net.sgen.max_q_kvar.at[index]
                    create_piecewise_linear_cost(net, index, "sgen",
                                                 np.array([[qmin, cost * qmin], [0, 0],
                                                           [qmax, cost * qmax]]), type="q")

    if "cost_per_kvar" in net.ext_grid:

        if not "piecewise_linear_cost" in net:
            for index, cost in net.ext_grid.cost_per_kvar.iteritems():
                if not np.isnan(cost):
                    qmin = net.ext_grid.min_q_kvar.at[index]
                    qmax = net.ext_grid.max_q_kvar.at[index]
                    create_piecewise_linear_cost(net, index, "ext_grid",
                                                 np.array([[qmin, cost * qmin], [0, 0],
                                                           [qmax, cost * qmax]]), type="q")

    if "tp_st_degree" not in net.trafo:
        net.trafo["tp_st_degree"] = np.nan
    if "tp_st_degree" not in net.trafo3w:
        net.trafo3w["tp_st_degree"] = np.nan
    if "tap_at_star_point" not in net.trafo3w:
        net.trafo3w["tap_at_star_point"] = False
    if "_pd2ppc_lookups" not in net:
        net._pd2ppc_lookups = {"bus": None,
                               "ext_grid": None,
                               "gen": None}
    if "_is_elements" not in net and "__is_elements" in net:
        net["_is_elements"] = copy.deepcopy(net["__is_elements"])
        net.pop("__is_elements", None)
    elif "_is_elements" not in net and "_is_elems" in net:
        net["_is_elements"] = copy.deepcopy(net["_is_elems"])
        net.pop("_is_elems", None)

    if "options" in net:
        if "recycle" in net["options"]:
            if "_is_elements" not in net["options"]["recycle"]:
                net["options"]["recycle"]["_is_elements"] = copy.deepcopy(
                    net["options"]["recycle"]["is_elems"])
                net["options"]["recycle"].pop("is_elems", None)

    if "const_z_percent" not in net.load or "const_i_percent" not in net.load:
        net.load["const_z_percent"] = np.zeros(net.load.shape[0])
        net.load["const_i_percent"] = np.zeros(net.load.shape[0])

    if "vn_kv" not in net["shunt"]:
        net.shunt["vn_kv"] = net.bus.vn_kv.loc[net.shunt.bus.values].values
    if "step" not in net["shunt"]:
        net.shunt["step"] = 1
    if "max_step" not in net["shunt"]:
        net.shunt["max_step"] = 1
    if "_pd2ppc_lookups" not in net:
        net["_pd2ppc_lookups"] = {"bus": None,
                                  "gen": None,
                                  "branch": None}
    net.version = float(__version__[:3])
    if "std_type" not in net.trafo3w:
        net.trafo3w["std_type"] = None

    if "time_resolution" not in net:
        # for storages
        time_resolution = 1.0

    new_net = create_empty_network()
    for key, item in net.items():
        if isinstance(item, pd.DataFrame):
            for col in item.columns:
                if key in new_net and col in new_net[key].columns:
                    if set(item.columns) == set(new_net[key]):
                        try:
                            net[key] = net[key].reindex(new_net[key].columns, axis=1)
                        except: #legacy for pandas <0.21
                            net[key] = net[key].reindex_axis(new_net[key].columns, axis=1)
                    if int(pd.__version__[2]) < 2:
                        net[key][col] = net[key][col].astype(new_net[key][col].dtype,
                                                             raise_on_error=False)
                    else:
                        net[key][col] = net[key][col].astype(new_net[key][col].dtype,
                                                             errors="ignore")
    if not "g_us_per_km" in net.line:
        net.line["g_us_per_km"] = 0.
    return net


def _pre_release_changes(net):
    from pandapower.std_types import add_basic_std_types, create_std_type, parameter_from_std_type
    from pandapower.powerflow import reset_results
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
    if "df" not in net.line.columns:
        net.line['df'] = 1.
    net["ext_grid"] = net["ext_grid"].rename(columns={"angle_degree": "va_degree",
                                                      "ua_degree": "va_degree",
                                                      "sk_max_mva": "s_sc_max_mva",
                                                      "sk_min_mva": "s_sc_min_mva"})
    net["line"]["type"].replace("f", "ol", inplace=True)
    net["line"]["type"].replace("k", "cs", inplace=True)
    net["trafo"] = net["trafo"].rename(columns={'trafotype': 'std_type', "type": "std_type",
                                                "un1_kv": "vn_hv_kv", "un2_kv": "vn_lv_kv",
                                                'vfe_kw': 'pfe_kw', "unh_kv": "vn_hv_kv",
                                                "unl_kv": "vn_lv_kv", "type": "std_type",
                                                'vfe_kw': 'pfe_kw', "uk_percent": "vsc_percent",
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
                                                    'vfe_kw': 'pfe_kw',
                                                    "vnh_kv": "vn_hv_kv", "vnm_kv": "vn_mv_kv",
                                                    "vnl_kv": "vn_lv_kv", "snh_kva": "sn_hv_kva",
                                                    "snm_kva": "sn_mv_kva", "snl_kva": "sn_lv_kva"})
    for element, old, new in [("trafo", "vnh_kv", "vn_hv_kv"),
                              ("trafo", "vnl_kv", "vn_lv_kv"),
                              ("trafo3w", "vnh_kv", "vn_hv_kv"),
                              ("trafo3w", "vnm_kv", "vn_mv_kv"),
                              ("trafo3w", "vnl_kv", "vn_lv_kv")]:
        for std_type, parameters in net.std_types[element].items():
            if old in parameters:
                net.std_types[element][std_type][new] = net.std_types[element][std_type].pop(old)
    if "name" not in net.switch.columns:
        net.switch["name"] = None
    net["switch"] = net["switch"].rename(columns={'element_type': 'et'})
    net["ext_grid"] = net["ext_grid"].rename(columns={'voltage': 'vm_pu', "u_pu": "vm_pu",
                                                      "sk_max": "sk_max_mva",
                                                      "ua_degree": "va_degree"})
    if "in_service" not in net["ext_grid"].columns:
        net["ext_grid"]["in_service"] = 1
    if "tp_phase_shifter" not in net["trafo"].columns:
        # infer to still have the same behavior
        net["trafo"]["tp_phase_shifter"] = False
        if "tp_st_degree" in net["trafo"]:
            is_tp_phase_shifter = \
                (net.trafo.tp_st_degree.values!=0)  & np.isfinite(net.trafo.tp_st_degree.values) \
                & ((net.trafo.tp_st_percent.values==0) | np.isnan(net.trafo.tp_st_percent.values))
            net["trafo"]["tp_phase_shifter"].values[is_tp_phase_shifter] = True
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
    if "parallel" not in net.trafo:
        net.trafo["parallel"] = 1
    if "df" not in net.trafo:
        net.trafo["df"] = 1.
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
    if "in_service" not in net["ward"]:
        net.ward["in_service"] = True
    net.switch.closed = net.switch.closed.astype(bool)


def add_column_from_node_to_elements(net, column, replace, elements=None, branch_bus=None):
    """
    Adds column data to elements, inferring them from the column data of buses they are
    connected to.

    INPUT:
        **net** (pandapowerNet) - the pandapower net that will be changed

        **column** (string) - name of column that should be copied from the bus table to the element
            table

        **replace** (boolean) - if True, an existing column in the element table will be overwritten

        **elements** (list) - list of elements that should get the column values from the bus table

        **branch_bus** (list) - defines which bus should be considered for branch elements.
            'branch_bus' must have the length of 2. One entry must be 'from_bus' or 'to_bus', the
            other 'hv_bus' or 'lv_bus'

    EXAMPLE:
        compare to add_zones_to_elements()
    """
    branch_bus = ["from_bus", "hv_bus"] if branch_bus is None else branch_bus
    if column not in net.bus.columns:
        raise ValueError("%s is not in net.bus.columns" % column)
    elements = elements if elements is not None else pp_elements(bus=False)
    elements_to_replace = elements if replace else [el for el in elements if column not in
                                                    net[el].columns]
    # bus elements
    for element, bus_type in element_bus_tuples(bus_elements=True, branch_elements=False):
        if element in elements_to_replace:
            net[element][column] = net["bus"][column].loc[net[element][bus_type]].values
    # branch elements
    to_validate = {}
    for element, bus_type in element_bus_tuples(bus_elements=False, branch_elements=True):
        if element in elements_to_replace:
            if bus_type in (branch_bus + ["bus"]):  # copy data, append branch_bus for switch.bus
                net[element][column] = net["bus"][column].loc[net[element][bus_type]].values
            else:  # save data for validation
                to_validate[element] = net["bus"][column].loc[net[element][bus_type]].values
    # validate branch elements, but do not validate double and switches at all
    already_validated = ["switch"]
    for element, bus_type in element_bus_tuples(bus_elements=False, branch_elements=True):
        if (element in elements_to_replace) & (element not in already_validated):
            already_validated += [element]
            crossing = sum(net[element][column].values != to_validate[element])
            if crossing > 0:
                logger.warning("There have been %i %ss with different " % (crossing, element) +
                               "%s data at from-/hv- and to-/lv-bus" % column)


def add_zones_to_elements(net, replace=True, elements=None, **kwargs):
    """ Adds zones to elements, inferring them from the zones of buses they are connected to. """
    elements = ["line", "trafo", "ext_grid", "switch"] if elements is None else elements
    add_column_from_node_to_elements(net, "zone", replace=replace, elements=elements, **kwargs)


def create_continuous_bus_index(net, start=0):
    """
    Creates a continuous bus index starting at zero and replaces all
    references of old indices by the new ones.
    """
    new_bus_idxs = list(np.arange(start, len(net.bus) + start))
    bus_lookup = dict(zip(net["bus"].index.values, new_bus_idxs))
    net.bus.index = new_bus_idxs

    for element, value in element_bus_tuples():
        net[element][value] = get_indices(net[element][value], bus_lookup)
    net["bus_geodata"].set_index(get_indices(net["bus_geodata"].index, bus_lookup), inplace=True)
    bb_switches = net.switch[net.switch.et == "b"]
    net.switch.loc[bb_switches.index, "element"] = get_indices(bb_switches.element, bus_lookup)
    return net


def set_scaling_by_type(net, scalings, scale_load=True, scale_sgen=True):
    """
    Sets scaling of loads and/or sgens according to a dictionary
    mapping type to a scaling factor. Note that the type-string is case
    sensitive.
    E.g. scaling = {"pv": 0.8, "bhkw": 0.6}

    :param net:
    :param scalings: A dictionary containing a mapping from element type to
    :param scale_load:
    :param scale_sgen:
    """
    if not isinstance(scalings, dict):
        raise UserWarning("The parameter scaling has to be a dictionary, "
                          "see docstring")

    def scaleit(what):
        et = net[what]
        et["scaling"] = [scale[t] if scale[t] is not None else s for t, s in
                         zip(et.type.values, et.scaling.values)]

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
    closed_switches = set()
    nl = net.switch[(net.switch.et == 'l') & (net.switch.closed == 0)]
    for _, switch in nl.groupby("element"):
        if len(switch.index) > 1:  # find all lines that have open switches at both ends
            # and close on of them
            net.switch.at[switch.index[0], "closed"] = 1
            closed_switches.add(switch.index[0])
    if len(closed_switches) > 0:
        logger.info('closed %d switches at line with 2 open switches (switches: %s)' % (
            len(closed_switches), closed_switches))


def drop_inactive_elements(net):
    """
    Drops any elements not in service AND any elements connected to inactive
    buses.
    """
    set_isolated_areas_out_of_service(net)
    drop_out_of_service_elements(net)


def drop_out_of_service_elements(net):
    # removes inactive lines and its switches and geodata
    inactive_lines = net.line[~net.line.in_service].index
    drop_lines(net, inactive_lines)

    inactive_trafos = net.trafo[~net.trafo.in_service].index
    drop_trafos(net, inactive_trafos, table='trafo')

    inactive_trafos3w = net.trafo3w[~net.trafo3w.in_service].index
    drop_trafos(net, inactive_trafos3w, table='trafo3w')

    do_not_delete = set(net.line.from_bus.values) | set(net.line.to_bus.values) | \
                    set(net.trafo.hv_bus.values) | set(net.trafo.lv_bus.values) | \
                    set(net.trafo3w.hv_bus.values) | set(net.trafo3w.mv_bus.values) | \
                    set(net.trafo3w.lv_bus.values)

    # removes inactive buses safely
    inactive_buses = set(net.bus[~net.bus.in_service].index) - do_not_delete
    drop_buses(net, inactive_buses, drop_elements=True)

    # TODO: the following is not necessary anymore?
    for element in net.keys():
        if element not in ["bus", "trafo", "trafo3w", "line", "_equiv_trafo3w"] \
                and isinstance(net[element], pd.DataFrame) \
                and "in_service" in net[element].columns:
            drop_idx = net[element].query("not in_service").index
            net[element].drop(drop_idx, inplace=True)
            if len(drop_idx) > 0:
                logger.info("dropped %d %s elements!" % (len(drop_idx), element))


def element_bus_tuples(bus_elements=True, branch_elements=True):
    """
    Utility function
    Provides the tuples of elements and corresponding columns for buses they are connected to
    :param bus_elements: whether tuples for bus elements e.g. load, sgen, ... are included
    :param branch_elements: whether branch elements e.g. line, trafo, ... are included
    :return: set of tuples with element names and column names
    """
    ebt = set()
    if bus_elements:
        ebt.update([("sgen", "bus"), ("load", "bus"), ("ext_grid", "bus"), ("gen", "bus"),
                    ("ward", "bus"), ("xward", "bus"), ("shunt", "bus"), ("measurement", "bus")])
    if branch_elements:
        ebt.update([("line", "from_bus"), ("line", "to_bus"), ("impedance", "from_bus"),
                    ("switch", "bus"), ("impedance", "to_bus"), ("trafo", "hv_bus"),
                    ("trafo", "lv_bus"), ("trafo3w", "hv_bus"), ("trafo3w", "mv_bus"),
                    ("trafo3w", "lv_bus"), ("dcline", "from_bus"), ("dcline", "to_bus")])
    return ebt


def pp_elements(bus=True, bus_elements=True, branch_elements=True):
    """ Returns the list of pandapower elements. """
    if bus:
        return set(["bus"] + [el[0] for el in element_bus_tuples(bus_elements, branch_elements)])
    else:
        return set([el[0] for el in element_bus_tuples(bus_elements, branch_elements)])


def drop_buses(net, buses, drop_elements=True):
    """
    Drops specified buses, their bus_geodata and by default safely drops all elements connected to
    them as well.
    """
    # drop busbus switches
    i = net["switch"][((net["switch"]["element"].isin(buses)) |
                       (net["switch"]["bus"].isin(buses))) & (net["switch"]["et"] == "b")].index
    net["switch"].drop(i, inplace=True)

    # drop buses and their geodata
    net["bus"].drop(buses, inplace=True)
    net["bus_geodata"].drop(set(buses) & set(net["bus_geodata"].index), inplace=True)
#    logger.info('dropped %d buses: %s' % (len(buses), buses))

    if drop_elements:
        for element, column in element_bus_tuples():
            if any(net[element][column].isin(buses)):
                eid = net[element][net[element][column].isin(buses)].index
                if element == 'line':
                    drop_lines(net, eid)
                elif element == 'trafo' or element == 'trafo3w':
                    drop_trafos(net, eid, table=element)
                else:
                    net[element].drop(eid, inplace=True)
#                    logger.info("dropped %s elements: %d" % (element, len(eid)))


def drop_elements_at_buses(net, buses):
    """
    drop elements connected to certain buses
    """
    # If there is a bus1 -bus2 switch, this will delete the switch when we select bus 2.
    if any(net['switch']['element'].isin(buses)):
        eid = net['switch'][net['switch']['element'].isin(buses)].index
        net['switch'].drop(eid, inplace=True)

    # drop elements connected to buses
    for element, column in element_bus_tuples():
        if any(net[element][column].isin(buses)):
            eid = net[element][net[element][column].isin(buses)].index
            if element == 'line':
                drop_lines(net, eid)
            elif element == 'trafo' or element == 'trafo3w':
                drop_trafos(net, eid, table=element)
            else:
                net[element].drop(eid, inplace=True)
                logger.info("dropped %s elements: %d" % (element, len(eid)))


def drop_trafos(net, trafos, table="trafo"):
    """
    Deletes all trafos and in the given list of indices and removes
    any switches connected to it.
    """
    if table not in ('trafo', 'trafo3w'):
        raise UserWarning("parameter 'table' must be 'trafo' or 'trafo3w'")
    # drop any switches
    if table == 'trafo':  # remove as soon as the trafo3w switches are implemented
        i = net["switch"].index[(net["switch"]["element"].isin(trafos)) &
                                (net["switch"]["et"] == "t")]
        net["switch"].drop(i, inplace=True)

    # drop the trafos
    net[table].drop(trafos, inplace=True)
    logger.info("dropped %d %s elements" % (len(trafos), table))


def drop_lines(net, lines):
    """
    Deletes all lines and their geodata in the given list of indices and removes
    any switches connected to it.
    """
    # drop any switches
    i = net["switch"][(net["switch"]["element"].isin(lines)) & (net["switch"]["et"] == "l")].index
    net["switch"].drop(i, inplace=True)

    # drop the lines+geodata
    net["line"].drop(lines, inplace=True)
    net["line_geodata"].drop(set(lines) & set(net["line_geodata"].index), inplace=True)
    logger.info("dropped %d lines" % len(lines))


def fuse_buses(net, b1, b2, drop=True):
    """
    Reroutes any connections to buses in b2 to the given bus b1. Additionally drops the buses b2,
    if drop=True (default).
    """
    try:
        b2.__iter__
        b2 = set(b2) - {b1}
    except:
        b2 = [b2]

    for element, value in element_bus_tuples():
        i = net[element][net[element][value].isin(b2)].index
        net[element].loc[i, value] = b1

    i = net["switch"][(net["switch"]["et"] == 'b') & (
        net["switch"]["element"].isin(b2))].index
    net["switch"].loc[i, "element"] = b1
    net["switch"].drop(net["switch"][(net["switch"]["bus"] == net["switch"]["element"]) &
                                     (net["switch"]["et"] == "b")].index, inplace=True)
    if drop:
        # drop_elements=False because the elements must be connected to new buses now
        drop_buses(net, b2, drop_elements=False)
    return net


def set_element_status(net, buses, in_service):
    """
    Sets buses and all elements connected to them in or out of service.
    """
    net.bus.loc[buses, "in_service"] = in_service

    for element in net.keys():
        if element not in ['bus'] and isinstance(net[element], pd.DataFrame) \
                and "in_service" in net[element].columns:
            idx = get_connected_elements(net, element, buses)
            net[element].loc[idx, 'in_service'] = in_service


def set_isolated_areas_out_of_service(net):
    """
    Set all isolated buses and all elements connected to isolated buses out of service.
    """
    closed_switches = set()
    unsupplied = unsupplied_buses(net)
    logger.info("set %d of %d unsupplied buses out of service" % (
        len(net.bus.loc[unsupplied].query('~in_service')), len(unsupplied)))
    set_element_status(net, unsupplied, False)

    # TODO: remove this loop after unsupplied_buses are fixed
    for tr3w in net.trafo3w.index.values:
        tr3w_buses = net.trafo3w.loc[tr3w, ['hv_bus', 'mv_bus', 'lv_bus']].values
        if not all(net.bus.loc[tr3w_buses, 'in_service'].values):
            net.trafo3w.loc[tr3w, 'in_service'] = False

    for element in ["line", "trafo"]:
        oos_elements = net.line[~net.line.in_service].index
        oos_switches = net.switch[(net.switch.et == element[0]) &
                                  (net.switch.element.isin(oos_elements))].index

        closed_switches.update([i for i in oos_switches.values if not net.switch.at[i, 'closed']])
        net.switch.loc[oos_switches, "closed"] = True

        for idx, bus in net.switch[
                    ~net.switch.closed & (net.switch.et == element[0])][["element", "bus"]].values:
            if not net.bus.in_service.at[next_bus(net, bus, idx, element)]:
                net[element].at[idx, "in_service"] = False
    if len(closed_switches) > 0:
        logger.info('closed %d switches: %s' % (len(closed_switches), closed_switches))


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
    p2.trafo3w = net.trafo3w[(net.trafo3w.hv_bus.isin(buses)) & (net.trafo3w.mv_bus.isin(buses)) &
                             (net.trafo3w.lv_bus.isin(buses))]
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
    # return a pandapowerNet
    if keep_everything_else:
        newnet = copy.deepcopy(net)
        newnet.update(p2)
        return pandapowerNet(newnet)
    p2["std_types"] = copy.deepcopy(net["std_types"])
    return pandapowerNet(p2)


def merge_nets(net1, net2, validate=True, tol=1e-9, **kwargs):
    """
    Function to concatenate two nets into one data structure. All element tables get new,
    continuous indizes in order to avoid duplicates.
    """
    net = copy.deepcopy(net1)
    net1 = copy.deepcopy(net1)
    net2 = copy.deepcopy(net2)
    create_continuous_bus_index(net2, start=net1.bus.index.max() + 1)
    if validate:
        runpp(net1, **kwargs)
        runpp(net2, **kwargs)

    def adapt_switches(net, element, offset=0):
        switches = net.switch[net.switch.et == element[0]]  # element[0] == "l" for "line", ect.
        new_index = [net[element].index.get_loc(ix) + offset
                     for ix in switches.element.values]
        if len(new_index):
            net.switch.loc[switches.index, "element"] = new_index

    for element, table in net.items():
        if element.startswith("_") or element.startswith("res"):
            continue
        if type(table) == pd.DataFrame and (len(table) > 0 or len(net2[element]) > 0):
            if element == "switch":
                adapt_switches(net2, "line", offset=len(net1.line))
                adapt_switches(net1, "line")
                adapt_switches(net2, "trafo", offset=len(net1.trafo))
                adapt_switches(net1, "trafo")
            if element == "line_geodata":
                ni = [net1.line.index.get_loc(ix) for ix in net1["line_geodata"].index]
                net1.line_geodata.set_index(np.array(ni), inplace=True)
                ni = [net2.line.index.get_loc(ix) + len(net1.line)
                      for ix in net2["line_geodata"].index]
                net2.line_geodata.set_index(np.array(ni), inplace=True)
            ignore_index = element not in ("bus", "bus_geodata", "line_geodata")
            dtypes = net1[element].dtypes
            net[element] = pd.concat([net1[element], net2[element]], ignore_index=ignore_index)
            _preserve_dtypes(net[element], dtypes)
    if validate:
        runpp(net, **kwargs)
        dev1 = max(abs(net.res_bus.loc[net1.bus.index].vm_pu.values - net1.res_bus.vm_pu.values))
        dev2 = max(abs(net.res_bus.iloc[len(net1.bus.index):].vm_pu.values -
                       net2.res_bus.vm_pu.values))
        if dev1 > tol or dev2 > tol:
            raise UserWarning("Deviation in bus voltages after merging: %.10f" % max(dev1, dev2))
    return net


# --- item/element selections

def get_element_index(net, element, name, exact_match=True):
    """
    Returns the element(s) identified by a name or regex and its element-table.

    INPUT:
      **net** - pandapower network

      **element** - Table to get indices from ("line", "bus", "trafo" etc.)

      **name** - Name of the element to match.

    OPTIONAL:
      **exact_match** (boolean, True) - True: Expects exactly one match, raises
                                                UserWarning otherwise.
                                        False: returns all indices containing the name

    OUTPUT:
      **index** - The indices of matching element(s).
    """
    if exact_match:
        idx = net[element][net[element]["name"] == name].index
        if len(idx) == 0:
            raise UserWarning("There is no %s with name %s" % (element, name))
        if len(idx) > 1:
            raise UserWarning("Duplicate %s names for %s" % (element, name))
        return idx[0]
    else:
        return net[element][net[element]["name"].str.contains(name)].index


def next_bus(net, bus, element_id, et='line', **kwargs):
    """
    Returns the index of the second bus an element is connected to, given a
    first one. E.g. the from_bus given the to_bus of a line.
    """
    if et == 'line':
        bc = ["from_bus", "to_bus"]
    elif et == 'trafo':
        bc = ["hv_bus", "lv_bus"]
    elif et == "switch" and list(net[et].loc[element_id,["et"]].values)==['b']:   # Raises error if switch is not a bus-bus switch
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
        **net** (pandapowerNet)

        **element** (string, name of the element table)

        **buses** (single integer or iterable of ints)

     OPTIONAL:
        **respect_switches** (boolean, True)    - True: open switches will be respected
                                                  False: open switches will be ignored
        **respect_in_service** (boolean, False) - True: in_service status of connected lines will be
                                                        respected
                                                  False: in_service status will be ignored
     OUTPUT:
        **connected_elements** (set) - Returns connected elements.

    """

    if not hasattr(buses, "__iter__"):
        buses = [buses]

    if element in ["line", "l"]:
        element = "l"
        element_table = net.line
        connected_elements = set(net.line.index[net.line.from_bus.isin(buses) |
                                                net.line.to_bus.isin(buses)])

    elif element in ["dcline"]:
        element_table = net.dcline
        connected_elements = set(net.dcline.index[net.dcline.from_bus.isin(buses) |
                                                  net.dcline.to_bus.isin(buses)])

    elif element in ["trafo"]:
        element = "t"
        element_table = net.trafo
        connected_elements = set(net["trafo"].index[(net.trafo.hv_bus.isin(buses)) |
                                                    (net.trafo.lv_bus.isin(buses))])
    elif element in ["trafo3w", "t3w"]:
        element = "t3w"
        element_table = net.trafo3w
        connected_elements = set(net["trafo3w"].index[(net.trafo3w.hv_bus.isin(buses)) |
                                                      (net.trafo3w.mv_bus.isin(buses)) |
                                                      (net.trafo3w.lv_bus.isin(buses))])
    elif element == "impedance":
        element_table = net.impedance
        connected_elements = set(net["impedance"].index[(net.impedance.from_bus.isin(buses)) |
                                                        (net.impedance.to_bus.isin(buses))])
    elif element in ["gen", "ext_grid", "xward", "shunt", "ward", "sgen", "load", "storage"]:
        element_table = net[element]
        connected_elements = set(element_table.index[(element_table.bus.isin(buses))])
    elif element in ['_equiv_trafo3w']:
        # ignore '_equiv_trafo3w'
        return {}
    else:
        raise UserWarning("Unknown element! ", element)

    if respect_switches and element in ["l", "t", "t3w"]:
        open_switches = get_connected_switches(net, buses, consider=element, status="open")
        if open_switches:
            open_and_connected = net.switch.loc[net.switch.index.isin(open_switches) &
                                                net.switch.element.isin(connected_elements)].index
            connected_elements -= set(net.switch.element[open_and_connected])

    if respect_in_service:
        connected_elements -= set(element_table[~element_table.in_service].index)

    return connected_elements


def get_connected_buses(net, buses, consider=("l", "s", "t", "t3"), respect_switches=True,
                        respect_in_service=False):
    """
     Returns buses connected to given buses. The source buses will NOT be returned.

     INPUT:
        **net** (pandapowerNet)

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
     OUTPUT:
        **cl** (set) - Returns connected buses.

    """
    if not hasattr(buses, "__iter__"):
        buses = [buses]

    cb = set()
    if "l" in consider:
        in_service_constr = net.line.in_service if respect_in_service else True
        opened_lines = set(net.switch.loc[(~net.switch.closed) & (net.switch.et == "l")
                                          ].element.unique()) if respect_switches else {}
        connected_fb_lines = set(net.line.index[
            (net.line.from_bus.isin(buses)) & ~net.line.index.isin(opened_lines) &
            (in_service_constr)])
        connected_tb_lines = set(net.line.index[
            (net.line.to_bus.isin(buses)) & ~net.line.index.isin(opened_lines) &
            (in_service_constr)])
        cb |= set(net.line[net.line.index.isin(connected_tb_lines)].from_bus)
        cb |= set(net.line[net.line.index.isin(connected_fb_lines)].to_bus)

    if "s" in consider:
        cs = get_connected_switches(net, buses, consider='b',
                                    status="closed" if respect_switches else "all")
        cb |= set(net.switch[net.switch.index.isin(cs)].element)
        cb |= set(net.switch[net.switch.index.isin(cs)].bus)

    if "t" in consider:
        in_service_constr = net.trafo.in_service if respect_in_service else True
        opened_trafos = set(net.switch.loc[(~net.switch.closed) & (net.switch.et == "t")
                                           ].element.unique()) if respect_switches else {}
        connected_hvb_trafos = set(net.trafo.index[
            (net.trafo.hv_bus.isin(buses)) & ~net.trafo.index.isin(opened_trafos) &
            (in_service_constr)])
        connected_lvb_trafos = set(net.trafo.index[
            (net.trafo.lv_bus.isin(buses)) & ~net.trafo.index.isin(opened_trafos) &
            (in_service_constr)])
        cb |= set(net.trafo.loc[connected_lvb_trafos].hv_bus.values)
        cb |= set(net.trafo.loc[connected_hvb_trafos].lv_bus.values)

    # Gives the lv mv and hv buses of a 3 winding transformer
    if "t3" in consider:
        ct3 = get_connected_elements(net, "trafo3w", buses, respect_switches, respect_in_service)
        cb |= set(net.trafo3w.loc[ct3].hv_bus.values)
        cb |= set(net.trafo3w.loc[ct3].mv_bus.values)
        cb |= set(net.trafo3w.loc[ct3].lv_bus.values)

    if respect_in_service:
        cb -= set(net.bus[~net.bus.in_service].index)

    return cb - set(buses)


def get_connected_buses_at_element(net, element, et, respect_in_service=False):
    """
     Returns buses connected to a given line, switch or trafo. In case of a bus switch, two buses
     will be returned, else one.

     INPUT:
        **net** (pandapowerNet)

        **element** (integer)

        **et** (string)                             - Type of the source element:
                                                      l: line
                                                      s: switch
                                                      t: trafo

     OPTIONAL:
        **respect_in_service** (boolean, False)     - True: in_service status of connected buses
                                                            will be respected
                                                      False: in_service status will be ignored
     OUTPUT:
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
        **net** (pandapowerNet)

        **buses** (single integer or iterable of ints)

    OPTIONAL:
        **respect_switches** (boolean, True)        - True: open switches will be respected
                                                     False: open switches will be ignored

        **respect_in_service** (boolean, False)     - True: in_service status of connected
                                                            buses will be respected

                                                      False: in_service status will be ignored
        **consider** (iterable, ("l", "s", "t"))    - Determines, which types of connections
                                                      will be considered.
                                                      l: lines
                                                      b: bus-bus-switches
                                                      t: trafos

        **status** (string, ("all", "closed", "open"))    - Determines, which switches will
                                                            be considered
    OUTPUT:
       **cl** (set) - Returns connected buses.

    """

    if not hasattr(buses, "__iter__"):
        buses = [buses]

    if status == "closed":
        switch_selection = net.switch.closed
    elif status == "open":
        switch_selection = ~net.switch.closed
    elif status == "all":
        switch_selection = np.full(len(net.switch), True, dtype=bool)
    else:
        logger.warn("Unknown switch status \"%s\" selected! "
                    "Selecting all switches by default." % status)

    cs = set()
    if 'b' in consider:
        cs |= set(net['switch'].index[
                      (net['switch']['bus'].isin(buses) | net['switch']['element'].isin(buses)) &
                      (net['switch']['et'] == 'b') & switch_selection])
    if 'l' in consider:
        cs |= set(net['switch'].index[(net['switch']['bus'].isin(buses)) & (
            net['switch']['et'] == 'l') & switch_selection])

    if 't' in consider:
        cs |= set(net['switch'].index[net['switch']['bus'].isin(buses) & (
            net['switch']['et'] == 't') & switch_selection])

    return cs


def pq_from_cosphi(s, cosphi, qmode, pmode):
    """
    Calculates P/Q values from rated apparent power and cosine(phi) values.

       - s: rated apparent power
       - cosphi: cosine phi of the
       - qmode: "ind" for inductive or "cap" for capacitive behaviour
       - pmode: "load" for load or "gen" for generation

    As all other pandapower functions this function is based on the consumer viewpoint. For active
    power, that means that loads are positive and generation is negative. For reactive power,
    inductive behaviour is modeled with positive values, capacitive behaviour with negative values.
    """
    if qmode == "ind":
        qsign = 1
    elif qmode == "cap":
        qsign = -1
    elif qmode == "ohm":
        qsign = 1
        if cosphi != 1:
            raise ValueError("qmode cannot be 'ohm' if cosphi is not 1.")
    else:
        raise ValueError("Unknown mode %s - specify 'ind' or 'cap'" % qmode)

    if pmode == "load":
        psign = 1
    elif pmode == "gen":
        psign = -1
    else:
        raise ValueError("Unknown mode %s - specify 'load' or 'gen'" % pmode)

    p = psign * s * cosphi
    q = qsign * np.sqrt(s ** 2 - p ** 2)
    return p, q


def cosphi_from_pq(p, q):
    """
    Analog to pq_from_cosphi, but other way around.
    In consumer viewpoint (pandapower): cap=overexcited and ind=underexcited
    """
    if p == 0:
        cosphi = np.nan
        logger.warn("A cosphi from p=0 is undefined.")
    else:
        cosphi = np.cos(np.arctan(q/p))
    s = (p**2 + q**2)**0.5
    pmode = ["undef", "load", "gen"][int(np.sign(p))]
    qmode = ["ohm", "ind", "cap"][int(np.sign(q))]
    return cosphi, s, qmode, pmode


def create_replacement_switch_for_branch(net, element, idx):
    """
    Creates a switch parallel to a branch, connecting the same buses as the branch.
    The switch is closed if the branch is in service and open if the branch is out of service.
    The in_service status of the original branch is not affected and should be set separately,
    if needed.

    :param net: pandapower network
    :param element: element table e. g. 'line', 'impedance'
    :param idx: index of the branch e. g. 0
    :return: None
    """
    bus_i = net[element].from_bus.at[idx]
    bus_j = net[element].to_bus.at[idx]
    in_service = net[element].in_service.at[idx]
    if element in ['line', 'trafo']:
        is_closed = all(
            net.switch.loc[(net.switch.element == idx) & (net.switch.et == element[0]), 'closed'])
        is_closed = is_closed and in_service
    else:
        is_closed = in_service

    switch_name = 'REPLACEMENT_%s_%d' % (element, idx)
    sid = create_switch(net, name=switch_name, bus=bus_i, element=bus_j, et='b', closed=is_closed,
                        type='CB')
    logger.debug('created switch %s (%d) as replacement for %s %s' %
                 (switch_name, sid, element, idx))


def replace_zero_branches_with_switches(net, elements=('line', 'impedance'),
                                        zero_length=True, zero_impedance=True, in_service_only=True,
                                        min_length_km=0, min_r_ohm_per_km=0, min_x_ohm_per_km=0,
                                        min_c_nf_per_km=0, min_rft_pu=0, min_xft_pu=0, min_rtf_pu=0,
                                        min_xtf_pu=0):
    """
    Creates a replacement switch for branches with zero impedance (line, impedance) and sets them
    out of service.

    :param net: pandapower network
    :param elements: a tuple of names of element tables e. g. ('line', 'impedance') or (line)
    :param zero_length: whether zero length lines will be affected
    :param zero_impedance: whether zero impedance branches will be affected
    :param in_service_only: whether the branches that are not in service will be affected
    :param min_length_km: threshhold for line length for a line to be considered zero line
    :param min_r_ohm_per_km: threshhold for line R' value for a line to be considered zero line
    :param min_x_ohm_per_km: threshhold for line X' value for a line to be considered zero line
    :param min_c_nf_per_km: threshhold for line C' for a line to be considered zero line
    :param min_rft_pu: threshhold for R from-to value for impedance to be considered zero impedance
    :param min_xft_pu: threshhold for X from-to value for impedance to be considered zero impedance
    :param min_rtf_pu: threshhold for R to-from value for impedance to be considered zero impedance
    :param min_xtf_pu: threshhold for X to-from value for impedance to be considered zero impedance
    :return:
    """

    if not isinstance(elements, tuple):
        raise TypeError(
            'input parameter "elements" must be a tuple, e.g. ("line", "impedance") or ("line")')

    for elm in elements:
        branch_zero = set()
        if elm == 'line' and zero_length:
            branch_zero.update(net[elm].loc[net[elm].length_km <= min_length_km].index.tolist())

        if elm == 'line' and zero_impedance:
            branch_zero.update(net[elm].loc[(net[elm].r_ohm_per_km <= min_r_ohm_per_km) &
                                            (net[elm].x_ohm_per_km <= min_x_ohm_per_km) &
                                            (net[elm].c_nf_per_km <= min_c_nf_per_km)
                                            ].index.tolist())

        if elm == 'impedance' and zero_impedance:
            branch_zero.update(net[elm].loc[(net[elm].rft_pu <= min_rft_pu) &
                                            (net[elm].xft_pu <= min_xft_pu) &
                                            (net[elm].rtf_pu <= min_rtf_pu) &
                                            (net[elm].xtf_pu <= min_xtf_pu)].index.tolist())

        k = 0
        for b in branch_zero:
            if in_service_only and ~net[elm].in_service.at[b]:
                continue
            create_replacement_switch_for_branch(net, element=elm, idx=b)
            net[elm].loc[b, 'in_service'] = False
            k += 1

        logger.info('set %d %ss out of service' % (k, elm))
