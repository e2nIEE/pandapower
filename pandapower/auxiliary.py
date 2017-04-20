# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

from collections import MutableMapping

import numpy as np
import pandas as pd
import scipy as sp
import six
from pandapower.idx_brch import F_BUS, T_BUS
from pandapower.idx_bus import BUS_I, BUS_TYPE, NONE, PD, QD

try:
    import pplog as logging
except:
    import logging

logger = logging.getLogger(__name__)


class ADict(dict, MutableMapping):
    def __init__(self, *args, **kwargs):
        super(ADict, self).__init__(*args, **kwargs)

        # to prevent overwrite of internal attributes by new keys
        # see _valid_name()
        self._setattr('_allow_invalid_attributes', False)

    def _build(self, obj, **kwargs):
        """
        We only want dict like elements to be treated as recursive AttrDicts.
        """
        return obj

    # --- taken from AttrDict

    def __getstate__(self):
        return self.copy(), self._allow_invalid_attributes

    def __setstate__(self, state):
        mapping, allow_invalid_attributes = state
        self.update(mapping)
        self._setattr('_allow_invalid_attributes', allow_invalid_attributes)

    @classmethod
    def _constructor(cls, mapping):
        return cls(mapping)

    # --- taken from MutableAttr

    def _setattr(self, key, value):
        """
        Add an attribute to the object, without attempting to add it as
        a key to the mapping (i.e. internals)
        """
        super(MutableMapping, self).__setattr__(key, value)

    def __setattr__(self, key, value):
        """
        Add an attribute.

        key: The name of the attribute
        value: The attributes contents
        """
        if self._valid_name(key):
            self[key] = value
        elif getattr(self, '_allow_invalid_attributes', True):
            super(MutableMapping, self).__setattr__(key, value)
        else:
            raise TypeError(
                "'{cls}' does not allow attribute creation.".format(
                    cls=self.__class__.__name__
                )
            )

    def _delattr(self, key):
        """
        Delete an attribute from the object, without attempting to
        remove it from the mapping (i.e. internals)
        """
        super(MutableMapping, self).__delattr__(key)

    def __delattr__(self, key, force=False):
        """
        Delete an attribute.

        key: The name of the attribute
        """
        if self._valid_name(key):
            del self[key]
        elif getattr(self, '_allow_invalid_attributes', True):
            super(MutableMapping, self).__delattr__(key)
        else:
            raise TypeError(
                "'{cls}' does not allow attribute deletion.".format(
                    cls=self.__class__.__name__
                )
            )

    def __call__(self, key):
        """
        Dynamically access a key-value pair.

        key: A key associated with a value in the mapping.

        This differs from __getitem__, because it returns a new instance
        of an Attr (if the value is a Mapping object).
        """
        if key not in self:
            raise AttributeError(
                "'{cls} instance has no attribute '{name}'".format(
                    cls=self.__class__.__name__, name=key
                )
            )

        return self._build(self[key])

    def __getattr__(self, key):
        """
        Access an item as an attribute.
        """
        if key not in self or not self._valid_name(key):
            raise AttributeError(
                "'{cls}' instance has no attribute '{name}'".format(
                    cls=self.__class__.__name__, name=key
                )
            )

        return self._build(self[key])

    @classmethod
    def _valid_name(cls, key):
        """
        Check whether a key is a valid attribute name.

        A key may be used as an attribute if:
         * It is a string
         * The key doesn't overlap with any class attributes (for Attr,
            those would be 'get', 'items', 'keys', 'values', 'mro', and
            'register').
        """
        return (
            isinstance(key, six.string_types) and
            not hasattr(cls, key)
        )


class pandapowerNet(ADict):
    def __init__(self, *args, **kwargs):
        super(pandapowerNet, self).__init__(*args, **kwargs)

    def __repr__(self):
        r = "This pandapower network includes the following parameter tables:"
        par = []
        res = []
        for tb in list(self.keys()):
            if isinstance(self[tb], pd.DataFrame) and len(self[tb]) > 0:
                if 'res_' in tb:
                    res.append(tb)
                else:
                    par.append(tb)
        for tb in par:
            r += "\n   - %s (%s elements)" % (tb, len(self[tb]))
        if res:
            r += "\n and the following results tables:"
            for tb in res:
                r += "\n   - %s (%s elements)" % (tb, len(self[tb]))
        return r


def _preserve_dtypes(df, dtypes):
    for item, dtype in list(dtypes.iteritems()):
        if df.dtypes.at[item] != dtype:
            try:
                df[item] = df[item].astype(dtype)
            except ValueError:
                df[item] = df[item].astype(float)


def get_free_id(df):
    """
    Returns next free ID in a dataframe
    """
    return np.int64(0) if len(df) == 0 else df.index.values.max() + 1


class ppException(Exception):
    """
    General pandapower custom parent exception.
    """
    pass


def _sum_by_group(bus, first_val, second_val):
    order = np.argsort(bus)
    bus = bus[order]
    index = np.ones(len(bus), 'bool')
    index[:-1] = bus[1:] != bus[:-1]
    bus = bus[index]
    first_val = first_val[order]
    first_val.cumsum(out=first_val)
    first_val = first_val[index]
    first_val[1:] = first_val[1:] - first_val[:-1]
    second_val = second_val[order]
    second_val.cumsum(out=second_val)
    second_val = second_val[index]
    second_val[1:] = second_val[1:] - second_val[:-1]
    return bus, first_val, second_val


def get_indices(selection, lookup, fused_indices=True):
    """
    Helper function during pd2mpc conversion. It resolves the mapping from a
    given selection of indices to the actual indices, using a dict lookup being
    passed as well.

    :param selection: Indices we want to select
    :param lookup: The mapping itself
    :param fused_indices: Flag which way the conversion is working.
    :return:
    """
    if fused_indices:
        return np.array([lookup[k] for k in selection], dtype="int")
    else:
        return np.array([lookup["before_fuse"][k] for k in selection], dtype="int")


def get_values(source, selection, lookup):
    """
    Returns values for a selection of values after a lookup.

    :param source: The array of values to select from.
    :param selection: An array of keys, for the selection.
    :param lookup: The mapping to resolve actual indices of the
    value array from the selection.
    :return:
    """
    return np.array([source[lookup[k]] for k in selection])


def _select_is_elements(net):
    """
    Selects certain "in_service" elements from net.
    This is quite time consuming so it is done once at the beginning


    @param net: pandapower Network
    @return: _is_elements Certain in service elements
    :rtype: object
    """
    recycle = net["_options"]["recycle"]

#    if recycle is not None and recycle["_is_elements"]:
#        if "_is_elements" not in net or net["_is_elements"] is None:
#            # sort elements according to their in service status
#            elems = ['bus', 'line']
#            for elm in elems:
#                net[elm] = net[elm].sort_values(by=['in_service'], ascending=0)
#
#            # select in service buses. needed for the other elements to be selected
#            bus_is = net["bus"]["in_service"].values.astype(bool)
#            line_is = net["line"]["in_service"].values.astype(bool)
#            bus_is_ind = net["bus"][bus_is].index
#            # check if in service elements are at in service buses
#            _is_elements = {
#                "gen": net['gen'][np.in1d(net["gen"].bus.values, bus_is_ind) \
#                                  & net["gen"]["in_service"].values.astype(bool)]
#                , "load": np.in1d(net["load"].bus.values, bus_is_ind) \
#                          & net["load"].in_service.values.astype(bool)
#                , "sgen": np.in1d(net["sgen"].bus.values, bus_is_ind) \
#                          & net["sgen"].in_service.values.astype(bool)
#                , "ward": np.in1d(net["ward"].bus.values, bus_is_ind) \
#                          & net["ward"].in_service.values.astype(bool)
#                , "xward": np.in1d(net["xward"].bus.values, bus_is_ind) \
#                           & net["xward"].in_service.values.astype(bool)
#                , "shunt": np.in1d(net["shunt"].bus.values, bus_is_ind) \
#                           & net["shunt"].in_service.values.astype(bool)
#                , "ext_grid": net["ext_grid"][np.in1d(net["ext_grid"].bus.values, bus_is_ind) \
#                                              & net["ext_grid"]["in_service"].values.astype(bool)]
#                , 'bus': net['bus'].iloc[:np.count_nonzero(bus_is)]
#                , 'line': net['line'].iloc[:np.count_nonzero(line_is)]
#            }
#        else:
#            # just update the elements
#            _is_elements = net['_is_elements']
#
#            bus_is_ind = _is_elements['bus'].index
#            # update elements
#            elems = ['gen', 'ext_grid']
#            for elm in elems:
#                _is_elements[elm] = net[elm][np.in1d(net[elm].bus.values, bus_is_ind) \
#                                             & net[elm]["in_service"].values.astype(bool)]
#
#    else:
    # select in service buses. needed for the other elements to be selected
    bus_is_mask = net["bus"]["in_service"].values.astype(bool)
    bus_is = net["bus"][bus_is_mask]
    bus_is_idx = bus_is.index
    line_is_mask = net["line"]["in_service"].values.astype(bool)
    # check if in service elements are at in service buses
    _is_elements = {
        "gen": np.in1d(net["gen"].bus.values, bus_is_idx) \
               & net["gen"]["in_service"].values.astype(bool)
        , "load": np.in1d(net["load"].bus.values, bus_is_idx) \
                  & net["load"].in_service.values.astype(bool)
        , "sgen": np.in1d(net["sgen"].bus.values, bus_is_idx) \
                  & net["sgen"].in_service.values.astype(bool)
        , "ward": np.in1d(net["ward"].bus.values, bus_is_idx) \
                  & net["ward"].in_service.values.astype(bool)
        , "xward": np.in1d(net["xward"].bus.values, bus_is_idx) \
                   & net["xward"].in_service.values.astype(bool)
        , "shunt": np.in1d(net["shunt"].bus.values, bus_is_idx) \
                   & net["shunt"].in_service.values.astype(bool)
        , "ext_grid": np.in1d(net["ext_grid"].bus.values, bus_is_idx) \
                   & net["ext_grid"]["in_service"].values.astype(bool)
        , "bus_is_idx": bus_is_idx
        , 'line': net['line'][line_is_mask]
    }

    return _is_elements

def _create_ppc2pd_bus_lookup(net):
    # pd to ppc lookup
    pd2ppc_bus_lookup = net["_pd2ppc_lookups"]["bus"]
    # valid entries in pd2ppc lookup
    valid_entries = pd2ppc_bus_lookup >= 0
    # init reverse (ppc2pd) lookup with -1
    ppc2pd_bus_lookup = np.ones(max(pd2ppc_bus_lookup[valid_entries]) + 1, dtype=int) * -1
    # index of pd2ppc lookup
    ind_pd2ppc_bus_lookup = np.array(range(len(pd2ppc_bus_lookup)), dtype=int)
    # update reverse lookup
    ppc2pd_bus_lookup[pd2ppc_bus_lookup[valid_entries]] = ind_pd2ppc_bus_lookup[valid_entries]
    # store reverse lookup innet
    net["_ppc2pd_lookups"]["bus"] = ppc2pd_bus_lookup

def _check_connectivity(ppc):
    """
    Checks if the ppc contains isolated buses. If yes this isolated buses are set out of service
    :param ppc: pypower case file
    :return:
    """
    nobranch = ppc['branch'].shape[0]
    nobus = ppc['bus'].shape[0]
    bus_from = ppc['branch'][:, F_BUS].real.astype(int)
    bus_to = ppc['branch'][:, T_BUS].real.astype(int)

    slacks = ppc['bus'][ppc['bus'][:, BUS_TYPE] == 3, BUS_I]

    # we create a "virtual" bus thats connected to all slack nodes and start the connectivity
    # search at this bus
    bus_from = np.hstack([bus_from, slacks])
    bus_to = np.hstack([bus_to, np.ones(len(slacks)) * nobus])

    adj_matrix = sp.sparse.coo_matrix((np.ones(nobranch + len(slacks)),
                                      (bus_from, bus_to)),
                                      shape=(nobus + 1, nobus + 1))

    reachable = sp.sparse.csgraph.breadth_first_order(adj_matrix, nobus, False, False)
    # TODO: the former impl. excluded ppc buses that are already oos, but is this necessary ?
    # if so: bus_not_reachable = np.hstack([ppc['bus'][:, BUS_TYPE] != 4, np.array([False])])
    bus_not_reachable = np.ones(ppc["bus"].shape[0] + 1, dtype=bool)
    bus_not_reachable[reachable] = False
    isolated_nodes = np.where(bus_not_reachable)[0]
    if len(isolated_nodes) > 0:
        logger.debug("There are isolated buses in the network!")
        # set buses in ppc out of service
        ppc['bus'][isolated_nodes, BUS_TYPE] = NONE

        pus = abs(ppc['bus'][isolated_nodes, PD] * 1e3).sum()
        qus = abs(ppc['bus'][isolated_nodes, QD] * 1e3).sum()
        if pus > 0 or qus > 0:
            logger.debug("%.0f kW active and %.0f kVar reactive power are unsupplied" % (pus, qus))
    else:
        pus = qus = 0
    return isolated_nodes, pus, qus


def _create_ppc2pd_bus_lookup(net):
    # pd to ppc lookup
    pd2ppc_bus_lookup = net["_pd2ppc_lookups"]["bus"]
    # valid entries in pd2ppc lookup
    valid_entries = pd2ppc_bus_lookup >= 0
    # init reverse (ppc2pd) lookup with -1
    ppc2pd_bus_lookup = np.ones(max(pd2ppc_bus_lookup[valid_entries]) + 1, dtype=int) * -1
    # index of pd2ppc lookup
    ind_pd2ppc_bus_lookup = np.array(range(len(pd2ppc_bus_lookup)), dtype=int)
    # update reverse lookup
    ppc2pd_bus_lookup[pd2ppc_bus_lookup[valid_entries]] = ind_pd2ppc_bus_lookup[valid_entries]
    # store reverse lookup innet
    net["_ppc2pd_lookups"]["bus"] = ppc2pd_bus_lookup


from numba import jit
@jit(nopython=True, cache=True)
def set_elements_oos(ti, tis, bis, lis):
    """iterates over elements; returns array where element is of service if element is oos in
    element table or bus is oos"""
    for i in range(len(ti)):
        if tis[i] and bis[ti[i]]:
            lis[i] = True


@jit(nopython=True, cache=True)
def set_isolated_buses_oos(bus_in_service, ppc_bus_isolated, bus_lookup):
    """determines out of service pp buses by also checking if fused to isolated ppc buses"""
    for k in range(len(bus_lookup)):
        if ppc_bus_isolated[bus_lookup[k]]:
            bus_in_service[k] = False


def _select_is_elements_numba(net, isolated_nodes=None):
    max_bus_idx = np.max(net["bus"].index.values)
    bus_in_service = np.zeros(max_bus_idx + 1, dtype=bool)
    bus_in_service[net["bus"].index.values] = net["bus"]["in_service"].values.astype(bool)
    if isolated_nodes is not None and len(isolated_nodes) > 0:
        ppc_bus_isolated = np.zeros(net["_ppc"]["bus"].shape[0], dtype=bool)
        ppc_bus_isolated[isolated_nodes] = True
        set_isolated_buses_oos(bus_in_service, ppc_bus_isolated, net["_pd2ppc_lookups"]["bus"])

    is_elements = dict()
    for element in ["load", "sgen", "gen", "ward", "xward", "shunt", "ext_grid"]:
        len_ = len(net[element].index)
        element_in_service = np.zeros(len_, dtype=bool)
        if len_ > 0:
            element_df = net[element]
            set_elements_oos(element_df["bus"].values, element_df["in_service"].values,
                             bus_in_service, element_in_service)
        is_elements[element] = element_in_service
    is_elements["bus_is_idx"] = net["bus"].index.values[bus_in_service[net["bus"].index.values]]
    is_elements["line"] = net["line"][net["line"]["in_service"].values.astype(bool)]
    return is_elements


def _add_ppc_options(net, calculate_voltage_angles, trafo_model, check_connectivity, mode,
                     copy_constraints_to_ppc, r_switch, init, enforce_q_lims, recycle, voltage_depend_loads=False):
    """
    creates dictionary for pf, opf and short circuit calculations from input parameters.
    """
    if recycle == None:
        recycle = dict(_is_elements=False, ppc=False, Ybus=False, bfsw=False)

    options = {
        "calculate_voltage_angles": calculate_voltage_angles
        , "trafo_model": trafo_model
        , "check_connectivity": check_connectivity
        , "mode": mode
        , "copy_constraints_to_ppc": copy_constraints_to_ppc
        , "r_switch": r_switch
        , "init": init
        , "enforce_q_lims": enforce_q_lims
        , "recycle": recycle
        , "voltage_depend_loads": voltage_depend_loads
    }
    _add_options(net, options)


def _add_pf_options(net, tolerance_kva, trafo_loading, numba, ac,
                    algorithm, max_iteration, **kwargs):
    """
    creates dictionary for pf, opf and short circuit calculations from input parameters.
    """

    options = {
        "tolerance_kva": tolerance_kva
        , "trafo_loading": trafo_loading
        , "numba": numba
        , "ac": ac
        , "algorithm": algorithm
        , "max_iteration": max_iteration
    }

    options.update(kwargs)  # update options with some algorithm-specific parameters
    _add_options(net, options)


def _add_opf_options(net, trafo_loading, ac, **kwargs):
    """
    creates dictionary for pf, opf and short circuit calculations from input parameters.
    """
    options = {
        "trafo_loading": trafo_loading
        , "ac": ac
    }

    options.update(kwargs)  # update options with some algorithm-specific parameters
    _add_options(net, options)


def _add_sc_options(net, fault, case, lv_tol_percent, tk_s, topology, r_fault_ohm,
                    x_fault_ohm, kappa, ip, ith, consider_sgens, branch_results, kappa_method):
    """
    creates dictionary for pf, opf and short circuit calculations from input parameters.
    """
    options = {
        "fault": fault
        , "case": case
        , "lv_tol_percent": lv_tol_percent
        , "tk_s": tk_s
        , "topology": topology
        , "r_fault_ohm": r_fault_ohm
        , "x_fault_ohm": x_fault_ohm
        , "kappa": kappa
        , "ip": ip
        , "ith": ith
        , "consider_sgens": consider_sgens
        , "branch_results": branch_results
        , "kappa_method": kappa_method
    }
    _add_options(net, options)


def _add_options(net, options):
    double_parameters = set(net._options.keys()) & set(options.keys())
    if len(double_parameters) > 0:
        raise UserWarning(
            "Parameters always have to be unique! The following parameters where specified twice: %s" % double_parameters)
    net._options.update(options)


def _clean_up(net):
    mode = net._options["mode"]
    res_bus = net["res_bus_sc"] if mode == "sc" else net["res_bus"]
    if len(net["trafo3w"]) > 0:
        buses_3w = net.trafo3w["ad_bus"].values
        res_bus.drop(buses_3w, inplace=True)
        net["bus"].drop(buses_3w, inplace=True)
        net["trafo3w"].drop(["ad_bus"], axis=1, inplace=True)

    if len(net["xward"]) > 0:
        xward_buses = net["xward"]["ad_bus"].values
        net["bus"].drop(xward_buses, inplace=True)
        res_bus.drop(xward_buses, inplace=True)
        net["xward"].drop(["ad_bus"], axis=1, inplace=True)

    if len(net["dcline"]) > 0:
        dc_gens = net.gen.index[(len(net.gen) - len(net.dcline) * 2):]
        net.gen.drop(dc_gens, inplace=True)
        net.res_gen.drop(dc_gens, inplace=True)


def _set_isolated_buses_out_of_service(net, ppc):
    # set disconnected buses out of service
    # first check if buses are connected to branches
    disco = np.setxor1d(ppc["bus"][:, 0].astype(int),
                        ppc["branch"][ppc["branch"][:, 10] == 1, :2].real.astype(int).flatten())

    # but also check if they may be the only connection to an ext_grid
    disco = np.setdiff1d(disco, ppc['bus'][ppc['bus'][:, 1] == 3, :1].real.astype(int))
    ppc["bus"][disco, 1] = 4


def _write_lookup_to_net(net, element, element_lookup):
    """
    Updates selected lookups in net
    """
    net["_pd2ppc_lookups"][element] = element_lookup


def _remove_isolated_elements_from_is_elements(net, isolated_nodes):
    if len(isolated_nodes) == 0:
        return
    ppc2pd_bus_lookup = net["_ppc2pd_lookups"]["bus"]
    _is_elements = net["_is_elements"]
    pp_nodes = [n for n in isolated_nodes if not (n > len(ppc2pd_bus_lookup)-1)]
    isolated_nodes_pp = ppc2pd_bus_lookup[pp_nodes]
    # remove isolated buses from _is_elements["bus"]
    _is_elements["bus_is_idx"] = np.setdiff1d(_is_elements["bus_is_idx"], isolated_nodes_pp)#_is_elements["bus"].drop(set(isolated_nodes_pp) & set(_is_elements["bus"].index))
    bus_is_ind = _is_elements["bus_is_idx"]
    # check if in service elements are at in service buses

    elems_to_update = ["load", "sgen", "gen", "ward", "xward", "shunt"]
    for elem in elems_to_update:
        _is_elements[elem] = np.in1d(net[elem].bus.values, bus_is_ind) \
                             & net[elem].in_service.values.astype(bool)

    net["_is_elements"] = _is_elements


def calculate_line_results(net, use_res_bus_est=False):
    """
    Calculates complex line currents, powers at both bus sides and saves them in the result table.
    Requires the res_bus or res_bus_est table of the network to be filled.
    :param net: pandapower network
    :param use_res_bus_est: use res_bus_est dataframe instead of res_bus
    :return: new dataframe, which can be assigned to either res_line or res_line_est
    """
    res_line = pd.DataFrame(columns=["p_from_kw", "q_from_kvar", "p_to_kw", "q_to_kvar", "pl_kw",
                                     "ql_kvar", "i_from_ka", "i_to_ka", "i_ka", "loading_percent"],
                            index=net.line.index)
    # calculate impedances and complex voltages
    Zij = net.line['length_km'] * (net.line['r_ohm_per_km'] + 1j * net.line['x_ohm_per_km'])
    Zcbij = 0.5j * 2 * np.pi * 50 * net.line['c_nf_per_km'] * 1e-9
    if use_res_bus_est:
        V = net.res_bus_est.vm_pu * net.bus.vn_kv * 1e3 * np.exp(
            1j * np.pi / 180 * net.res_bus_est.va_degree)
    else:
        V = net.res_bus.vm_pu * net.bus.vn_kv * 1e3 * np.exp(
            1j * np.pi / 180 * net.res_bus.va_degree)
    fb = net.line.from_bus
    tb = net.line.to_bus
    # calculate line currents of from bus side
    line_currents_from = ((V[fb].values - V[tb].values) / np.sqrt(3) / Zij + V[fb].values
                          * Zcbij).values
    open_lines_from = net.switch.element.loc[(net.switch.et == 'l') & (net.switch.closed == False)]
    line_currents_from[open_lines_from.values] = 0.
    charging_from = open_lines_from[open_lines_from.index[
        net.line.to_bus.loc[open_lines_from].values ==
        net.switch.bus.loc[(net.switch.et == 'l') & (net.switch.closed == False)].values]].values
    line_currents_from[charging_from] = V[net.line.ix[charging_from].from_bus].values \
                                        * Zcbij[charging_from] * (1 + Zij[charging_from])
    # calculate line currents on to bus side
    line_currents_to = ((V[tb].values - V[fb].values) / np.sqrt(3) / Zij + V[tb].values
                        * Zcbij).values
    open_lines_to = net.switch.element.loc[(net.switch.et == 'l') & (net.switch.closed == False)]
    line_currents_to[open_lines_to.values] = 0.
    charging_to = open_lines_to[open_lines_to.index[
        net.line.from_bus.loc[open_lines_to].values ==
        net.switch.bus.loc[(net.switch.et == 'l') & (net.switch.closed == False)].values]].values
    line_currents_to[charging_to] = V[net.line.ix[charging_to].to_bus] * Zcbij[charging_to] \
                                    * (1 + Zij[charging_to])
    # derive other values
    line_powers_from = np.sqrt(3) * V[fb].values * np.conj(line_currents_from) / 1e3
    line_powers_to = np.sqrt(3) * V[tb].values * np.conj(line_currents_to) / 1e3
    res_line.i_from_ka = np.abs(line_currents_from) / 1e3
    res_line.i_to_ka = np.abs(line_currents_to) / 1e3
    res_line.i_ka = np.fmax(res_line.i_from_ka, res_line.i_to_ka)
    res_line.loading_percent = res_line.i_ka * 100. / net.line.max_i_ka.values \
                               / net.line.df.values / net.line.parallel.values
    res_line.p_from_kw = line_powers_from.real
    res_line.q_from_kvar = line_powers_from.imag
    res_line.p_to_kw = line_powers_to.real
    res_line.q_to_kvar = line_powers_to.imag
    res_line.pl_kw = res_line.p_from_kw + res_line.p_to_kw
    res_line.ql_kvar = res_line.q_from_kvar + res_line.q_to_kvar
    return res_line

