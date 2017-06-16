# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

# Additional copyright for modified code by Brendan Curran-Johnson (ADict class):
# Copyright (c) 2013 Brendan Curran-Johnson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# (https://github.com/bcj/AttrDict/blob/master/LICENSE.txt)

from collections import MutableMapping

import numpy as np
import pandas as pd
import scipy as sp
import six

from pandapower.idx_brch import F_BUS, T_BUS
from pandapower.idx_bus import BUS_I, BUS_TYPE, NONE, PD, QD

try:
    from numba import _version as numba_version
except:
    pass

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

    def __repr__(self):  # pragma: no cover
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
    return np.array([source[lookup[np.int(k)]] for k in selection])


def _select_is_elements(net):
    """
    Selects certain "in_service" elements from net.
    This is quite time consuming so it is done once at the beginning


    @param net: pandapower Network
    @return: _is_elements Certain in service elements
    :rtype: object
    """
    #    recycle = net["_options"]["recycle"]

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


from numba import jit


@jit(nopython=True, cache=True)
def set_elements_oos(ti, tis, bis, lis):  # pragma: no cover
    """iterates over elements; returns array where element is of service if element is oos in
    element table or bus is oos"""
    for i in range(len(ti)):
        if tis[i] and bis[ti[i]]:
            lis[i] = True


@jit(nopython=True, cache=True)
def set_isolated_buses_oos(bus_in_service, ppc_bus_isolated, bus_lookup):  # pragma: no cover
    """determines out of service pp buses by also checking if fused to isolated ppc buses"""
    for k in range(len(bus_lookup)):
        if ppc_bus_isolated[bus_lookup[k]]:
            bus_in_service[k] = False


def _select_is_elements_numba(net, isolated_nodes=None):
    # is missing sgen_controllable and load_controllable
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

    if net["_options"]["mode"] == "opf":
        is_elements["load_controllable"] = net._is_elements["load_controllable"]
        is_elements["sgen_controllable"] = net._is_elements["sgen_controllable"]
    return is_elements


def _add_ppc_options(net, calculate_voltage_angles, trafo_model, check_connectivity, mode,
                     copy_constraints_to_ppc, r_switch, init, enforce_q_lims, recycle, delta=1e-10,
                     voltage_depend_loads=False):
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
        , "delta": delta
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


def _check_if_numba_is_installed(numba, check_connectivity):
    try:
        # get numba Version (in order to use it it must be > 0.25)
        nb_version = float(numba_version.version_version[:4])
        if nb_version < 0.25:
            logger.warning('Warning: Numba version too old -> Upgrade to a version > 0.25. Numba is disabled\n')
            numba = False

    except:
        logger.warning('Warning: Numba cannot be imported.'
                       ' Numba is disabled. Call runpp() with numba=False to avoid this warning!\n')
        if check_connectivity:
            logger.warning('Connectivity check is based on numba and is disabled since numba could not be imported')
            check_connectivity = False
        numba = False

    return numba, check_connectivity
