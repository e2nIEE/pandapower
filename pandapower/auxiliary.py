# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import pandas as pd
import numpy as np
from collections import MutableMapping
import six

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


class PandapowerNet(ADict):

    def __init__(self, *args, **kwargs):
        super(PandapowerNet, self).__init__(*args, **kwargs)

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
    
def _select_is_elements(net, recycle=None):

    """
    Selects certain "in_service" elements from net.
    This is quite time consuming so it is done once at the beginning


    @param net: Pandapower Network
    @return: is_elems Certain in service elements
    """

    if recycle is not None and recycle["is_elems"]:
        if "_is_elems" not in net or net["_is_elems"] is None:
            # sort elements according to their in service status
            elems = ['bus', 'line']
            for elm in elems:
                net[elm] = net[elm].sort_values(by=['in_service'], ascending=0)

            # select in service buses. needed for the other elements to be selected
            bus_is = net["bus"]["in_service"].values.astype(bool)
            line_is = net["line"]["in_service"].values.astype(bool)
            bus_is_ind = net["bus"][bus_is].index
            # check if in service elements are at in service buses
            is_elems = {
                "gen": net['gen'][np.in1d(net["gen"].bus.values, bus_is_ind) \
                                  & net["gen"]["in_service"].values.astype(bool)]
                , "load": np.in1d(net["load"].bus.values, bus_is_ind) \
                          & net["load"].in_service.values.astype(bool)
                , "sgen": np.in1d(net["sgen"].bus.values, bus_is_ind) \
                          & net["sgen"].in_service.values.astype(bool)
                , "ward": np.in1d(net["ward"].bus.values, bus_is_ind) \
                          & net["ward"].in_service.values.astype(bool)
                , "xward": np.in1d(net["xward"].bus.values, bus_is_ind) \
                           & net["xward"].in_service.values.astype(bool)
                , "shunt": np.in1d(net["shunt"].bus.values, bus_is_ind) \
                           & net["shunt"].in_service.values.astype(bool)
                , "ext_grid": net["ext_grid"][np.in1d(net["ext_grid"].bus.values, bus_is_ind) \
                                        & net["ext_grid"]["in_service"].values.astype(bool)]
                , 'bus': net['bus'].iloc[:np.count_nonzero(bus_is)]
                , 'line': net['line'].iloc[:np.count_nonzero(line_is)]
            }
        else:
            # just update the elements
            is_elems = net['_is_elems']

            bus_is_ind = is_elems['bus'].index
            #update elements
            elems = ['gen', 'ext_grid']
            for elm in elems:
                is_elems[elm] = net[elm][np.in1d(net[elm].bus.values, bus_is_ind) \
                                     & net[elm]["in_service"].values.astype(bool)]

    else:
        # select in service buses. needed for the other elements to be selected
        bus_is = net["bus"]["in_service"].values.astype(bool)
        line_is = net["line"]["in_service"].values.astype(bool)
        bus_is_ind = net["bus"][bus_is].index
        # check if in service elements are at in service buses
        is_elems = {
            "gen" : net['gen'][np.in1d(net["gen"].bus.values, bus_is_ind) \
                    & net["gen"]["in_service"].values.astype(bool)]
            , "load" : np.in1d(net["load"].bus.values, bus_is_ind) \
                    & net["load"].in_service.values.astype(bool)
            , "sgen" : np.in1d(net["sgen"].bus.values, bus_is_ind) \
                    & net["sgen"].in_service.values.astype(bool)
            , "ward" : np.in1d(net["ward"].bus.values, bus_is_ind) \
                    & net["ward"].in_service.values.astype(bool)
            , "xward" : np.in1d(net["xward"].bus.values, bus_is_ind) \
                    & net["xward"].in_service.values.astype(bool)
            , "shunt" : np.in1d(net["shunt"].bus.values, bus_is_ind) \
                    & net["shunt"].in_service.values.astype(bool)
            , "ext_grid" : net["ext_grid"][np.in1d(net["ext_grid"].bus.values, bus_is_ind) \
                    & net["ext_grid"]["in_service"].values.astype(bool)]
            , 'bus': net['bus'][bus_is]
            , 'line': net['line'][line_is]
        }

    return is_elems
    
def _clean_up(net):
    if len(net["trafo3w"]) > 0:
        buses_3w = net.trafo3w["ad_bus"].values
        net["res_bus"].drop(buses_3w, inplace=True)
        net["bus"].drop(buses_3w, inplace=True)
        net["trafo3w"].drop(["ad_bus"], axis=1, inplace=True)

    if len(net["xward"]) > 0:
        xward_buses = net["xward"]["ad_bus"].values
        net["bus"].drop(xward_buses, inplace=True)
        net["res_bus"].drop(xward_buses, inplace=True)
        net["xward"].drop(["ad_bus"], axis=1, inplace=True)


def _set_isolated_buses_out_of_service(net, ppc):
    # set disconnected buses out of service
    # first check if buses are connected to branches
    disco = np.setxor1d(ppc["bus"][:, 0].astype(int),
                        ppc["branch"][ppc["branch"][:, 10] == 1, :2].real.astype(int).flatten())

    # but also check if they may be the only connection to an ext_grid
    disco = np.setdiff1d(disco, ppc['bus'][ppc['bus'][:, 1] == 3, :1].real.astype(int))
    ppc["bus"][disco, 1] = 4
    
