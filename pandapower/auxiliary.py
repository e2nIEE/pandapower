# -*- coding: utf-8 -*-

"""
__author__ = 'jdollichon, tdess, lthurner, ascheidler'

"""
from __future__ import division
import pandas as pd
from attrdict import AttrDict
import numpy as np

class PandapowerNet(AttrDict):
    def __init__(self, *args, **kwargs):
        super(PandapowerNet, self).__init__(*args, **kwargs)

    def __repr__(self):
        r = "This pandapower network includes the following parameter tables:"
        par = []
        res = []
        for tb in list(self.keys()):
            if isinstance(self[tb], pd.DataFrame) and len(self[tb])>0:
                if 'res_' in tb:
                    res.append(tb)
                else:
                    par.append(tb)
        for tb in par:
            r += "\n   - %s (%s elements)" % (tb,len(self[tb]))
        if res:
            r += "\n and the following results tables:"
            for tb in res:
                r += "\n   - %s (%s elements)" % (tb,len(self[tb]))
        return r

    def _build(self, obj, **kwargs):
        """
        We only want dict like elements to be treated as recursive AttrDicts.
        """
        return obj

def _preserve_dtypes(df, dtypes):
    for item, dtype in dtypes.items():
        if df.dtypes.at[item] != dtype:
            try:
                df[item] = df[item].astype(dtype)
            except ValueError:
                df[item] = df[item].astype(float)
                
def get_free_id(df):
    """
    Returns next free ID in a dataframe
    """
    return np.int64(0) if len(df)==0 else df.index.values.max() + 1
        

class HpException(Exception):
    """
    General honeypot-wide custom parent exception.
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