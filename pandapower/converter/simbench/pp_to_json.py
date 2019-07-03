"""
This functions allow saving pandapower grids including profile extension to excel files.
"""

from copy import deepcopy
import pandas as pd
import pandapower as pp
import os

__author__ = 'smeinecke'


def to_json(net1, path=None, keep=True):
    """
    Replaces additional dict data in net["profiles"] and net["std_types"] into a DataFrame in net
    to prevent hp.pandapower.io_utils - WARNING: Attribute net.profiles could not be saved !
    """
    net = deepcopy(net1) if keep else net1
    # make dict to DataFrame changes
    elm = 'profiles'
    for key in net[elm].keys():
        net[key + '_' + elm] = net[elm][key]
    del net["profiles"]
    elm = 'std_types'
    for key in ['renewables', 'load']:
        if key in net[elm].keys():
            net[elm + '_' + key] = pd.DataFrame(net[elm][key])
            del net["std_types"][key]
    # save to json
    if isinstance(path, str):
        pp.to_json(net, path)
    return net


if __name__ == '__main__':
    if 1:
        pass
    else:
        pass
