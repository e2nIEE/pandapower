"""
Created on Fri Jul  1 13:40:07 2016

@author: thurner
"""
import os
import shutil

from pandapower.std_types import available_std_types
from pandapower.network import pandapowerNet
from pandapower.network_structure import get_structure_dict

file_path = os.path.dirname(os.path.realpath(__file__))


def save_std_types(_):
    path = os.path.join(file_path, '..', 'std_types', 'tables')
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)
    print("Generating std_types csv files")

    structure_dict = get_structure_dict(required_only=False)
    net = pandapowerNet(name="save_pp_std_types")

    for type_ in net.std_types.keys():
        types = available_std_types(net, type_)
        if type_ in structure_dict:
            columns = [c for c in types.columns if structure_dict[type_]]
        else:
            columns = types.columns  # fuse is not contained in structure dict
        types = types.reindex(columns, axis=1)
        types.to_csv(os.path.join(path, f"{type_}_std_types.csv"), sep=";")


def setup(app):
    app.connect('builder-inited', save_std_types)
    return {'version': '0.1'}


if __name__ == '__main__':
    save_std_types(None)
