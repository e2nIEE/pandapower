# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:07:58 2019

@author: thurner
"""

from pandapower import pp_dir
import pandapower as pp
import numpy as np
import os
import pytest

def test_convert_format():
    path = os.path.join(pp_dir, "test", "test_files", "old_versions")
    for subdir, dirs, files in os.walk(path):
        for file in files:
            version = file.split("_")[1].split(".json")[0]
            filename = os.path.join(subdir, file)
            try:
                net = pp.from_json(filename, convert=False)
            except:
                raise UserWarning("Can not load network saved in pandapower version %s"%version)
            net.res_bus.sort(inplace=True)
            net.bus.sort(inplace=True)
            vm_pu_old = net.res_bus.vm_pu.copy()
            pp.convert_format(net)
            net.trafo["tap_step_degree"] = np.nan
            try:
                pp.runpp(net)
            except:
                raise UserWarning("Can not run power flow in network saved with pandapower version %s"%version)
            if not np.allclose(vm_pu_old.values, net.res_bus.vm_pu.values):
                raise UserWarning("Power flow results mismatch with pandapower version %s"%version)
                
if __name__ == '__main__':
#    path = os.path.join(pp_dir, "test", "test_files", "old_versions", "example_1.6.1.json")
#    net = pp.from_json(path, convert=False)
    pytest.main(__file__)