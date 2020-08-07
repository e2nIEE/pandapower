# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pytest
import copy
from numpy import array
from pandapower.converter.pypower import from_ppc

import pandapower as pp
from pandapower.convert_format import convert_format
from pandapower.networks import case5

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def case5_pm_matfile_I():
    
    mpc = {"branch": array, "bus": array, "gen": array, "gencost": array,
           "version": 2, "baseMVA": 100.0}
    # mpc.version = '2';
    # mpc.baseMVA = 100.0;

    # %% bus data
    # %	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
    mpc["bus"] = array([
        [1, 2, 0.0, 0.0, 0.0, 0.0, 1, 1.00000, 2.80377, 230.0, 1, 1.10000, 0.90000],
        [2, 1, 300.0, 98.61, 0.0, 0.0, 1, 1.08407, 0.73465, 230.0, 1, 1.10000, 0.90000],
        [3, 2, 300.0, 98.61, 0.0, 0.0, 1, 1.00000, 0.55972, 230.0, 1, 1.10000, 0.90000],
        [4, 3, 400.0, 131.47, 0.0, 0.0, 1, 1.00000, 0.00000, 230.0, 1, 1.10000, 0.90000],
        [10, 2, 0.0, 0.0, 0.0, 0.0, 1, 1.00000, 3.59033, 230.0, 1, 1.10000, 0.90000],
        ])


# %% generator data
# %	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin
    mpc["gen"] = array([
       [1, 40.0, 30.0, 30.0, -30.0, 1.07762, 100.0, 1, 40.0, 0.0],
       [1, 170.0, 127.5, 127.5, -127.5, 1.07762, 100.0, 1, 170.0, 0.0],
       [3, 324.49, 390.0, 390.0, -390.0, 1.1, 100.0, 1, 520.0, 0.0],
       [4, 0.0, -10.802, 150.0, -150.0, 1.06414, 100.0, 1, 200.0, 0.0],
       [10, 470.69,	-165.039, 450.0, -450.0, 1.06907, 100.0, 1, 600.0, 0.0],
    ])
# %% generator cost data
# %	2	startup	shutdown	n	c(n-1)	...	c0
    mpc["gencost"] = array([
        [2, 0.0, 0.0, 3, 0.000000, 14.000000, 0.000000],
        [2, 0.0, 0.0, 3, 0.000000, 15.000000, 0.000000],
        [2, 0.0, 0.0, 3, 0.000000, 30.000000, 0.000000],
        [2, 0.0, 0.0, 3, 0.000000, 40.000000, 0.000000],
        [2, 0.0, 0.0, 3, 0.000000, 10.000000, 0.000000],
    ])

# %% branch data
# %	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
    mpc["branch"] = array([
       [1, 2, 0.00281, 0.0281, 0.00712,    400.0, 0.0, 0.0, 0.0, 0.0, 1, -30.0, 30.0],
       [1, 4, 0.00304, 0.0304, 0.00658,    426, 0.0, 0.0, 0.0, 0.0, 1, -30.0, 30.0],
       [1, 10, 0.00064, 0.0064, 0.03126,   426, 0.0, 0.0, 0.0, 0.0, 1, -30.0, 30.0],
       [2, 3, 0.00108, 0.0108, 0.01852,    426, 0.0, 0.0, 0.0, 0.0, 1, -30.0, 30.0],
       [3, 4, 0.00297, 0.0297, 0.00674,    426, 0.0, 0.0, 1.05, 1.0, 1, -30.0, 30.0],
       [3, 4, 0.00297, 0.0297, 0.00674,    426, 0.0, 0.0, 1.05, - 1.0, 1, -30.0, 30.0],
       [4, 10, 0.00297, 0.0297, 0.00674,   240.0, 0.0, 0.0, 0.0, 0.0, 1, -30.0, 30.0],
    ])

    net = from_ppc(mpc, f_hz= 50)
    
    return net
    
    
def test_case5_pm_constrain_ext_grid_vm_pu():
    
    net = case5_pm_matfile_I()
    pp.runpp(net)
    

    pp.runpm_ac_opf(net, calculate_voltage_angles=True, correct_pm_network_data=False,
            delete_buffer_file=False,
            pm_file_path="case5_clm_matfile_va.json", opf_flow_lim="I",
            constrain_ext_grid_vm_pu = True)

    
    assert np.isclose(net.res_cost, 17082.8)
    
    
    pp.runpm_ac_opf(net, calculate_voltage_angles=True, correct_pm_network_data=False,
                    delete_buffer_file=False,
                    pm_file_path="case5_clm_matfile_va.json", opf_flow_lim="I",
                    constrain_ext_grid_vm_pu = False)
    
    
    assert np.isclose(net.res_cost, 17015.5635)
    
    
def test_case5_pm_vs_pp():
    
    net = case5_pm_matfile_I()
    pp.runpp(net)
    

    pp.runpm_ac_opf(net, calculate_voltage_angles=True, correct_pm_network_data=False,
            delete_buffer_file=False,
            pm_file_path="case5_clm_matfile_va.json", opf_flow_lim="I")

    
    assert np.isclose(net.res_cost, 17082.8)
    
    pp.runopp(net)
    
    assert np.isclose(net.res_cost, 17082.8)
        


if __name__ == "__main__":
    test_case5_pm_constrain_ext_grid_vm_pu()
   # pytest.main([__file__, "-xs"])
#     test_case5_pm_matfile_I()
