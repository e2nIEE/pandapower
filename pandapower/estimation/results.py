# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.


from pandapower.results import reset_results

    
def _copy_power_flow_results(net):
    # copy old power flow results (if they exist) into res_*_power_flow tables for backup
    elements_to_init = ["bus", "ext_grid", "line", "load", "sgen", "trafo", "trafo3w",
                        "shunt", "impedance", "gen", "ward", "xward", "dcline"]
    for element in elements_to_init:
        res_name = "res_" + element
        res_name_pf = res_name + "_power_flow"
        if res_name in net:
            net[res_name_pf] = (net[res_name]).copy()
    reset_results(net)


def _rename_results(net):
    elements_to_init = ["bus", "ext_grid", "line", "load", "sgen", "trafo", "trafo3w",
                        "shunt", "impedance", "gen", "ward", "xward", "dcline"]
    # rename res_* tables to res_*_est and then res_*_power_flow to res_*
    for element in elements_to_init:
        res_name = "res_" + element
        res_name_pf = res_name + "_power_flow"
        res_name_est = res_name + "_est"
        net[res_name_est] = net[res_name]
        if res_name_pf in net:
            net[res_name] = net[res_name_pf]
        else:
            del net[res_name]
