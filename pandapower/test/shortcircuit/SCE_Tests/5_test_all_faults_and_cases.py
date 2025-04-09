# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pandas as pd
import numpy as np
from pandapower.shortcircuit.calc_sc import calc_sc
from pandapower.file_io import from_json


def test_shortcircuit():
    net = from_json('4_bus_radial_grid.json')
    net.line.rename(columns={'temperature_degree_celsius': 'endtemp_degree'}, inplace=True)
    excel_file = '2_Short_Circuit_Results_PF_all.xlsx'
    sheets = pd.ExcelFile(excel_file).sheet_names
    dataframes = {}
    for sheet in sheets:
        pf_results = pd.read_excel(excel_file, sheet_name=sheet)
        pf_results = pf_results.drop(columns=['Netz'])
        pf_results = pf_results.drop(index=0)
        if "3ph" in sheet:
            pf_results.columns = ['name', 'ikss_ka', 'skss_mw', 'rk_ohm', 'xk_ohm']
        dataframes[sheet] = pf_results

    columns_to_check = ["ikss_ka", "skss_mw", "rk_ohm", "xk_ohm"]
    rtol = {"ikss_ka": 0, "skss_mw": 0, "rk_ohm": 0, "xk_ohm": 0}
    atol = {"ikss_ka": 1e-6, "skss_mw": 1e-5, "rk_ohm": 1e-5, "xk_ohm": 1e-5}
    faults = ["3ph"]
    # faults = ["3ph", "2ph", "1ph"]
    cases = ["max", "min"]

    for fault in faults:
        for case in cases:
            selected_sheet = f"{fault}_{case}"
            selected_pf_results = dataframes[selected_sheet]
            calc_sc(net, fault=fault, case=case, branch_results=False, ip=False)
            net.res_bus_sc["name"] = net.bus.name
            # Spalte 'name' an den Anfang stellen
            cols = ['name'] + [col for col in net.res_bus_sc.columns if col != 'name']
            net.res_bus_sc = net.res_bus_sc[cols]
            # DataFrame nach der Spalte 'name' sortieren
            net.res_bus_sc.sort_values(by='name', inplace=True)

            for bus in net.bus.name:
                for column in columns_to_check:
                    assert np.isclose(net.res_bus_sc[net.bus.name == bus][column].values[0],
                                      selected_pf_results[selected_pf_results.name == bus][column].values[0],
                                      rtol=rtol[column], atol=atol[column])


test_shortcircuit()
