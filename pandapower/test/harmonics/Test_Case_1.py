import os
import pytest
import pandas as pd
import pandapower as pp
from pandapower.harmonics.balanced import balanced_thd_voltage
from pandapower.plotting.simple_plot import simple_plot

def test_harmonics_case_1():
    path = os.path.join(pp.pp_dir, "test", "test_files", "harmonics")

    nodes = pd.read_excel(os.path.join(path, 'Test_1f_Loads.xlsx'))
    lines = pd.read_excel(os.path.join(path,'Test_1f_lines.xlsx'))

    har_a = [11,10,9,8,7,6,5]
    har_a_angle = [250,240,230,220,210,200,190]

    harmonics = [1, 3, 5, 7, 9, 11, 13, 15]
    # harmonics = [1, 3]

    net = pp.create_empty_network()

    for i in range(0, len(nodes['Naziv'])):
        pp.create_bus(net, 10, nodes['Naziv'][i], int(nodes['ID'][i]))

    pp.create_ext_grid(net, bus = 0, s_sc_max_mva = 10, s_sc_min_mva = 10, rx_max=5, rx_min=0, r0x0_max=5, x0x_max=3)

    for i in range (0, len(lines['Length'])):
        pp.create_line_from_parameters(net,
                                       lines['Start Node'][i],
                                       lines['End Node'][i],
                                       lines['Length'][i],
                                       lines['R1[ohm/km]'][i],
                                       lines['X1[ohm/km]'][i],
                                       1e3*lines['C1[uF/km]'][i],
                                       lines['Imax[kA]'][i],
                                       r0_ohm_per_km=lines['R0[ohm/km]'][i],
                                       x0_ohm_per_km=lines['X0[ohm/km]'][i],
                                       c0_nf_per_km=0)

    for i in range(0, len(nodes['Naziv'])):
        pp.create_load(net, nodes['ID'][i], nodes['P [kW]'][i]/1000, nodes['Q [kVAr]'][i]/1000)

    a, b = balanced_thd_voltage(net, harmonics, har_a, har_a_angle, analysis_type = 'balanced_positive')
    # simple_plot(net, line_width = 2.5, trafo_size=2.0, bus_size = 1.5, ext_grid_size = 1.5, line_color = 'black')

if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
