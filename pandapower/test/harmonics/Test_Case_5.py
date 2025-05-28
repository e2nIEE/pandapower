import os
import pytest
import pandas as pd
import pandapower as pp
import numpy as np
import cmath
from pandapower.harmonics.unbalanced import unbalanced_thd_voltage
from pandapower.plotting import simple_plot


def test_harmonics_case_5():
    path = os.path.join(pp.pp_dir, "test", "test_files", "harmonics")

    # Defining the path of files containing network parameters - Modified CIGRE LV network
    bus = pd.read_excel(os.path.join(path, 'CIGRE_LV.xlsx'), sheet_name = 'Bus')
    line = pd.read_excel(os.path.join(path, 'CIGRE_LV.xlsx'), sheet_name = 'Line')
    load = pd.read_excel(os.path.join(path, 'CIGRE_LV.xlsx'), sheet_name = 'Load')

    # 3rd, 5th, 7th, 9th, 11th, 13th, 15th harmonic currents (percentage of fundamental harmonic and angles)
    har_a = [8,7,6,5,4,3,2]
    har_a_angle = [320,310,300,290,280,270,260]
    har_b = [13,11.5,10,8.5,7,5.5,4]
    har_b_angle = [300,290,280,270,280,290,300]
    har_c = [10,9.25,8.5,7.75,7,6.25,5.5]
    har_c_angle = [250,255,260,265,255,245,235]

    #LC technology harmonic pattern, e.g., PV - does not need to be defined, or more than one LC technology can be used
    har_a_lc = [10,8,6,4,2,1.5,1]
    har_a_lc_angle = [300,320,280,260,250,270,220]
    har_b_lc = [15,14,13,12,11,10,9]
    har_b_lc_angle = [310,297,222,200,180,144,127]
    har_c_lc = [10,9,8.5,7,6.5,5,4.5]
    har_c_lc_angle = [230,245,160,165,155,145,135]

    harmonics = [1, 3, 5, 7, 9, 11, 13, 15]

    a1 = cmath.rect(1, 2/3*cmath.pi)
    a2 = cmath.rect(1, 4/3*cmath.pi)
    s_base = 1000000
    matrix_A = np.matrix([[1, 1, 1], [1, a2, a1], [1, a1, a2]])

    net = pp.create_empty_network()


    for i in range(0, len(bus['name'])):
        coord = (bus['x'][i], bus['y'][i])
        pp.create_bus(net, bus['vn_kv'][i], name = bus['name'][i], index = bus['id'][i]-1, geodata = coord)

    pp.create_ext_grid(net, bus = 0, s_sc_max_mva = 10, s_sc_min_mva = 10, rx_max=1, rx_min=1, r0x0_max=1, x0x_max=1)


    for i in range(0, len(load['name'])):
        #We create both residential load and the LC technology, e.g., PV. For the purpose of creating and veryfing
        #the tool, we determine the PV with the power of 3 kW, at the phase A at every node
        #If no LC technology is used, the second asymmetric load at the same node is not needed
        #Also, more than two loads can be created
        pp.create_asymmetric_load(net, load['bus'][i]-1, p_a_mw = load['p_a_mw'][i], q_a_mvar = load['q_a_mvar'][i], p_b_mw = load['p_b_mw'][i],\
                                  q_b_mvar = load['q_b_mvar'][i], p_c_mw = load['p_c_mw'][i], q_c_mvar = load['q_c_mvar'][i])

        pp.create_asymmetric_load(net, load['bus'][i]-1, p_a_mw = -10/1000, q_a_mvar = 0, p_b_mw = 0,\
                              q_b_mvar = 0, p_c_mw = 0, q_c_mvar = 0)

    for i in range(0, len(line['name'])):
        pp.create_line_from_parameters(net, line['from_bus'][i]-1, line['to_bus'][i]-1, line['length_km'][i],\
                                      line['r_ohm_per_km'][i], line['x_ohm_per_km'][i], 0, line['max_i_ka'][i],\
                                      r0_ohm_per_km = line['r0_ohm_per_km'][i], x0_ohm_per_km = line['x0_ohm_per_km'][i],
                                      c0_nf_per_km = 0, name = line['name'][i])

    pp.runpp_3ph(net)

    thd_a, thd_b, thd_c, har_vol_a, har_vol_b, har_vol_c = unbalanced_thd_voltage(net, harmonics, har_a, har_a_angle, har_b, har_b_angle, har_c, har_c_angle,\
                                            har_a_lc, har_a_lc_angle, har_b_lc, har_b_lc_angle, har_c_lc, har_c_lc_angle,\
                                            analysis_type = "unbalanced")

    # # simple_plot(net, line_width = 2.5, trafo_size=2.0, bus_size = 1.5, ext_grid_size = 1.5, line_color = 'black')

if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
