import os
import cmath
import pytest
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as nw
from pandapower.harmonics.balanced import balanced_thd_voltage
from pandapower.harmonics.unbalanced import unbalanced_thd_voltage, unbalanced_harmonic_current_voltage
from pandapower.plotting.simple_plot import simple_plot

def test_harmonics_case_1():
    path = os.path.join(pp.pp_dir, "test", "test_files", "harmonics")

    nodes = pd.read_excel(os.path.join(path, 'Test_1f_Loads.xlsx'))
    lines = pd.read_excel(os.path.join(path,'Test_1f_lines.xlsx'))

    har_a = [11, 10, 9, 8, 7, 6, 5]
    har_a_angle = [250, 240, 230, 220, 210, 200, 190]

    harmonics = [1, 3, 5, 7, 9, 11, 13, 15]
    # harmonics=[1, 3]

    net = pp.create_empty_network()

    for i in range(0, len(nodes['Naziv'])):
        pp.create_bus(net, 10, nodes['Naziv'][i], int(nodes['ID'][i]))

    pp.create_ext_grid(net, bus=0, s_sc_max_mva=10, s_sc_min_mva=10, rx_max=5, rx_min=0, r0x0_max=5, x0x_max=3)

    for i in range (0, len(lines['Length'])):
        pp.create_line_from_parameters(net,
                                       from_bus=lines['Start Node'][i],
                                       to_bus=lines['End Node'][i],
                                       length_km=lines['Length'][i],
                                       r_ohm_per_km=lines['R1[ohm/km]'][i],
                                       x_ohm_per_km=lines['X1[ohm/km]'][i],
                                       c_nf_per_km=1e3*lines['C1[uF/km]'][i],
                                       max_i_ka=lines['Imax[kA]'][i],
                                       r0_ohm_per_km=lines['R0[ohm/km]'][i],
                                       x0_ohm_per_km=lines['X0[ohm/km]'][i],
                                       c0_nf_per_km=0)

    for i in range(0, len(nodes['Naziv'])):
        pp.create_load(net, nodes['ID'][i], nodes['P [kW]'][i]/1000, nodes['Q [kVAr]'][i]/1000)

    a, b = balanced_thd_voltage(net, harmonics, har_a, har_a_angle, analysis_type='balanced_positive')
    # simple_plot(net, line_width=2.5, trafo_size=2.0, bus_size=1.5, ext_grid_size=1.5, line_color='black')


def test_harmonics_case_2():
    har_a = [8.5, 7.5, 7, 6.5, 5.5, 5, 4.5]
    har_a_angle = [170, 155, 140, 125, 110, 95, 80]

    harmonics = [1, 3, 5, 7, 9, 11, 13, 15]

    net = nw.create_kerber_landnetz_kabel_1()

    net.ext_grid["s_sc_max_mva"]=5
    net.ext_grid["rx_max"]=3
    net.ext_grid["r0x0_max"]=3
    net.ext_grid["x0x_max"]=2

    net.trafo.i0_percent=1.25
    net.trafo["vector_group"]='YNyn'
    net.trafo["vk0_percent"]=net.trafo["vk_percent"]
    net.trafo["vkr0_percent"]=net.trafo["vkr_percent"]
    net.trafo["mag0_percent"]=100
    net.trafo["mag0_rx"]=0
    net.trafo["si0_hv_partial"]=100

    net.line['r0_ohm_per_km']=3*net.line['r_ohm_per_km']
    net.line['x0_ohm_per_km']=4*net.line['x_ohm_per_km']
    net.line['c0_nf_per_km']=0

    # pp.runpp_3ph(net)

    a, b = balanced_thd_voltage(net, harmonics, har_a, har_a_angle, analysis_type='balanced_positive')

    #simple_plot(net, line_width=2.5, trafo_size=2.0, bus_size=1.5, ext_grid_size=1.5, line_color='black')


def test_harmonics_case_3():
    net=pp.create_empty_network()

    pp.create_bus(net, 110, index=0)

    for i in range(1, 5):
        pp.create_bus(net, 10, index=i)

    pp.create_bus(net, 0.4, index=5)

    pp.create_ext_grid(net, bus=0, s_sc_max_mva=10, rx_max=3, r0x0_max=3, x0x_max= 4)

    pp.create_transformer_from_parameters(net, hv_bus=0, lv_bus=1, sn_mva=6, vn_hv_kv=110, vn_lv_kv=10,
                                          vkr_percent=0, vk_percent=4,pfe_kw=0, i0_percent=0, vector_group='Yyn',
                                          shift_degree=0, vkr0_percent=0.8, vk0_percent=4, mag0_percent=100,
                                          mag0_rx=100, si0_hv_partial=100)

    pp.create_line_from_parameters(net, from_bus=1, to_bus=2, length_km=1, r_ohm_per_km=0.61,
                                   x_ohm_per_km=0.355, c_nf_per_km=0, max_i_ka=170, r0_ohm_per_km=0.76,
                                   x0_ohm_per_km=1.7, c0_nf_per_km=0)
    pp.create_line_from_parameters(net, from_bus=2, to_bus=3, length_km=1.5, r_ohm_per_km=0.61,
                                   x_ohm_per_km=0.355, c_nf_per_km=0, max_i_ka=170, r0_ohm_per_km=0.76,
                                   x0_ohm_per_km=1.7, c0_nf_per_km=0)
    pp.create_line_from_parameters(net, from_bus=3, to_bus=4, length_km=1.5, r_ohm_per_km=0.61,
                                   x_ohm_per_km=0.355, c_nf_per_km=0, max_i_ka=170, r0_ohm_per_km=0.76,
                                   x0_ohm_per_km=1.7, c0_nf_per_km=0)

    pp.create_transformer_from_parameters(net, hv_bus=4, lv_bus=5, sn_mva=0.63, vn_hv_kv=10, vn_lv_kv=0.4,
                                          vkr_percent=0, vk_percent=4,pfe_kw=0, i0_percent=0, vector_group='Yyn',
                                          shift_degree=0, vkr0_percent=0, vk0_percent=4, mag0_percent=100,
                                          mag0_rx=100, si0_hv_partial=100)

    pp.create_load(net, 2, p_mw=0.35,  q_mvar=0.115039)
    pp.create_load(net, 3, p_mw=0.375, q_mvar=0.123257)
    pp.create_load(net, 5, p_mw=0.02,  q_mvar=0.006574)

    har_a = [11, 10, 9, 8, 7, 6, 5]
    har_a_angle = [250, 240, 230, 220, 210, 200, 190]
    harmonics = [1, 3, 5, 7, 9, 11, 13, 15]

    a, b = balanced_thd_voltage(net, harmonics, har_a, har_a_angle, analysis_type='balanced_positive')

    # simple_plot(net, line_width=2.5, trafo_size=2.0, bus_size=1.5, ext_grid_size=1.5, line_color='black')


def test_harmonics_case_4():
    path = os.path.join(pp.pp_dir, "test", "test_files", "harmonics")

    # Defining the path of files containing network parameters - Modified CIGRE LV network
    bus  = pd.read_excel(os.path.join(path, 'CIGRE_LV.xlsx'), sheet_name='Bus')
    line = pd.read_excel(os.path.join(path, 'CIGRE_LV.xlsx'), sheet_name='Line')
    load = pd.read_excel(os.path.join(path, 'CIGRE_LV.xlsx'), sheet_name='Load')

    # 3rd, 5th, 7th, 9th, 11th, 13th, 15th harmonic currents (percentage of fundamental harmonic and angles)
    har_a = [8, 7, 6, 5, 4, 3, 2]
    har_a_angle = [320, 310, 300, 290, 280, 270, 260]
    har_b = [13, 11.5, 10, 8.5, 7, 5.5, 4]
    har_b_angle = [300, 290, 280, 270, 280, 290, 300]
    har_c = [10, 9.25, 8.5, 7.75, 7, 6.25, 5.5]
    har_c_angle = [250, 255, 260, 265, 255, 245, 235]

    #LC technology harmonic pattern, e.g., PV - does not need to be defined, or more than one LC technology can be used
    har_a_lc = [10, 8, 6, 4, 2, 1.5, 1]
    har_a_lc_angle = [300, 320, 280, 260, 250, 270, 220]
    har_b_lc = [15, 14, 13, 12, 11, 10, 9]
    har_b_lc_angle = [310, 297, 222, 200, 180, 144, 127]
    har_c_lc = [10, 9, 8.5, 7, 6.5, 5, 4.5]
    har_c_lc_angle = [230, 245, 160, 165, 155, 145, 135]

    harmonics = [1, 3, 5, 7, 9, 11, 13, 15]

    net = pp.create_empty_network()

    for i in range(0, len(bus['name'])):
        coord=(bus['x'][i], bus['y'][i])
        pp.create_bus(net, vn_kv=bus['vn_kv'][i], name=bus['name'][i], index=bus['id'][i]-1, geodata=coord)

    pp.create_ext_grid(net, bus=0, s_sc_max_mva=10, s_sc_min_mva=10, rx_max=1, rx_min=1, r0x0_max=1, x0x_max=1)

    for i in range(0, len(load['name'])):
        #We create both residential load and the LC technology, e.g., PV. For the purpose of creating and veryfing
        #the tool, we determine the PV with the power of 3 kW, at the phase A at every node
        #If no LC technology is used, the second asymmetric load at the same node is not needed
        #Also, more than two loads can be created
        pp.create_asymmetric_load(net, load['bus'][i]-1, p_a_mw=load['p_a_mw'][i], q_a_mvar=load['q_a_mvar'][i], p_b_mw=load['p_b_mw'][i],\
                                  q_b_mvar=load['q_b_mvar'][i], p_c_mw=load['p_c_mw'][i], q_c_mvar=load['q_c_mvar'][i])

        pp.create_asymmetric_load(net, load['bus'][i]-1, p_a_mw=0/1000, q_a_mvar=0, p_b_mw=0,\
                              q_b_mvar=0, p_c_mw=0, q_c_mvar=0)

    for i in range(0, len(line['name'])):
        pp.create_line_from_parameters(net, line['from_bus'][i]-1, line['to_bus'][i]-1, line['length_km'][i],\
                                      line['r_ohm_per_km'][i], line['x_ohm_per_km'][i], 0, line['max_i_ka'][i],\
                                      r0_ohm_per_km=line['r0_ohm_per_km'][i], x0_ohm_per_km=line['x0_ohm_per_km'][i],
                                      c0_nf_per_km=0, name=line['name'][i])

    pp.runpp_3ph(net)

    thd_a, thd_b, thd_c, har_vol_a, har_vol_b, har_vol_c = unbalanced_thd_voltage(net, harmonics, har_a, har_a_angle,
                                                                                  har_b, har_b_angle, har_c, har_c_angle,
                                                                                  har_a_lc, har_a_lc_angle, har_b_lc,
                                                                                  har_b_lc_angle, har_c_lc, har_c_lc_angle,
                                                                                  analysis_type="unbalanced")

    # simple_plot(net, line_width=2.5, trafo_size=2.0, bus_size=1.5, ext_grid_size=1.5, line_color='black')

    voltages = []

    for i in range(0, np.shape(har_vol_a)[0]):
        for j in range(0, np.shape(har_vol_a)[1]):
            voltages.append(har_vol_a[i,j])
            voltages.append(har_vol_b[i,j])
            voltages.append(har_vol_c[i,j])

    pp_volt=[]

    for i in net.res_bus_3ph.index:
        pp_volt.append(net.res_bus_3ph.vm_a_pu[i])
        pp_volt.append(net.res_bus_3ph.vm_b_pu[i])
        pp_volt.append(net.res_bus_3ph.vm_c_pu[i])


def test_harmonics_case_5():
    path = os.path.join(pp.pp_dir, "test", "test_files", "harmonics")

    # Defining the path of files containing network parameters - Modified CIGRE LV network
    bus = pd.read_excel(os.path.join(path, 'CIGRE_LV.xlsx'), sheet_name='Bus')
    line = pd.read_excel(os.path.join(path, 'CIGRE_LV.xlsx'), sheet_name='Line')
    load = pd.read_excel(os.path.join(path, 'CIGRE_LV.xlsx'), sheet_name='Load')

    # 3rd, 5th, 7th, 9th, 11th, 13th, 15th harmonic currents (percentage of fundamental harmonic and angles)
    har_a = [8, 7, 6, 5, 4, 3, 2]
    har_a_angle = [320, 310, 300, 290, 280, 270, 260]
    har_b = [13, 11.5, 10, 8.5, 7, 5.5, 4]
    har_b_angle = [300, 290, 280, 270, 280, 290, 300]
    har_c = [10, 9.25, 8.5, 7.75, 7, 6.25, 5.5]
    har_c_angle = [250, 255, 260, 265, 255, 245, 235]

    #LC technology harmonic pattern, e.g., PV - does not need to be defined, or more than one LC technology can be used
    har_a_lc = [10, 8, 6, 4, 2, 1.5, 1]
    har_a_lc_angle = [300, 320, 280, 260, 250, 270, 220]
    har_b_lc = [15, 14, 13, 12, 11, 10, 9]
    har_b_lc_angle = [310, 297, 222, 200, 180, 144, 127]
    har_c_lc = [10, 9, 8.5, 7, 6.5, 5, 4.5]
    har_c_lc_angle = [230, 245, 160, 165, 155, 145, 135]

    harmonics = [1, 3, 5, 7, 9, 11, 13, 15]

    a1 = cmath.rect(1, 2/3*cmath.pi)
    a2 = cmath.rect(1, 4/3*cmath.pi)
    s_base = 1000000
    matrix_A = np.matrix([[1, 1, 1], [1, a2, a1], [1, a1, a2]])

    net = pp.create_empty_network()


    for i in range(0, len(bus['name'])):
        coord = (bus['x'][i], bus['y'][i])
        pp.create_bus(net, bus['vn_kv'][i], name=bus['name'][i], index=bus['id'][i]-1, geodata=coord)

    pp.create_ext_grid(net, bus=0, s_sc_max_mva=10, s_sc_min_mva=10, rx_max=1, rx_min=1, r0x0_max=1, x0x_max=1)


    for i in range(0, len(load['name'])):
        #We create both residential load and the LC technology, e.g., PV. For the purpose of creating and veryfing
        #the tool, we determine the PV with the power of 3 kW, at the phase A at every node
        #If no LC technology is used, the second asymmetric load at the same node is not needed
        #Also, more than two loads can be created
        pp.create_asymmetric_load(net, load['bus'][i]-1, p_a_mw=load['p_a_mw'][i], q_a_mvar=load['q_a_mvar'][i], p_b_mw=load['p_b_mw'][i],
                                  q_b_mvar=load['q_b_mvar'][i], p_c_mw=load['p_c_mw'][i], q_c_mvar=load['q_c_mvar'][i])

        pp.create_asymmetric_load(net, load['bus'][i]-1, p_a_mw=-10/1000, q_a_mvar=0, p_b_mw=0,\
                              q_b_mvar=0, p_c_mw=0, q_c_mvar=0)

    for i in range(0, len(line['name'])):
        pp.create_line_from_parameters(net, line['from_bus'][i]-1, line['to_bus'][i]-1, line['length_km'][i],
                                      line['r_ohm_per_km'][i], line['x_ohm_per_km'][i], 0, line['max_i_ka'][i],
                                      r0_ohm_per_km=line['r0_ohm_per_km'][i], x0_ohm_per_km=line['x0_ohm_per_km'][i],
                                      c0_nf_per_km=0, name=line['name'][i])

    pp.runpp_3ph(net)

    thd_a, thd_b, thd_c, har_vol_a, har_vol_b, har_vol_c = unbalanced_thd_voltage(net, harmonics, har_a, har_a_angle,
                                                                                  har_b, har_b_angle, har_c, har_c_angle,
                                                                                  har_a_lc, har_a_lc_angle, har_b_lc,
                                                                                  har_b_lc_angle, har_c_lc, har_c_lc_angle,
                                                                                  analysis_type="unbalanced")

    # # simple_plot(net, line_width=2.5, trafo_size=2.0, bus_size=1.5, ext_grid_size=1.5, line_color='black')


def test_harmonics_case_6():
    net = pp.create_empty_network()

    pp.create_bus(net, 10, index=0)

    # 3rd, 5th, 7th, 9th, 11th, 13th, 15th harmonic currents (percentage of fundamental harmonic and angles)
    har_a = [8, 7, 6, 5, 4, 3, 2]
    har_a_angle = [320, 310, 300, 290, 280, 270, 260]
    har_b = [13, 11.5, 10, 8.5, 7, 5.5, 4]
    har_b_angle = [300, 290, 280, 270, 280, 290, 300]
    har_c = [10, 9.25, 8.5, 7.75, 7, 6.25, 5.5]
    har_c_angle = [250, 255, 260, 265, 255, 245, 235]

    #LC technology harmonic pattern, e.g., PV - does not need to be defined, or more than one LC technology can be used
    har_a_lc = [10, 8, 6, 4, 2, 1.5, 1]
    har_a_lc_angle = [300, 320, 280, 260, 250, 270, 220]
    har_b_lc = [15, 14, 13, 12, 11, 10, 9]
    har_b_lc_angle = [310, 297, 222, 200, 180, 144, 127]
    har_c_lc = [10, 9, 8.5, 7, 6.5, 5, 4.5]
    har_c_lc_angle = [230, 245, 160, 165, 155, 145, 135]

    harmonics = [1, 3, 5, 7, 9, 11, 13, 15]

    for i in range(1, 7):
        pp.create_bus(net, 0.4, index=i)

    pp.create_ext_grid(net, bus=0, s_sc_max_mva=5, rx_max=3, r0x0_max=3, x0x_max=3)

    pp.create_transformer_from_parameters(net, hv_bus=0, lv_bus=1, sn_mva=0.4, vn_hv_kv=10, vn_lv_kv=0.4, vkr_percent=0,
                                        vk_percent=4,pfe_kw=0, i0_percent=0, vector_group='YNyn',shift_degree=0,
                                        vkr0_percent=0, vk0_percent=4, mag0_percent=100, mag0_rx=100, si0_hv_partial=0.9)

    pp.create_line_from_parameters(net, from_bus=1, to_bus=2, length_km=0.3, r_ohm_per_km=0.203, x_ohm_per_km=0.08,
                                    c_nf_per_km=0, max_i_ka=275, r0_ohm_per_km=0.812, x0_ohm_per_km=0.24, c0_nf_per_km=0)
    pp.create_line_from_parameters(net, from_bus=2, to_bus=3, length_km=0.1, r_ohm_per_km=0.606, x_ohm_per_km=0.083,
                                    c_nf_per_km=0, max_i_ka=144, r0_ohm_per_km=2.424, x0_ohm_per_km=0.249, c0_nf_per_km=0)
    pp.create_line_from_parameters(net, from_bus=2, to_bus=4, length_km=0.3, r_ohm_per_km=0.203, x_ohm_per_km=0.08,
                                    c_nf_per_km=0, max_i_ka=275, r0_ohm_per_km=0.812, x0_ohm_per_km=0.24, c0_nf_per_km=0)
    pp.create_line_from_parameters(net, from_bus=4, to_bus=5, length_km=0.1, r_ohm_per_km=0.606, x_ohm_per_km=0.083,
                                    c_nf_per_km=0, max_i_ka=144, r0_ohm_per_km=2.424, x0_ohm_per_km=0.249, c0_nf_per_km=0)
    pp.create_line_from_parameters(net, from_bus=4, to_bus=6, length_km=0.1, r_ohm_per_km=0.606, x_ohm_per_km=0.083,
                                    c_nf_per_km=0, max_i_ka=144, r0_ohm_per_km=2.424, x0_ohm_per_km=0.249, c0_nf_per_km=0)

    pp.create_asymmetric_load(net, 3, p_a_mw=5/1000, p_b_mw=6/1000, p_c_mw=4/1000)
    pp.create_asymmetric_load(net, 3, p_a_mw=0, p_b_mw=0, p_c_mw=0)
    pp.create_asymmetric_load(net, 5, p_a_mw=6/1000, p_b_mw=7/1000, p_c_mw=8/1000)
    pp.create_asymmetric_load(net, 5, p_a_mw=0, p_b_mw=0, p_c_mw=0)
    pp.create_asymmetric_load(net, 6, p_a_mw=6/1000, p_b_mw=8/1000, p_c_mw=5/1000)
    pp.create_asymmetric_load(net, 6, p_a_mw=0, p_b_mw=0, p_c_mw=0)

    # pp.runpp_3ph(net)

    # simple_plot(net, line_width=2.5, trafo_size=2.0, bus_size=1.5, ext_grid_size=1.5, line_color='black')

    pp_volt = []

    for i in net.res_bus_3ph.index:
        pp_volt.append(net.res_bus_3ph.vm_a_pu[i]*100)
        pp_volt.append(net.res_bus_3ph.vm_b_pu[i]*100)
        pp_volt.append(net.res_bus_3ph.vm_c_pu[i]*100)

    thd_a, thd_b, thd_c, har_vol_a, har_vol_b, har_vol_c = unbalanced_thd_voltage(net, harmonics, har_a,har_a_angle,
                                                                                  har_b, har_b_angle, har_c, har_c_angle,
                                                                                  har_a_lc, har_a_lc_angle, har_b_lc,
                                                                                  har_b_lc_angle, har_c_lc, har_c_lc_angle,
                                                                                  analysis_type="unbalanced")

    a, b, c, d = unbalanced_harmonic_current_voltage(net, harmonics, har_a, har_a_angle, har_b, har_b_angle, har_c,
                                                     har_c_angle, har_a_lc, har_a_lc_angle, har_b_lc, har_b_lc_angle,
                                                     har_c_lc, har_c_lc_angle, analysis_type="unbalanced")

    # y0=net._ppc0['internal']['Ybus'].todense()
    # y1=net._ppc1['internal']['Ybus'].todense()


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
