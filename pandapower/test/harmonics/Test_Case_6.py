import pytest
import pandapower as pp
from pandapower.plotting import simple_plot
from pandapower.harmonics.unbalanced import unbalanced_thd_voltage, unbalanced_harmonic_current_voltage

def test_harmonics_case_6():
    net = pp.create_empty_network()

    pp.create_bus(net, 10, index = 0)

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

    for i in range(1, 7):
        pp.create_bus(net, 0.4, index = i)

    pp.create_ext_grid(net, bus = 0, s_sc_max_mva = 5, rx_max = 3, r0x0_max = 3, x0x_max= 3)

    pp.create_transformer_from_parameters(net, hv_bus = 0, lv_bus = 1, sn_mva = 0.4, vn_hv_kv = 10, vn_lv_kv = 0.4, vkr_percent = 0, \
                                        vk_percent = 4,pfe_kw = 0, i0_percent = 0, vector_group = 'Yyn',shift_degree = 0,\
                                        vkr0_percent = 0, vk0_percent = 4, mag0_percent = 100, mag0_rx = 100, si0_hv_partial = 0.9)

    pp.create_line_from_parameters(net, from_bus = 1, to_bus = 2, length_km = 0.3, r_ohm_per_km = 0.203, x_ohm_per_km = 0.08, \
                                    c_nf_per_km = 0, max_i_ka = 275, r0_ohm_per_km = 0.812, x0_ohm_per_km = 0.24, c0_nf_per_km = 0)
    pp.create_line_from_parameters(net, from_bus = 2, to_bus = 3, length_km = 0.1, r_ohm_per_km = 0.606, x_ohm_per_km = 0.083, \
                                    c_nf_per_km = 0, max_i_ka = 144, r0_ohm_per_km = 2.424, x0_ohm_per_km = 0.249, c0_nf_per_km = 0)
    pp.create_line_from_parameters(net, from_bus = 2, to_bus = 4, length_km = 0.3, r_ohm_per_km = 0.203, x_ohm_per_km = 0.08, \
                                    c_nf_per_km = 0, max_i_ka = 275, r0_ohm_per_km = 0.812, x0_ohm_per_km = 0.24, c0_nf_per_km = 0)
    pp.create_line_from_parameters(net, from_bus = 4, to_bus = 5, length_km = 0.1, r_ohm_per_km = 0.606, x_ohm_per_km = 0.083, \
                                    c_nf_per_km = 0, max_i_ka = 144, r0_ohm_per_km = 2.424, x0_ohm_per_km = 0.249, c0_nf_per_km = 0)
    pp.create_line_from_parameters(net, from_bus = 4, to_bus = 6, length_km = 0.1, r_ohm_per_km = 0.606, x_ohm_per_km = 0.083, \
                                    c_nf_per_km = 0, max_i_ka = 144, r0_ohm_per_km = 2.424, x0_ohm_per_km = 0.249, c0_nf_per_km = 0)

    pp.create_asymmetric_load(net, 3, p_a_mw = 5/1000, p_b_mw = 6/1000, p_c_mw = 4/1000)
    pp.create_asymmetric_load(net, 3, p_a_mw = 0, p_b_mw = 0, p_c_mw = 0)
    pp.create_asymmetric_load(net, 5, p_a_mw = 6/1000, p_b_mw = 7/1000, p_c_mw = 8/1000)
    pp.create_asymmetric_load(net, 5, p_a_mw = 0, p_b_mw = 0, p_c_mw = 0)
    pp.create_asymmetric_load(net, 6, p_a_mw = 6/1000, p_b_mw = 8/1000, p_c_mw = 5/1000)
    pp.create_asymmetric_load(net, 6, p_a_mw = 0, p_b_mw = 0, p_c_mw = 0)

    # pp.runpp_3ph(net)

    # simple_plot(net, line_width = 2.5, trafo_size=2.0, bus_size = 1.5, ext_grid_size = 1.5, line_color = 'black')

    pp_volt = []

    for i in net.res_bus_3ph.index:
        pp_volt.append(net.res_bus_3ph.vm_a_pu[i]*100)
        pp_volt.append(net.res_bus_3ph.vm_b_pu[i]*100)
        pp_volt.append(net.res_bus_3ph.vm_c_pu[i]*100)

    thd_a, thd_b, thd_c, har_vol_a, har_vol_b, har_vol_c = \
                                            unbalanced_thd_voltage(net, harmonics, har_a,
                                            har_a_angle, har_b, har_b_angle, har_c, har_c_angle,
                                            har_a_lc, har_a_lc_angle, har_b_lc, har_b_lc_angle, har_c_lc, har_c_lc_angle,
                                            analysis_type = "unbalanced")

    a, b, c, d = unbalanced_harmonic_current_voltage(net, harmonics, har_a,
                                                     har_a_angle, har_b, har_b_angle, har_c, har_c_angle,
                                                     har_a_lc, har_a_lc_angle, har_b_lc, har_b_lc_angle, har_c_lc, har_c_lc_angle,
                                                     analysis_type = "unbalanced")

    # y0 = net._ppc0['internal']['Ybus'].todense()
    # y1 = net._ppc1['internal']['Ybus'].todense()

if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
