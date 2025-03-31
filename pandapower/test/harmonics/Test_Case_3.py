import pytest
import pandapower as pp
from pandapower.harmonics.balanced import balanced_thd_voltage
from pandapower.plotting import simple_plot

def test_harmonics_case_3():
    net = pp.create_empty_network()

    pp.create_bus(net, 110, index = 0)

    for i in range(1, 5):
        pp.create_bus(net, 10, index = i)

    pp.create_bus(net, 0.4, index = 5)

    pp.create_ext_grid(net, bus = 0, s_sc_max_mva = 10, rx_max = 3, r0x0_max = 3, x0x_max= 4)

    pp.create_transformer_from_parameters(net, hv_bus = 0, lv_bus = 1, sn_mva = 6, vn_hv_kv = 110, vn_lv_kv = 10, vkr_percent = 0, \
                                           vk_percent = 4,pfe_kw = 0, i0_percent = 0, vector_group = 'Yyn', shift_degree = 0,\
                                           vkr0_percent = 0.8, vk0_percent = 4, mag0_percent = 100, mag0_rx = 100, si0_hv_partial = 100)

    pp.create_line_from_parameters(net, from_bus = 1, to_bus = 2, length_km = 1, r_ohm_per_km = 0.61, x_ohm_per_km = 0.355, \
                                   c_nf_per_km = 0, max_i_ka = 170, r0_ohm_per_km = 0.76, x0_ohm_per_km = 1.7, c0_nf_per_km = 0)
    pp.create_line_from_parameters(net, from_bus = 2, to_bus = 3, length_km = 1.5, r_ohm_per_km = 0.61, x_ohm_per_km = 0.355, \
                                   c_nf_per_km = 0, max_i_ka = 170, r0_ohm_per_km = 0.76, x0_ohm_per_km = 1.7, c0_nf_per_km = 0)
    pp.create_line_from_parameters(net, from_bus = 3, to_bus = 4, length_km = 1.5, r_ohm_per_km = 0.61, x_ohm_per_km = 0.355, \
                                   c_nf_per_km = 0, max_i_ka = 170, r0_ohm_per_km = 0.76, x0_ohm_per_km = 1.7, c0_nf_per_km = 0)

    pp.create_transformer_from_parameters(net, hv_bus = 4, lv_bus = 5, sn_mva = 0.63, vn_hv_kv = 10, vn_lv_kv = 0.4, vkr_percent = 0, \
                                        vk_percent = 4,pfe_kw = 0, i0_percent = 0, vector_group = 'Yyn',shift_degree = 0,\
                                        vkr0_percent = 0, vk0_percent = 4, mag0_percent = 100, mag0_rx = 100, si0_hv_partial = 100)

    pp.create_load(net, 2, p_mw = 0.35, q_mvar = 0.115039)
    pp.create_load(net, 3, p_mw = 0.375, q_mvar = 0.123257)
    pp.create_load(net, 5, p_mw = 0.02, q_mvar = 0.006574)

    har_a = [11,10,9,8,7,6,5]
    har_a_angle = [250,240,230,220,210,200,190]
    harmonics = [1, 3, 5, 7, 9, 11, 13, 15]

    a, b = balanced_thd_voltage(net, harmonics, har_a, har_a_angle, analysis_type = 'balanced_positive')

    # simple_plot(net, line_width = 2.5, trafo_size=2.0, bus_size = 1.5, ext_grid_size = 1.5, line_color = 'black')

if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
