import pytest
import pandapower as pp
import pandapower.networks as pn
from pandapower.harmonics.balanced import balanced_thd_voltage
from pandapower.plotting import simple_plot

def test_harmonics_case_2():
    har_a = [8.5,7.5,7,6.5,5.5,5,4.5]
    har_a_angle = [170,155,140,125,110,95,80]

    harmonics = [1, 3, 5, 7, 9, 11, 13, 15]

    net = pn.create_kerber_landnetz_kabel_1()

    net.ext_grid["s_sc_max_mva"] = 5
    net.ext_grid["rx_max"] = 3
    net.ext_grid["r0x0_max"] = 3
    net.ext_grid["x0x_max"] = 2

    net.trafo.i0_percent = 1.25
    net.trafo["vector_group"] = 'YNyn'
    net.trafo["vk0_percent"] = net.trafo["vk_percent"]
    net.trafo["vkr0_percent"] = net.trafo["vkr_percent"]
    net.trafo["mag0_percent"] = 100
    net.trafo["mag0_rx"] = 0
    net.trafo["si0_hv_partial"] = 100

    net.line['r0_ohm_per_km'] = 3*net.line['r_ohm_per_km']
    net.line['x0_ohm_per_km'] = 4*net.line['x_ohm_per_km']
    net.line['c0_nf_per_km'] = 0

    # pp.runpp_3ph(net)

    a, b = balanced_thd_voltage(net, harmonics, har_a, har_a_angle, analysis_type = 'balanced_positive')

    #simple_plot(net, line_width = 2.5, trafo_size=2.0, bus_size = 1.5, ext_grid_size = 1.5, line_color = 'black')

if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
