# -*- coding: utf-8 -*-

import pandapower as pp
import pandapower.shortcircuit as sc
import pandapower.test
import copy
import numpy as np
import os
import pytest

def check_results(net, vc, result):
    res_ika = net.res_bus_sc[net.bus.zone==vc].ikss_ka.values
    if not np.allclose(result, res_ika):
        raise ValueError("Incorrect results for vector group %s"%vc, res_ika, result)

def add_network(net, vector_group):
    b1 = pp.create_bus(net, 110, zone=vector_group)
    b2 = pp.create_bus(net, 20, zone=vector_group)
    b3 = pp.create_bus(net, 20, zone=vector_group)
    b4 = pp.create_bus(net, 20, zone=vector_group)

    pp.create_ext_grid(net, b1, s_sc_max_mva=100, s_sc_min_mva=80, rx_min=0.20, rx_max=0.35)
    net.ext_grid["r0x0_max"] = 0.4
    net.ext_grid["x0x_max"] = 1.0

    pp.create_std_type(net, {"r_ohm_per_km": 0.122, "x_ohm_per_km": 0.112, "c_nf_per_km": 304,
                         "max_i_ka": 0.421, "endtemp_degree": 70.0, "r0_ohm_per_km": 0.244,
                         "x0_ohm_per_km": 0.336, "c0_nf_per_km": 2000}, "unsymmetric_line_type")
    pp.create_line(net, b2, b3, length_km=10, std_type="unsymmetric_line_type")
    pp.create_line(net, b3, b4, length_km=15, std_type="unsymmetric_line_type")
    
    
    transformer_type = copy.copy(pp.load_std_type(net, "25 MVA 110/20 kV v1.4.3 and older","trafo"))
    transformer_type.update({"vsc0_percent": 5, "vscr0_percent": 0.4, "mag0_percent": 10,
                             "mag0_rx": 0.4, "mag0_rx": 0.4, "si0_hv_partial": 0.9,
                             "vector_group": vector_group})
    pp.create_std_type(net, transformer_type, vector_group, "trafo")
    pp.create_transformer(net, b1, b2, std_type=vector_group)
    pp.add_zero_impedance_parameters(net)

def test_1ph_shortcircuit():
    results = {
            "YNyn": [0.58405824654, 2.4517609153, 1.6614556629, 1.111187189]
           , "Dyn": [0.52209347316, 2.9376424501, 1.8966571675, 1.2175499508]
           , "YNd": [0.7326427258, 0.77018941065, 0.7718741164, 0.84675601192]
           , "Yyn": [0.52209347316, 1.7178242081, 1.2607678015, 0.9082963158]
           , "YNy": [0.58918837022, 0.77018941065, 0.7718741164, 0.84675601192]
           }
    net = pp.create_empty_network()
    for vc in results.keys():
         add_network(net, vc)
         try:
             sc.calc_sc(net, fault="1ph", case="max")
         except:
             raise UserWarning("Did not converge after adding transformer with vector group %s"%vc)
    
    for vc, result in results.items():
        check_results(net, vc, result)

def test_iec60909_example_4():
    file = os.path.join(os.path.dirname(pandapower.test.__file__), "test_files",
                        "IEC60909-4_example.json")
    net = pp.from_json(file)
    sc.calc_sc(net, fault="1ph")
    assert np.isclose(net.res_bus_sc[net.bus.name=="Q"].ikss_ka.values[0], 10.05957231)
    assert np.isclose(net.res_bus_sc[net.bus.name=="T2LV"].ikss_ka.values[0], 34.467353142)
    assert np.isclose(net.res_bus_sc[net.bus.name=="F1"].ikss_ka.values[0], 35.53066312)
    assert np.isclose(net.res_bus_sc[net.bus.name=="F2"].ikss_ka.values[0], 34.89135137)
    assert np.isclose(net.res_bus_sc[net.bus.name=="F3"].ikss_ka.values[0], 5.0321033105)
    assert np.isclose(net.res_bus_sc[net.bus.name=="Cable/Line IC"].ikss_ka.values[0], 16.362586813)

        
if __name__ == "__main__":
    pytest.main(["test_1ph.py"])
