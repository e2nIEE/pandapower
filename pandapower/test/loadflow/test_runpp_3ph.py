# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:06:25 2018
Tests 3 phase power flow algorithm
@author: sghosh
"""
from numpy import complex128
import pandapower as pp
import numpy as np
import pytest
from pandapower.pf.runpp_3ph import combine_X012
from pandapower.create import create_asymmetric_load, create_load
from pandapower.pf.runpp_3ph import runpp_3ph
import copy
from pandapower.pypower.makeYbus import makeYbus 
from pandapower.pf.runpp_3ph import I0_from_V012,I1_from_V012,I2_from_V012


@pytest.fixture
def net():
    V_base = 110              # 110kV Base Voltage
    kVA_base = 100         # 100 MVA
#    I_base = (kVA_base/V_base) * 1e-3           # in kA
    net = pp.create_empty_network(sn_mva = kVA_base )
    
    pp.create_bus(net, vn_kv = V_base, index=1)
    pp.create_bus(net, vn_kv = V_base, index=5)
    
    pp.create_ext_grid(net, bus=1, vm_pu= 1.0, s_sc_max_mva=5000, rx_max=0.1)
    net.ext_grid["r0x0_max"] = 0.1
    net.ext_grid["x0x_max"] = 1.0
    
    pp.create_std_type(net, {"r0_ohm_per_km": 0.0848, "x0_ohm_per_km": 0.4649556, "c0_nf_per_km":
                             230.6,"max_i_ka": 0.963, "r_ohm_per_km": 0.0212,
                             "x_ohm_per_km": 0.1162389, "c_nf_per_km":  230},
                       "example_type")
    pp.create_line(net, from_bus = 1, to_bus = 5, length_km = 50.0, std_type="example_type")

    create_asymmetric_load(net, 5, p_A_mw=50, q_A_mvar=50, p_B_mw=10, q_B_mvar=15,
                       p_C_mw=10, q_C_mvar=5)
    return net


def check_it(net):
    assert np.allclose(net.res_bus_3ph.vmA_pu[~np.isnan(net.res_bus_3ph.vmA_pu)],
                       np.array([0.96742893, 0.74957533]),atol=1e-4)
    assert np.allclose(net.res_bus_3ph.vmB_pu[~np.isnan(net.res_bus_3ph.vmB_pu)],
                       np.array([1.01302766, 1.09137945]),atol=1e-4)
    assert np.allclose(net.res_bus_3ph.vmC_pu[~np.isnan(net.res_bus_3ph.vmC_pu)],
                       np.array([1.019784, 1.05124282]),atol=1e-4)

    assert abs(net.res_line_3ph.iA_from_ka.values[0] - 1.34212045) < 1e-4
    assert abs(net.res_line_3ph.iA_to_ka.values[0]   - 1.48537916) < 1e-4
    
    assert abs(net.res_line_3ph.iB_from_ka.values[0] - 0.13715552) < 1e-4
    assert abs(net.res_line_3ph.iB_to_ka.values[0]   - 0.26009611) < 1e-4
               
    assert abs(net.res_line_3ph.iC_from_ka.values[0] - 0.22838401) < 1e-4
    assert abs(net.res_line_3ph.iC_to_ka.values[0]   - 0.1674634) < 1e-4
        
    assert abs(net.res_line_3ph.p_A_from_mw.values[0]   - 55.70772301) < 1e-4
    assert abs(net.res_line_3ph.p_A_to_mw.values[0]     - (-49.999992954)) < 1e-4
    assert abs(net.res_line_3ph.q_A_from_mvar.values[0] - 60.797262682) < 1e-4
    assert abs(net.res_line_3ph.q_A_to_mvar.values[0]   - (-49.999959283)) < 1e-4
               
    assert abs(net.res_line_3ph.p_B_from_mw.values[0]   - 8.7799379802) < 1e-4
    assert abs(net.res_line_3ph.p_B_to_mw.values[0]     - (-9.9999996625)) < 1e-4           
    assert abs(net.res_line_3ph.q_B_from_mvar.values[0] - (-0.88093549983)) < 1e-4
    assert abs(net.res_line_3ph.q_B_to_mvar.values[0]   - (-15.000000238)) < 1e-4
               
    assert abs(net.res_line_3ph.p_C_from_mw.values[0]   - 9.3739293122) < 1e-4
    assert abs(net.res_line_3ph.p_C_to_mw.values[0]     - (-10.000000161)) < 1e-4
    assert abs(net.res_line_3ph.q_C_from_mvar.values[0] - (-11.441663679)) < 1e-4
    assert abs(net.res_line_3ph.q_C_to_mvar.values[0]   - (-4.9999997418)) < 1e-4
           	        
    assert abs(net.res_line_3ph.loading_percentA.values[0] - 154.2452) < 1e-2
    assert abs(net.res_line_3ph.loading_percentB.values[0] - 27.00894) < 1e-2
    assert abs(net.res_line_3ph.loading_percentC.values[0] - 23.71589) < 1e-2
    assert abs(net.res_line_3ph.loading_percent.values[0]  - 154.2452) < 1e-2


def test_2bus_network(net):
    "#-o---o"
    pp.add_zero_impedance_parameters(net)
    assert runpp_3ph(net)[3]["success"]
    check_it(net)    


def test_2bus_network_single_isolated_busses(net):
    "#-o---o o x"
    pp.create_bus(net, vn_kv=110)
    pp.create_bus(net, vn_kv=110, in_service=False)
    pp.add_zero_impedance_parameters(net)
    assert runpp_3ph(net)[3]["success"]
    check_it(net)    


def test_2bus_network_isolated_net_part(net):
    "#-o---o o---o"
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)
    pp.create_line(net, from_bus=b1, to_bus=b2, length_km = 50.0, std_type="example_type")
    create_asymmetric_load(net, b2, p_A_mw=50, q_A_mvar=50, p_B_mw=10, q_B_mvar=15,
                   p_C_mw=10, q_C_mvar=5)
    pp.add_zero_impedance_parameters(net)
    assert runpp_3ph(net)[3]["success"]
    check_it(net)

def test_2bus_network_singel_oos_bus(net):
    "#-o---x---o"
    b1 = pp.create_bus(net, vn_kv=110)
    net.bus.loc[5, "in_service"] = False
    pp.create_line(net, from_bus=5, to_bus=b1, length_km = 10.0, std_type="example_type")
    create_asymmetric_load(net, b1, p_A_mw=-5, q_A_mvar=5, p_B_mw=-1, q_B_mvar=1.5,
                    p_C_mw=-1, q_C_mvar=.5)
    pp.add_zero_impedance_parameters(net)
    assert runpp_3ph(net)[3]["success"]


def test_4bus_network():
    V_base = 110                     # 110kV Base Voltage
    kVA_base = 100000                      # 100 MVA

    net = pp.create_empty_network(sn_mva = kVA_base )
    # =============================================================================
    # Main Program
    # =============================================================================
    busn  =  pp.create_bus(net, vn_kv = V_base, name = "busn")
    busk  =  pp.create_bus(net, vn_kv = V_base, name = "busk")
    busm =  pp.create_bus(net, vn_kv = V_base, name = "busm")
    busp =  pp.create_bus(net, vn_kv = V_base, name = "busp")
    pp.create_ext_grid(net, bus=busn, vm_pu=1.0, name="Grid Connection", s_sc_max_mva=5000, 
                       rx_max=0.1)
    net.ext_grid["r0x0_max"] = 0.1
    net.ext_grid["x0x_max"] = 1.0
    
    pp.create_std_type(net, {"r0_ohm_per_km": 0.0848, "x0_ohm_per_km": 0.4649556, "c0_nf_per_km":\
        230.6,   "max_i_ka": 0.963, "r_ohm_per_km": 0.0212, "x_ohm_per_km": 0.1162389,
                 "c_nf_per_km":  230}, "example_type1")
    
    pp.create_std_type(net, {"r0_ohm_per_km" : .3048, "x0_ohm_per_km" : 0.6031856, 
                             "c0_nf_per_km" : 140.3, "max_i_ka": 0.531, 
                              "r_ohm_per_km" : .0762, "x_ohm_per_km" : 0.1507964
                            , "c_nf_per_km" : 140}, "example_type2")
    pp.create_std_type(net, {"r0_ohm_per_km" : .154, "x0_ohm_per_km" : 0.5277876
                             , "c0_nf_per_km" : 170.4, "max_i_ka": 0.741, 
                              "r_ohm_per_km" : .0385, "x_ohm_per_km" : 0.1319469
                             , "c_nf_per_km" : 170}, "example_type3")
    
    pp.create_std_type(net, {"r0_ohm_per_km" : .1005, "x0_ohm_per_km" : 0.4900884
                             , "c0_nf_per_km":  200.5, "max_i_ka" : 0.89
                             , "r_ohm_per_km": .0251, "x_ohm_per_km" : 0.1225221
                             , "c_nf_per_km" : 210}, "example_type4")
    
    pp.create_line(net, from_bus = busn, to_bus = busm, length_km = 1.0, std_type="example_type3")
    pp.create_line(net, from_bus = busn, to_bus = busp, length_km = 1.0, std_type="example_type3")
    pp.create_line(net, from_bus = busn, to_bus = busk, length_km = 1.0, std_type="example_type4")
    pp.create_line(net, from_bus = busk, to_bus = busm, length_km = 1.0, std_type="example_type1")
    pp.create_line(net, from_bus = busk, to_bus = busp, length_km = 1.0, std_type="example_type2")
    pp.add_zero_impedance_parameters(net)
    
    create_asymmetric_load(net, busk, p_A_mw=50, q_A_mvar=20, p_B_mw=80, q_B_mvar=60,
                       p_C_mw=20, q_C_mvar=5)
    create_asymmetric_load(net, busm, p_A_mw=50, q_A_mvar=50, p_B_mw=10, q_B_mvar=15,
                       p_C_mw=10, q_C_mvar=5)
    create_asymmetric_load(net, busp, p_A_mw=50, q_A_mvar=20, p_B_mw=60, q_B_mvar=20,
                       p_C_mw=10, q_C_mvar=5)
    assert runpp_3ph(net)[3]["success"]
    
    v_a_pf = np.array([0.98085729,  0.97828577,  0.97774307,  0.9780892])
    v_b_pf = np.array([0.97711997,  0.97534651,  0.97648197,  0.97586805])
    v_c_pf = np.array([1.04353786,  1.04470864,  1.04421233,  1.04471106])
    
    assert np.allclose(net.res_bus_3ph.vmA_pu,v_a_pf,atol=1e-8 )
    assert np.allclose(net.res_bus_3ph.vmB_pu,v_b_pf,atol=1e-8 )
    assert np.allclose(net.res_bus_3ph.vmC_pu,v_c_pf,atol=1e-8 )
    
    i_a_f_pf = np.array([0.98898804851	,    0.87075816277		,	0.95760407055	,	0.21780921494 	,	0.03712221482])
    i_b_f_pf = np.array([0.68943734300	,	1.03463205150 	,	1.14786582480	,	0.42795802661	,	0.10766244499])
    i_c_f_pf = np.array([0.19848961274	,	0.19072621839	,	0.24829126422	,	0.03706411747	,	0.03093504641])

    i_a_t_pf = np.array([0.99093993074  ,   0.87210778953,	0.95975383019,0.22229618812	,0.03446870607])
    i_b_t_pf = np.array([0.69146383889	,	1.03599166860	,	1.15028039760	,	0.42603286136 	,	0.10500385951])
    i_c_t_pf = np.array([0.19966503490	,	0.19188990522 	,	0.24975552990	,	0.03771702877 	,	0.03179428313])
    assert np.allclose(net.res_line_3ph.iA_from_ka,i_a_f_pf,atol=1e-4)
    assert np.allclose(net.res_line_3ph.iB_from_ka,i_b_f_pf,atol=1e-4)
    assert np.allclose(net.res_line_3ph.iC_from_ka,i_c_f_pf,atol=1e-4)
    
    assert np.allclose(net.res_line_3ph.iA_to_ka,i_a_t_pf,atol=1e-4)
    assert np.allclose(net.res_line_3ph.iB_to_ka,i_b_t_pf,atol=1e-4)
    assert np.allclose(net.res_line_3ph.iC_to_ka,i_c_t_pf,atol=1e-4)
    
    p_a_f_pf =np.array([49.87434308300	,	49.59359423400	,	50.87938854400	,	0.23292404144	,	0.52956690003])
    p_b_f_pf =np.array([33.86579548300	,	58.53676841800	,	57.53628872500	,	-23.88471674400	,	1.44846451670])
    p_c_f_pf =np.array([12.44659879200	,	11.97553940900	,	15.54470530900	,	-2.45255094990	,	-1.98645639180]) 
    
    q_a_f_pf =np.array([36.16562612500	,	21.96967200100	,	31.13888556700	,	13.53037091600	,	-2.24476446080])
    q_b_f_pf =np.array([26.14426519000	,	26.37559958400	,	41.99378843000	,	-11.49972059800	,	-6.50971484920])
    q_c_f_pf =np.array([4.25746427990	,	4.04458872690	,	5.39758513360	,	0.17971664973	,	0.51637909987]) 
    
    p_a_t_pf =np.array([-49.75842138300	,	-49.47110288600	,	-50.76249094400	,	-0.24157861683	,	-0.52889711581]) 
    p_b_t_pf =np.array([-33.90236497100	,	-58.55284704700	,	-57.56374776900	,	23.90236496800	,	-1.44715294780])
    p_c_t_pf =np.array([-12.45155362100	,	-11.98669515800	,	-15.56099267200	,	2.45155361280	,	1.98669514890]) 
    
    q_a_t_pf =np.array([-36.19862687800	,	-22.07474008400	,	-31.28560645900	,	-13.80137312400	,	2.07474008110]) 
    q_b_t_pf =np.array([-26.25675246200	,	-26.34476810500	,	-41.99056453000	,	11.25675246600	,	6.34476811700])
    q_c_t_pf =np.array([-4.50384237760	,	-4.29078446670	,	-5.69609575190	,	-0.49615762485	,	-0.70921553583]) 
    
    assert np.allclose(net.res_line_3ph.p_A_from_mw,p_a_f_pf,atol=1e-2)
    assert np.allclose(net.res_line_3ph.p_B_from_mw,p_b_f_pf,atol=1e-2)
    assert np.allclose(net.res_line_3ph.p_C_from_mw,p_c_f_pf,atol=1e-2)
    
    assert np.allclose(net.res_line_3ph.q_A_from_mvar,q_a_f_pf,atol=1e-2)
    assert np.allclose(net.res_line_3ph.q_B_from_mvar,q_b_f_pf,atol=1e-2)
    assert np.allclose(net.res_line_3ph.q_C_from_mvar,q_c_f_pf,atol=1e-2)
    
    assert np.allclose(net.res_line_3ph.p_A_to_mw,p_a_t_pf,atol=1e-2)
    assert np.allclose(net.res_line_3ph.p_B_to_mw,p_b_t_pf,atol=1e-2)
    assert np.allclose(net.res_line_3ph.p_C_to_mw,p_c_t_pf,atol=1e-2)
    
    assert np.allclose(net.res_line_3ph.q_A_to_mvar,q_a_t_pf,atol=1e-2)
    assert np.allclose(net.res_line_3ph.q_B_to_mvar,q_b_t_pf,atol=1e-2)
    assert np.allclose(net.res_line_3ph.q_C_to_mvar,q_c_t_pf,atol=1e-2)
    
    load_a_pf = np.array([133.73010000000	,	117.69340000000	,	107.83750000000	,	23.08372000000	,	6.99100100000]) 
    load_b_pf = np.array([93.31496000000	,	139.80990000000	,	129.24500000000	,	44.44009000000	,	20.27541000000]) 
    load_c_pf = np.array([26.94535000000	,	25.89607000000	,	28.06242000000	,	3.91661800000	,	5.98762400000]) 
    load_pf = np.array([133.73010000000	,	139.80990000000	,	129.24500000000	,	44.44009000000	,	20.27541000000]) 
    assert np.allclose(net.res_line_3ph.loading_percentA ,load_a_pf ,atol=1e-2)
    assert np.allclose(net.res_line_3ph.loading_percentB ,load_b_pf,atol=1e-2 ) 	
    assert np.allclose(net.res_line_3ph.loading_percentC ,load_c_pf,atol=1e-2 ) 
    assert np.allclose(net.res_line_3ph.loading_percent ,load_pf,atol=1e-2 )



def test_in_serv_load():
    V_base = 110                     # 110kV Base Voltage
    MVA_base = 100                      # 100 MVA

    net = pp.create_empty_network(sn_mva = MVA_base )
    
    busn  =  pp.create_bus(net, vn_kv = V_base, name = "busn", index=1)
    busk  =  pp.create_bus(net, vn_kv = V_base, name = "busk", index=5)
    pp.create_bus(net, vn_kv=20., in_service=False)
    pp.create_bus(net, vn_kv=20., in_service=True)
    
    
    pp.create_ext_grid(net, bus=busn, vm_pu= 1.0, name="Grid Connection",
                       s_sc_max_mva=5000, rx_max=0.1)
    net.ext_grid["r0x0_max"] = 0.1
    net.ext_grid["x0x_max"] = 1.0
    
    pp.create_std_type(net, {"r0_ohm_per_km": 0.0848, "x0_ohm_per_km": 0.4649556, "c0_nf_per_km":\
        230.6,"max_i_ka": 0.963, "r_ohm_per_km": 0.0212, "x_ohm_per_km": 0.1162389,
                 "c_nf_per_km":  230}, "example_type")
    
    create_asymmetric_load(net, busk, p_A_mw=50, q_A_mvar=50, p_B_mw=10, q_B_mvar=15,
                       p_C_mw=10, q_C_mvar=5)
    
    pp.create_line(net, from_bus = busn, to_bus = busk, length_km = 50.0, std_type="example_type")
    
    pp.add_zero_impedance_parameters(net)
    
    assert runpp_3ph(net)[3]["success"]
    
    assert np.allclose(net.res_bus_3ph.vmA_pu[~np.isnan(net.res_bus_3ph.vmA_pu)],
                                              np.array([0.96742893, 0.74957533]))
    assert np.allclose(net.res_bus_3ph.vmB_pu[~np.isnan(net.res_bus_3ph.vmB_pu)],
                                              np.array([1.01302766, 1.09137945]))
    assert np.allclose(net.res_bus_3ph.vmC_pu[~np.isnan(net.res_bus_3ph.vmC_pu)],
                                              np.array([1.019784, 1.05124282]))

    assert abs(net.res_line_3ph.iA_from_ka.values[0] - 1.34212045) < 1e-5
    assert abs(net.res_line_3ph.iA_to_ka.values[0]   - 1.48537916) < 1e-5
    
    assert abs(net.res_line_3ph.iB_from_ka.values[0] - 0.13715552) < 1e-5
    assert abs(net.res_line_3ph.iB_to_ka.values[0]   - 0.26009611) < 1e-5
               
    assert abs(net.res_line_3ph.iC_from_ka.values[0] - 0.22838401) < 1e-5
    assert abs(net.res_line_3ph.iC_to_ka.values[0]   - 0.1674634) < 1e-5
        
    assert abs(net.res_line_3ph.p_A_from_mw.values[0]   - 55.70772301) < 1e-4
    assert abs(net.res_line_3ph.p_A_to_mw.values[0]     - (-49.999992954)) < 1e-4
    assert abs(net.res_line_3ph.q_A_from_mvar.values[0] - 60.797262682) < 1e-4
    assert abs(net.res_line_3ph.q_A_to_mvar.values[0]   - (-49.999959283)) < 1e-4
               
    assert abs(net.res_line_3ph.p_B_from_mw.values[0]  - 8.7799379802) < 1e-4
    assert abs(net.res_line_3ph.p_B_to_mw.values[0]     - (-9.9999996625)) < 1e-4           
    assert abs(net.res_line_3ph.q_B_from_mvar.values[0] - (-0.88093549983)) < 1e-4
    assert abs(net.res_line_3ph.q_B_to_mvar.values[0]   - (-15.000000238)) < 1e-4
               
    assert abs(net.res_line_3ph.p_C_from_mw.values[0]   - 9.3739293122) < 1e-4
    assert abs(net.res_line_3ph.p_C_to_mw.values[0]     - (-10.000000161)) < 1e-4
    assert abs(net.res_line_3ph.q_C_from_mvar.values[0] - (-11.441663679)) < 1e-4
    assert abs(net.res_line_3ph.q_C_to_mvar.values[0]   - (-4.9999997418)) < 1e-4
           	        
    assert abs(net.res_line_3ph.loading_percentA.values[0] - 154.2452) < 1e-2
    assert abs(net.res_line_3ph.loading_percentB.values[0] - 27.00894) < 1e-2
    assert abs(net.res_line_3ph.loading_percentC.values[0] - 23.71589) < 1e-2
    assert abs(net.res_line_3ph.loading_percent.values[0]  - 154.2452) < 1e-2
    
    create_asymmetric_load(net, busk, p_A_mw=50, q_A_mvar=100, p_B_mw=29, q_B_mvar=38,
                   p_C_mw=10, q_C_mvar=5, in_service=False)

    assert runpp_3ph(net)[3]["success"]
    
    assert np.allclose(net.res_bus_3ph.vmA_pu[~np.isnan(net.res_bus_3ph.vmA_pu)],
                                              np.array([0.96742893, 0.74957533]))
    assert np.allclose(net.res_bus_3ph.vmB_pu[~np.isnan(net.res_bus_3ph.vmB_pu)],
                                              np.array([1.01302766, 1.09137945]))
    assert np.allclose(net.res_bus_3ph.vmC_pu[~np.isnan(net.res_bus_3ph.vmC_pu)],
                                              np.array([1.019784, 1.05124282]))

    assert abs(net.res_line_3ph.iA_from_ka.values[0] - 1.34212045) < 1e-5
    assert abs(net.res_line_3ph.iA_to_ka.values[0]   - 1.48537916) < 1e-5
    
    assert abs(net.res_line_3ph.iB_from_ka.values[0] - 0.13715552) < 1e-5
    assert abs(net.res_line_3ph.iB_to_ka.values[0]   - 0.26009611) < 1e-5
               
    assert abs(net.res_line_3ph.iC_from_ka.values[0] - 0.22838401) < 1e-5
    assert abs(net.res_line_3ph.iC_to_ka.values[0]   - 0.1674634) < 1e-5
        
    assert abs(net.res_line_3ph.p_A_from_mw.values[0]   - 55.70772301) < 1e-4
    assert abs(net.res_line_3ph.p_A_to_mw.values[0]     - (-49.999992954)) < 1e-4
    assert abs(net.res_line_3ph.q_A_from_mvar.values[0] - 60.797262682) < 1e-4
    assert abs(net.res_line_3ph.q_A_to_mvar.values[0]   - (-49.999959283)) < 1e-4
               
    assert abs(net.res_line_3ph.p_B_from_mw.values[0]  - 8.7799379802) < 1e-4
    assert abs(net.res_line_3ph.p_B_to_mw.values[0]     - (-9.9999996625)) < 1e-4           
    assert abs(net.res_line_3ph.q_B_from_mvar.values[0] - (-0.88093549983)) < 1e-4
    assert abs(net.res_line_3ph.q_B_to_mvar.values[0]   - (-15.000000238)) < 1e-4
               
    assert abs(net.res_line_3ph.p_C_from_mw.values[0]   - 9.3739293122) < 1e-4
    assert abs(net.res_line_3ph.p_C_to_mw.values[0]     - (-10.000000161)) < 1e-4
    assert abs(net.res_line_3ph.q_C_from_mvar.values[0] - (-11.441663679)) < 1e-4
    assert abs(net.res_line_3ph.q_C_to_mvar.values[0]   - (-4.9999997418)) < 1e-4
           	        
    assert abs(net.res_line_3ph.loading_percentA.values[0] - 154.2452) < 1e-2
    assert abs(net.res_line_3ph.loading_percentB.values[0] - 27.00894) < 1e-2
    assert abs(net.res_line_3ph.loading_percentC.values[0] - 23.71589) < 1e-2
    assert abs(net.res_line_3ph.loading_percent.values[0]  - 154.2452) < 1e-2



def test_3ph_bus_mapping_order():
    net = pp.create_empty_network()
    
    b2 = pp.create_bus(net, vn_kv=0.4, index=4)
    pp.create_bus(net, vn_kv=0.4, in_service=False, index=3)
    b1 = pp.create_bus(net, vn_kv=0.4, index=7)
    
    pp.create_ext_grid(net, b1, vm_pu=1.0, s_sc_max_mva=10, rx_max=0.1)
    net.ext_grid["x0x_max"] = 1.
    net.ext_grid["r0x0_max"] = 0.1
    pp.create_std_type(net, {"r_ohm_per_km":0.1013, "x_ohm_per_km": 0.06911504,
                             "c_nf_per_km": 690, "g_us_per_km": 0, "max_i_ka": 0.44,
                             "c0_nf_per_km": 312.4, "r0_ohm_per_km": 0.4053,
                             "x0_ohm_per_km": 0.2764602},"N2XRY 3x185sm 0.6/1kV")
    
    pp.create_line(net, b1, b2, 1.0, std_type="N2XRY 3x185sm 0.6/1kV", index=4)
    pp.create_line(net, b1, b2, 1.0, std_type="N2XRY 3x185sm 0.6/1kV", index=3, in_service=False)
    pp.create_line(net, b1, b2, 1.0, std_type="N2XRY 3x185sm 0.6/1kV", index=7)
    pp.add_zero_impedance_parameters(net)
    pp.create_load(net, b2, p_mw=0.030, q_mvar=0.030)
    pp.runpp(net)
    assert runpp_3ph(net)[3]["success"]
    
    assert np.allclose(net.res_bus_3ph.vmA_pu.values, net.res_bus.vm_pu.values, equal_nan=True)
    assert net.res_bus_3ph.index.tolist() == net.res_bus.index.tolist()
    
    assert net.res_line_3ph.index.tolist() == net.res_line.index.tolist()
    assert np.allclose(net.res_line.p_from_mw, net.res_line_3ph.p_A_from_mw +
                                               net.res_line_3ph.p_B_from_mw +
                                               net.res_line_3ph.p_C_from_mw )
    assert np.allclose(net.res_line.loading_percent, net.res_line_3ph.loading_percentA)  

    
def test_3ph_two_bus_line_powerfactory():
    net = pp.create_empty_network()
    
    b1 = pp.create_bus(net, vn_kv=0.4)
    b2 = pp.create_bus(net, vn_kv=0.4)
    
    pp.create_ext_grid(net, b1, vm_pu=1.0, s_sc_max_mva=10, rx_max=0.1)
    net.ext_grid["x0x_max"] = 1.
    net.ext_grid["r0x0_max"] = 0.1
    pp.create_std_type(net, {"r_ohm_per_km":0.1013, "x_ohm_per_km": 0.06911504,
                             "c_nf_per_km": 690, "g_us_per_km": 0, "max_i_ka": 0.44,
                             "c0_nf_per_km": 312.4, "r0_ohm_per_km": 0.4053,
                             "x0_ohm_per_km": 0.2764602}, "N2XRY 3x185sm 0.6/1kV")
    
    pp.create_line(net, b1, b2, 0.4, std_type="N2XRY 3x185sm 0.6/1kV")
    pp.add_zero_impedance_parameters(net)
    pp.create_load(net, b2, p_mw=0.010, q_mvar=0.010)
    pp.create_asymmetric_load(net, b2, p_A_mw=0.020, q_A_mvar=0.010, p_B_mw=0.015, q_B_mvar=0.005, p_C_mw=0.025,
                       q_C_mvar=0.010)
    
    assert runpp_3ph(net)[3]["success"]
    
    assert np.allclose(net.res_bus_3ph.vmA_pu, np.array([0.99939853552, 0.97401782343]))
    assert np.allclose(net.res_bus_3ph.vmB_pu, np.array([1.0013885141, 0.98945593737]))
    assert np.allclose(net.res_bus_3ph.vmC_pu, np.array([0.99921580141, 0.96329605983]))

    assert abs(net.res_line_3ph.iA_from_ka.values[0] - 0.11946088987) < 1e-5
    assert abs(net.res_line_3ph.iA_to_ka.values[0]   - 0.1194708224) < 1e-5
    
    assert abs(net.res_line_3ph.iB_from_ka.values[0] - 0.08812337783) < 1e-5
    assert abs(net.res_line_3ph.iB_to_ka.values[0]   - 0.088131567331) < 1e-5
               
    assert abs(net.res_line_3ph.iC_from_ka.values[0] - 0.14074226065) < 1e-5
    assert abs(net.res_line_3ph.iC_to_ka.values[0]   - 0.14075063601) < 1e-5
        
    assert abs(net.res_line_3ph.p_A_from_mw.values[0]   - 0.023810539354) < 1e-2
    assert abs(net.res_line_3ph.p_A_to_mw.values[0]     + 0.023333142958) < 1e-2
    assert abs(net.res_line_3ph.q_A_from_mvar.values[0] - 0.013901720672) < 1e-2
    assert abs(net.res_line_3ph.q_A_to_mvar.values[0]   + 0.013332756527) < 1e-2
               
    assert abs(net.res_line_3ph.p_B_from_mw.values[0]   - 0.01855791658) < 1e-2
    assert abs(net.res_line_3ph.p_B_to_mw.values[0]     + 0.018333405987) < 1e-2           
    assert abs(net.res_line_3ph.q_B_from_mvar.values[0] - 0.008421814704) < 1e-2
    assert abs(net.res_line_3ph.q_B_to_mvar.values[0]   + 0.008333413919) < 1e-2
               
    assert abs(net.res_line_3ph.p_C_from_mw.values[0]   - 0.029375192747) < 1e-2
    assert abs(net.res_line_3ph.p_C_to_mw.values[0]     + 0.028331643666) < 1e-2
    assert abs(net.res_line_3ph.q_C_from_mvar.values[0] - 0.013852398586) < 1e-2
    assert abs(net.res_line_3ph.q_C_to_mvar.values[0]   + 0.013332422725) < 1e-2
               
    assert abs(net.res_line_3ph.loading_percentA.values[0] - 27.1525) < 1e-2
    assert abs(net.res_line_3ph.loading_percentB.values[0] - 20.0299) < 1e-2
    assert abs(net.res_line_3ph.loading_percentC.values[0] - 31.98878) < 1e-2
    assert abs(net.res_line_3ph.loading_percent.values[0]  - 31.98878) < 1e-2
    
def check_results(net, vc, result):
    res_vm_kv = np.concatenate(
            (
            net.res_bus_3ph[(net.bus.zone==vc)&(net.bus.in_service)].vmA_pu
#           , net.res_bus_3ph[(net.bus.zone==vc)&(net.bus.in_service)].vaA_degree,
            ,net.res_bus_3ph[(net.bus.zone==vc)&(net.bus.in_service)].vmB_pu
#           , net.res_bus_3ph[(net.bus.zone==vc)&(net.bus.in_service)].vaB_degree,
            ,net.res_bus_3ph[(net.bus.zone==vc)&(net.bus.in_service)].vmC_pu
#           , net.res_bus_3ph[(net.bus.zone==vc)&(net.bus.in_service)].vaC_degree
            )
            ,axis =0)
    if not np.allclose(result, res_vm_kv,atol=1e-4):
        raise ValueError("Incorrect results for vector group %s"%vc, res_vm_kv, result)
        
def make_nw(net,bushv,tap_ps,case,vector_group):
    b1 = pp.create_bus(net, bushv, zone=vector_group, index=pp.get_free_id(net.bus))
    b2 = pp.create_bus(net, 0.4, zone=vector_group)
    b3 = pp.create_bus(net, 0.4, zone=vector_group)
    
    pp.create_ext_grid(net, b1, s_sc_max_mva=10000, s_sc_min_mva=8000, rx_min=0.1, rx_max=0.1)
    net.ext_grid["r0x0_max"] = 0.1
    net.ext_grid["x0x_max"] = 1.0
    
    pp.create_std_type(net, {"sn_mva": 1.6,
                "vn_hv_kv": 10,
                "vn_lv_kv": 0.4,
                "vk_percent": 6,
                "vkr_percent": 0.78125,
                "pfe_kw": 2.7,
                "i0_percent": 0.16875,
                "shift_degree": 0,
                "vector_group": vector_group,
                "tap_side": "lv",
                "tap_neutral": 0,
                "tap_min": -2,
                "tap_max": 2,
                "tap_step_degree": 0,
                "tap_step_percent": 2.5,
                "tap_phase_shifter": False,
                "vk0_percent": 6, 
                "vkr0_percent": 0.78125, 
                "mag0_percent": 100,
                "mag0_rx": 0.,
                "si0_hv_partial": 0.9,}, vector_group, "trafo")
    
    t1=pp.create_transformer(net, b1, b2, std_type=vector_group, parallel=1,tap_pos=tap_ps,
    					  index=pp.get_free_id(net.trafo)+1)
    
    
    pp.create_std_type(net, {"r_ohm_per_km": 0.1941, "x_ohm_per_km": 0.07476991,
                        "c_nf_per_km": 1160., "max_i_ka": 0.421,
                        "endtemp_degree": 70.0, "r0_ohm_per_km": 0.7766,
                        "x0_ohm_per_km": 0.2990796,
                        "c0_nf_per_km":  496.2}, "unsymmetric_line_type")
    
            
    pp.create_line(net, b2, b3, length_km=0.5, std_type="unsymmetric_line_type",
    			   index=pp.get_free_id(net.line)+1)
    
    if case == "bal_wye":
        ##Symmetric Load
        pp.create_load(net,b3,0.08,0.012,type='wye')
    elif case == "delta_wye":
        #Unsymmetric Light Load
        create_asymmetric_load(net, b3, p_A_mw=0.0044, q_A_mvar=0.0013, p_B_mw=0.0044, q_B_mvar=0.0013,
                               p_C_mw=0.0032, q_C_mvar=0.0013,type='wye')
        create_asymmetric_load(net, b3, p_A_mw=0.0300, q_A_mvar=0.0048, p_B_mw=0.0280, q_B_mvar=0.0036,
                       p_C_mw=0.027, q_C_mvar=0.0043, type ='delta')
        
    elif case == "wye":
        ##Unsymmetric Heavy Load
        create_asymmetric_load(net, b3, p_A_mw=0.0300, q_A_mvar=0.0048, p_B_mw=0.0280, q_B_mvar=0.0036,
                               p_C_mw=0.027, q_C_mvar=0.0043,type=case)
    elif case == "delta":
        create_asymmetric_load(net, b3, p_A_mw=0.0300, q_A_mvar=0.0048, p_B_mw=0.0280, q_B_mvar=0.0036,
                       p_C_mw=0.027, q_C_mvar=0.0043,type=case)
    
    pp.add_zero_impedance_parameters(net) 
    return t1
    
def test_trafo_asym():
    
# =============================================================================
# TODO: Check why there is formation of 2x1 Y0 bus matrix for other vector groups
# It has something to do with Y sh for these groups    
# =============================================================================
    results = get_PowerFactory_Results()   # Results taken out from PF using a different script   
    for bushv in [10,11]:
        for tap_ps in [0,1]:
            for loadtyp in ["delta","wye","delta_wye","bal_wye"]:
                for vc in ["YNyn","Dyn","Yzn"]:#,"Yyn"]:
                    net = pp.create_empty_network(sn_mva = 100) 
                    make_nw(net,bushv,tap_ps,loadtyp,vc)
                    assert runpp_3ph(net)[3]["success"]
                    check_results(net,vc,results[bushv][tap_ps][loadtyp][vc])


def test_2trafos():
    net = pp.create_empty_network() 
    make_nw(net,10.,0.,"wye","YNyn")
    make_nw(net,10.,0., "wye","YNyn")
    assert runpp_3ph(net)[3]["success"]
    assert np.allclose(net.res_ext_grid_3ph.iloc[0].values, net.res_ext_grid_3ph.iloc[1].values)
    

def test_3ph_isolated_nodes():
    V_base = 110  # 110kV Base Voltage
    MVA_base = 100  # 100 MVA
    net = pp.create_empty_network(sn_mva=MVA_base)

    busn = pp.create_bus(net, vn_kv=V_base, name="busn", index=1)
    busx = pp.create_bus(net, vn_kv=20., in_service=True, index=2, name="busx")
    busk = pp.create_bus(net, vn_kv=V_base, name="busk", index=5)
    busl = pp.create_bus(net, vn_kv=V_base, name="busl", index=6)
    pp.create_bus(net, vn_kv=20., in_service=False, index=3)
    busy = pp.create_bus(net, vn_kv=20., in_service=True, index=0, name="busy")

    pp.create_ext_grid(net, bus=busn, vm_pu=1.0, name="Grid Connection",
                       s_sc_max_mva=5000, rx_max=0.1)
    net.ext_grid["r0x0_max"] = 0.1
    net.ext_grid["x0x_max"] = 1.0

    pp.create_std_type(net, {"r0_ohm_per_km": 0.0848, "x0_ohm_per_km": 0.4649556, "c0_nf_per_km": \
        230.6, "max_i_ka": 0.963, "r_ohm_per_km": 0.0212, "x_ohm_per_km": 0.1162389,
                             "c_nf_per_km": 230}, "example_type")

    # Loads on supplied buses
    create_asymmetric_load(net, busk, p_A_mw=50, q_A_mvar=50, p_B_mw=10, q_B_mvar=15,
                    p_C_mw=10, q_C_mvar=5)
    create_load(net, bus=busl, p_mw=7, q_mvar=0.070, name="Load 1")

    # Loads on unsupplied buses
    # create_load(net, bus=busy, p_mw=0, q_mvar=0, name="Load Y")
    create_load(net, bus=busy, p_mw=70, q_mvar=70, name="Load Y")
    # create_asymmetric_load(net, busx, p_A_mw=5000, q_A_mvar=5000, p_B_mw=1000, q_B_mvar=1500,
    #                 p_C_mw=1000, q_C_mvar=500, name="Load X")

    pp.create_line(net, from_bus=busn, to_bus=busk, length_km=50.0, std_type="example_type")
    pp.create_line(net, from_bus=busl, to_bus=busk, length_km=50.0, std_type="example_type")

    pp.add_zero_impedance_parameters(net)

    r = runpp_3ph(net)

    assert r[3]["success"]
    assert np.allclose(net.res_bus_3ph.T[[0, 2, 3]].T[["vmA_pu", "vaA_degree", "vmB_pu",
                       "vaB_degree", "vmC_pu", "vaC_degree"]], np.nan, equal_nan=True)
    assert np.allclose(net.res_bus_3ph.T[[0, 2, 3]].T[["p_A_mw", "q_A_mvar", "p_B_mw", "q_B_mvar",
                       "p_C_mw", "q_C_mvar"]], 0.0)

def get_PowerFactory_Results():
    results=\
    { 
    	10:
    	{ 
    
    		0:
    		{ 
    
    			'delta' :
    			{ 
    
    				'Yyn': np.array
    				([ 
    #10,0,deltaYyn
    #BusTr_HV,Tr_LV,Load
    					1.0000001787261197,					0.9990664471050634,					0.9408623912831601,
    					0.9999997973033823,					0.9989329879720452,					0.9398981202882926,
    					1.000000023970535,					0.9990124767159095,					0.9422153531204793,
    
    					 ] )
    ,
    				'YNyn': np.array
    				([ 
    #10,0,deltaYNyn
    #BusTr_HV,Tr_LV,Load
    					1.0000001786899793,					0.9990638105447855,					0.9408586320432043,
    					0.9999997971517767,					0.9989338020819162,					0.9398997093459485,
    					1.000000024158281,					0.9990142941344189,					0.9422174830541402,
    
    					 ] )
    ,
    				'Dyn': np.array
    				([ 
    #10,0,deltaDyn
    #BusTr_HV,Tr_LV,Load
    					1.000000178603741,					0.9990638106892,					0.9408586322473715,
    					0.9999997971832201,					0.9989338020666364,					0.9398997093074486,
    					1.000000024213076,					0.9990142940055439,					0.9422174828921106,
    
    					 ] )
    ,
    				'Yzn': np.array
    				([ 
    #10,0,deltaYzn
    #BusTr_HV,Tr_LV,Load
    					1.000000178603741,					0.9990638106892,					0.9408586322473715,
    					0.9999997971832201,					0.9989338020666364,					0.9398997093074486,
    					1.000000024213076,					0.9990142940055439,					0.9422174828921106,
    
    					 ] )
    ,
    			},
    			'wye' :
    			{ 
    
    				'Yyn': np.array
    				([ 
    #10,0,wyeYyn
    #BusTr_HV,Tr_LV,Load
    					0.9999998021362442,					0.9915031010358111,					0.9206318374527404,
    					0.9999997791045989,					1.0143417780460269,					0.9616365638634155,
    					1.000000418759289,					0.9913387390190033,					0.9408558778822637,
    
    					 ] )
    ,
    				'YNyn': np.array
    				([ 
    #10,0,wyeYNyn
    #BusTr_HV,Tr_LV,Load
    					0.9999997083766274,					0.9988968962217385,					0.9287452455114519,
    					1.0000001672319114,					0.999061839981782,					0.9452915718541725,
    					1.0000001243918462,					0.9990504923797096,					0.9488965582258678,
    
    					 ] )
    ,
    				'Dyn': np.array
    				([ 
    #10,0,wyeDyn
    #BusTr_HV,Tr_LV,Load
    					0.9999999599731432,					0.9988963012384348,					0.9287445940341739,
    					0.999999734429128,					0.9990625733649781,					0.9452923634430362,
    					1.000000305597812,					0.9990503538577492,					0.9488964199625295,
    
    					 ] )
    ,
    				'Yzn': np.array
    				([ 
    #10,0,wyeYzn
    #BusTr_HV,Tr_LV,Load
    					0.9999999599731432,					0.9988963012384348,					0.9287445940341739,
    					0.999999734429128,					0.9990625733649781,					0.9452923634430362,
    					1.000000305597812,					0.9990503538577492,					0.9488964199625295,
    
    					 ] )
    ,
    			},
    			'delta_wye' :
    			{ 
    
    				'Yyn': np.array
    				([ 
    #10,0,delta_wyeYyn
    #BusTr_HV,Tr_LV,Load
    					1.000000289039923,					0.9945259444558469,					0.9241479442057374,
    					0.9999996598061066,					1.0028660964609941,					0.9332827547884484,
    					1.0000000511540714,					0.9989227003917809,					0.9366758414321353,
    
    					 ] )
    ,
    				'YNyn': np.array
    				([ 
    #10,0,delta_wyeYNyn
    #BusTr_HV,Tr_LV,Load
    					1.0000001633660651,					0.9988186334488024,					0.9284513283443013,
    					0.9999997731436624,					0.9986857571039884,					0.9290168825920521,
    					1.0000000634904662,					0.9987917974558278,					0.9366076053493121,
    
    					 ] )
    ,
    				'Dyn': np.array
    				([ 
    #10,0,delta_wyeDyn
    #BusTr_HV,Tr_LV,Load
    					1.0000002947774138,					0.9988183812973129,					0.928451074375663,
    					0.9999996601592913,					0.9986859152711799,					0.9290170457925304,
    					1.0000000450633972,					0.9987918914643369,					0.936607696605823,
    
    					 ] )
    ,
    				'Yzn': np.array
    				([ 
    #10,0,delta_wyeYzn
    #BusTr_HV,Tr_LV,Load
    					1.0000002947774138,					0.9988183812973129,					0.928451074375663,
    					0.9999996601592913,					0.9986859152711799,					0.9290170457925304,
    					1.0000000450633972,					0.9987918914643369,					0.936607696605823,
    
    					 ] )
    ,
    			},
    			'bal_wye' :
    			{ 
    
    				'Yyn': np.array
    				([ 
    #10,0,bal_wyeYyn
    #BusTr_HV,Tr_LV,Load
    					0.9999999999999879,					0.9990668908275987,					0.9446728357045939,
    					0.9999999999999739,					0.9990668910254652,					0.9446728363197381,
    					1.0000000000000384,					0.9990668908667012,					0.9446728362625954,
    
    					 ] )
    ,
    				'YNyn': np.array
    				([ 
    #10,0,bal_wyeYNyn
    #BusTr_HV,Tr_LV,Load
    					0.9999999999999863,					0.9990668909016067,					0.9446728357836535,
    					0.9999999999999772,					0.9990668908990621,					0.9446728361848189,
    					1.0000000000000362,					0.9990668909190944,					0.9446728363184529,
    
    					 ] )
    ,
    				'Dyn': np.array
    				([ 
    #10,0,bal_wyeDyn
    #BusTr_HV,Tr_LV,Load
    					0.999999999999989,					0.999066890901618,					0.9446728357836652,
    					0.9999999999999737,					0.999066890899081,					0.9446728361848393,
    					1.0000000000000375,					0.999066890919066,					0.9446728363184226,
    
    					 ] )
    ,
    				'Yzn': np.array
    				([ 
    #10,0,bal_wyeYzn
    #BusTr_HV,Tr_LV,Load
    					0.999999999999989,					0.999066890901618,					0.9446728357836652,
    					0.9999999999999737,					0.999066890899081,					0.9446728361848393,
    					1.0000000000000375,					0.999066890919066,					0.9446728363184226,
    
    					 ] )
    ,
    			},
    		},
    		1:
    		{ 
    
    			'delta' :
    			{ 
    
    				'Yyn': np.array
    				([ 
    #10,1,deltaYyn
    #BusTr_HV,Tr_LV,Load
    					1.0000001795040512,					1.0240495841864894,					0.9674397511496959,
    					0.9999997971910463,					1.0239111614639989,					0.9664923222986317,
    					1.0000000233049395,					1.0239935208058917,					0.9687543048259518,
    
    					 ] )
    ,
    				'YNyn': np.array
    				([ 
    #10,1,deltaYNyn
    #BusTr_HV,Tr_LV,Load
    					1.0000001782704175,					1.0240459468337655,					0.9674352916726019,
    					0.9999997977852046,					1.0239130527637306,					0.9664952324047731,
    					1.0000000239444145,					1.023995255504894,					0.9687558295327158,
    
    					 ] )
    ,
    				'Dyn': np.array
    				([ 
    #10,1,deltaDyn
    #BusTr_HV,Tr_LV,Load
    					1.0000001782214243,					1.024045946940332,					0.967435291834159,
    					0.9999997978066542,					1.0239130527420286,					0.9664952323430777,
    					1.0000000239719584,					1.023995255420507,					0.9687558294364838,
    
    					 ] )
    ,
    				'Yzn': np.array
    				([ 
    #10,1,deltaYzn
    #BusTr_HV,Tr_LV,Load
    					1.0000001782214243,					1.024045946940332,					0.967435291834159,
    					0.9999997978066542,					1.0239130527420286,					0.9664952323430777,
    					1.0000000239719584,					1.023995255420507,					0.9687558294364838,
    
    					 ] )
    ,
    			},
    			'wye' :
    			{ 
    
    				'Yyn': np.array
    				([ 
    #10,1,wyeYyn
    #BusTr_HV,Tr_LV,Load
    					0.9999998049723338,					1.0163471727161444,					0.9474851372085454,
    					0.9999997835047069,					1.0396033478524176,					0.9883119194148919,
    					1.0000004115230865,					1.016177862041642,					0.9670415224711911,
    
    					 ] )
    ,
    				'YNyn': np.array
    				([ 
    #10,1,wyeYNyn
    #BusTr_HV,Tr_LV,Load
    					0.9999997111904564,					1.023876123903735,					0.9557104532156954,
    					1.000000169840967,					1.024045000904823,					0.97172789408756,
    					1.0000001189689527,					1.024030547850082,					0.9752090807560196,
    
    					 ] )
    ,
    				'Dyn': np.array
    				([ 
    #10,1,wyeDyn
    #BusTr_HV,Tr_LV,Load
    					0.9999999610844935,					1.0238755180281829,					0.9557097928361534,
    					0.9999997396431541,					1.0240457481759326,					0.9717286975282872,
    					1.0000002992724317,					1.0240304063318828,					0.975208939465858,
    
    					 ] )
    ,
    				'Yzn': np.array
    				([ 
    #10,1,wyeYzn
    #BusTr_HV,Tr_LV,Load
    					0.9999999610844935,					1.0238755180281829,					0.9557097928361534,
    					0.9999997396431541,					1.0240457481759326,					0.9717286975282872,
    					1.0000002992724317,					1.0240304063318828,					0.975208939465858,
    
    					 ] )
    ,
    			},
    			'delta_wye' :
    			{ 
    
    				'Yyn': np.array
    				([ 
    #10,1,delta_wyeYyn
    #BusTr_HV,Tr_LV,Load
    					1.0000002896605282,					1.0194026014413138,					0.9509830141499932,
    					0.9999996606572187,					1.0279455302463374,					0.9603073239465667,
    					1.0000000496823542,					1.0238970684816717,					0.9633884768515291,
    
    					 ] )
    ,
    				'YNyn': np.array
    				([ 
    #10,1,delta_wyeYNyn
    #BusTr_HV,Tr_LV,Load
    					1.0000001631049464,					1.0237965435008547,					0.9553922424619002,
    					0.9999997741736003,					1.0236607923322103,					0.9559358029296258,
    					1.000000062721646,					1.0237688359303385,					0.9633200580357987,
    
    					 ] )
    ,
    				'Dyn': np.array
    				([ 
    #10,1,delta_wyeDyn
    #BusTr_HV,Tr_LV,Load
    					1.0000002940160242,					1.023796285978077,					0.9553919829548445,
    					0.9999996614657936,					1.0236609541452617,					0.9559359697011912,
    					1.000000044518284,					1.0237689316654306,					0.9633201512377196,
    
    					 ] )
    ,
    				'Yzn': np.array
    				([ 
    #10,1,delta_wyeYzn
    #BusTr_HV,Tr_LV,Load
    					1.0000002940160242,					1.023796285978077,					0.9553919829548445,
    					0.9999996614657936,					1.0236609541452617,					0.9559359697011912,
    					1.000000044518284,					1.0237689316654306,					0.9633201512377196,
    
    					 ] )
    ,
    			},
    			'bal_wye' :
    			{ 
    
    				'Yyn': np.array
    				([ 
    #10,1,bal_wyeYyn
    #BusTr_HV,Tr_LV,Load
    					0.99999999999999,					1.02404859308445,					0.971134029249497,
    					0.9999999999999845,					1.0240485931685195,					0.9711340295967834,
    					1.0000000000000258,					1.0240485931044616,					0.9711340295607079,
    
    					 ] )
    ,
    				'YNyn': np.array
    				([ 
    #10,1,bal_wyeYNyn
    #BusTr_HV,Tr_LV,Load
    					0.9999999999999892,					1.0240485931151249,					0.9711340292823146,
    					0.9999999999999865,					1.024048593114567,					0.9711340295398108,
    					1.0000000000000244,					1.0240485931277552,					0.9711340295848808,
    
    					 ] )
    ,
    				'Dyn': np.array
    				([ 
    #10,1,bal_wyeDyn
    #BusTr_HV,Tr_LV,Load
    					0.9999999999999902,					1.024048593115119,					0.9711340292823075,
    					0.9999999999999848,					1.0240485931145844,					0.9711340295398292,
    					1.0000000000000249,					1.024048593127728,					0.9711340295848522,
    
    					 ] )
    ,
    				'Yzn': np.array
    				([ 
    #10,1,bal_wyeYzn
    #BusTr_HV,Tr_LV,Load
    					0.9999999999999902,					1.024048593115119,					0.9711340292823075,
    					0.9999999999999848,					1.0240485931145844,					0.9711340295398292,
    					1.0000000000000249,					1.024048593127728,					0.9711340295848522,
    
    					 ] )
    ,
    			},
    		},
    	},
    	11:
    	{ 
    
    		0:
    		{ 
    
    			'delta' :
    			{ 
    
    				'Yyn': np.array
    				([ 
    #11,0,deltaYyn
    #BusTr_HV,Tr_LV,Load
    					1.0000001770832512,					1.0991666419999009,					1.046863039382953,
    					0.9999997998271506,					1.0990478952608114,					1.0459974904307656,
    					1.0000000230896342,					1.0991196058562567,					1.0480820977965253,
    
    					 ] )
    ,
    				'YNyn': np.array
    				([ 
    #11,0,deltaYNyn
    #BusTr_HV,Tr_LV,Load
    					1.000000177064337,					1.0991653032170863,					1.0468611006390927,
    					0.9999997997417357,					1.0990483460592901,					1.0459983357170173,
    					1.0000000231939636,					1.0991204912844936,					1.0480831713683516,
    
    					 ] )
    ,
    				'Dyn': np.array
    				([ 
    #11,0,deltaDyn
    #BusTr_HV,Tr_LV,Load
    					1.0000001770170086,					1.099165303280019,					1.046861100729514,
    					0.9999997997589116,					1.0990483460550085,					1.0459983357036897,
    					1.0000000232241157,					1.0991204912259542,					1.0480831712929268,
    
    					 ] )
    ,
    				'Yzn': np.array
    				([ 
    #11,0,deltaYzn
    #BusTr_HV,Tr_LV,Load
    					1.0000001770170086,					1.099165303280019,					1.046861100729514,
    					0.9999997997589116,					1.0990483460550085,					1.0459983357036897,
    					1.0000000232241157,					1.0991204912259542,					1.0480831712929268,
    
    					 ] )
    ,
    			},
    			'wye' :
    			{ 
    
    				'Yyn': np.array
    				([ 
    #11,0,wyeYyn
    #BusTr_HV,Tr_LV,Load
    					0.9999998409135958,					1.0924753274233265,					1.0291805067306592,
    					0.9999997887228856,					1.112638254093763,					1.0649872145063082,
    					1.0000003703636224,					1.0923417509837368,					1.0468846408299153,
    
    					 ] )
    ,
    				'YNyn': np.array
    				([ 
    #11,0,wyeYNyn
    #BusTr_HV,Tr_LV,Load
    					0.9999997198861459,					1.0990179190476412,					1.0362148303868974,
    					1.0000001764446427,					1.0991669773561135,					1.0507765134998273,
    					1.0000001036695618,					1.0991473807202723,					1.0539233691792418,
    
    					 ] )
    ,
    				'Dyn': np.array
    				([ 
    #11,0,wyeDyn
    #BusTr_HV,Tr_LV,Load
    					0.9999999645965844,					1.0990174387140366,					1.036214314982853,
    					0.9999997540341666,					1.0991675482923782,					1.0507771199594842,
    					1.0000002813693196,					1.0991472900387962,					1.0539232794875342,
    
    					 ] )
    ,
    				'Yzn': np.array
    				([ 
    #11,0,wyeYzn
    #BusTr_HV,Tr_LV,Load
    					0.9999999645965844,					1.0990174387140366,					1.036214314982853,
    					0.9999997540341666,					1.0991675482923782,					1.0507771199594842,
    					1.0000002813693196,					1.0991472900387962,					1.0539232794875342,
    
    					 ] )
    ,
    			},
    			'delta_wye' :
    			{ 
    
    				'Yyn': np.array
    				([ 
    #11,0,delta_wyeYyn
    #BusTr_HV,Tr_LV,Load
    					1.0000002867915057,					1.09511471406464,					1.0320045668742739,
    					0.9999996655448716,					1.102582851029247,					1.0401766570762196,
    					1.0000000476637207,					1.0990187740288424,					1.0431968194073924,
    
    					 ] )
    ,
    				'YNyn': np.array
    				([ 
    #11,0,delta_wyeYNyn
    #BusTr_HV,Tr_LV,Load
    					1.0000001623852481,					1.0989490480618516,					1.0358488170212126,
    					0.9999997776678232,					1.098829878782537,					1.0363599386677118,
    					1.0000000599471168,					1.0989238972185933,					1.0431472226133363,
    
    					 ] )
    ,
    				'Dyn': np.array
    				([ 
    #11,0,delta_wyeDyn
    #BusTr_HV,Tr_LV,Load
    					1.000000291479138,					1.0989488469146447,					1.0358486145520418,
    					0.9999996659434413,					1.0988300000349813,					1.0363600632236267,
    					1.0000000425775202,					1.098923977128452,					1.0431473008280179,
    
    					 ] )
    ,
    				'Yzn': np.array
    				([ 
    #11,0,delta_wyeYzn
    #BusTr_HV,Tr_LV,Load
    					1.000000291479138,					1.0989488469146447,					1.0358486145520418,
    					0.9999996659434413,					1.0988300000349813,					1.0363600632236267,
    					1.0000000425775202,					1.098923977128452,					1.0431473008280179,
    
    					 ] )
    ,
    			},
    			'bal_wye' :
    			{ 
    
    				'Yyn': np.array
    				([ 
    #11,0,bal_wyeYyn
    #BusTr_HV,Tr_LV,Load
    					0.999999999999994,					1.0991663222840553,					1.0502483483014522,
    					0.999999999999986,					1.0991663223629755,					1.0502483485683893,
    					1.00000000000002,					1.0991663223022374,					1.0502483485566558,
    
    					 ] )
    ,
    				'YNyn': np.array
    				([ 
    #11,0,bal_wyeYNyn
    #BusTr_HV,Tr_LV,Load
    					0.9999999999999934,					1.0991663223142185,					1.050248348333234,
    					0.9999999999999878,					1.0991663223125718,					1.0502483485153113,
    					1.000000000000019,					1.0991663223224817,					1.0502483485779557,
    
    					 ] )
    ,
    				'Dyn': np.array
    				([ 
    #11,0,bal_wyeDyn
    #BusTr_HV,Tr_LV,Load
    					0.9999999999999944,					1.099166322314217,					1.0502483483332314,
    					0.999999999999986,					1.0991663223125883,					1.050248348515329,
    					1.0000000000000195,					1.099166322322463,					1.0502483485779364,
    
    					 ] )
    ,
    				'Yzn': np.array
    				([ 
    #11,0,bal_wyeYzn
    #BusTr_HV,Tr_LV,Load
    					0.9999999999999944,					1.099166322314217,					1.0502483483332314,
    					0.999999999999986,					1.0991663223125883,					1.050248348515329,
    					1.0000000000000195,					1.099166322322463,					1.0502483485779364,
    
    					 ] )
    ,
    			},
    		},
    		1:
    		{ 
    
    			'delta' :
    			{ 
    
    				'Yyn': np.array
    				([ 
    #11,1,deltaYyn
    #BusTr_HV,Tr_LV,Load
    					1.000000177759738,					1.1266508599188314,					1.075749945733859,
    					0.9999997996753168,					1.1265276819882335,					1.0748995015125222,
    					1.0000000225649812,					1.1266018378562361,					1.076934372664356,
    
    					 ] )
    ,
    				'YNyn': np.array
    				([ 
    #11,1,deltaYNyn
    #BusTr_HV,Tr_LV,Load
    					1.000000176730594,					1.1266486259211201,					1.0757473443700512,
    					0.9999998002521623,					1.1265290107226675,					1.0749013345769867,
    					1.0000000230172796,					1.1266027366684568,					1.0769351304583261,
    
    					 ] )
    ,
    				'Dyn': np.array
    				([ 
    #11,1,deltaDyn
    #BusTr_HV,Tr_LV,Load
    					1.0000001767039686,					1.1266486259729462,					1.0757473444450258,
    					0.9999998002646232,					1.1265290107113315,					1.0749013345478544,
    					1.0000000230314439,					1.126602736628164,					1.0769351304141572,
    
    					 ] )
    ,
    				'Yzn': np.array
    				([ 
    #11,1,deltaYzn
    #BusTr_HV,Tr_LV,Load
    					1.0000001767039686,					1.1266486259729462,					1.0757473444450258,
    					0.9999998002646232,					1.1265290107113315,					1.0749013345478544,
    					1.0000000230314439,					1.126602736628164,					1.0769351304141572,
    
    					 ] )
    ,
    			},
    			'wye' :
    			{ 
    
    				'Yyn': np.array
    				([ 
    #11,1,wyeYyn
    #BusTr_HV,Tr_LV,Load
    					0.9999998425139852,					1.1198215550651343,					1.0582701679876008,
    					0.999999792808548,					1.1404037383383383,					1.0940119347447643,
    					1.000000364677568,					1.119678656475928,					1.0754147798091545,
    
    					 ] )
    ,
    				'YNyn': np.array
    				([ 
    #11,1,wyeYNyn
    #BusTr_HV,Tr_LV,Load
    					0.9999997220234313,					1.1264984365036237,					1.065423794124721,
    					1.0000001785338588,					1.126651120595415,					1.0795452055229118,
    					1.0000000994430542,					1.126629015453866,					1.0825891788506536,
    
    					 ] )
    ,
    				'Dyn': np.array
    				([ 
    #11,1,wyeDyn
    #BusTr_HV,Tr_LV,Load
    					0.9999999654333293,					1.1264979466596041,					1.0654232703853377,
    					0.9999997580954444,					1.1266517031402583,					1.079545822405393,
    					1.0000002764712945,					1.1266289226736226,					1.0825890870214312,
    
    					 ] )
    ,
    				'Yzn': np.array
    				([ 
    #11,1,wyeYzn
    #BusTr_HV,Tr_LV,Load
    					0.9999999654333293,					1.1264979466596041,					1.0654232703853377,
    					0.9999997580954444,					1.1266517031402583,					1.079545822405393,
    					1.0000002764712945,					1.1266289226736226,					1.0825890870214312,
    
    					 ] )
    ,
    			},
    			'delta_wye' :
    			{ 
    
    				'Yyn': np.array
    				([ 
    #11,1,delta_wyeYyn
    #BusTr_HV,Tr_LV,Load
    					1.0000002872593454,					1.122503013135439,					1.061107915739188,
    					0.9999996662661563,					1.1301536319129346,					1.069448792307849,
    					1.0000000464745962,					1.1264944198323028,					1.0721922685731713,
    
    					 ] )
    ,
    				'YNyn': np.array
    				([ 
    #11,1,delta_wyeYNyn
    #BusTr_HV,Tr_LV,Load
    					1.0000001621739123,					1.126428316031026,					1.0650458103409908,
    					0.9999997785161929,					1.1263065012425137,					1.0655375147447366,
    					1.0000000593100822,					1.12640238251751,					1.0721435619381965,
    
    					 ] )
    ,
    				'Dyn': np.array
    				([ 
    #11,1,delta_wyeDyn
    #BusTr_HV,Tr_LV,Load
    					1.0000002908474748,					1.1264281104824707,					1.0650456033928053,
    					0.9999996670234566,					1.1263066253385652,					1.065537642082384,
    					1.0000000421291677,					1.126402463985756,					1.0721436418376473,
    
    					 ] )
    ,
    				'Yzn': np.array
    				([ 
    #11,1,delta_wyeYzn
    #BusTr_HV,Tr_LV,Load
    					1.0000002908474748,					1.1264281104824707,					1.0650456033928053,
    					0.9999996670234566,					1.1263066253385652,					1.065537642082384,
    					1.0000000421291677,					1.126402463985756,					1.0721436418376473,
    
    					 ] )
    ,
    			},
    			'bal_wye' :
    			{ 
    
    				'Yyn': np.array
    				([ 
    #11,1,bal_wyeYyn
    #BusTr_HV,Tr_LV,Load
    					0.9999999999999946,					1.126649305937712,					1.0790357881145098,
    					0.9999999999999919,					1.1266493059651883,					1.0790357882640247,
    					1.0000000000000135,					1.1266493059449603,					1.0790357882526134,
    
    					 ] )
    ,
    				'YNyn': np.array
    				([ 
    #11,1,bal_wyeYNyn
    #BusTr_HV,Tr_LV,Load
    					0.9999999999999944,					1.126649305947411,					1.079035788124742,
    					0.9999999999999928,					1.126649305946962,					1.0790357882450081,
    					1.000000000000013,					1.1266493059535365,					1.079035788261449,
    
    					 ] )
    ,
    				'Dyn': np.array
    				([ 
    #11,1,bal_wyeDyn
    #BusTr_HV,Tr_LV,Load
    					0.9999999999999944,					1.1266493059473897,					1.0790357881247188,
    					0.9999999999999922,					1.1266493059469642,					1.079035788245011,
    					1.0000000000000133,					1.1266493059535063,					1.0790357882614174,
    
    					 ] )
    ,
    				'Yzn': np.array
    				([ 
    #11,1,bal_wyeYzn
    #BusTr_HV,Tr_LV,Load
    					0.9999999999999944,					1.1266493059473897,					1.0790357881247188,
    					0.9999999999999922,					1.1266493059469642,					1.079035788245011,
    					1.0000000000000133,					1.1266493059535063,					1.0790357882614174,
    
    					 ] )
    ,
    			},
    		},
    	},
    
    }

    return results
if __name__ == "__main__":
    pytest.main(["test_runpp_3ph.py"])