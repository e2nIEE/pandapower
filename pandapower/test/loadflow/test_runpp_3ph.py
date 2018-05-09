# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:06:25 2018
Tests 3 phase power flow algorithm
@author: sghosh
"""
import pandapower as pp
import numpy as np
import pytest
from pandapower.pf.runpp_3ph import combine_X012
from pandapower.create import create_load_3ph
from pandapower.pf.runpp_3ph import runpp_3ph,show_results
     
def results_2bus_PowerFactory():
    Sabc_sl_sp =  np.matrix( [
        [55707.684189 + 60797.066456j],
        [8779.9399188 - 880.93186592j],
        [9373.9326305 - 11441.658401j]
        ]
        ,dtype = np.complex 
        ) #kW and kVAr

    Sabc_pq_sp =  np.matrix(   [
                         [49999.976033 + 49999.946905j] 
                        ,[9999.9987591 + 15000.000944j] 
                        ,[10000.000590 + 4999.9990418j]
                        ]
                    ,dtype = np.complex 
                    ) #kW and kVAr
    Sabc_powerFactory = np.concatenate((Sabc_sl_sp,Sabc_pq_sp),axis =1)
    # =============================================================================
	# Slack Current I012 in kA as per Power Factory 
	# =============================================================================
						
    Ia_sl_pf = np.matrix(1.3421204457* np.exp(1j*np.deg2rad(-48.552565134)))
    Ib_sl_pf = np.matrix(0.1371555175	 * np.exp(1j*np.deg2rad(-113.7410795)))
    Ic_sl_pf = np.matrix(0.22838401431* np.exp(1j*np.deg2rad(171.14429027)))
    Iabc_sl_pf = combine_X012(Ia_sl_pf,Ib_sl_pf,Ic_sl_pf)

	# =============================================================================
	#  PQ  Current I012 in kA as per Power Factory 
	# =============================================================================

    Ia_pf = np.matrix(1.4853791557	* np.exp(1j*np.deg2rad(-54.01018178)))
    Ib_pf = np.matrix(0.26009610688	* np.exp(1j*np.deg2rad(179.58428912)))
    Ic_pf = np.matrix(0.16746340142	* np.exp(1j*np.deg2rad(99.329437604)))
    Iabc_pq_pf = combine_X012(Ia_pf,Ib_pf,Ic_pf)
    Iabc_powerFactory = np.concatenate((Iabc_sl_pf,Iabc_pq_pf),axis =1)
	# =============================================================================
	# Slack bus Voltages Vabc in kV as per Power Factory 
	# =============================================================================
    Va_sl_pf = np.matrix(61.439988828	* np.exp(1j*np.deg2rad(-1.051252102)))
    Vb_sl_pf = np.matrix(64.335896865	* np.exp(1j*np.deg2rad(-119.47065404)))
    Vc_sl_pf = np.matrix(64.764982202	* np.exp(1j*np.deg2rad(120.47139943)))
    Vabc_sl_pf = combine_X012(Va_sl_pf,Vb_sl_pf,Vc_sl_pf)


	# =============================================================================
	# PQ Bus Voltages in kV as per Power Factory 
	# =============================================================================
    Va_pf = np.matrix(47.604427027	* np.exp(1j*np.deg2rad(-9.0101984693)))
    Vb_pf = np.matrix(69.311904321	* np.exp(1j*np.deg2rad(-124.10577346)))
    Vc_pf = np.matrix(66.76288605	* np.exp(1j*np.deg2rad(125.89448304)))

    Vabc_pq_pf = combine_X012(Va_pf,Vb_pf,Vc_pf)

    Vabc_powerFactory = np.concatenate((Vabc_sl_pf,Vabc_pq_pf),axis =1)

    return Sabc_powerFactory, Vabc_powerFactory, Iabc_powerFactory
    
def test_2bus_network():
    # =============================================================================
    # Base Value Assignmeent
    # =============================================================================
    V_base = 110                     # 110kV Base Voltage
    kVA_base = 100000                      # 100 MVA
#    I_base = (kVA_base/V_base) * 1e-3           # in kA
    V_base_res = V_base/np.sqrt(3)
    I_base_res = kVA_base/V_base_res*1e-3
    net = pp.create_empty_network(sn_kva = kVA_base )
    
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
    
    create_load_3ph(net, busk, p_kw_A=50000, q_kvar_A=50000, p_kw_B=10000, q_kvar_B=15000,
                       p_kw_C=10000, q_kvar_C=5000)
    
    pp.create_line(net, from_bus = busn, to_bus = busk, length_km = 50.0, std_type="example_type")
    
    pp.add_zero_impedance_parameters(net)
    
    count,V012_it,I012_it,ppci0,Y1_pu = runpp_3ph(net)
    
    V_abc_new,I_abc_new,Sabc_new = show_results(V_base,kVA_base,count,ppci0,Y1_pu,V012_it,I012_it)
    Sabc_powerFactory, Vabc_powerFactory, Iabc_powerFactory = results_2bus_PowerFactory()
    assert np.allclose(abs(Sabc_powerFactory*1e-3),abs(Sabc_new)*1e-3,atol = 1e-4)
    assert np.allclose(Vabc_powerFactory,(V_abc_new*V_base_res),atol=1.e-4)
    assert np.allclose(abs(Iabc_powerFactory),abs(I_abc_new*I_base_res),atol=1.e-4)

def results_4bus_PowerFactory():
    Sabc_sl_sp =  np.matrix(   [						
    	[150.34732586*1000+89.274183693*1000j ,		149.93885263*1000+94.513653203*1000j,	\
      39.966843513*1000+13.69963814*1000j]					
    	]					
    						
    	                    ).T #kW and kVAr					
    						
    Sabc_pq_sp =  np.matrix(   [					
					
    	[50000+20000j ,		79999.999999+60000j,		20000+5000j],
    	[50000+50000j ,		9999.9999998+15000j,		10000+5000j],
    	[50000+20000j ,		59999.999999+20000j,		10000+5000j]
    		]			
    		                    ,dtype = np.complex 			
    		                    ).T #kW and kVAr			

    Sabc_powerFactory = np.concatenate((Sabc_sl_sp,Sabc_pq_sp),axis =1)

	# =============================================================================
	# Slack Current I012 in kA as per Power Factory 
	# =============================================================================
    Iabc_pq_pf =  np.matrix(   [					\
    	[0.86676467552*np.exp(1j*np.deg2rad(-23.16800712)),	\
      1.614392047*np.exp(1j*np.deg2rad(-158.37122826)),	\
      0.31071857716*np.exp(1j*np.deg2rad(108.4669283))],
    	
      [1.1387494998*np.exp(1j*np.deg2rad(-46.403411358)),	\
      0.29070024235*np.exp(1j*np.deg2rad(-177.76002958)),	\
      0.16859090052*np.exp(1j*np.deg2rad(95.969503627))],
    	
      [0.86693887872*np.exp(1j*np.deg2rad(-23.183031658)),		\
      1.0204854981*np.exp(1j*np.deg2rad(-139.95476824)),		\
      0.16851041836*np.exp(1j*np.deg2rad(95.960819649))]
    					
    		]			
    		                    ,dtype = np.complex 			
    		                    ).T #kW and kVAr			


	# =============================================================================
	#  PQ  Current I012 in kA as per Power Factory 
	# =============================================================================

    Iabc_sl_pf =  np.matrix(   [					
    	[2.806982184*np.exp(1j*np.deg2rad(-32.037382293)),		\
      2.85617664*np.exp(1j*np.deg2rad(-153.546895528)),		\
      0.637503143*np.exp(1j*np.deg2rad(103.573568845))]				
    	]				
    	                    ,dtype = np.complex 				
    	                    ).T #kW and kVAr				
    	
					

    Iabc_powerFactory = np.concatenate((Iabc_sl_pf,Iabc_pq_pf),axis = 1)
	# =============================================================================
	# Slack bus Voltages Vabc in kV as per Power Factory 
	# =============================================================================
    Vabc_sl_pf =  np.matrix(   [						
    	[62.292804335*np.exp(1j*np.deg2rad(-1.336121658)),		\
      62.055452267*np.exp(1j*np.deg2rad(-121.321697076)),		\
      66.273554938*np.exp(1j*np.deg2rad(122.494006073))]	
				
    	]					
    	                    ,dtype = np.complex 					
    	                    ).T #kW and kVAr					


	# =============================================================================
	# PQ Bus Voltages in kV as per Power Factory 
	# =============================================================================

    Vabc_pq_pf =  np.matrix(   [					
					
	[62.129490959*np.exp(1j*np.deg2rad(-1.366597634)),		\
  61.942822493*np.exp(1j*np.deg2rad(-121.501330612)),		\
  66.347909792*np.exp(1j*np.deg2rad(122.503171764))],
	[62.095024525*np.exp(1j*np.deg2rad(-1.403411358)),		\
  62.014934118*np.exp(1j*np.deg2rad(-121.450097103)),		\
  66.316389871*np.exp(1j*np.deg2rad(122.534554804))],
	[62.117006623*np.exp(1j*np.deg2rad(-1.381622172)),		\
  61.975945099*np.exp(1j*np.deg2rad(-121.519819412)),		\
  66.3480632*np.exp(1j*np.deg2rad(122.525870826))]
					
		]			
		                    ,dtype = np.complex 			
		                    ).T #kW and kVAr			


    Vabc_powerFactory = np.concatenate((Vabc_sl_pf,Vabc_pq_pf),axis =1)
    return Sabc_powerFactory, Vabc_powerFactory, Iabc_powerFactory 

def test_4bus_network():
    V_base = 110                     # 110kV Base Voltage
    kVA_base = 100000                      # 100 MVA
#    I_base = (kVA_base/V_base) * 1e-3           # in kA
    V_base_res = V_base/np.sqrt(3)
    I_base_res = kVA_base/V_base_res*1e-3
    net = pp.create_empty_network(sn_kva = kVA_base )
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
    
    create_load_3ph(net, busk, p_kw_A=50000, q_kvar_A=20000, p_kw_B=80000, q_kvar_B=60000,
                       p_kw_C=20000, q_kvar_C=5000)
    create_load_3ph(net, busm, p_kw_A=50000, q_kvar_A=50000, p_kw_B=10000, q_kvar_B=15000,
                       p_kw_C=10000, q_kvar_C=5000)
    create_load_3ph(net, busp, p_kw_A=50000, q_kvar_A=20000, p_kw_B=60000, q_kvar_B=20000,
                       p_kw_C=10000, q_kvar_C=5000)
    count,V012_new,I012_new,ppci0,Y1_pu = runpp_3ph(net)
    V_abc_new,I_abc_new,Sabc_new = show_results(V_base,kVA_base,count,ppci0,Y1_pu,V012_new,I012_new)
    Sabc_powerFactory, Vabc_powerFactory, Iabc_powerFactory = results_4bus_PowerFactory()

    assert np.allclose(abs(Sabc_powerFactory*1.e-3),abs(Sabc_new)*1.e-3,atol = 1.e-4)
    assert np.allclose(Vabc_powerFactory,(V_abc_new*V_base_res),atol=1.e-4)
    assert np.allclose(abs(Iabc_powerFactory),abs(I_abc_new*I_base_res),atol=1.e-4)
def test_in_serv_load():
    V_base = 110                     # 110kV Base Voltage
    kVA_base = 100000                      # 100 MVA
#    I_base = (kVA_base/V_base) * 1e-3           # in kA
    V_base_res = V_base/np.sqrt(3)
    I_base_res = kVA_base/V_base_res*1e-3
    net = pp.create_empty_network(sn_kva = kVA_base )
    
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
    
    create_load_3ph(net, busk, p_kw_A=50000, q_kvar_A=50000, p_kw_B=10000, q_kvar_B=15000,
                       p_kw_C=10000, q_kvar_C=5000)
    
    pp.create_line(net, from_bus = busn, to_bus = busk, length_km = 50.0, std_type="example_type")
    
    pp.add_zero_impedance_parameters(net)
    
    count,V012_it,I012_it,ppci0,Y1_pu = runpp_3ph(net)
    
    V_abc_new,I_abc_new,Sabc_new = show_results(V_base,kVA_base,count,ppci0,Y1_pu,V012_it,I012_it)
    Sabc_powerFactory, Vabc_powerFactory, Iabc_powerFactory = \
    results_2bus_PowerFactory()
    assert np.allclose(abs(Sabc_powerFactory*1e-3),abs(Sabc_new)*1e-3,atol = 1e-4)
    assert np.allclose(Vabc_powerFactory,(V_abc_new*V_base_res),atol=1.e-4)
    assert np.allclose(abs(Iabc_powerFactory),abs(I_abc_new*I_base_res),atol=1.e-4)
    
    create_load_3ph(net, busk, p_kw_A=50000, q_kvar_A=100000, p_kw_B=29000, q_kvar_B=38000,
                   p_kw_C=10000, q_kvar_C=5000, in_service=False)

    count,V012_it,I012_it,ppci0,Y1_pu = runpp_3ph(net)
    
    V_abc_new,I_abc_new,Sabc_new = show_results(V_base,kVA_base,count,ppci0,Y1_pu,V012_it,I012_it)
    Sabc_powerFactory, Vabc_powerFactory, Iabc_powerFactory = \
    results_2bus_PowerFactory()
    assert np.allclose(abs(Sabc_powerFactory*1e-3),abs(Sabc_new)*1e-3,atol = 1e-4)
    assert np.allclose(Vabc_powerFactory,(V_abc_new*V_base_res),atol=1.e-4)
    assert np.allclose(abs(Iabc_powerFactory),abs(I_abc_new*I_base_res),atol=1.e-4)
    create_load_3ph(net, busk, p_kw_A=50000, q_kvar_A=50000, p_kw_B=50000, q_kvar_B=50000,
           p_kw_C=50000, q_kvar_C=50000, in_service=True)
    count,V012_it,I012_it,ppci0,Y1_pu = runpp_3ph(net)
    
    V_abc_new,I_abc_new,Sabc_new = show_results(V_base,kVA_base,count,ppci0,Y1_pu,V012_it,I012_it)
    Sabc_powerFactory, Vabc_powerFactory, Iabc_powerFactory = \
    results_2bus_PowerFactory()
    assert not np.allclose(abs(Sabc_powerFactory*1e-3),abs(Sabc_new)*1e-3,atol = 1e-4)
    assert not np.allclose(Vabc_powerFactory,(V_abc_new*V_base_res),atol=1.e-4)
    assert not np.allclose(abs(Iabc_powerFactory),abs(I_abc_new*I_base_res),atol=1.e-4)
    
if __name__ == "__main__":
    pytest.main(["test_runpp_3ph.py"])