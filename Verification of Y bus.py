# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:04:07 2018

@author: sghosh
"""

import pandapower as pp
import numpy as np

from pandapower.pd2ppc import _pd2ppc
from pandapower.pd2ppc_zero import _pd2ppc_zero
from pandapower.pf.makeYbus import makeYbus
from pandapower.auxiliary import combine_X012,sequence_to_phase,phase_to_sequence
from pandapower.build_bus import _add_ext_grid_sc_impedance
# =============================================================================
# Grid Base values as per Power Factory 
# =============================================================================
V_base = 110/np.sqrt(3)                     # 110kV Base Voltage
kVA_base = 100000                           # 100 MVA
I_base = (kVA_base/V_base) * 1e-3           # in kA

# =============================================================================
# Slack Current I012 in kA as per Power Factory 
# =============================================================================
					
Iabc_sl_pf =  np.matrix(   [					
	[2.806982184*np.exp(1j*np.deg2rad(-32.037382293)),		2.85617664*np.exp(1j*np.deg2rad(-153.546895528)),		0.637503143*np.exp(1j*np.deg2rad(103.573568845))]
					
	]				
	                    ,dtype = np.complex 				
	                    ).T #kW and kVAr				

Iabc_sl_pf_pu = np.divide(Iabc_sl_pf,I_base)
I012_sl_pf_pu = phase_to_sequence(Iabc_sl_pf_pu)
# =============================================================================
#  PQ  Current I012 in kA as per Power Factory 
# =============================================================================

Iabc_pq_pf =  np.matrix(   [					
					
	[0.86676467552*np.exp(1j*np.deg2rad(-23.16800712)),		1.614392047*np.exp(1j*np.deg2rad(-158.37122826)),		0.31071857716*np.exp(1j*np.deg2rad(108.4669283))],
	[1.1387494998*np.exp(1j*np.deg2rad(-46.403411358)),		0.29070024235*np.exp(1j*np.deg2rad(-177.76002958)),		0.16859090052*np.exp(1j*np.deg2rad(95.969503627))],
	[0.86693887872*np.exp(1j*np.deg2rad(-23.183031658)),		1.0204854981*np.exp(1j*np.deg2rad(-139.95476824)),		0.16851041836*np.exp(1j*np.deg2rad(95.960819649))]
					
		]			
		                    ,dtype = np.complex 			
		                    ).T #kW and kVAr			

Iabc_pq_pf_pu = np.divide(Iabc_pq_pf,I_base)
I012_pq_pf_pu = phase_to_sequence(Iabc_pq_pf_pu)
# =============================================================================
# Slack bus Voltages Vabc in kV as per Power Factory 				
# =============================================================================
Vabc_sl_pf =  np.matrix(   [					
	[62.292804335*np.exp(1j*np.deg2rad(-1.336121658)),		62.055452267*np.exp(1j*np.deg2rad(-121.321697076)),		66.273554938*np.exp(1j*np.deg2rad(122.494006073))]
					
	]				
	                    ,dtype = np.complex 				
	                    ).T #kW and kVAr				

#In per Unit
Vabc_sl_pf_pu = np.divide(Vabc_sl_pf,V_base)
V012_sl_pf_pu = phase_to_sequence(Vabc_sl_pf_pu)
# =============================================================================
# PQ Bus Voltages in kV as per Power Factory 
# =============================================================================
Vabc_pq_pf =  np.matrix(   [					
					
	[62.129490959*np.exp(1j*np.deg2rad(-1.366597634)),		61.942822493*np.exp(1j*np.deg2rad(-121.501330612)),		66.347909792*np.exp(1j*np.deg2rad(122.503171764))],
	[62.095024525*np.exp(1j*np.deg2rad(-1.403411358)),		62.014934118*np.exp(1j*np.deg2rad(-121.450097103)),		66.316389871*np.exp(1j*np.deg2rad(122.534554804))],
	[62.117006623*np.exp(1j*np.deg2rad(-1.381622172)),		61.975945099*np.exp(1j*np.deg2rad(-121.519819412)),		66.3480632*np.exp(1j*np.deg2rad(122.525870826))]
					
		]			
		                    ,dtype = np.complex 			
		                    ).T #kW and kVAr			


#In per Unit
Vabc_pq_pf_pu = np.divide(Vabc_pq_pf,V_base)
V012_pq_pf_pu = phase_to_sequence(Vabc_pq_pf_pu)
# =============================================================================
# Calculated Power from V_PowerFactory x conj( I_PowerFactory)
# =============================================================================
Sabc_pq_pf_pu = np.multiply(Vabc_pq_pf_pu,Iabc_pq_pf_pu.conjugate())
Sabc_sl_pf_pu = np.multiply(Vabc_sl_pf_pu,Iabc_sl_pf_pu.conjugate())

# =============================================================================
# Y Bus formation for  Network
# =============================================================================
net = pp.create_empty_network(sn_kva = kVA_base )
busn  =  pp.create_bus(net, vn_kv = V_base, name = "busn")
busk  =  pp.create_bus(net, vn_kv = V_base, name = "busk")
busm =  pp.create_bus(net, vn_kv = V_base, name = "busm")
busp =  pp.create_bus(net, vn_kv = V_base, name = "busp")
pp.create_ext_grid(net, bus=busn, vm_pu=1.0, name="Grid Connection", s_sc_max_mva=5000/3, rx_max=0.1)
net.ext_grid["r0x0_max"] = 0.1
net.ext_grid["x0x_max"] = 1.0

pp.create_std_type(net, {"r0_ohm_per_km": 0.0848, "x0_ohm_per_km": 0.4649556, "c0_nf_per_km":  230.6,
             "max_i_ka": 0.963, "r_ohm_per_km": 0.0212, "x_ohm_per_km": 0.1162389,
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

net._options = {'calculate_voltage_angles': 'auto', 'check_connectivity': True, 'init': 'auto',
    'r_switch': 0.0,'voltage_depend_loads': False, 'mode': "pf",'copy_constraints_to_ppc': False}
_, ppci1 = _pd2ppc(net)

_, ppci2 = _pd2ppc(net)
_add_ext_grid_sc_impedance(net, ppci2)

_, ppci0 = _pd2ppc_zero(net)

ppci0['bus'][0,4] = 0
ppci0['bus'][0,5] = 0
ppci1['bus'][0,4] = 0
ppci1['bus'][0,5] = 0
ppci2['bus'][0,4] = 0
ppci2['bus'][0,5] = 0

Y0_pu,_,_ = makeYbus(ppci0["baseMVA"], ppci0["bus"], ppci0["branch"])

Y1_pu,_,_ = makeYbus(ppci1["baseMVA"], ppci1["bus"], ppci1["branch"])

Y2_pu,_,_ = makeYbus(ppci2["baseMVA"], ppci2["bus"], ppci2["branch"])
    

V0_pf_all = np.concatenate(
                        (np.transpose(V012_sl_pf_pu[0,:])
                        ,np.transpose(V012_pq_pf_pu[0,:]))
                        ,axis =0
                        )
V1_pf_all = np.concatenate(
                        (np.transpose(V012_sl_pf_pu[1,:])
                        ,np.transpose(V012_pq_pf_pu[1,:]))
                        ,axis =0
                        )

V2_pf_all = np.concatenate(
                        (np.transpose(V012_sl_pf_pu[2,:])
                        ,np.transpose(V012_pq_pf_pu[2,:]))
                        ,axis =0
                        )

I0_pf_all = np.matmul(Y0_pu.todense(),V0_pf_all)
I1_pf_all = np.matmul(Y1_pu.todense(),V1_pf_all)
I2_pf_all = np.matmul(Y2_pu.todense(),V2_pf_all)

# =============================================================================
# Current using V_PowerFactory
# =============================================================================
I012_pf_all = combine_X012(I0_pf_all,I1_pf_all,I2_pf_all)
V012_pf_all = combine_X012(V0_pf_all,V1_pf_all,V2_pf_all)

Vabc_pf_all = sequence_to_phase(V012_pf_all)
Iabc_pf_all = sequence_to_phase(I012_pf_all)
# =============================================================================
# Power Using V_PowerFactory
# =============================================================================
Sabc_using_V_case1 = np.multiply(Vabc_pf_all,Iabc_pf_all.conjugate())
		
#Sabc_pf_sp =  np.matrix(   [
#                         [49999.976033 + 49999.946905j] 
#                        ,[9999.9987591 + 15000.000944j] 
#                        ,[10000.000590 + 4999.9990418j]
#                        ]
#                    ,dtype = np.complex 
#                    ) #kW and kVAr

print ('\nY0 Values \n')
print (Y0_pu)
print ('\n Y2 Values \n')
print (Y2_pu)

print ('\nV Power Factory * conj(I Power Factory)\n')
print (Sabc_pq_pf_pu*kVA_base)
print (Sabc_sl_pf_pu*kVA_base)
print ('\n V Power Factory * conj(I from V Power Factory)')
print ('\n Please See the change in Slack\n')

print ('\n 1. With out addition of Generator Impedance')
print (Sabc_using_V_case1*kVA_base)
