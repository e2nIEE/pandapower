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
					
Ia_sl_pf = np.matrix(2.402756264* np.exp(1j*np.deg2rad(-28.263023464)))
Ib_sl_pf = np.matrix(2.730048953	 * np.exp(1j*np.deg2rad(-153.053312530)))
Ic_sl_pf = np.matrix(1.101638324* np.exp(1j*np.deg2rad(90.480307239)))
Iabc_sl_pf = combine_X012(Ia_sl_pf,Ib_sl_pf,Ic_sl_pf)
Iabc_sl_pf_pu = np.divide(Iabc_sl_pf,I_base)
I012_sl_pf_pu = phase_to_sequence(Iabc_sl_pf_pu)
# =============================================================================
#  PQ  Current I012 in kA as per Power Factory 
# =============================================================================

Ia_pf = np.matrix([	
	[0.85361334928*np.exp(1j*np.deg2rad(-22.597021296))],
	[1.1217345202*np.exp(1j*np.deg2rad(-45.833804321))],
	[0.54690896502*np.exp(1j*np.deg2rad(-2.4474461727))]
])	

Ib_pf = np.matrix([
[1.609734392*np.exp(1j*np.deg2rad(-158.13889773))],
[0.2898457303*np.exp(1j*np.deg2rad(-177.53818591))],
[0.90680096195*np.exp(1j*np.deg2rad(-137.21052139))]
])

Ic_pf = np.matrix([
[0.31635572984*np.exp(1j*np.deg2rad(107.76170403))],
[0.17162062815*np.exp(1j*np.deg2rad(95.275718806))],
[0.65033987738*np.exp(1j*np.deg2rad(79.45206248))]
])

Iabc_pq_pf = combine_X012(Ia_pf,Ib_pf,Ic_pf)
Iabc_pq_pf_pu = np.divide(Iabc_pq_pf,I_base)
I012_pq_pf_pu = phase_to_sequence(Iabc_pq_pf_pu)
# =============================================================================
# Slack bus Voltages Vabc in kV as per Power Factory 				
# =============================================================================
Va_sl_pf = np.matrix(63.21203928*np.exp(1j*np.deg2rad(-0.76941527813)))
Vb_sl_pf = np.matrix(62.242151768	* np.exp(1j*np.deg2rad(-121.11499304)))
Vc_sl_pf = np.matrix(65.121473346	* np.exp(1j*np.deg2rad(121.81276191)))
Vabc_sl_pf = combine_X012(Va_sl_pf,Vb_sl_pf,Vc_sl_pf)
#In per Unit
Vabc_sl_pf_pu = np.divide(Vabc_sl_pf,V_base)
V012_sl_pf_pu = phase_to_sequence(Vabc_sl_pf_pu)
# =============================================================================
# PQ Bus Voltages in kV as per Power Factory 
# =============================================================================
Va_pf = np.matrix([[63.14137033	* np.exp(1j*np.deg2rad(-0.82125690967))],
                   [63.08669858	* np.exp(1j*np.deg2rad(-0.79561180923))],
                  [63.036910109	* np.exp(1j*np.deg2rad(-0.83380432097))]])
Vb_pf = np.matrix([[62.154586369	* np.exp(1j*np.deg2rad(-121.23987672))],
                  [62.122049759	* np.exp(1j*np.deg2rad(-121.26900008))],
                  [62.197764162	* np.exp(1j*np.deg2rad(-121.22825343))]])
Vc_pf = np.matrix([[65.098044778	* np.exp(1j*np.deg2rad(121.78828016))],
                  [65.165654304	* np.exp(1j*np.deg2rad(121.79794749))],
                  [65.145664643	* np.exp(1j*np.deg2rad(121.84076998))]])

Vabc_pq_pf = combine_X012(Va_pf,Vb_pf,Vc_pf)

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
