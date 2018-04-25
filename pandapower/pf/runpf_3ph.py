# -*- coding: utf-8 -*-
"""
# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

@author: sghosh (Intern :Feb 2018-July 2018)
"""


import pandapower as pp
import numpy as np

from pandapower.create import create_load_3ph
from pandapower.pd2ppc import _pd2ppc
from pandapower.pd2ppc_zero import _pd2ppc_zero
from pandapower.pf.makeYbus import makeYbus
from pandapower.auxiliary import X012_to_X0, X012_to_X2, combine_X012,sequence_to_phase
from pandapower.auxiliary import phase_to_sequence, I0_from_V012, I1_from_V012, I2_from_V012
from pandapower.auxiliary import V1_from_ppc, V_from_I,S_from_VI
from pandapower.pf.run_newton_raphson_pf import _run_newton_raphson_pf
from pandapower.build_bus import _add_ext_grid_sc_impedance
from pandapower.pf.bustypes import bustypes


# =============================================================================
# Base Value Assignmeent
# =============================================================================
V_base = 110                     # 110kV Base Voltage
kVA_base = 100000                      # 100 MVA
I_base = (kVA_base/V_base) * 1e-3           # in kA

net = pp.create_empty_network(sn_kva = kVA_base )
# =============================================================================
# Index Values
# =============================================================================


# =============================================================================
# Sequence Network Parameters
# =============================================================================

busn  =  pp.create_bus(net, vn_kv = V_base, name = "busn")
busk  =  pp.create_bus(net, vn_kv = V_base, name = "busk")

pp.create_ext_grid(net, bus=busn, vm_pu= 1.0, name="Grid Connection",
                   s_sc_max_mva=5000, rx_max=0.1)
net.ext_grid["r0x0_max"] = 0.1
net.ext_grid["x0x_max"] = 1.0

pp.create_std_type(net, {"r0_ohm_per_km": 0.0848, "x0_ohm_per_km": 0.4649556, "c0_nf_per_km":  230.6,
             "max_i_ka": 0.963, "r_ohm_per_km": 0.0212, "x_ohm_per_km": 0.1162389,
             "c_nf_per_km":  230}, "example_type")


# =============================================================================
# PA_busk = 50000
# QA_busk = 20000
# PB_busk = 80000
# QB_busk = 60000
# PC_busk = 20000
# QC_busk = 5000
# 
# PA_busm = 50000
# QA_busm = 50000
# PB_busm = 10000
# QB_busm = 15000
# PC_busm = 10000
# QC_busm = 5000
# 
# PA_busp = 50000
# QA_busp = 20000
# PB_busp = 60000
# QB_busp = 20000
# PC_busp = 10000
# QC_busp = 5000
# =============================================================================



create_load_3ph(net, busk, p_kw_A=50000, q_kvar_A=50000, p_kw_B=10000, q_kvar_B=15000,
                   p_kw_C=10000, q_kvar_C=5000)

pp.create_line(net, from_bus = busn, to_bus = busk, length_km = 50.0, std_type="example_type")

pp.add_zero_impedance_parameters(net)

net._options = {'calculate_voltage_angles': 'auto', 'check_connectivity': True, 'init': 'auto',
    'r_switch': 0.0,'voltage_depend_loads': False, 'mode': "pf_3ph",'copy_constraints_to_ppc': False,
    'enforce_q_lims': False, 'numba': True, 'recycle': {'Ybus': False, '_is_elements': False, 'bfsw': False, 'ppc': False},
    "tolerance_kva": 1e-5, "max_iteration": 10}

# =============================================================================
# Y Bus formation for Sequence Networks
# =============================================================================
ppc1, ppci1 = _pd2ppc(net)

ppc2, ppci2 = _pd2ppc(net)
_add_ext_grid_sc_impedance(net, ppci2)

ppc0, ppci0 = _pd2ppc_zero(net)

Y0_pu,_,_ = makeYbus(ppci0["baseMVA"], ppci0["bus"], ppci0["branch"])

Y1_pu,_,_ = makeYbus(ppci1["baseMVA"], ppci1["bus"], ppci1["branch"])

Y2_pu,_,_ = makeYbus(ppci2["baseMVA"], ppci2["bus"], ppci2["branch"])

sl_bus,pv_bus,pq_bus = bustypes(ppc1['bus'],ppc1['gen'])

# =============================================================================
# 3 Phase Load converted to Sabc matrix
# =============================================================================
l = net["load_3ph"]
if len(l) > 0:
    Sa,Sb,Sc = np.array([0+0j]), np.array([0+0j]), np.array([0+0j])
    
    Sa = np.matrix(np.hstack([Sa,l["p_kw_A"].values + l["q_kvar_A"].values*1j]))
    Sb = np.matrix(np.hstack([Sb,l["p_kw_B"].values + l["q_kvar_B"].values*1j]))
    Sc = np.matrix(np.hstack([Sc,l["p_kw_C"].values + l["q_kvar_C"].values*1j]))
    Sabc =  -np.concatenate((Sa,Sb,Sc),axis=0)

Sabc_pu = np.divide(Sabc,kVA_base)

# =============================================================================
# Initial voltage values
# =============================================================================
nb = len(net.bus)
V012_it = np.concatenate(    
                        (
                         np.matrix(np.zeros((1,nb),dtype=np.complex))
                        ,np.matrix(np.ones((1,nb),dtype=np.complex))
                        ,np.matrix(np.zeros((1,nb),dtype=np.complex))
                        )
                    ,axis =0
                    ) 

Vabc_it = sequence_to_phase(V012_it)

# =============================================================================
# Initialise iteration variables
# =============================================================================
count = 0
S_mismatch = np.matrix([[True],[True]],dtype =bool)

# =============================================================================
#             Iteration using Power mismatch criterion
# =============================================================================
while (S_mismatch > 1e-6).all():
# =============================================================================
#     Voltages and Current transformation for PQ and Slack bus
# =============================================================================
    Iabc_it = np.divide(Sabc_pu, Vabc_it).conjugate()
    I012_it = phase_to_sequence(Iabc_it)
    
    I0_pu_it = X012_to_X0(I012_it)
    I2_pu_it = X012_to_X2(I012_it)
    npq = len(pq_bus)
    V1_for_S1 = np.matrix(np.zeros((1,npq),dtype = np.complex))
    I1_for_S1 = np.matrix(np.zeros((1,npq),dtype = np.complex))
    S1 = np.matrix(np.zeros((1,npq),dtype = np.complex))
    
    for load_id in net.load_3ph.index.values:
        for bus_id in pq_bus:
            if net.load_3ph.bus[load_id] == bus_id:
                V1_for_S1[0,load_id] = V012_it[:,load_id+1][1,:]
                I1_for_S1[0,load_id] = -I012_it[:,load_id+1][1,:]
                S1[0,load_id] = np.multiply(V1_for_S1[0,load_id],I1_for_S1[0,load_id].conjugate())
            
            # =============================================================================
            # Current used to find S1 Positive sequence power    
            # =============================================================================
                ppc1["bus"][bus_id, 2] = np.real(S1[0,load_id])*kVA_base*1e-3
                ppc1["bus"][bus_id, 3] = np.imag(S1[0,load_id])*kVA_base*1e-3

    
    _run_newton_raphson_pf(ppc1, net._options)

    I1_from_V_it = -np.transpose(I1_from_V012(V012_it,Y1_pu))
    s_from_voltage = S_from_VI(V1_for_S1, I1_from_V_it[0,pq_bus])
    
    V1_pu_it = V1_from_ppc(ppc1)
    V0_pu_it = V_from_I(Y0_pu,I0_pu_it)
    V2_pu_it = V_from_I(Y2_pu,I2_pu_it)
# =============================================================================
#     This current is YV for the present iteration
# =============================================================================
    V012_new = combine_X012(V0_pu_it,V1_pu_it,V2_pu_it)

    V_abc_new = sequence_to_phase(V012_new)

# =============================================================================
#     Mismatch from Sabc to Vabc Needs to be done tomorrow
# =============================================================================
    S_mismatch = np.abs(S1 - s_from_voltage)
    V012_it = V012_new
    Vabc_it = sequence_to_phase(V012_it)
    count+= 1

print("\n No of Iterations: %u"%count)
Iabc = sequence_to_phase(I012_it)
Vabc = Vabc_it
Sabc = S_from_VI(Vabc,Iabc)

print ('\n\n Final  Values Pandapower ')
V_base_res = V_base/np.sqrt(3)
I_base_res = (kVA_base/V_base_res) * 1e-3 

ppc0["bus"][0, 4] = 0
ppc0["bus"][0, 5] = 0

Y0_pu,_,_ = makeYbus(ppc0["baseMVA"], ppc0["bus"], ppc0["branch"])
#Y0_pu = Y0_pu.todense()
I012_new = combine_X012(I0_from_V012(V012_new,Y0_pu),
                        I1_from_V012(V012_new,Y1_pu),
                        I2_from_V012(V012_new,Y1_pu))
I_abc_new = sequence_to_phase(I012_new)

Sabc_new = S_from_VI(V_abc_new,I_abc_new)*kVA_base
print ('\n SABC New using I=YV\n')
print (Sabc_new)

print (' \n Voltage  ABC\n')
print (abs(V_abc_new)*V_base_res)

print ('\n Current  ABC\n')
print (abs(I_abc_new)*I_base_res)



print ('\n Power factory Values \n ')


   
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

print ('\n SABC \n')
print (Sabc_powerFactory)

 # 

print (' \n Voltage  ABC\n')
print (abs(Vabc_powerFactory)*V_base_res)

print ('\n Current  ABC\n')
print (abs(Iabc_powerFactory)*I_base_res)

print ('\nDifference between Power Factory and pandapower values\n')

print ('\n Power difference')
print ((abs(Sabc_powerFactory)-abs(Sabc_new))/1000,'MVA\n')

print ('\n Current  difference')
print (abs(Iabc_powerFactory)-abs(I_abc_new*I_base_res),'kA\n')

print ('\n Voltage difference')
print (abs(Vabc_powerFactory)- abs(V_abc_new*V_base_res),'kV\n')