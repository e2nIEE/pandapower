# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 17:23:14 2018

@author: sghosh
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 09:53:51 2018

@author: sghosh
"""

import pandapower as pp
import numpy as np

from pandapower.pd2ppc import _pd2ppc
from pandapower.pd2ppc_zero import _pd2ppc_zero
from pandapower.pf.makeYbus import makeYbus

# =============================================================================
# Base Value Assignmeent
# =============================================================================
V_base = 110                     # 110kV Base Voltage
kVA_base = 100000                      # 100 MVA
I_base = (kVA_base/V_base) * 1e-3           # in kA
#Z_base = V_base/I_base

net = pp.create_empty_network(sn_kva = kVA_base )
# =============================================================================
# Index Values
# =============================================================================

pq_bus = [1]
sl_bus = [0]
load_id = [0]
# =============================================================================
# Convert to three decoupled sequence networks 
# =============================================================================

def X012_to_X0(X012):
    return np.transpose(X012[0,:]) 
def X012_to_X1(X012):
    return np.transpose(X012[1,:])
def X012_to_X2(X012):
    return np.transpose(X012[2,:])
# =============================================================================
# Three decoupled sequence network to 012 matrix conversion
# =============================================================================

def combine_X012(X0,X1,X2):
    return np.transpose(np.concatenate((X0,X1,X2),axis=1))

# =============================================================================
# Symmetrical transformation matrix 
# Tabc : 012 > abc
# T012 : abc >012
# =============================================================================

def sequence_to_phase(X012):
    Tabc = np.matrix(
    [
    [1*np.exp(1j*np.deg2rad(0)), 1*np.exp(1j*np.deg2rad(0))   , 1*np.exp(1j*np.deg2rad(0))]
    ,[1*np.exp(1j*np.deg2rad(0)), 1*np.exp(1j*np.deg2rad(-120)), 1*np.exp(1j*np.deg2rad(120))]
    ,[1*np.exp(1j*np.deg2rad(0)), 1*np.exp(1j*np.deg2rad(120)) , 1*np.exp(1j*np.deg2rad(-120))]
    ],dtype = np.complex)
    return np.matmul(Tabc,X012)
def phase_to_sequence(Xabc):
    Tabc = np.matrix(
            [
            [1*np.exp(1j*np.deg2rad(0)),1*np.exp(1j*np.deg2rad(0))   ,1*np.exp(1j*np.deg2rad(0))]
            ,[1*np.exp(1j*np.deg2rad(0)),1*np.exp(1j*np.deg2rad(-120)),1*np.exp(1j*np.deg2rad(120))]
            ,[1*np.exp(1j*np.deg2rad(0)),1*np.exp(1j*np.deg2rad(120)) ,1*np.exp(1j*np.deg2rad(-120))]
            ],dtype = np.complex)
    return np.matmul(np.linalg.inv(Tabc),Xabc)

# =============================================================================
# Calculating Current Voltage and Power
# =============================================================================

def I0_from_V012(V012,Y):
    return np.matmul(Y,X012_to_X0(V012))
def I1_from_V012(V012,Y):
    return np.matmul(Y,X012_to_X1(V012))
def I2_from_V012(V012,Y):
    return np.matmul(Y,X012_to_X2(V012))
           
def V1_from_ppc(ppc):
    return np.transpose(
            np.matrix(
            ppc1["bus"][:,7] * np.exp(1j*np.deg2rad(ppc1["bus"][:,8]))
                                ,dtype = np.complex

            )
            )
            
def V_from_I(Y,I):
    return np.matmul(np.linalg.inv(Y),I)
def I_from_V(Y,V):
    return np.matmul(Y, V)

def S_from_VI(V,I):
    return np.multiply(V,I.conjugate())

# =============================================================================
# Y0 formation for Zero Sequence Network
# =============================================================================


bus1 = pp.create_bus(net, vn_kv = V_base/np.sqrt(3), name = "Bus k")
bus2 = pp.create_bus(net, vn_kv = V_base/np.sqrt(3), name = "Bus m")
pp.create_ext_grid(net, bus=bus1, vm_pu= 1.0, name="Grid Connection",
                   s_sc_max_mva=5000, rx_max=0.1)
net.ext_grid["r0x0_max"] = 0.1
net.ext_grid["x0x_max"] = 1.0

pp.create_std_type(net, {"r0_ohm_per_km": 0.0848, "x0_ohm_per_km": 0.4649556, "c0_nf_per_km": 230.6,
             "max_i_ka": 0.963, "r_ohm_per_km": 0.0212, "x_ohm_per_km": 0.1162389,
             "c_nf_per_km":  230}, "example_type")
pp.create_load(net, bus2, p_kw=10, q_kvar=20)
pp.create_line(net, from_bus=bus1, to_bus=bus2, length_km = 50.0, std_type="example_type")
pp.add_zero_impedance_parameters(net)

net._options = {'calculate_voltage_angles': 'auto', 'check_connectivity': True, 'init': 'auto',
    'r_switch': 0.0,'voltage_depend_loads': False, 'mode': "pf",'copy_constraints_to_ppc': False,
    'enforce_q_lims': False, 'numba': True, 'recycle': {'Ybus': False, '_is_elements': False, 'bfsw': False, 'ppc': False},
    "tolerance_kva": 1e-5, "max_iteration": 10}
ppc1, ppci1 = _pd2ppc(net)

from pandapower.build_bus import _add_ext_grid_sc_impedance

ppc2, ppci2 = _pd2ppc(net)
_add_ext_grid_sc_impedance(net, ppci2)

#ppc['bus'][0,4] = np.real(Y2_pf_eg)
#ppc['bus'][0,5] = np.imag(Y2_pf_eg)

ppc0, ppci0 = _pd2ppc_zero(net)

#ppc0['bus'][0,4] = np.real(Y0_pf_eg)
#ppc0['bus'][0,5] = np.imag(Y0_pf_eg)

Y0_pu,_,_ = makeYbus(ppci0["baseMVA"], ppci0["bus"], ppci0["branch"])
Y0_pu = Y0_pu.todense()

Y1_pu,_,_ = makeYbus(ppci1["baseMVA"], ppci1["bus"], ppci1["branch"])
Y1_pu = Y1_pu.todense()

Y2_pu,_,_ = makeYbus(ppci2["baseMVA"], ppci2["bus"], ppci2["branch"])
Y2_pu = Y2_pu.todense()

Sabc =  -1*np.matrix(   [
                         [0,50000+50000j] 
                        ,[0,10000+15000j] 
                        ,[0,10000+5000j]
                        ]
                    ,dtype = np.complex 
                    ) #kW and kVAr

#Sabc = np.append(-np.matrix.sum(Sabc,axis =1),Sabc,axis =1)

Sabc_pu = np.divide(Sabc,kVA_base)

# =============================================================================
# Initial voltage values
# =============================================================================
V012_it = np.matrix(    [
                         [0,0]
                        ,[1,1]
                        ,[0,0]
                        ]
                    ,dtype = np.complex
                    ) 

Vabc_it = sequence_to_phase(V012_it)




# =============================================================================
# Initialise iteration variables
# =============================================================================
count = 0
#Vabc_it = Vabc_pu
#V012_it = V012_pu
#I012_it = I012_pu
#
#V1_pu_it = bus_to_pos_seq(V012_pu)

S_mismatch = np.matrix([[True],[True]],dtype =bool)
'''
=============================================================================
            Iteration using Power mismatch criterion
=============================================================================
'''
while (S_mismatch > 1e-8).all():
    print("\n count %u"%count)
# =============================================================================
#     Voltages and Current transformation for PQ and Slack bus
# =============================================================================
    Iabc_it = np.divide(Sabc_pu, Vabc_it).conjugate()
    I012_it = phase_to_sequence(Iabc_it)
    
    I0_pu_it = X012_to_X0(I012_it)
    I2_pu_it = X012_to_X2(I012_it)
    
    V1_for_S1 = V012_it[:,pq_bus][1,:]
    I1_for_S1 = -I012_it[:,pq_bus][1,:]
    
    S1 = np.multiply(V1_for_S1,I1_for_S1.conjugate())
        
#    V012_pu_it = np.transpose(np.concatenate((
#            V0_pu_it, 
#            V1_pu_it, 
#            V2_pu_it), axis=1))
#    I012_pu_it = np.transpose(np.concatenate((I0_pu_it,I1_pu_it,I2_pu_it),axis=1)) 
    
# =============================================================================
# Current used to find S1 Positive sequence power    
# =============================================================================
    ppc1["bus"][pq_bus, 2] = np.real(S1)*kVA_base*1e-3
    ppc1["bus"][pq_bus, 3] = np.imag(S1)*kVA_base*1e-3

    from pandapower.pf.run_newton_raphson_pf import _run_newton_raphson_pf
    _run_newton_raphson_pf(ppc1, net._options)

    I1_from_V_it = -I1_from_V012(V012_it,Y1_pu)
    s_from_voltage = S_from_VI(V1_for_S1, I1_from_V_it[pq_bus,:])
    
    V1_pu_it = V1_from_ppc(ppc1)
    V0_pu_it = V_from_I(Y0_pu,I0_pu_it)
    V2_pu_it = V_from_I(Y2_pu,I2_pu_it)
# =============================================================================
#     This current is YV for the present iteration
# =============================================================================

    
#    print (s_from_voltage*kVA_base)
#    print (V0_pu_it)
#    print (V2_pu_it)
    V012_new = combine_X012(V0_pu_it,V1_pu_it,V2_pu_it)


    V_abc_new = sequence_to_phase(V012_new)

     
#    I1_from_V_new = I0_from_V012(V012_new, Y0_pu)
    print('V it')
    print (abs(V012_it))
    print('V New')
    print(abs(V012_new))
    
    print ('I it')
    print (I012_it)
#    print ('I 012 New')
#    print (I012_new)
#    print (S_from_VI(V_abc_new,I_abc_new)*kVA_base)
# =============================================================================
#     Mismatch from Sabc to Vabc Needs to be done tomorrow
# =============================================================================
    S_mismatch = np.abs(S1 - s_from_voltage)
    print ('\n S Mismatch \n')
    print (S_mismatch)
#    V_mismatch = V012_new-V012_it
    V012_it = V012_new
    Vabc_it = sequence_to_phase(V012_it)
#    Sabc_pu[:,sl_bus] = S_from_VI(Vabc_it[:,sl_bus],I_abc_new[:,sl_bus])
    
#    Iabc_from_V = sequence_to_phase(combine_X012(I0_pu_it,I1_from_V_new,I2_pu_it))
#    Sabc_from_V = S_from_VI(Vabc_it,Iabc_from_V)
#    print ('\n Power abc\n')
#    print (abs(Sabc_from_V*kVA_base))
    count+= 1
#    if count > 30:
#        break
Iabc = sequence_to_phase(I012_it)
Vabc = Vabc_it
Sabc = S_from_VI(Vabc,Iabc)
print ('\n\n Final Power Values ')
print (Sabc*kVA_base)



print('V012 New')
print(abs(V012_new))
print ('I Using S/V')
print (I012_it)

ppc0["bus"][0, 4] = 0
ppc0["bus"][0, 5] = 0

Y0_pu,_,_ = makeYbus(ppc0["baseMVA"], ppc0["bus"], ppc0["branch"])
Y0_pu = Y0_pu.todense()
I012_new = combine_X012(I0_from_V012(V012_new,Y0_pu),
                        I1_from_V012(V012_new,Y1_pu),
                        I2_from_V012(V012_new,Y1_pu))
I_abc_new = sequence_to_phase(I012_new)

print ('I using YV')
print (I012_new)

print ('\n SABC New using I=YV\n')
print (S_from_VI(V_abc_new,I_abc_new)*kVA_base)

print (' \n Voltage  ABC\n')
print (abs(V_abc_new)*V_base)

print ('\n Current  ABC\n')
print (abs(I_abc_new)*I_base)

