# -*- coding: utf-8 -*-
"""
# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

@author: sghosh (Intern : Feb 2018-July 2018)
"""



import numpy as np
import scipy as sp

from pandapower.pd2ppc import _pd2ppc
from pandapower.pd2ppc_zero import _pd2ppc_zero
from pandapower.pf.makeYbus import makeYbus

from pandapower.auxiliary import _sum_by_group
from pandapower.pf.run_newton_raphson_pf import _run_newton_raphson_pf
from pandapower.build_bus import _add_ext_grid_sc_impedance
from pandapower.pf.bustypes import bustypes


# =============================================================================
# Functions for 3 Phase Unbalanced Load Flow
# =============================================================================
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

def phase_shift_unit_operator(angle_deg):
    return 1*np.exp(1j*np.deg2rad(angle_deg))

a = phase_shift_unit_operator(120)
asq = phase_shift_unit_operator(-120)
Tabc = np.matrix(
    [
    [1, 1, 1],
    [1, asq, a],
    [1, a, asq]
    ], dtype=np.complex)
    
T012 = np.divide(np.matrix(
    [
    [1, 1, 1],
    [1, a, asq],
    [1, asq, a]
    ],dtype = np.complex),3)
    
def sequence_to_phase(X012):
    return np.matmul(Tabc,X012)

def phase_to_sequence(Xabc):

    return np.matmul(T012,Xabc)

# =============================================================================
# Calculating Sequence Current from sequence Voltages 
# =============================================================================

def I0_from_V012(V012,Y):
    V0 = X012_to_X0(V012)
    if type(Y) == sp.sparse.csr.csr_matrix:
        return np.matmul(Y.todense(),V0)
    else:
        return np.matmul(Y,V0)
def I1_from_V012(V012,Y):
    V1 = X012_to_X1(V012)
    if type(Y) == sp.sparse.csr.csr_matrix:
        return np.matmul(Y.todense(),V1)
    else:
        return np.matmul(Y,V1)
def I2_from_V012(V012,Y):
    V2 = X012_to_X2(V012)
    if type(Y) == sp.sparse.csr.csr_matrix:
        return np.matmul(Y.todense(),V2)
    else:
        return np.matmul(Y,V2)
           
def V1_from_ppc(ppc):
    return np.transpose(
            np.matrix(
            ppc["bus"][:,7] * np.exp(1j*np.deg2rad(ppc["bus"][:,8]))
                                ,dtype = np.complex
            )
            )
            
def V_from_I(Y,I):
    return np.transpose(np.matrix(sp.sparse.linalg.spsolve(Y,I)))
def I_from_V(Y,V):
    if type(Y) == sp.sparse.csr.csr_matrix:
        return np.matmul(Y.todense(), V)
    else:
        return np.matmul(Y, V)

# =============================================================================
# Calculating Power 
# =============================================================================
def S_from_VI(V,I):
    return np.multiply(V,I.conjugate())
# =============================================================================
# Mapping load for positive sequence loads
# =============================================================================
def load_mapping(net):
    _is_elements = net["_is_elements"]
    b = np.array([0], dtype=int)
    SA,SB,SC = np.array([0]), np.array([]), np.array([])
    q_a,QA = np.array([0]), np.array([])
    p_a,PA = np.array([0]), np.array([])
    q_b,QB = np.array([0]), np.array([])
    p_b,PB = np.array([0]), np.array([])
    q_c,QC = np.array([0]), np.array([])
    p_c,PC = np.array([0]), np.array([])
    
    l3 = net["load_3ph"]
    if len(l3) > 0:
        vl =_is_elements["load_3ph"] * l3["scaling"].values.T
        q_a = np.hstack([q_a, l3["q_kvar_A"].values *vl])
        p_a = np.hstack([p_a, l3["p_kw_A"].values *vl])
        q_b = np.hstack([q_b, l3["q_kvar_B"].values *vl])
        p_b = np.hstack([p_b, l3["p_kw_B"].values*vl])
        q_c = np.hstack([q_c, l3["q_kvar_C"].values *vl])
        p_c = np.hstack([p_c, l3["p_kw_C"].values*vl])            
        b = np.hstack([b, l3["bus"].values])

    sgen_3ph = net["sgen_3ph"]
    if len(sgen_3ph) > 0:
        vl = _is_elements["sgen_3ph"] * sgen_3ph["scaling"].values.T
        q_a = np.hstack([q_a, sgen_3ph["q_kvar_A"].values*vl ])
        p_a = np.hstack([p_a, sgen_3ph["p_kw_A"].values*vl ])
        q_b = np.hstack([q_b, sgen_3ph["q_kvar_B"].values *vl])
        p_b = np.hstack([p_b, sgen_3ph["p_kw_B"].values*vl ])
        q_c = np.hstack([q_c, sgen_3ph["q_kvar_C"].values*vl ])
        p_c = np.hstack([p_c, sgen_3ph["p_kw_C"].values *vl])
        b = np.hstack([b, sgen_3ph["bus"].values])
    # For Network Symmetric loads with unsymmetric loads
#    Since the bus values of ppc values are not known, it is added again, fresh
    l = net["load"]
    if len(l) > 0:
        vl = _is_elements["load"] * l["scaling"].values.T
        q_a = np.hstack([q_a, l["q_kvar"].values/3*vl])
        p_a = np.hstack([p_a, l["p_kw"].values/3*vl])
        q_b = np.hstack([q_b, l["q_kvar"].values/3*vl])
        p_b = np.hstack([p_b, l["p_kw"].values/3*vl])
        q_c = np.hstack([q_c, l["q_kvar"].values/3*vl])
        p_c = np.hstack([p_c, l["p_kw"].values/3*vl])       
        b = np.hstack([b, l["bus"].values])

    sgen = net["sgen"]
    if len(sgen) > 0:
        vl = _is_elements["load"] * l["scaling"].values.T
        q_a = np.hstack([q_a, sgen["q_kvar"].values/3*vl])
        p_a = np.hstack([p_a, sgen["p_kw"].values/3*vl])
        q_b = np.hstack([q_b, sgen["q_kvar"].values/3*vl])
        p_b = np.hstack([p_b, sgen["p_kw"].values/3*vl])
        q_c = np.hstack([q_c, sgen["q_kvar"].values/3*vl])
        p_c = np.hstack([p_c, sgen["p_kw"].values/3*vl])
        b = np.hstack([b, sgen["bus"].values/3])
    if b.size:
        bus_lookup = net["_pd2ppc_lookups"]["bus"]
        b = bus_lookup[b]
        b, PA, QA = _sum_by_group(b, p_a,q_a*1j )
        b, PB, QB = _sum_by_group(b, p_b,q_b*1j )
        b, PC, QC = _sum_by_group(b, p_c,q_c*1j )
        b,SA,SB,SC = bus_lookup,PA+QA,PB+QB,PC+QC
    return np.vstack([SA,SB,SC])

# =============================================================================
# 3 phase algorithm function
# =============================================================================
def runpp_3ph(net):
    
    # =============================================================================
    # Y Bus formation for Sequence Networks
    # =============================================================================
    net._options = {'calculate_voltage_angles': 'auto', 'check_connectivity': True, 'init': 'auto',
        'r_switch': 0.0,'voltage_depend_loads': False, 'mode': "pf_3ph",'copy_constraints_to_ppc': False,
        'enforce_q_lims': False, 'numba': True, 'recycle': {'Ybus': False, '_is_elements': False, 'bfsw': False, 'ppc': False},
        "tolerance_kva": 1e-5, "max_iteration": 10}
    _, ppci1 = _pd2ppc(net)
    
    _, ppci2 = _pd2ppc(net)
    _add_ext_grid_sc_impedance(net, ppci2)
    
    _, ppci0 = _pd2ppc_zero(net)
    
    Y0_pu,_,_ = makeYbus(ppci0["baseMVA"], ppci0["bus"], ppci0["branch"])
    
    Y1_pu,_,_ = makeYbus(ppci1["baseMVA"], ppci1["bus"], ppci1["branch"])
    
    Y2_pu,_,_ = makeYbus(ppci2["baseMVA"], ppci2["bus"], ppci2["branch"])
    
    sl_bus,pv_bus,pq_bus = bustypes(ppci1['bus'],ppci1['gen'])

# =============================================================================
# Initial voltage values
# =============================================================================
    nb = ppci1["bus"].shape[0]
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
    Sabc = load_mapping(net)
    # =============================================================================
    #             Iteration using Power mismatch criterion
    # =============================================================================
    while (S_mismatch > 1e-6).any():
    # =============================================================================
    #     Voltages and Current transformation for PQ and Slack bus
    # =============================================================================
        Sabc_pu = -np.divide(Sabc,net.sn_kva)
        Iabc_it = np.divide(Sabc_pu, Vabc_it).conjugate()
        I012_it = phase_to_sequence(Iabc_it)
        
        I0_pu_it = X012_to_X0(I012_it)
        I2_pu_it = X012_to_X2(I012_it)
    
        V1_for_S1= V012_it[1,:]
        I1_for_S1 = -I012_it[1,:]
        S1 = np.multiply(V1_for_S1,I1_for_S1.conjugate())
                
        # =============================================================================
        # Current used to find S1 Positive sequence power    
        # =============================================================================
        
        ppci1["bus"][pq_bus, 2] = np.real(S1[:,pq_bus])*net.sn_kva*1e-3
        ppci1["bus"][pq_bus, 3] = np.imag(S1[:,pq_bus])*net.sn_kva*1e-3
        
        _run_newton_raphson_pf(ppci1, net._options)
    
        I1_from_V_it = -np.transpose(I1_from_V012(V012_it,Y1_pu))
        s_from_voltage = S_from_VI(V1_for_S1, I1_from_V_it)
        
        V1_pu_it = V1_from_ppc(ppci1)
        V0_pu_it = V_from_I(Y0_pu,I0_pu_it)
        V2_pu_it = V_from_I(Y2_pu,I2_pu_it)
    # =============================================================================
    #     This current is YV for the present iteration
    # =============================================================================
        V012_new = combine_X012(V0_pu_it,V1_pu_it,V2_pu_it)
    
#        V_abc_new = sequence_to_phase(V012_new)
    
    # =============================================================================
    #     Mismatch from Sabc to Vabc Needs to be done tomorrow
    # =============================================================================
        S_mismatch = np.abs(S1[:,pq_bus] - s_from_voltage[:,pq_bus])
        V012_it = V012_new
        Vabc_it = sequence_to_phase(V012_it)
        count+= 1
#    Iabc = sequence_to_phase(I012_it)
#    Vabc = Vabc_it
#    Sabc = S_from_VI(Vabc,Iabc)    
    return count,V012_it,I012_it,ppci0,Y1_pu

def show_results(V_base,kVA_base,count,ppci0,Y1_pu,V012_new,I012_new):
    V_base_res = V_base/np.sqrt(3)
    I_base_res = kVA_base/V_base_res * 1e-3 
    print("\n No of Iterations: %u"%count)
    print ('\n\n Final  Values Pandapower ')
    ppci0["bus"][0, 4] = 0
    ppci0["bus"][0, 5] = 0
    Y0_pu,_,_ = makeYbus(ppci0["baseMVA"], ppci0["bus"], ppci0["branch"])
#Y0_pu = Y0_pu.todense()
    I012_new = combine_X012(I0_from_V012(V012_new,Y0_pu),
                        I1_from_V012(V012_new,Y1_pu),
                        I2_from_V012(V012_new,Y1_pu))
    I_abc_new = sequence_to_phase(I012_new)
    V_abc_new = sequence_to_phase(V012_new)
    Sabc_new = S_from_VI(V_abc_new,I_abc_new)*kVA_base
    print ('\n SABC New using I=YV\n')
    print (Sabc_new)
    print (' \n Voltage  ABC\n')
    print (abs(V_abc_new)*V_base_res)

    print ('\n Current  ABC\n')
    print (abs(I_abc_new)*I_base_res)
    return V_abc_new,I_abc_new,Sabc_new
   

	
# =============================================================================
# Main Program
# =============================================================================
# =============================================================================
# Base Value Assignmeent
# =============================================================================
#V_base = 110                     # 110kV Base Voltage
#kVA_base = 100000                      # 100 MVA
#I_base = (kVA_base/V_base) * 1e-3           # in kA
#
#net = pp.create_empty_network(sn_kva = kVA_base )
#
#busn  =  pp.create_bus(net, vn_kv = V_base, name = "busn", index=1)
#busk  =  pp.create_bus(net, vn_kv = V_base, name = "busk", index=5)
#pp.create_bus(net, vn_kv=20., in_service=False)
#pp.create_bus(net, vn_kv=20., in_service=True)
#
#
#pp.create_ext_grid(net, bus=busn, vm_pu= 1.0, name="Grid Connection",
#                   s_sc_max_mva=5000, rx_max=0.1)
#net.ext_grid["r0x0_max"] = 0.1
#net.ext_grid["x0x_max"] = 1.0
#
#pp.create_std_type(net, {"r0_ohm_per_km": 0.0848, "x0_ohm_per_km": 0.4649556, "c0_nf_per_km":  230.6,
#             "max_i_ka": 0.963, "r_ohm_per_km": 0.0212, "x_ohm_per_km": 0.1162389,
#             "c_nf_per_km":  230}, "example_type")
#
#create_load_3ph(net, busk, p_kw_A=50000, q_kvar_A=50000, p_kw_B=10000, q_kvar_B=15000,
#                   p_kw_C=10000, q_kvar_C=5000)
#
#pp.create_line(net, from_bus = busn, to_bus = busk, length_km = 50.0, std_type="example_type")
#
#pp.add_zero_impedance_parameters(net)
#
#count,V012_it,I012_it,ppci0,Y1_pu = runpp_3ph(net)
#show_results(V_base,count,ppci0,Y1_pu,V012_it,I012_it)


