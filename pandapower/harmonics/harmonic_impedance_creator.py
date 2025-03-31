#Script for creating impedance matrix for higher order frequencies

import pandapower as pp
import numpy as np
import math
import cmath
from pandapower.pd2ppc import _pd2ppc
from pandapower.pd2ppc_zero import _pd2ppc_zero

def harmonic_imp_creator(net, harmon_order, analysis_type):
    harmonics = harmon_order

    a1 = cmath.rect(1, 2/3*cmath.pi)
    a2 = cmath.rect(1, 4/3*cmath.pi)
    
    matrix_A = np.matrix([[1, 1, 1], [1, a2, a1], [1, a1, a2]])
    
    harmonic_matrices = []
    harmonic_ext_matrices = []
    
    #In case of unbalanced harmonic analyses, values for zero sequence system need to be defined.
    #If they are unknown, they are assumed to be same as values for positive sequence system.
    if 'r0_ohm_per_km' not in net.line:
        
        net.line['r0_ohm_per_km'] = net.line['r_ohm_per_km']
       
    if 'x0_ohm_per_km' not in net.line:
        net.line['x0_ohm_per_km'] = net.line['x_ohm_per_km']
    
    if 'c0_nf_per_km' not in net.line:
        net.line['c0_nf_per_km'] = net.line['c_nf_per_km']
    
    #Defining parameters needed for LF calculations
    
    net["_options"] = {"mode":"pf", "check_connectivity":True, "calculate_voltage_angles":True, "init_vm_pu":True,
                    "init_va_degree":True, "consider_line_temperature":False, "voltage_depend_loads":False,
                    "trafo3w_losses":"hv", "neglect_open_switch_branches":False, "p_lim_default":1000000,
                    "delta":0, "q_lim_default":1000000, "trafo_model":"t", "distributed_slack": False, "tdpf": False}
    net["_isolated_buses"] = []
    net["_is_elements"] = None
    
    u_base = net.bus.vn_kv[net.ext_grid.bus[0]]
    
    #Calculation of external grid's impedance and the network's impedance matrix for every harmonic order 
    #Already developed pandapower functionalities are used for matrices creation
    for h in range(0, len(harmonics)):
                
        ppc_0, ppci_0 = _pd2ppc_zero(net, None)
        
        ybus_0, yf_0, yt_0 = pp.pypower.makeYbus.makeYbus(ppci_0["baseMVA"], ppci_0["bus"], ppci_0["branch"])
    
        ppc_1, ppci_1 = _pd2ppc(net, 1)

        ybus_1, yf_1, yt_1 = pp.pypower.makeYbus.makeYbus(ppci_1["baseMVA"], ppci_1["bus"], ppci_1["branch"])
    
        ppc_2, ppci_2 = _pd2ppc(net, 2)

        ybus_2, yf_2, yt_2 = pp.pypower.makeYbus.makeYbus(ppci_2["baseMVA"], ppci_2["bus"], ppci_2["branch"])
        
        #Full sequence admittance matrices            
        zero_full = ybus_0.todense()
        pos_full = ybus_1.todense()
        neg_full = ybus_2.todense()
        
        #Removing referent node related row and column of each matrix
        #It is assumed that first row and column are referent node related
        zero = np.delete(np.delete(zero_full, 0, 0), 0, 1)
        pos = np.delete(np.delete(pos_full, 0, 0), 0, 1)
        neg = np.delete(np.delete(neg_full, 0, 0), 0, 1)
           
        #External grid impedance calculation
        
        z_pos_ohm = u_base**2/net.ext_grid.s_sc_max_mva
        delta_pos = math.atan(1/net.ext_grid.rx_max)
        r_pos = z_pos_ohm*math.cos(delta_pos)
        x_pos = z_pos_ohm*math.sin(delta_pos)*harmonics[h]
        z_pos = complex(r_pos, x_pos)

        z_zer_ohm = z_pos_ohm*net.ext_grid.x0x_max 
        delta_zer = math.atan(1/net.ext_grid.r0x0_max)
        r_zer = z_zer_ohm*math.cos(delta_zer)
        x_zer = z_zer_ohm*math.sin(delta_zer)*harmonics[h]
        z_zer = complex(r_zer, x_zer)
        
        #Depenent on the analysis type, impedances are created
        if analysis_type == 'unbalanced':
            
            phase_mat_y = np.zeros([3*(np.shape(zero)[0]), 3*(np.shape(zero)[1])], dtype = complex)
            
            z_ext_0 = np.zeros([3,3], dtype = complex)
            
            #Z012 matrix is created for the referent node
            z_ext_0[0, 0] = z_zer/(u_base**2/3)
            z_ext_0[1, 1] = z_pos/(u_base**2/3)
            z_ext_0[2, 2] = z_pos/(u_base**2/3)
            
            #Transformation from the sequence to the phase system    
            z_ext_abc = np.matmul(np.matmul(np.linalg.inv(matrix_A), z_ext_0), matrix_A)
                    
            for i in range(0, np.shape(zero)[0]):
                for j in range(0, np.shape(zero)[1]):
                    #Y012 is created for every other element of Y0,1,2 matrices
                    y_012 = np.zeros([3,3], dtype = complex)
                    
                    z = zero[i, j]  
                    p = pos[i,j]
                    n = neg[i,j]
                    
                    y_012[0,0] = z
                    y_012[1,1] = p
                    y_012[2,2] = n
                    
                    #Transformation from the sequence to the phase system
                    y_abc = np.matmul(np.matmul(np.linalg.inv(matrix_A), y_012), matrix_A)
                    
                    for row in range (0, 3):
                        for col in range(0, 3):
                            phase_mat_y[3*i+row, 3*j+col] = y_abc[row, col]
            #Z matrix as an inverse of the Y matrix               
            phase_mat_z = np.linalg.inv(phase_mat_y)*3
            
            #Dependent on the harmonic order, reactive part of the impedance (reactance) is calculated as h*X
            for r in range(0, np.shape(phase_mat_z)[0]):
                for c in range(0, np.shape(phase_mat_z)[1]):
                    aux = phase_mat_z[r,c]
                    res = np.real(aux) 
                    reac = np.imag(aux) * harmonics[h]
                    imp = complex(res, reac)
                    phase_mat_z[r,c] = imp
            
            #Appending the referent node and network impedances
            harmonic_matrices.append(phase_mat_z)
            harmonic_ext_matrices.append(z_ext_abc)
        
        elif analysis_type == 'balanced_positive':
            
            #Only positive sequence system is observed
            z_pos_net = np.linalg.inv(pos)
            
            #Dependent on the harmonic order, reactive part of the impedance (reactance) is calculated as h*X
            for r in range(0, np.shape(z_pos_net)[0]):
                for c in range(0, np.shape(z_pos_net)[1]):
                    aux = z_pos_net[r,c]
                    res = np.real(aux)
                    reac = np.imag(aux) * harmonics[h]
                    imp = complex(res, reac)
                    z_pos_net[r,c] = imp
            
            #Appending the referent node and network impedances
            harmonic_matrices.append(z_pos_net)
            harmonic_ext_matrices.append(z_pos/(u_base**2))
        
        elif analysis_type == 'balanced_all':
            
            #Balanced network but all sequence systems are used
            #Dependent on the harmonic order, zero, positive or negative sequence impedance are considered
            z_pos_net = np.linalg.inv(pos)
            z_neg_net = np.linalg.inv(neg)
            z_zero_net = np.linalg.inv(zero)
            
            #Dependent on the harmonic order, reactive part of the impedance (reactance) is calculated as h*X
            for r in range(0, np.shape(z_pos_net)[0]):
                for c in range(0, np.shape(z_pos_net)[1]):
                    aux_p = z_pos_net[r,c]
                    res_p = np.real(aux_p)
                    reac_p = np.imag(aux_p) * harmonics[h]
                    imp_p = complex(res_p, reac_p)
                    z_pos_net[r,c] = imp_p
                    
                    aux_n = z_neg_net[r,c]
                    res_n = np.real(aux_n)
                    reac_n = np.imag(aux_n) * harmonics[h]
                    imp_n = complex(res_n, reac_n)
                    z_neg_net[r,c] = imp_n
                    
                    aux_z = z_zero_net[r,c]
                    res_z = np.real(aux_z)
                    reac_z = np.imag(aux_z) * harmonics[h]
                    imp_z = complex(res_z, reac_z)
                    z_zero_net[r,c] = imp_z
            
            #Appending the referent node and network impedances
            if harmonics[h] % 3 == 0:
                harmonic_matrices.append(z_zero_net)
                harmonic_ext_matrices.append(z_zer/(u_base**2))
            elif harmonics[h] % 3 == 1:
                harmonic_matrices.append(z_pos_net)
                harmonic_ext_matrices.append(z_pos/(u_base**2))
            elif harmonics[h] % 3 == 2:
                harmonic_matrices.append(z_neg_net)
                harmonic_ext_matrices.append(z_pos/(u_base**2))
                
    return(harmonic_matrices, harmonic_ext_matrices)
        