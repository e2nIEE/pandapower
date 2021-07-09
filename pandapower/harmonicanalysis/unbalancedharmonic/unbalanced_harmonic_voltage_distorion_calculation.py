import numpy as np
import math
import pandapower.harmonicanalysis.unbalancedharmonic.unbalanced_harmonic_voltages_currents_calculation as har_vol_calc

#formatting harmonic voltages in table and calculation of THD
#Nodes need to be sorted 0, 1, 2, 3, 4... Node with index 0 needs to be referent node (External grid is connected to this node)

def unbalanced_thd_voltage(network,harmonics, har_a, har_a_angle, har_b, har_b_angle, har_c, har_c_angle,\
                                        har_a_lc, har_a_lc_angle, har_b_lc, har_b_lc_angle, har_c_lc, har_c_lc_angle,\
                                        analysis_type):
    
    harmonics_voltage_0, harmonics_voltage, harmonics_current_0, harmonics_current = \
                                        har_vol_calc.harmonic_current_voltage_calculator(network, harmonics, \
                                        har_a, har_a_angle, har_b, har_b_angle, har_c, har_c_angle,\
                                        har_a_lc, har_a_lc_angle, har_b_lc, har_b_lc_angle, har_c_lc, har_c_lc_angle,\
                                        analysis_type)
    
    thd_a = []
    thd_b = []
    thd_c = []
    
    
    for i in range(0, np.shape(harmonics_voltage)[0], 3):
        sum_thd = 0
        for j in range(0, np.shape(harmonics_voltage)[1]):
            sum_thd += harmonics_voltage[i, j]**2
        thd_a.append(sum_thd)
    
    for i in range(1, np.shape(harmonics_voltage)[0], 3):
        sum_thd = 0
        for j in range(0, np.shape(harmonics_voltage)[1]):
            sum_thd += harmonics_voltage[i, j]**2
        thd_b.append(sum_thd)
        
    for i in range(2, np.shape(harmonics_voltage)[0], 3):
        sum_thd = 0
        for j in range(0, np.shape(harmonics_voltage)[1]):
            sum_thd += harmonics_voltage[i, j]**2
        thd_c.append(sum_thd)

    for i in range(0, len(thd_a)):
        thd_a[i] = math.sqrt(thd_a[i])/(network.res_bus_3ph.vm_a_pu[i+1])*100
        thd_b[i] = math.sqrt(thd_b[i])/(network.res_bus_3ph.vm_b_pu[i+1])*100
        thd_c[i] = math.sqrt(thd_c[i])/(network.res_bus_3ph.vm_c_pu[i+1])*100
               
    sum_thd_a_0 = 0
    sum_thd_b_0 = 0
    sum_thd_c_0 = 0
    
    for i in range(0, np.shape(harmonics_voltage_0)[1]):
        sum_thd_a_0 += harmonics_voltage_0[0, i]**2
        sum_thd_b_0 += harmonics_voltage_0[1, i]**2 
        sum_thd_c_0 += harmonics_voltage_0[2, i]**2
           
    thd_a.insert(0, math.sqrt(sum_thd_a_0)/network.res_bus_3ph.vm_a_pu[0]*100)
    thd_b.insert(0, math.sqrt(sum_thd_b_0)/network.res_bus_3ph.vm_b_pu[0]*100)
    thd_c.insert(0, math.sqrt(sum_thd_c_0)/network.res_bus_3ph.vm_c_pu[0]*100)  
    
    harmonics_voltage_a = np.zeros([int(len(network.bus.name)*3/3), len(har_a)], dtype = float)
    harmonics_voltage_b = np.zeros([int(len(network.bus.name)*3/3), len(har_a)], dtype = float)
    harmonics_voltage_c = np.zeros([int(len(network.bus.name)*3/3), len(har_a)], dtype = float)
    #Harmonic voltage (percentage) of each phase and each harmoni
    for i in range(0, np.shape(harmonics_voltage_a)[0] - 1):
        for j in range(0, np.shape(harmonics_voltage)[1]):
            harmonics_voltage_a[i+1, j] = harmonics_voltage[3*i, j]*100 * (1/network.res_bus_3ph.vm_a_pu[i+1])
            harmonics_voltage_b[i+1, j] = harmonics_voltage[3*i+1, j]*100 * (1/network.res_bus_3ph.vm_b_pu[i+1])
            harmonics_voltage_c[i+1, j] = harmonics_voltage[3*i+2, j]*100 * (1/network.res_bus_3ph.vm_c_pu[i+1])

    
    for i in range(0, np.shape(harmonics_voltage_0)[1]):
            harmonics_voltage_a[0, i] = harmonics_voltage_0[0, i]*100 * (1/network.res_bus_3ph.vm_a_pu[0])
            harmonics_voltage_b[0, i] = harmonics_voltage_0[1, i]*100 * (1/network.res_bus_3ph.vm_b_pu[0]) 
            harmonics_voltage_c[0, i] = harmonics_voltage_0[2, i]*100 * (1/network.res_bus_3ph.vm_c_pu[0])

    return thd_a, thd_b, thd_c, harmonics_voltage_a, harmonics_voltage_b, harmonics_voltage_c