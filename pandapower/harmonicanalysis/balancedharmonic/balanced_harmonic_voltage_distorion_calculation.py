import numpy as np
import math
import pandapower.harmonicanalysis.balancedharmonic.balanced_harmonic_voltages_currents_calculation as har_vol_calc

#formatting harmonic voltages in table and calculation of THD
def balanced_thd_voltage(net, harmonics, har, har_angle, analysis_type):
        
    harmonics_voltage_0, harmonics_voltage = har_vol_calc.harmonic_current_voltage_calculator(net, harmonics, har,\
                                                                                              har_angle, analysis_type)
    
    #Nodes need to be sorted 0, 1, 2, 3, 4... Node with index 0 needs to be referent node (External grid is connected to this node)
    
    thd = []
    for i in range(0, np.shape(harmonics_voltage)[0]):
        sum_thd = 0
        for j in range(0, np.shape(harmonics_voltage)[1]):
            sum_thd += harmonics_voltage[i, j]**2
        thd.append(sum_thd)
        
    for i in range(0, len(thd)):
        thd[i] = math.sqrt(thd[i])/(net.res_bus.vm_pu[i+1])*100 
    
    sum_thd_0 = 0

    for i in range(0, len(harmonics_voltage_0)):
        sum_thd_0 += harmonics_voltage_0[i]**2
    
    thd.insert(0, math.sqrt(sum_thd_0)/net.res_bus.vm_pu[0]*100)
    
    harmonics_voltage_res = np.zeros([int(len(net.bus.name)), len(har)], dtype = float)
    
    for i in range(0, np.shape(harmonics_voltage)[0]):
        for j in range(0, np.shape(harmonics_voltage)[1]):
            harmonics_voltage_res[i+1, j] = harmonics_voltage[i, j]*100
    
    for i in range(0, len(harmonics_voltage_0)):
            harmonics_voltage_res[0, i] = harmonics_voltage_0[i]*100
    
    return thd, harmonics_voltage_res