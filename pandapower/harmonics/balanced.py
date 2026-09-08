import math
import cmath
import numpy as np
from pandapower import runpp
from pandapower.auxiliary import pandapowerNet
import pandapower.harmonics.harmonic_impedance_creator as hic


# formatting harmonic voltages in table and calculation of THD
def balanced_thd_voltage(net: pandapowerNet,
                         harmonics: list[int],
                         har: list[float], har_angle: list[float],
                         analysis_type: str):

    harmonics_voltage_0, harmonics_voltage = balanced_harmonic_current_voltage(net, harmonics, har, har_angle, analysis_type)

    # Nodes need to be sorted 0, 1, 2, 3, 4... Node with index 0 needs to be referent node (External grid is connected to this node)

    thd = []
    for i in range(0, np.shape(harmonics_voltage)[0]):
        sum_thd = 0
        for j in range(0, np.shape(harmonics_voltage)[1]):
            sum_thd += harmonics_voltage[i, j] ** 2
        thd.append(sum_thd)

    for i in range(0, len(thd)):
        thd[i] = math.sqrt(thd[i]) / (net.res_bus.vm_pu[i + 1]) * 100

    sum_thd_0 = 0

    for i in range(0, len(harmonics_voltage_0)):
        sum_thd_0 += harmonics_voltage_0[i] ** 2

    thd.insert(0, math.sqrt(sum_thd_0) / net.res_bus.vm_pu[0] * 100)

    harmonics_voltage_res = np.zeros([int(len(net.bus.name)), len(har)], dtype=float)

    for i in range(0, np.shape(harmonics_voltage)[0]):
        for j in range(0, np.shape(harmonics_voltage)[1]):
            harmonics_voltage_res[i + 1, j] = harmonics_voltage[i, j] * 100

    for i in range(0, len(harmonics_voltage_0)):
        harmonics_voltage_res[0, i] = harmonics_voltage_0[i] * 100

    return thd, harmonics_voltage_res


# calculation of harmonic voltages from harmonic currents
# at the moment it is possible to define only one harmonic patter which is same for every harmonic source
def balanced_harmonic_current_voltage(net: pandapowerNet,
                                      harmonics: list[int],
                                      har: list[float], har_angle: list[float],
                                      analysis_type: str):

    delta_harmonics_voltage = np.zeros([len(net.bus.name) - 1, len(har)], dtype=complex)
    harmonics_voltage = np.zeros([len(net.bus.name) - 1, len(har)], dtype=float)
    harmonic_cur_val = []
    harmonic_cur_ang = []
    u_harmonics_0 = []
    u_harmonics_0_ang = []
    harmonic_cur_0 = []
    harmonic_cur_0_ang = []

    har_matrices, har_ext_matrix = hic.harmonic_imp_creator(net, harmonics, analysis_type)

    for h in range(0, len(harmonics)):
        mat_z = har_matrices[h]
        z_ext = har_ext_matrix[h]

        if harmonics[h] == 1:
            runpp(net)

            current = []
            current_0 = []

            s = complex(net.res_bus.p_mw[net.ext_grid.bus[0]], net.res_bus.q_mvar[net.ext_grid.bus[0]])

            current_0.append(np.conjugate(s / (math.sqrt(3) * (cmath.rect(net.res_bus.vm_pu[net.ext_grid.bus[0]],
                                                                          net.res_bus.va_degree[
                                                                              net.ext_grid.bus[0]])))))

            for i in net.res_bus.index:
                connected = 0

                if i != net.ext_grid.bus[0]:
                    for j in range(0, len(net.load.index)):

                        if net.load.bus[j] == net.res_bus.index[i]:
                            connected = 1

                            s = complex(net.res_bus.p_mw[i], net.res_bus.q_mvar[i])

                    if connected == 1:
                        current.append(np.conjugate(s \
                                                    / (math.sqrt(3) * (
                            cmath.rect(net.res_bus.vm_pu[i], net.res_bus.va_degree[i])))))
                    else:
                        current.append(0 + 0j)
        else:

            harmonics_current = []
            harmonics_angle = []

            for i in range(0, len(net.bus.name)):
                harmonics_current.append(har[h - 1])
                # Only positive sequence system is considered. harmonics[h]*har_a_angle[h-1] + 240 can be appended
                # but it does not change the magnitude of the voltage, only the angle.
                if analysis_type == 'balanced_positive':

                    if harmonics[h] % 3 == 0:
                        harmonics_angle.append(harmonics[h] * har_angle[h - 1] + 240)
                    elif harmonics[h] % 3 == 1:
                        harmonics_angle.append(harmonics[h] * har_angle[h - 1] + 240)
                    elif harmonics[h] % 3 == 2:
                        harmonics_angle.append(harmonics[h] * har_angle[h - 1] + 240)

                elif analysis_type == 'balanced_all':

                    if harmonics[h] % 3 == 0:
                        harmonics_angle.append(harmonics[h] * har_angle[h - 1])
                    elif harmonics[h] % 3 == 1:
                        harmonics_angle.append(harmonics[h] * har_angle[h - 1] + 240)
                    elif harmonics[h] % 3 == 2:
                        harmonics_angle.append(harmonics[h] * har_angle[h - 1] + 120)

            har_cur = []

            for i in range(0, len(current)):
                har_cur.append(
                    -abs(current[i]) * cmath.rect(harmonics_current[i] / 100, harmonics_angle[i] * cmath.pi / 180))
                harmonic_cur_val.append(abs(har_cur[i]))
                harmonic_cur_ang.append(cmath.phase(har_cur[i]) * 180 / cmath.pi)

            sum_a = 0

            for a in range(0, len(har_cur)):
                sum_a += har_cur[a]

            har_cur_0 = []
            har_cur_0.append(sum_a)

            harmonic_cur_0.append(abs(sum_a))
            harmonic_cur_0_ang.append(cmath.phase(sum_a) * 180 / cmath.pi)

            u_har_0 = (z_ext * har_cur_0[0] * math.sqrt(3))

            u_harmonics_0.append(abs(u_har_0))
            u_harmonics_0_ang.append(cmath.phase(u_har_0) * 180 / cmath.pi)

            delta_har_vol = np.matmul(mat_z, har_cur) * math.sqrt(3)

            for i in range(0, np.shape(delta_harmonics_voltage)[0]):
                delta_harmonics_voltage[i, h - 1] = (np.transpose(delta_har_vol)[i])

            for i in range(0, np.shape(harmonics_voltage)[0]):
                harmonics_voltage[i, h - 1] = abs(u_har_0 + delta_harmonics_voltage[i, h - 1])

    return u_harmonics_0, harmonics_voltage

