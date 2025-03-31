import math
import cmath
import numpy as np
import pandapower as pp
import pandapower.harmonics.harmonic_impedance_creator as hic
#import pandapower.harmonics.unbalanced.voltages_currents_calculation as har_vol_calc


# formatting harmonic voltages in table and calculation of THD
# Nodes need to be sorted 0, 1, 2, 3, 4... Node with index 0 needs to be referent node (External grid is connected to this node)

def unbalanced_thd_voltage(network, harmonics, har_a, har_a_angle, har_b, har_b_angle, har_c, har_c_angle,
                           har_a_lc, har_a_lc_angle, har_b_lc, har_b_lc_angle, har_c_lc, har_c_lc_angle,
                           analysis_type):
    harmonics_voltage_0, harmonics_voltage, harmonics_current_0, harmonics_current = \
        unbalanced_harmonic_current_voltage(network, harmonics,
                                            har_a, har_a_angle, har_b, har_b_angle, har_c, har_c_angle,
                                            har_a_lc, har_a_lc_angle, har_b_lc, har_b_lc_angle, har_c_lc,
                                            har_c_lc_angle,
                                            analysis_type)

    thd_a = []
    thd_b = []
    thd_c = []

    for i in range(0, np.shape(harmonics_voltage)[0], 3):
        sum_thd = 0
        for j in range(0, np.shape(harmonics_voltage)[1]):
            sum_thd += harmonics_voltage[i, j] ** 2
        thd_a.append(sum_thd)

    for i in range(1, np.shape(harmonics_voltage)[0], 3):
        sum_thd = 0
        for j in range(0, np.shape(harmonics_voltage)[1]):
            sum_thd += harmonics_voltage[i, j] ** 2
        thd_b.append(sum_thd)

    for i in range(2, np.shape(harmonics_voltage)[0], 3):
        sum_thd = 0
        for j in range(0, np.shape(harmonics_voltage)[1]):
            sum_thd += harmonics_voltage[i, j] ** 2
        thd_c.append(sum_thd)

    for i in range(0, len(thd_a)):
        thd_a[i] = math.sqrt(thd_a[i]) / (network.res_bus_3ph.vm_a_pu[i + 1]) * 100
        thd_b[i] = math.sqrt(thd_b[i]) / (network.res_bus_3ph.vm_b_pu[i + 1]) * 100
        thd_c[i] = math.sqrt(thd_c[i]) / (network.res_bus_3ph.vm_c_pu[i + 1]) * 100

    sum_thd_a_0 = 0
    sum_thd_b_0 = 0
    sum_thd_c_0 = 0

    for i in range(0, np.shape(harmonics_voltage_0)[1]):
        sum_thd_a_0 += harmonics_voltage_0[0, i] ** 2
        sum_thd_b_0 += harmonics_voltage_0[1, i] ** 2
        sum_thd_c_0 += harmonics_voltage_0[2, i] ** 2

    thd_a.insert(0, math.sqrt(sum_thd_a_0) / network.res_bus_3ph.vm_a_pu[0] * 100)
    thd_b.insert(0, math.sqrt(sum_thd_b_0) / network.res_bus_3ph.vm_b_pu[0] * 100)
    thd_c.insert(0, math.sqrt(sum_thd_c_0) / network.res_bus_3ph.vm_c_pu[0] * 100)

    harmonics_voltage_a = np.zeros([int(len(network.bus.name) * 3 / 3), len(har_a)], dtype=float)
    harmonics_voltage_b = np.zeros([int(len(network.bus.name) * 3 / 3), len(har_a)], dtype=float)
    harmonics_voltage_c = np.zeros([int(len(network.bus.name) * 3 / 3), len(har_a)], dtype=float)
    # Harmonic voltage (percentage) of each phase and each harmoni
    for i in range(0, np.shape(harmonics_voltage_a)[0] - 1):
        for j in range(0, np.shape(harmonics_voltage)[1]):
            harmonics_voltage_a[i + 1, j] = harmonics_voltage[3 * i, j] * 100 * (1 / network.res_bus_3ph.vm_a_pu[i + 1])
            harmonics_voltage_b[i + 1, j] = harmonics_voltage[3 * i + 1, j] * 100 * (
                        1 / network.res_bus_3ph.vm_b_pu[i + 1])
            harmonics_voltage_c[i + 1, j] = harmonics_voltage[3 * i + 2, j] * 100 * (
                        1 / network.res_bus_3ph.vm_c_pu[i + 1])

    for i in range(0, np.shape(harmonics_voltage_0)[1]):
        harmonics_voltage_a[0, i] = harmonics_voltage_0[0, i] * 100 * (1 / network.res_bus_3ph.vm_a_pu[0])
        harmonics_voltage_b[0, i] = harmonics_voltage_0[1, i] * 100 * (1 / network.res_bus_3ph.vm_b_pu[0])
        harmonics_voltage_c[0, i] = harmonics_voltage_0[2, i] * 100 * (1 / network.res_bus_3ph.vm_c_pu[0])

    return thd_a, thd_b, thd_c, harmonics_voltage_a, harmonics_voltage_b, harmonics_voltage_c


def unbalanced_harmonic_current_voltage(network, harmonics, har_a, har_a_angle, har_b, har_b_angle, har_c, har_c_angle, \
                                        har_a_lc, har_a_lc_angle, har_b_lc, har_b_lc_angle, har_c_lc, har_c_lc_angle, \
                                        analysis_type):
    delta_harmonics_voltage = np.zeros([len(network.bus.name) * 3 - 3, len(har_a)], dtype=complex)
    harmonics_voltage = np.zeros([len(network.bus.name) * 3 - 3, len(har_a)], dtype=float)
    harmonics_current = np.zeros([len(network.bus.name) * 3 - 3, len(har_a)], dtype=float)
    u_harmonics_0 = np.zeros([3, len(har_a)], dtype=complex)
    i_harmonics_0 = np.zeros([3, len(har_a)], dtype=float)

    har_matrices, har_ext_matrix = hic.harmonic_imp_creator(network, harmonics, analysis_type)

    s_base = network.sn_mva * 1e6

    for h in range(0, len(harmonics)):
        phase_mat_z = har_matrices[h]
        z_ext_abc = har_ext_matrix[h]

        if harmonics[h] == 1:

            pp.runpp_3ph(network)

            current = []
            current_res = []
            current_pv = []
            current_0 = []

            # p.u. values
            s_a = complex(network.res_bus_3ph.p_a_mw[network.ext_grid.bus[0]] * 1000000 / s_base,
                          network.res_bus_3ph.q_a_mvar[network.ext_grid.bus[0]] * 1000000 / s_base)
            s_b = complex(network.res_bus_3ph.p_b_mw[network.ext_grid.bus[0]] * 1000000 / s_base,
                          network.res_bus_3ph.q_b_mvar[network.ext_grid.bus[0]] * 1000000 / s_base)
            s_c = complex(network.res_bus_3ph.p_c_mw[network.ext_grid.bus[0]] * 1000000 / s_base,
                          network.res_bus_3ph.q_c_mvar[network.ext_grid.bus[0]] * 1000000 / s_base)

            current_0.append(np.conjugate(s_a / (cmath.rect(network.res_bus_3ph.vm_a_pu[network.ext_grid.bus[0]],
                                                            network.res_bus_3ph.va_a_degree[
                                                                network.ext_grid.bus[0]] * cmath.pi / 180))))
            current_0.append(np.conjugate(s_b / (cmath.rect(network.res_bus_3ph.vm_b_pu[network.ext_grid.bus[0]],
                                                            network.res_bus_3ph.va_b_degree[
                                                                network.ext_grid.bus[0]] * cmath.pi / 180))))
            current_0.append(np.conjugate(s_c / (cmath.rect(network.res_bus_3ph.vm_c_pu[network.ext_grid.bus[0]],
                                                            network.res_bus_3ph.va_c_degree[
                                                                network.ext_grid.bus[0]] * cmath.pi / 180))))

            for i in network.res_bus_3ph.index:
                connected = 0

                if i != network.ext_grid.bus[0]:
                    for j in range(0, len(network.asymmetric_load.index), 2):
                        # The assumption is that two harmonic sources are connected to the node
                        # It can be meodified, additional sources can be added, or number of sources at every node can be reduced
                        if network.asymmetric_load.bus[j] == i:
                            connected = 1  # an asymmetric load is connected to the observed node

                            ##Residential
                            s_a_res = complex(network.asymmetric_load.p_a_mw[j] * 1000000 / s_base,
                                              network.asymmetric_load.q_a_mvar[j] * 1000000 / s_base)
                            s_b_res = complex(network.asymmetric_load.p_b_mw[j] * 1000000 / s_base,
                                              network.asymmetric_load.q_b_mvar[j] * 1000000 / s_base)
                            s_c_res = complex(network.asymmetric_load.p_c_mw[j] * 1000000 / s_base,
                                              network.asymmetric_load.q_c_mvar[j] * 1000000 / s_base)

                            # PV
                            s_a_pv = complex(network.asymmetric_load.p_a_mw[j + 1] * 1000000 / s_base,
                                             network.asymmetric_load.q_a_mvar[j + 1] * 1000000 / s_base)
                            s_b_pv = complex(network.asymmetric_load.p_b_mw[j + 1] * 1000000 / s_base,
                                             network.asymmetric_load.q_b_mvar[j + 1] * 1000000 / s_base)
                            s_c_pv = complex(network.asymmetric_load.p_c_mw[j + 1] * 1000000 / s_base,
                                             network.asymmetric_load.q_c_mvar[j + 1] * 1000000 / s_base)

                            break

                    if connected == 1:
                        # If the node is a node where a load/generator is connected we calculate the currents
                        # Else currents at that node are 0
                        current_res.append(np.conjugate(s_a_res / (cmath.rect(network.res_bus_3ph.vm_a_pu[i],
                                                                              network.res_bus_3ph.va_a_degree[
                                                                                  i] * cmath.pi / 180))))
                        current_res.append(np.conjugate(s_b_res / (cmath.rect(network.res_bus_3ph.vm_b_pu[i],
                                                                              network.res_bus_3ph.va_b_degree[
                                                                                  i] * cmath.pi / 180))))
                        current_res.append(np.conjugate(s_c_res / (cmath.rect(network.res_bus_3ph.vm_c_pu[i],
                                                                              network.res_bus_3ph.va_c_degree[
                                                                                  i] * cmath.pi / 180))))

                        current_pv.append(np.conjugate(s_a_pv / (cmath.rect(network.res_bus_3ph.vm_a_pu[i],
                                                                            network.res_bus_3ph.va_a_degree[
                                                                                i] * cmath.pi / 180))))
                        current_pv.append(np.conjugate(s_b_pv / (cmath.rect(network.res_bus_3ph.vm_b_pu[i],
                                                                            network.res_bus_3ph.va_b_degree[
                                                                                i] * cmath.pi / 180))))
                        current_pv.append(np.conjugate(s_c_pv / (cmath.rect(network.res_bus_3ph.vm_c_pu[i],
                                                                            network.res_bus_3ph.va_c_degree[
                                                                                i] * cmath.pi / 180))))

                    else:
                        current_res.append(0 + 0j)
                        current_res.append(0 + 0j)
                        current_res.append(0 + 0j)

                        current_pv.append(0 + 0j)
                        current_pv.append(0 + 0j)
                        current_pv.append(0 + 0j)

                    # p.u. values of current in each node. Must be equal to s_res + s_pv
                    s_a = complex(network.res_bus_3ph.p_a_mw[i] * 1000000 / s_base,
                                  network.res_bus_3ph.q_a_mvar[i] * 1000000 / s_base)
                    s_b = complex(network.res_bus_3ph.p_b_mw[i] * 1000000 / s_base,
                                  network.res_bus_3ph.q_b_mvar[i] * 1000000 / s_base)
                    s_c = complex(network.res_bus_3ph.p_c_mw[i] * 1000000 / s_base,
                                  network.res_bus_3ph.q_c_mvar[i] * 1000000 / s_base)

                    current.append(np.conjugate(s_a / (cmath.rect(network.res_bus_3ph.vm_a_pu[i],
                                                                  network.res_bus_3ph.va_a_degree[
                                                                      i] * cmath.pi / 180))))
                    current.append(np.conjugate(s_b / (cmath.rect(network.res_bus_3ph.vm_b_pu[i],
                                                                  network.res_bus_3ph.va_b_degree[
                                                                      i] * cmath.pi / 180))))
                    current.append(np.conjugate(s_c / (cmath.rect(network.res_bus_3ph.vm_c_pu[i],
                                                                  network.res_bus_3ph.va_c_degree[
                                                                      i] * cmath.pi / 180))))

        else:
            harmonics_current_res = []
            harmonics_angle_res = []

            harmonics_current_pv = []
            harmonics_angle_pv = []

            for i in range(0, len(network.bus.name)):
                # Residential household devices harmonics
                harmonics_current_res.append(har_a[h - 1])
                harmonics_current_res.append(har_b[h - 1])
                harmonics_current_res.append(har_c[h - 1])

                # PV harmonics
                harmonics_current_pv.append(har_a_lc[h - 1])
                harmonics_current_pv.append(har_b_lc[h - 1])
                harmonics_current_pv.append(har_c_lc[h - 1])

                # Harmonic angles are defined. The harmonic angles are not same for 3rd, 5th, 7th etc.

                if harmonics[h] % 3 == 0:
                    harmonics_angle_res.append(harmonics[h] * har_a_angle[h - 1])
                    harmonics_angle_res.append(harmonics[h] * har_b_angle[h - 1])
                    harmonics_angle_res.append(harmonics[h] * har_c_angle[h - 1])

                    harmonics_angle_pv.append(harmonics[h] * har_a_lc_angle[h - 1])
                    harmonics_angle_pv.append(harmonics[h] * har_b_lc_angle[h - 1])
                    harmonics_angle_pv.append(harmonics[h] * har_c_lc_angle[h - 1])
                elif harmonics[h] % 3 == 1:
                    harmonics_angle_res.append(harmonics[h] * har_a_angle[h - 1])
                    harmonics_angle_res.append(harmonics[h] * har_b_angle[h - 1] + 240)
                    harmonics_angle_res.append(harmonics[h] * har_c_angle[h - 1] + 120)

                    harmonics_angle_pv.append(harmonics[h] * har_a_lc_angle[h - 1])
                    harmonics_angle_pv.append(harmonics[h] * har_b_lc_angle[h - 1] + 240)
                    harmonics_angle_pv.append(harmonics[h] * har_c_lc_angle[h - 1] + 120)
                elif harmonics[h] % 3 == 2:
                    harmonics_angle_res.append(harmonics[h] * har_a_angle[h - 1])
                    harmonics_angle_res.append(harmonics[h] * har_b_angle[h - 1] + 120)
                    harmonics_angle_res.append(harmonics[h] * har_c_angle[h - 1] + 240)

                    harmonics_angle_pv.append(harmonics[h] * har_a_lc_angle[h - 1])
                    harmonics_angle_pv.append(harmonics[h] * har_b_lc_angle[h - 1] + 120)
                    harmonics_angle_pv.append(harmonics[h] * har_c_lc_angle[h - 1] + 240)

            har_cur_pv = []
            har_cur_res = []
            har_cur = []

            for i in range(0, len(current)):
                # Only the absolute value of the fundamnetal harmonic is taken
                # since in the most researches and measurments the angle of the fundamental harmonic is taken to be zero
                # We caclulate harmonic currents of hosuehold devices, PVs, and we create harmonic current at the node
                # as the sum of currents of all harmonic sources at the node
                har_cur_res.append(-abs(current_res[i]) * cmath.rect(harmonics_current_res[i] / 100,
                                                                     harmonics_angle_res[i] * cmath.pi / 180))
                har_cur_pv.append(-abs(current_pv[i]) * cmath.rect(harmonics_current_pv[i] / 100,
                                                                   harmonics_angle_pv[i] * cmath.pi / 180))
                har_cur.append(har_cur_pv[i] + har_cur_res[i])

            sum_a = 0
            sum_b = 0
            sum_c = 0

            # At the first node, current is equal to the sum of harmonic currents at all other nodes
            for a in range(0, len(har_cur), 3):
                sum_a += har_cur[a]
                sum_b += har_cur[a + 1]
                sum_c += har_cur[a + 2]

            har_cur_0 = []

            har_cur_0.append(sum_a)
            har_cur_0.append(sum_b)
            har_cur_0.append(sum_c)

            # Calculation of the harmonic current of the first node
            u_har_0 = np.matmul(z_ext_abc, har_cur_0)

            for i in range(0, np.shape(u_har_0)[1]):
                u_harmonics_0[i, h - 1] = u_har_0[0, i]
                i_harmonics_0[i, h - 1] = abs(har_cur_0[i])

            # Calculating the harmonics voltage drop, the difference between the harmonic of slack and all other nodes
            delta_har_vol = np.matmul(phase_mat_z, har_cur)

            for i in range(0, np.shape(delta_harmonics_voltage)[0]):
                delta_harmonics_voltage[i, h - 1] = delta_har_vol[i]
                harmonics_current[i, h - 1] = abs(har_cur[i])

            # Calculating harmonic voltage from the harmonic of a slack node and differences
            for i in range(0, np.shape(delta_harmonics_voltage)[0], 3):
                harmonics_voltage[i, h - 1] = abs(u_harmonics_0[0, h - 1] + delta_harmonics_voltage[i, h - 1])
                harmonics_voltage[i + 1, h - 1] = abs(u_harmonics_0[1, h - 1] + delta_harmonics_voltage[i + 1, h - 1])
                harmonics_voltage[i + 2, h - 1] = abs(u_harmonics_0[2, h - 1] + delta_harmonics_voltage[i + 2, h - 1])

    harmonics_voltage_0 = abs(u_harmonics_0)

    return harmonics_voltage_0, harmonics_voltage, i_harmonics_0, harmonics_current