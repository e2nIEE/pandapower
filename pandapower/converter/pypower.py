from math import pi, nan
from numpy import sign, append, zeros, max, array
from pandas import Series, DataFrame
 
from pandapower.runpf import _runpf

import pandapower as pp
from pandapower.runpp import _pd2ppc, _select_is_elements

try:
    import pplog as log
except:
    import logging as log
    
logger = log.getLogger(__name__)

def from_ppc(ppc, f_hz=50):
    """
    This function converts pypower case files to pandapower net structure.

    INPUT:

        **ppc** - The pypower case file.

    OPTIONAL:

        **f_hz** - The frequency of the network.

    OUTPUT:

        **net**

    EXAMPLE:

        import converter as cv

        net = cv.from_ppc(ppc)
    """
    # --- catch common failures
    if Series(ppc['bus'][:, 9] <= 0).any():
        logger.error('There are false baseKV given in the pypower case file.')

    # --- general_parameters
    baseMVA = ppc['baseMVA']  # MVA
    omega = pi * f_hz  # 1/s

    net = pp.create_empty_network(f_hz=f_hz)

    # --- bus data -> create buses, sgen, load, shunt
    for i in range(len(ppc['bus'])):
        # create buses
        pp.create_bus(net, name=int(ppc['bus'][i, 0]), vn_kv=ppc['bus'][i, 9], type="b",
                      zone=ppc['bus'][i, 6], in_service=bool(ppc['bus'][i, 1] != 4),
                      max_vm_pu=ppc['bus'][i, 11], min_vm_pu=ppc['bus'][i, 12])
        # create sgen, load
        if ppc['bus'][i, 2] > 0:
            pp.create_load(net, i, p_kw=ppc['bus'][i, 2]*1e3, q_kvar=ppc['bus'][i, 3]*1e3)
        elif ppc['bus'][i, 2] < 0:
            pp.create_sgen(net, i, p_kw=ppc['bus'][i, 2]*1e3, q_kvar=ppc['bus'][i, 3]*1e3,
                           type="")
        elif ppc['bus'][i, 3] != 0:
            pp.create_load(net, i, p_kw=ppc['bus'][i, 2]*1e3, q_kvar=ppc['bus'][i, 3]*1e3)
        # create shunt
        if ppc['bus'][i, 4] != 0 or ppc['bus'][i, 5] != 0:
            pp.create_shunt(net, i, p_kw=-ppc['bus'][i, 4]*1e3,
                            q_kvar=-ppc['bus'][i, 5]*1e3)
    # unused data: Vm, Va (partwise: in ext_grid), zone

    # --- gen data -> create ext_grid, gen, sgen
    # prepare gen data -> no duplicates
    GEN = DataFrame(ppc['gen'])
    GEN_uniq = GEN.drop_duplicates(subset=[0])
    dupl = GEN[0].duplicated()
    GEN_dupl = GEN[dupl]
    if len(GEN_dupl) > 0:
        logger.debug('There are several generators at one bus.')
    for i in GEN_dupl.index:
        GEN_bus = int(GEN_dupl[0][i])
        current_bus_idx = pp.get_element_index(net, 'bus', name=GEN_bus)
        current_bus_type = int(ppc['bus'][current_bus_idx, 1])
        # check different vm_pu values for gen at the same bus
        if GEN_dupl[5][i] != GEN_uniq[GEN_uniq[0] == GEN_bus][5].values[0]:
            logger.error('Several generators at one bus have different vm_pu values.')
        # set in_service
        if (GEN[GEN[0] == GEN_bus][7] > 0).any():
            GEN_uniq.loc[GEN_uniq[GEN_uniq[0] == GEN_bus].index, 7] = 1
        # sum up active powers and power limits as well as reactive power limits
        for j in [1, 3, 4, 8, 9]:
            GEN_uniq.loc[GEN_uniq[GEN_uniq[0] == GEN_bus].index, j] = \
                GEN[(GEN[0] == GEN_bus) & (GEN[7] > 0)][j].sum()

    for i in GEN_uniq.index:
        GEN_bus = int(GEN_uniq[0][i])
        current_bus_idx = pp.get_element_index(net, 'bus', name=GEN_bus)
        current_bus_type = int(ppc['bus'][current_bus_idx, 1])
        # create ext_grid
        if current_bus_type == 3:
            pp.create_ext_grid(net, bus=current_bus_idx, vm_pu=GEN_uniq[5][i],
                               va_degree=ppc['bus'][current_bus_idx, 8],
                               in_service=bool(GEN_uniq[7][i] > 0))
        # create gen
        elif current_bus_type == 2:
            pp.create_gen(net, bus=current_bus_idx, vm_pu=GEN_uniq[5][i],
                          p_kw=-GEN_uniq[1][i]*1e3, in_service=bool(GEN_uniq[7][i] > 0),
                          max_p_kw=-GEN_uniq[8][i]*1e3, min_p_kw=-GEN_uniq[9][i]*1e3,
                          max_q_kvar=GEN_uniq[3][i]*1e3,
                          min_q_kvar=GEN_uniq[4][i]*1e3, controllable=True)
            if GEN_uniq[4][i] > GEN_uniq[3][i]:
                logger.warning('min_q_kvar must be less than max_q_kvar.')
            if -GEN_uniq[9][i] < -GEN_uniq[8][i]:
                logger.warning('max_p_kw must be less than min_p_kw.')
        # create sgen
        elif current_bus_type == 1:
            pp.create_sgen(net, bus=current_bus_idx, p_kw=-GEN_uniq[1][i]*1e3,
                           q_kvar=-GEN_uniq[2][i]*1e3, type="",
                           in_service=bool(GEN_uniq[7][i] > 0),
                           max_p_kw=-GEN_uniq[8][i]*1e3, min_p_kw=-GEN_uniq[9][i]*1e3,
                           max_q_kvar=GEN_uniq[3][i]*1e3,
                           min_q_kvar=GEN_uniq[4][i]*1e3, controllable=True)
    # unused data: Vg (partwise: in ext_grid and gen), mBase, Pc1, Pc2, Qc1min, Qc1max, Qc2min,
    # Qc2max, ramp_agc, ramp_10, ramp_30,ramp_q, apf

    # --- branch data -> create line, trafo
    for i in range(len(ppc['branch'])):
        if ppc['branch'][i, 8] == 0:
            from_bus = pp.get_element_index(net, 'bus', name=int(ppc['branch'][i, 0]))
            to_bus = pp.get_element_index(net, 'bus', name=int(ppc['branch'][i, 1]))
            Zni = ppc['bus'][to_bus, 9]**2/baseMVA  # ohm

            pp.create_line_from_parameters(
                net, from_bus=from_bus, to_bus=to_bus, length_km=1,
                r_ohm_per_km=ppc['branch'][i, 2]*Zni, x_ohm_per_km=ppc['branch'][i, 3]*Zni,
                c_nf_per_km=ppc['branch'][i, 4]/Zni/omega*1e9/2,
                imax_ka=ppc['branch'][i, 5]/ppc['bus'][to_bus, 9], type='ol',
                in_service=bool(ppc['branch'][i, 10]))
        else:
            from_bus = pp.get_element_index(net, 'bus', name=int(ppc['branch'][i, 0]))
            to_bus = pp.get_element_index(net, 'bus', name=int(ppc['branch'][i, 1]))
            from_vn_kv = ppc['bus'][from_bus, 9]
            to_vn_kv = ppc['bus'][to_bus, 9]
            if from_vn_kv >= to_vn_kv:
                hv_bus = from_bus
                vn_hv_kv = from_vn_kv
                lv_bus = to_bus
                vn_lv_kv = to_vn_kv
                if from_vn_kv == to_vn_kv:
                    logger.debug('A transformer voltage is on both side the same.')
            else:
                hv_bus = to_bus
                vn_hv_kv = to_vn_kv
                lv_bus = from_bus
                vn_lv_kv = from_vn_kv
            rk = ppc['branch'][i, 2]
            xk = ppc['branch'][i, 3]
            zk = (rk**2+xk**2)**0.5
            sn = ppc['branch'][i, 5]*1e3
            ratio_1 = ppc['branch'][i, 8] - 1
            i0_percent = ppc['branch'][i, 4]*100*baseMVA*1e3/sn
            if i0_percent > 0:
                logger.warning('The transformer always behaves inductive but the susceptance is '
                               'positive')

            pp.create_transformer_from_parameters(
                net, hv_bus=hv_bus, lv_bus=lv_bus, sn_kva=sn, vn_hv_kv=vn_hv_kv, vn_lv_kv=vn_lv_kv,
                vsc_percent=zk*sn/1e3, vscr_percent=rk*sn/1e3, pfe_kw=0,
                i0_percent=i0_percent, shift_degree=ppc['branch'][i, 9],
                tp_side="hv" if ratio_1 else nan, tp_st_percent=abs(ratio_1) if ratio_1 else nan,
                tp_pos=sign(ratio_1) if ratio_1 else nan)
    # unused data: rateB, rateC

    # gencost is currently unconverted

    return net

def to_ppc(net, trafo_model="t", calculate_voltage_angles=True):
    is_elems = _select_is_elements(net, None)
    ppc, ppci, bus_lookup = _pd2ppc(net, is_elems, calculate_voltage_angles, True,
                                       trafo_model=trafo_model, init_results=False)
    return ppc

def validate_ppc_conversion(net, ppc, trafo_model="t", calculate_voltage_angles=True):
    """
    This function validates the pypower case files to pandapower net structure conversion via a \
    comparison of loadflow calculations.

    INPUT:

        **ppc** - The pypower case file.

        **net** - The pandapower network.

    EXAMPLE:

        import converter as cv

        from pypower import case4gs

        ppc = case4gs.case4gs()

        net = cv.from_ppc(ppc, f_hz=60)

        cv.validate_ppc_conversion(ppc, net)
    """
    ppc_res, sucesss = _runpf(ppc, recycle=dict(is_elems=False, ppc=False, Ybus=False))

    ppc_res_branch = ppc_res['branch'][:, 13:17]
    ppc_res_bus = ppc_res['bus'][:, 7:9]
    ppc_res_gen = ppc_res['gen'][:, 1:3]

    try:
        pp.runpp(net, init="dc", calculate_voltage_angles=calculate_voltage_angles, 
                 trafo_model=trafo_model)
    except:
        if ppc_res['success'] == 1 and ~net.converged:
            logger.info('The validation of ppc conversion fails because the pandapower net power '
                        'flow do not convert.')
        elif ppc_res['success'] != 1 and net.converged:
            logger.info('The validation of ppc conversion fails because the power flow of the '
                        'pypower case do not convert.')
        elif ppc_res['success'] != 1 and ~net.converged:
            logger.info('The power flow of both, the pypower case and the pandapower net, do not '
                        'convert.')
    if ppc_res['success'] == 1 and net.converged:
        pp_res_bus = array(net.res_bus[['vm_pu', 'va_degree']])

        pp_res_gen = zeros([1, 2])
        for i in range(len(ppc_res['gen'])):
            current_bus_idx = pp.get_element_index(net, 'bus', name=int(ppc_res['gen'][i, 0]))
            current_bus_type = int(ppc_res['bus'][current_bus_idx, 1])
            # create ext_grid
            if current_bus_type == 3:
                pp_res_gen = append(pp_res_gen, array(net.res_ext_grid[
                    net.ext_grid.bus == current_bus_idx][['p_kw', 'q_kvar']]), 0)
            # create gen
            elif current_bus_type == 2:
                pp_res_gen = append(pp_res_gen, array(net.res_gen[
                    net.gen.bus == current_bus_idx][['p_kw', 'q_kvar']]), 0)
            # create sgen
            elif current_bus_type == 1:
                pp_res_gen = append(pp_res_gen, array(net.res_sgen[
                    net.sgen.bus == current_bus_idx][['p_kw', 'q_kvar']]), 0)
        pp_res_gen = pp_res_gen[1:, :]

        pp_res_branch = zeros([1, 4])
        for i in range(len(ppc_res['branch'])):
            if ppc['branch'][i, 8] == 0:
                from_bus = pp.get_element_index(net, 'bus', name=int(ppc['branch'][i, 0]))
                to_bus = pp.get_element_index(net, 'bus', name=int(ppc['branch'][i, 1]))
                pp_res_branch = append(pp_res_branch, array(net.res_line[
                    (net.line.from_bus == from_bus) & (net.line.to_bus == to_bus)]
                        [['p_from_kw', 'q_from_kvar', 'p_to_kw', 'q_to_kvar']]), 0)
            else:
                hv_bus = pp.get_element_index(net, 'bus', name=int(ppc['branch'][i, 0]))
                lv_bus = pp.get_element_index(net, 'bus', name=int(ppc['branch'][i, 1]))
                pp_res_branch = append(pp_res_branch, array(net.res_trafo[
                    (net.trafo.hv_bus == hv_bus) & (net.trafo.lv_bus == lv_bus)]
                        [['p_hv_kw', 'q_hv_kvar', 'p_lv_kw', 'q_lv_kvar']]), 0)
        pp_res_branch = pp_res_branch[1:, :]

        if not ppc_res['success'] == net.converged:
            logger.info('Pypower powerflow or pandapower powerflow converges whereas the other one '
                        'does not.')
        diff_res_bus = ppc_res_bus - pp_res_bus
        logger.info("Maximum voltage magnitude difference between pypower and pandapower: "
                    "%.3f pu" % max(abs(diff_res_bus[:, 0])))
        logger.info("Maximum voltage angle difference between pypower and pandapower: "
                    "%.3f degree" % max(abs(diff_res_bus[:, 1])))
        diff_res_branch = ppc_res_branch - pp_res_branch*1e-3
        logger.info("Maximum branch flow active power difference between pypower and pandapower: "
                    "%.3f MW" % max(abs(diff_res_branch[:, [0, 2]])))
        logger.info("Maximum branch flow reactive power difference between pypower and pandapower: "
                    "%.3f MVAr" % max(abs(diff_res_branch[:, [1, 3]])))
        diff_res_gen = ppc_res_gen + pp_res_gen*1e-3
        logger.info("Maximum active power generation difference between pypower and pandapower: "
                    "%.3f MW" % max(abs(diff_res_gen[:, 0])))
        logger.info("Maximum reactive power generation difference between pypower and pandapower: "
                    "%.3f MVAr" % max(abs(diff_res_gen[:, 1])))
        if (max(abs(diff_res_bus[:, 0])) < 1e-3) & (max(abs(diff_res_bus[:, 1])) < 1e-3) & \
            (max(abs(diff_res_branch[:, [0, 2]])) < 1e-3) & \
            (max(abs(diff_res_branch[:, [1, 3]])) < 1e-3) & \
                (ppc_res_gen + pp_res_gen*1e-3 < 1e1).all():
                logger.info("The active/reactive power generation difference possibly is because of"
                            " a pypower fault. If you have an access, please validate the results "
                            "via matpower loadflow.")

if __name__ == "__main__":
    import pandapower.networks as nw
    net = nw.mv_oberrhein()
    ppc = to_ppc(net, calculate_voltage_angles=False)
    validate_ppc_conversion(net, ppc)