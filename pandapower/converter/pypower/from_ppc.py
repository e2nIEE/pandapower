# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from math import pi
from numpy import sign, nan, append, zeros, array, power, sqrt
from numpy import max as max_
from pandas import Series, DataFrame, concat

import pandapower as pp

try:
    import pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)

try:
    from pypower import ppoption, runpf
    ppopt = ppoption.ppoption(VERBOSE=0, OUT_ALL=0)
    pypower_import = True
except ImportError:
    pypower_import = False


def _create_costs(net, ppc, gen_lookup, type, idx):
    if ppc['gencost'][idx, 0] == 1:
        if not len(ppc['gencost'][idx, 4:]) == 2*ppc['gencost'][idx, 3]:
            logger.error("In gencost line %s, the number n does not fit to the number of values" %
                         idx)
        pp.create_piecewise_linear_cost(net, gen_lookup.element.at[idx],
                                        gen_lookup.element_type.at[idx],
                                        ppc['gencost'][idx, 4:], type)
    elif ppc['gencost'][idx, 0] == 2:
        if len(ppc['gencost'][idx, 4:]) == ppc['gencost'][idx, 3]:
            n = len(ppc['gencost'][idx, 4:])
            values = ppc['gencost'][idx, 4:] / power(1e3, array(range(n))[::-1])
        else:
            logger.error("In gencost line %s, the number n does not fit to the number of values" %
                         idx)
        pp.create_polynomial_cost(net, gen_lookup.element.at[idx], gen_lookup.element_type.at[idx],
                                  values, type)
    else:
        logger.info("Cost mode of gencost line %s is unknown." % idx)


def from_ppc(ppc, f_hz=50, validate_conversion=False):
    """
    This function converts pypower case files to pandapower net structure.

    INPUT:

        **ppc** : The pypower case file.

    OPTIONAL:

        **f_hz** (float, 50) - The frequency of the network.

        **validate_conversion** (bool, False) - If True, validate_from_ppc is run after conversion.
            For running the validation, the ppc must already contain the pypower
            powerflow results or pypower must be importable.

    OUTPUT:

        **net** : pandapower net.

    EXAMPLE:

        import pandapower.converter as pc

        from pypower import case4gs

        ppc_net = case4gs.case4gs()

        pp_net = pc.from_ppc(ppc_net, f_hz=60)

    """
    # --- catch common failures
    if Series(ppc['bus'][:, 9] <= 0).any():
        logger.info('There are false baseKV given in the pypower case file.')

    # --- general_parameters
    baseMVA = ppc['baseMVA']  # MVA
    omega = pi * f_hz  # 1/s
    MAX_VAL = 99999.

    net = pp.create_empty_network(f_hz=f_hz, sn_kva=baseMVA*1e3)

    # --- bus data -> create buses, sgen, load, shunt
    for i in range(len(ppc['bus'])):
        # create buses
        pp.create_bus(net, name=int(ppc['bus'][i, 0]), vn_kv=ppc['bus'][i, 9], type="b",
                      zone=ppc['bus'][i, 6], in_service=bool(ppc['bus'][i, 1] != 4),
                      max_vm_pu=ppc['bus'][i, 11], min_vm_pu=ppc['bus'][i, 12])
        # create sgen, load
        if ppc['bus'][i, 2] > 0:
            pp.create_load(net, i, p_kw=ppc['bus'][i, 2] * 1e3, q_kvar=ppc['bus'][i, 3] * 1e3,
                           controllable=False)
        elif ppc['bus'][i, 2] < 0:
            pp.create_sgen(net, i, p_kw=ppc['bus'][i, 2] * 1e3, q_kvar=ppc['bus'][i, 3] * 1e3,
                           type="", controllable=False)
        elif ppc['bus'][i, 3] != 0:
            pp.create_load(net, i, p_kw=ppc['bus'][i, 2] * 1e3, q_kvar=ppc['bus'][i, 3] * 1e3,
                           controllable=False)
        # create shunt
        if ppc['bus'][i, 4] != 0 or ppc['bus'][i, 5] != 0:
            pp.create_shunt(net, i, p_kw=ppc['bus'][i, 4] * 1e3,
                            q_kvar=-ppc['bus'][i, 5] * 1e3)
    # unused data of ppc: Vm, Va (partwise: in ext_grid), zone

    # --- gen data -> create ext_grid, gen, sgen
    gen_lookup = DataFrame(nan, columns=['element', 'element_type'],
                           index=range(len(ppc['gen'][:, 0])))
    for i in range(len(ppc['gen'])):
        # if in ppc is only one gen -> numpy initially uses one dim array -> change to two dim array
        if len(ppc["gen"].shape) == 1:
            ppc["gen"] = array(ppc["gen"], ndmin=2)
        current_bus_idx = pp.get_element_index(net, 'bus', name=int(ppc['gen'][i, 0]))
        current_bus_type = int(ppc['bus'][current_bus_idx, 1])
        # create ext_grid
        if current_bus_type == 3:
            if len(pp.get_connected_elements(net, 'ext_grid', current_bus_idx)) > 0:
                logger.info('At bus %d an ext_grid already exists. ' % current_bus_idx +
                            'Because of that generator %d ' % i +
                            'is converted not as an ext_grid but as a sgen')
                current_bus_type = 1
            else:
                gen_lookup.element.loc[i] = pp.create_ext_grid(
                    net, bus=current_bus_idx, vm_pu=ppc['gen'][i, 5],
                    va_degree=ppc['bus'][current_bus_idx, 8], in_service=bool(ppc['gen'][i, 7] > 0),
                    max_p_kw=-ppc['gen'][i, 9] * 1e3, min_p_kw=-ppc['gen'][i, 8] * 1e3,
                    max_q_kvar=ppc['gen'][i, 3] * 1e3, min_q_kvar=ppc['gen'][i, 4] * 1e3)
                gen_lookup.element_type.loc[i] = 'ext_grid'
                if ppc['gen'][i, 4] > ppc['gen'][i, 3]:
                    logger.info('min_q_kvar of gen %d must be less than max_q_kvar but is not.' % i)
                if -ppc['gen'][i, 9] < -ppc['gen'][i, 8]:
                    logger.info('max_p_kw of gen %d must be less than min_p_kw but is not.' % i)
        # create gen
        elif current_bus_type == 2:
            gen_lookup.element.loc[i] = pp.create_gen(
                net, bus=current_bus_idx, vm_pu=ppc['gen'][i, 5], p_kw=-ppc['gen'][i, 1] * 1e3,
                in_service=bool(ppc['gen'][i, 7] > 0), controllable=True,
                max_p_kw=-ppc['gen'][i, 9] * 1e3, min_p_kw=-ppc['gen'][i, 8] * 1e3,
                max_q_kvar=ppc['gen'][i, 3] * 1e3, min_q_kvar=ppc['gen'][i, 4] * 1e3)
            gen_lookup.element_type.loc[i] = 'gen'
            if ppc['gen'][i, 1] < 0:
                logger.info('p_kw of gen %d must be less than zero but is not.' % i)
            if ppc['gen'][i, 4] > ppc['gen'][i, 3]:
                logger.info('min_q_kvar of gen %d must be less than max_q_kvar but is not.' % i)
            if -ppc['gen'][i, 9] < -ppc['gen'][i, 8]:
                logger.info('max_p_kw of gen %d must be less than min_p_kw but is not.' % i)
        # create sgen
        if current_bus_type == 1:
            gen_lookup.element.loc[i] = pp.create_sgen(
                net, bus=current_bus_idx, p_kw=-ppc['gen'][i, 1] * 1e3,
                q_kvar=-ppc['gen'][i, 2] * 1e3, type="", in_service=bool(ppc['gen'][i, 7] > 0),
                max_p_kw=-ppc['gen'][i, 9] * 1e3, min_p_kw=-ppc['gen'][i, 8] * 1e3,
                max_q_kvar=ppc['gen'][i, 3] * 1e3, min_q_kvar=ppc['gen'][i, 4] * 1e3,
                controllable=True)
            gen_lookup.element_type.loc[i] = 'sgen'
            if ppc['gen'][i, 1] < 0:
                logger.info('p_kw of sgen %d must be less than zero but is not.' % i)
            if ppc['gen'][i, 4] > ppc['gen'][i, 3]:
                logger.info('min_q_kvar of gen %d must be less than max_q_kvar but is not.' % i)
            if -ppc['gen'][i, 9] < -ppc['gen'][i, 8]:
                logger.info('max_p_kw of gen %d must be less than min_p_kw but is not.' % i)
    # unused data of ppc: Vg (partwise: in ext_grid and gen), mBase, Pc1, Pc2, Qc1min, Qc1max,
    # Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30,ramp_q, apf

    # --- branch data -> create line, trafo
    for i in range(len(ppc['branch'])):
        from_bus = pp.get_element_index(net, 'bus', name=int(ppc['branch'][i, 0]))
        to_bus = pp.get_element_index(net, 'bus', name=int(ppc['branch'][i, 1]))

        from_vn_kv = ppc['bus'][from_bus, 9]
        to_vn_kv = ppc['bus'][to_bus, 9]
        if (from_vn_kv == to_vn_kv) & ((ppc['branch'][i, 8] == 0) | (ppc['branch'][i, 8] == 1)) & \
           (ppc['branch'][i, 9] == 0):
            Zni = ppc['bus'][to_bus, 9]**2/baseMVA  # ohm
            max_i_ka = ppc['branch'][i, 5]/ppc['bus'][to_bus, 9]/sqrt(3)
            if max_i_ka == 0.0:
                max_i_ka = MAX_VAL
                logger.debug("ppc branch rateA is zero -> Using MAX_VAL instead to calculate " +
                             "maximum branch flow")
            pp.create_line_from_parameters(
                net, from_bus=from_bus, to_bus=to_bus, length_km=1,
                r_ohm_per_km=ppc['branch'][i, 2]*Zni, x_ohm_per_km=ppc['branch'][i, 3]*Zni,
                c_nf_per_km=ppc['branch'][i, 4]/Zni/omega*1e9/2,
                max_i_ka=max_i_ka, type='ol',
                in_service=bool(ppc['branch'][i, 10]))

        else:
            if from_vn_kv >= to_vn_kv:
                hv_bus = from_bus
                vn_hv_kv = from_vn_kv
                lv_bus = to_bus
                vn_lv_kv = to_vn_kv
                tp_side = 'hv'
            else:
                hv_bus = to_bus
                vn_hv_kv = to_vn_kv
                lv_bus = from_bus
                vn_lv_kv = from_vn_kv
                tp_side = 'lv'
                if from_vn_kv == to_vn_kv:
                    logger.warning('The pypower branch %d (from_bus, to_bus)=(%d, %d) is considered'
                                   ' as a transformer because of a ratio != 0 | 1 but it connects '
                                   'the same voltage level', i, ppc['branch'][i, 0],
                                   ppc['branch'][i, 1])
            rk = ppc['branch'][i, 2]
            xk = ppc['branch'][i, 3]
            zk = (rk ** 2 + xk ** 2) ** 0.5
            sn = ppc['branch'][i, 5] * 1e3
            if sn == 0.0:
                sn = MAX_VAL
                logger.debug("ppc branch rateA is zero -> Using MAX_VAL instead to calculate " +
                             "apparent power")
            ratio_1 = 0 if ppc['branch'][i, 8] == 0 else (ppc['branch'][i, 8] - 1) * 100
            i0_percent = -ppc['branch'][i, 4] * 100 * baseMVA * 1e3 / sn
            if i0_percent < 0:
                logger.info('A transformer always behaves inductive consumpting but the '
                            'susceptance of pypower branch %d (from_bus, to_bus)=(%d, %d) is '
                            'positive.', i, ppc['branch'][i, 0], ppc['branch'][i, 1])

            pp.create_transformer_from_parameters(
                net, hv_bus=hv_bus, lv_bus=lv_bus, sn_kva=sn, vn_hv_kv=vn_hv_kv,
                vn_lv_kv=vn_lv_kv, vsc_percent=sign(xk) * zk * sn / 1e3 * 100 / baseMVA,
                vscr_percent=rk * sn / 1e3 * 100 / baseMVA,
                pfe_kw=0, i0_percent=i0_percent, shift_degree=ppc['branch'][i, 9],
                tp_st_percent=abs(ratio_1) if ratio_1 else nan,
                tp_pos=sign(ratio_1) if ratio_1 else nan,
                tp_side=tp_side if ratio_1 else None, tp_mid=0 if ratio_1 else nan)
    # unused data of ppc: rateB, rateC

    # --- gencost -> create polynomial_cost, piecewise_cost
    if 'gencost' in ppc:
        if len(ppc['gencost'].shape) == 1:
            # reshape gencost if only one gencost is given -> no indexError
            ppc['gencost'] = ppc['gencost'].reshape((1, ppc['gencost'].shape[0]))
        if ppc['gencost'].shape[0] <= gen_lookup.shape[0]:
            idx_p = range(ppc['gencost'].shape[0])
            idx_q = []
        elif ppc['gencost'].shape[0] > gen_lookup.shape[0]:
            idx_p = range(gen_lookup.shape[0])
            idx_q = range(gen_lookup.shape[0], ppc['gencost'].shape[0])
        if ppc['gencost'].shape[0] >= 2*gen_lookup.shape[0]:
            idx_p = range(gen_lookup.shape[0])
            idx_q = range(gen_lookup.shape[0], 2*gen_lookup.shape[0])
        for idx in idx_p:
            _create_costs(net, ppc, gen_lookup, 'p', idx)
        for idx in idx_q:
            _create_costs(net, ppc, gen_lookup, 'q', idx)

    # areas are unconverted

    if validate_conversion:
        logger.setLevel(logging.DEBUG)
        if not validate_from_ppc(ppc, net):
            logger.error("Validation failed.")

    return net


def validate_from_ppc(ppc_net, pp_net, max_diff_values={
    "vm_pu": 1e-6, "va_degree": 1e-5, "p_branch_kw": 1e-3, "q_branch_kvar": 1e-3, "p_gen_kw": 1e-3,
        "q_gen_kvar": 1e-3}):
    """
    This function validates the pypower case files to pandapower net structure conversion via a \
    comparison of loadflow calculation results. (Hence the opf cost conversion is not validated.)

    INPUT:

        **ppc_net** - The pypower case file, which must already contain the pypower powerflow
            results or pypower must be importable.

        **pp_net** - The pandapower network.

    OPTIONAL:

        **max_diff_values** - Dict of maximal allowed difference values. The keys must be
        'vm_pu', 'va_degree', 'p_branch_kw', 'q_branch_kvar', 'p_gen_kw' and 'q_gen_kvar' and
        the values floats.

    OUTPUT:

        **conversion_success** - conversion_success is returned as False if pypower or pandapower
        cannot calculate a powerflow or if the maximum difference values (max_diff_values )
        cannot be hold.

    EXAMPLE:

        import pandapower.converter as pc

        pp_net = cv.from_ppc(ppc_net, f_hz=50)

        conversion_success = cv.validate_from_ppc(ppc_net, pp_net)

    NOTE:

        The user has to take care that the loadflow results already are included in the provided \
        ppc_net.
    """
    # --- check pypower powerflow success, if possible
    if pypower_import:
        try:
            ppc_net = runpf.runpf(ppc_net, ppopt)[0]
        except:
            logger.debug("The pypower runpf did not work.")
    ppc_success = True
    if 'success' in ppc_net.keys():
        if ppc_net['success'] != 1:
            ppc_success = False
            logger.error("The given ppc data indicates an unsuccessful pypower powerflow: " +
                         "'ppc_net['success'] != 1'")
    if (ppc_net['branch'].shape[1] < 17):
        ppc_success = False
        logger.error("The shape of given ppc data indicates missing pypower powerflow results.")

    # --- try to run a pandapower powerflow
    try:
        pp.runpp(pp_net, init="dc", calculate_voltage_angles=True, trafo_model="pi")
    except pp.LoadflowNotConverged:
        try:
            pp.runpp(pp_net, calculate_voltage_angles=True, init="flat", trafo_model="pi")
        except pp.LoadflowNotConverged:
            try:
                pp.runpp(pp_net, trafo_model="pi")
            except pp.LoadflowNotConverged:
                logger.error('The pandapower powerflow does not converge.')

    # --- prepare powerflow result comparison by reordering pp results as they are in ppc results
    if not ((ppc_success) & (pp_net.converged)):
        return False

    # --- store pypower powerflow results
    ppc_res_branch = ppc_net['branch'][:, 13:17]
    ppc_res_bus = ppc_net['bus'][:, 7:9]
    ppc_res_gen = ppc_net['gen'][:, 1:3]

    # --- pandapower bus result table
    pp_res_bus = array(pp_net.res_bus[['vm_pu', 'va_degree']])

    # --- pandapower gen result table
    pp_res_gen = zeros([1, 2])
    # consideration of parallel generators via storing how much generators have been considered
    # each node
    already_used_gen = Series(zeros([pp_net.bus.shape[0]]), index=pp_net.bus.index).astype(int)
    GENS = DataFrame(ppc_net['gen'][:, [0]].astype(int))
    change_q_compare = []
    for i, j in GENS.iterrows():
        current_bus_idx = pp.get_element_index(pp_net, 'bus', name=j[0])
        current_bus_type = int(ppc_net['bus'][current_bus_idx, 1])
        # ext_grid
        if current_bus_type == 3:
            if already_used_gen.at[current_bus_idx] == 0:
                pp_res_gen = append(pp_res_gen, array(pp_net.res_ext_grid[
                    pp_net.ext_grid.bus == current_bus_idx][['p_kw', 'q_kvar']])[
                    already_used_gen.at[current_bus_idx]].reshape((1, 2)), 0)
                already_used_gen.at[current_bus_idx] += 1
            else:
                pp_res_gen = append(pp_res_gen, array(pp_net.res_sgen[
                    pp_net.sgen.bus == current_bus_idx][['p_kw', 'q_kvar']])[
                    already_used_gen.at[current_bus_idx]-1].reshape((1, 2)), 0)
                already_used_gen.at[current_bus_idx] += 1
                change_q_compare += [j[0]]
        # gen
        elif current_bus_type == 2:
            pp_res_gen = append(pp_res_gen, array(pp_net.res_gen[
                pp_net.gen.bus == current_bus_idx][['p_kw', 'q_kvar']])[
                already_used_gen.at[current_bus_idx]].reshape((1, 2)), 0)
            if already_used_gen.at[current_bus_idx] > 0:
                change_q_compare += [j[0]]
            already_used_gen.at[current_bus_idx] += 1
        # sgen
        elif current_bus_type == 1:
            pp_res_gen = append(pp_res_gen, array(pp_net.res_sgen[
                pp_net.sgen.bus == current_bus_idx][['p_kw', 'q_kvar']])[
                already_used_gen.at[current_bus_idx]].reshape((1, 2)), 0)
            already_used_gen.at[current_bus_idx] += 1
    pp_res_gen = pp_res_gen[1:, :]  # delete initial zero row

    # --- pandapower branch result table
    pp_res_branch = zeros([1, 4])
    # consideration of parallel branches via storing how much branches have been considered
    # each node-to-node-connection
    init1 = concat([pp_net.line.from_bus, pp_net.line.to_bus], axis=1).drop_duplicates()
    init2 = concat([pp_net.trafo.hv_bus, pp_net.trafo.lv_bus], axis=1).drop_duplicates()
    init1['hv_bus'] = nan
    init1['lv_bus'] = nan
    init2['from_bus'] = nan
    init2['to_bus'] = nan
    already_used_branches = concat([init1, init2], axis=0)
    already_used_branches['number'] = zeros([already_used_branches.shape[0], 1]).astype(int)
    BRANCHES = DataFrame(ppc_net['branch'][:, [0, 1, 8, 9]])
    for i in BRANCHES.index:
        from_bus = pp.get_element_index(pp_net, 'bus', name=int(ppc_net['branch'][i, 0]))
        to_bus = pp.get_element_index(pp_net, 'bus', name=int(ppc_net['branch'][i, 1]))
        from_vn_kv = ppc_net['bus'][from_bus, 9]
        to_vn_kv = ppc_net['bus'][to_bus, 9]
        ratio = BRANCHES[2].at[i]
        angle = BRANCHES[3].at[i]
        # from line results
        if (from_vn_kv == to_vn_kv) & ((ratio == 0) | (ratio == 1)) & (angle == 0):
            pp_res_branch = append(pp_res_branch, array(pp_net.res_line[
                (pp_net.line.from_bus == from_bus) &
                (pp_net.line.to_bus == to_bus)]
                [['p_from_kw', 'q_from_kvar', 'p_to_kw', 'q_to_kvar']])[
                int(already_used_branches.number.loc[
                   (already_used_branches.from_bus == from_bus) &
                   (already_used_branches.to_bus == to_bus)].values)].reshape(1, 4), 0)
            already_used_branches.number.loc[(already_used_branches.from_bus == from_bus) &
                                             (already_used_branches.to_bus == to_bus)] += 1
        # from trafo results
        else:
            if from_vn_kv >= to_vn_kv:
                pp_res_branch = append(pp_res_branch, array(pp_net.res_trafo[
                    (pp_net.trafo.hv_bus == from_bus) &
                    (pp_net.trafo.lv_bus == to_bus)]
                    [['p_hv_kw', 'q_hv_kvar', 'p_lv_kw', 'q_lv_kvar']])[
                    int(already_used_branches.number.loc[
                        (already_used_branches.hv_bus == from_bus) &
                        (already_used_branches.lv_bus == to_bus)].values)].reshape(1, 4), 0)
                already_used_branches.number.loc[(already_used_branches.hv_bus == from_bus) &
                                                 (already_used_branches.lv_bus == to_bus)] += 1
            else:  # switch hv-lv-connection of pypower connection buses
                pp_res_branch = append(pp_res_branch, array(pp_net.res_trafo[
                    (pp_net.trafo.hv_bus == to_bus) &
                    (pp_net.trafo.lv_bus == from_bus)]
                    [['p_lv_kw', 'q_lv_kvar', 'p_hv_kw', 'q_hv_kvar']])[
                    int(already_used_branches.number.loc[
                        (already_used_branches.hv_bus == to_bus) &
                        (already_used_branches.lv_bus == from_bus)].values)].reshape(1, 4), 0)
                already_used_branches.number.loc[
                    (already_used_branches.hv_bus == to_bus) &
                    (already_used_branches.lv_bus == from_bus)] += 1
    pp_res_branch = pp_res_branch[1:, :]  # delete initial zero row

    # --- do the powerflow result comparison
    diff_res_bus = ppc_res_bus - pp_res_bus
    diff_res_branch = ppc_res_branch - pp_res_branch * 1e-3
    diff_res_gen = ppc_res_gen + pp_res_gen * 1e-3
    # comparison of buses with several generator units only as q sum
    GEN_uniq = GENS.drop_duplicates()
    for i in GEN_uniq.loc[GEN_uniq[0].isin(change_q_compare)].index:
        next_is = GEN_uniq.index[GEN_uniq.index > i]
        if len(next_is) > 0:
            next_i = next_is[0]
        else:
            next_i = GENS.index[-1] + 1
        if (next_i - i) > 1:
            diff_res_gen[i:next_i, 1] = sum(diff_res_gen[i:next_i, 1])
    # logger info
    logger.debug("Maximum voltage magnitude difference between pypower and pandapower: "
                 "%.2e pu" % max_(abs(diff_res_bus[:, 0])))
    logger.debug("Maximum voltage angle difference between pypower and pandapower: "
                 "%.2e degree" % max_(abs(diff_res_bus[:, 1])))
    logger.debug("Maximum branch flow active power difference between pypower and pandapower: "
                 "%.2e kW" % max_(abs(diff_res_branch[:, [0, 2]] * 1e3)))
    logger.debug("Maximum branch flow reactive power difference between pypower and "
                 "pandapower: %.2e kVAr" % max_(abs(diff_res_branch[:, [1, 3]] * 1e3)))
    logger.debug("Maximum active power generation difference between pypower and pandapower: "
                 "%.2e kW" % max_(abs(diff_res_gen[:, 0] * 1e3)))
    logger.debug("Maximum reactive power generation difference between pypower and pandapower: "
                 "%.2e kVAr" % max_(abs(diff_res_gen[:, 1] * 1e3)))
    if (max_(abs(diff_res_bus[:, 0])) < 1e-3) & (max_(abs(diff_res_bus[:, 1])) < 1e-3) & \
            (max_(abs(diff_res_branch[:, [0, 2]])) < 1e-3) & \
            (max_(abs(diff_res_branch[:, [1, 3]])) < 1e-3) & \
            (max_(abs(diff_res_gen)) > 1e-1).any():
        logger.debug("The active/reactive power generation difference possibly results "
                     "because of a pypower error. Please validate "
                     "the results via pypower loadflow.")  # this occurs e.g. at ppc case9
    # give a return
    if isinstance(max_diff_values, dict):
        if Series(['q_gen_kvar', 'p_branch_kw', 'q_branch_kvar', 'p_gen_kw', 'va_degree',
                   'vm_pu']).isin(Series(list(max_diff_values.keys()))).all():
            return (max_(abs(diff_res_bus[:, 0])) < max_diff_values['vm_pu']) & \
                    (max_(abs(diff_res_bus[:, 1])) < max_diff_values['va_degree']) & \
                    (max_(abs(diff_res_branch[:, [0, 2]])) < max_diff_values['p_branch_kw'] /
                        1e3) & \
                    (max_(abs(diff_res_branch[:, [1, 3]])) < max_diff_values['q_branch_kvar'] /
                        1e3) & \
                    (max_(abs(diff_res_gen[:, 0])) < max_diff_values['p_gen_kw'] / 1e3) & \
                    (max_(abs(diff_res_gen[:, 1])) < max_diff_values['q_gen_kvar'] / 1e3)
        else:
            logger.debug('Not all requried dict keys are provided.')
    else:
        logger.debug("'max_diff_values' must be a dict.")
