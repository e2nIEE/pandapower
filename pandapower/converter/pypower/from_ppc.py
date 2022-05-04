# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from math import pi
import numpy as np
import pandas as pd
from pandapower.pypower.idx_bus import \
    BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN
from pandapower.pypower.idx_gen import \
    GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN
from pandapower.pypower.idx_brch import \
    F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS, ANGMIN, ANGMAX
from pandapower.pypower.idx_cost import COST, NCOST
import pandapower as pp

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)

try:
    from pypower import ppoption, runpf, runopf, rundcpf, rundcopf
    ppopt = ppoption.ppoption(VERBOSE=0, OUT_ALL=0)
    pypower_import = True
except ImportError:
    pypower_import = False

ppc_elms = ["bus", "branch", "gen"]


def from_ppc(ppc, f_hz=50, validate_conversion=False, **kwargs):
    """
    This function converts pypower case files to pandapower net structure.

    INPUT:

        **ppc** : The pypower case file.

    OPTIONAL:

        **f_hz** (float, 50) - The frequency of the network.

        **validate_conversion** (bool, False) - If True, validate_from_ppc is run after conversion.
            For running the validation, the ppc must already contain the pypower
            powerflow results or pypower must be installed.

        ****kwargs** keyword arguments for

            - validate_from_ppc if validate_conversion is True

            - tap_side

    OUTPUT:

        **net** : pandapower net.

    EXAMPLE:

        import pandapower.converter as pc

        from pypower import case4gs

        ppc_net = case4gs.case4gs()

        net = pc.from_ppc(ppc_net, f_hz=60)

    """
    # --- catch common failures
    if pd.Series(ppc['bus'][:, BASE_KV] <= 0).any():
        logger.info('There are false baseKV given in the pypower case file.')

    net = pp.create_empty_network(f_hz=f_hz, sn_mva=ppc["baseMVA"])


    _from_ppc_bus(net, ppc)
    gen_lookup = _from_ppc_gen(net, ppc)
    _from_ppc_branch(net, ppc, f_hz, **kwargs)
    _from_ppc_gencost(net, ppc, gen_lookup)

    # areas are unconverted

    if validate_conversion:
        logger.setLevel(logging.DEBUG)
        if not validate_from_ppc(ppc, net, **kwargs):
            logger.error("Validation failed.")

    net._options = {}
    net._options["gen_lookup"] = gen_lookup

    return net


def _from_ppc_bus(net, ppc):
    """ bus data -> create buses, sgen, load, shunt """

    # create buses
    idx_buses = pp.create_buses(
        net, ppc['bus'].shape[0], name=ppc['bus'][:, BUS_I].astype(int),
        vn_kv=ppc['bus'][:, BASE_KV], type="b", zone=ppc['bus'][:, ZONE],
        in_service=(ppc['bus'][:, BUS_TYPE] != 4).astype(bool),
        max_vm_pu=ppc['bus'][:, VMAX], min_vm_pu=ppc['bus'][:, VMIN])

    # create loads
    is_load = (ppc['bus'][:, PD] > 0) | ((ppc['bus'][:, PD] == 0) & (ppc['bus'][:, QD] != 0))
    pp.create_loads(net, idx_buses[is_load], p_mw=ppc['bus'][is_load, PD], q_mvar=ppc['bus'][
        is_load, QD], controllable=False)

    # create sgens
    is_sgen = ppc['bus'][:, PD] < 0
    pp.create_sgens(net, idx_buses[is_sgen], p_mw=-ppc['bus'][is_sgen, PD], q_mvar=-ppc['bus'][
        is_sgen, QD], type="", controllable=False)

    # create shunts
    is_shunt = (ppc['bus'][:, GS] != 0) | (ppc['bus'][:, BS] != 0)
    pp.create_shunts(net, idx_buses[is_shunt], p_mw=ppc['bus'][is_shunt, GS],
                     q_mvar=-ppc['bus'][is_shunt, BS])

    # unused data of ppc: Vm, Va (partwise: in ext_grid), zone


def _from_ppc_gen(net, ppc):
    """ gen data -> create ext_grid, gen, sgen """
    n_gen = ppc["gen"].shape[0]

    # if in ppc is only one gen -> numpy initially uses one dim array -> change to two dim array
    if len(ppc["gen"].shape) == 1:
        ppc["gen"] = np.array(ppc["gen"], ndmin=2)

    bus_pos = _get_bus_pos(ppc, ppc["gen"][:, GEN_BUS])

    # determine which gen should considered as ext_grid, gen or sgen
    is_ext_grid, is_gen, is_sgen = _gen_to_which(ppc, bus_pos=bus_pos)

    # take VG of the last gen of each bus
    vg_bus_lookup = pd.DataFrame({"vg": ppc["gen"][:, VG], "bus": bus_pos})
    # vg_bus_lookup = vg_bus_lookup.drop_duplicates(subset=["bus"], keep="last").set_index("bus")["vg"]
    vg_bus_lookup = vg_bus_lookup.drop_duplicates(subset=["bus"]).set_index("bus")["vg"]

    # create ext_grid
    idx_eg = list()
    for i in np.arange(n_gen, dtype=int)[is_ext_grid]:
        idx_eg.append(pp.create_ext_grid(
            net, bus=bus_pos[i], vm_pu=vg_bus_lookup.at[bus_pos[i]],
            va_degree=ppc['bus'][bus_pos[i], VA],
            in_service=(ppc['gen'][i, GEN_STATUS] > 0).astype(bool),
            max_p_mw=ppc['gen'][i, PMAX], min_p_mw=ppc['gen'][i, PMIN],
            max_q_mvar=ppc['gen'][i, QMAX], min_q_mvar=ppc['gen'][i, QMIN]))

    # create gen
    idx_gen = pp.create_gens(
        net, buses=bus_pos[is_gen], vm_pu=vg_bus_lookup.loc[bus_pos[is_gen]].values,
        p_mw=ppc['gen'][is_gen, PG],
        in_service=(ppc['gen'][is_gen, GEN_STATUS] > 0), controllable=True,
        max_p_mw=ppc['gen'][is_gen, PMAX], min_p_mw=ppc['gen'][is_gen, PMIN],
        max_q_mvar=ppc['gen'][is_gen, QMAX], min_q_mvar=ppc['gen'][is_gen, QMIN])

    # create sgen
    idx_sgen = pp.create_sgens(
        net, buses=bus_pos[is_sgen], p_mw=ppc['gen'][is_sgen, PG],
        q_mvar=ppc['gen'][is_sgen, QG], type="",
        in_service=(ppc['gen'][is_sgen, GEN_STATUS] > 0),
        max_p_mw=ppc['gen'][is_sgen, PMAX], min_p_mw=ppc['gen'][is_sgen, PMIN],
        max_q_mvar=ppc['gen'][is_sgen, QMAX], min_q_mvar=ppc['gen'][is_sgen, QMIN],
        controllable=True)

    neg_p_gens = np.arange(n_gen, dtype=int)[(ppc['gen'][:, PG] < 0) & (is_gen | is_sgen)]
    neg_p_lim_false = np.arange(n_gen, dtype=int)[ppc['gen'][:, PMIN] > ppc['gen'][:, PMAX]]
    neg_q_lim_false = np.arange(n_gen, dtype=int)[ppc['gen'][:, QMIN] > ppc['gen'][:, QMAX]]
    if len(neg_p_gens):
        logger.info(f'These gen have PG < 0 and are not converted to ext_grid: {neg_p_gens}.')
    if len(neg_p_lim_false):
        logger.info(f'These gen have PMIN > PMAX: {neg_p_lim_false}.')
    if len(neg_q_lim_false):
        logger.info(f'These gen have QMIN > QMAX: {neg_q_lim_false}.')

    # unused data of ppc: Vg (partwise: in ext_grid and gen), mBase, Pc1, Pc2, Qc1min, Qc1max,
    # Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30,ramp_q, apf

    # gen_lookup
    gen_lookup = pd.DataFrame({
        'element': np.r_[idx_eg, idx_gen, idx_sgen],
        'element_type': ["ext_grid"]*sum(is_ext_grid) + ["gen"]*sum(is_gen) + ["sgen"]*sum(is_sgen)
        })
    return gen_lookup


def _from_ppc_branch(net, ppc, f_hz, **kwargs):
    """ branch data -> create line, trafo """

    # --- general_parameters
    baseMVA = ppc['baseMVA']  # MVA
    omega = pi * f_hz  # 1/s
    MAX_VAL = 99999.

    from_bus = _get_bus_pos(ppc, ppc['branch'][:, F_BUS].real.astype(int))
    to_bus = _get_bus_pos(ppc, ppc['branch'][:, T_BUS].real.astype(int))
    from_vn_kv = ppc['bus'][from_bus, BASE_KV]
    to_vn_kv = ppc['bus'][to_bus, BASE_KV]

    is_line, to_vn_is_leq = _branch_to_which(ppc, from_vn_kv=from_vn_kv, to_vn_kv=to_vn_kv)

    # --- create line
    Zni = ppc['bus'][to_bus, BASE_KV]**2/baseMVA  # ohm
    max_i_ka = ppc['branch'][:, 5]/ppc['bus'][to_bus, BASE_KV]/np.sqrt(3)
    i_is_zero = np.isclose(max_i_ka, 0)
    if np.any(i_is_zero):
        max_i_ka[i_is_zero] = MAX_VAL
        logger.debug("ppc branch rateA is zero -> Using MAX_VAL instead to calculate " +
                     "maximum branch flow")
    pp.create_lines_from_parameters(
        net, from_buses=from_bus[is_line], to_buses=to_bus[is_line], length_km=1,
        r_ohm_per_km=(ppc['branch'][is_line, BR_R]*Zni[is_line]).real,
        x_ohm_per_km=(ppc['branch'][is_line, BR_X]*Zni[is_line]).real,
        c_nf_per_km=(ppc['branch'][is_line, BR_B]/Zni[is_line]/omega*1e9/2).real,
        max_i_ka=max_i_ka[is_line].real, type='ol', max_loading_percent=100,
        in_service=ppc['branch'][is_line, BR_STATUS].real.astype(bool))

    # --- create transformer
    if not np.all(is_line):
        hv_bus = from_bus[~is_line]
        vn_hv_kv = from_vn_kv[~is_line]
        lv_bus = to_bus[~is_line]
        vn_lv_kv = to_vn_kv[~is_line]
        if not np.all(to_vn_is_leq):
            hv_bus[~to_vn_is_leq] = to_bus[~is_line][~to_vn_is_leq]
            vn_hv_kv[~to_vn_is_leq] = to_vn_kv[~is_line][~to_vn_is_leq]
            lv_bus[~to_vn_is_leq] = from_bus[~is_line][~to_vn_is_leq]
            vn_lv_kv[~to_vn_is_leq] = from_vn_kv[~is_line][~to_vn_is_leq]
        same_vn = to_vn_kv[~is_line] == from_vn_kv[~is_line]
        if np.any(same_vn):
            logger.warning(
                f'There are {sum(same_vn)} branches which are considered as trafos - due to ratio '
                f'unequal 0 or 1 - but connect same voltage levels.')
        rk = ppc['branch'][~is_line, BR_R].real
        xk = ppc['branch'][~is_line, BR_X].real
        zk = (rk ** 2 + xk ** 2) ** 0.5
        sn = ppc['branch'][~is_line, RATE_A].real
        sn_is_zero = np.isclose(sn, 0)
        if np.any(sn_is_zero):
            sn[sn_is_zero] = MAX_VAL
            logger.debug("ppc branch rateA is zero -> Using MAX_VAL instead to calculate " +
                            "apparent power")
        tap_side = kwargs.get("tap_side", "hv")
        if isinstance(tap_side, str):
            tap_side_is_hv = np.array([tap_side == "hv"]*sum(~is_line))
        else:
            tap_side_is_hv = tap_side == "hv"
        ratio_1 = ppc['branch'][~is_line, TAP].real
        ratio_is_zero = np.isclose(ratio_1, 0)
        ratio_1[~ratio_is_zero & ~tap_side_is_hv] **= -1
        ratio_1[~ratio_is_zero] -= 1
        i0_percent = -ppc['branch'][~is_line, 4].real * 100 * baseMVA / sn
        is_neg_i0_percent = i0_percent < 0
        if np.any(is_neg_i0_percent):
            logger.info(
                'Transformers always behave inductive consumpting but the susceptance of pypower '
                f'branches {np.arange(len(is_neg_i0_percent), dtype=int)[is_neg_i0_percent]} '
                f'(hv_bus, lv_bus)=({hv_bus[is_neg_i0_percent]}, {hv_bus[is_neg_i0_percent]}) '
                'is positive.')
        vk_percent = np.sign(xk) * zk * sn * 100 / baseMVA
        vk_percent[~tap_side_is_hv] /= (1+ratio_1[~tap_side_is_hv])**2
        vkr_percent = rk * sn * 100 / baseMVA
        vkr_percent[~tap_side_is_hv] /= (1+ratio_1[~tap_side_is_hv])**2

        pp.create_transformers_from_parameters(
            net, hv_buses=hv_bus, lv_buses=lv_bus, sn_mva=sn,
            vn_hv_kv=vn_hv_kv, vn_lv_kv=vn_lv_kv,
            vk_percent=vk_percent, vkr_percent=vkr_percent,
            max_loading_percent=100, pfe_kw=0, i0_percent=i0_percent,
            shift_degree=ppc['branch'][~is_line, SHIFT].real,
            tap_step_percent=np.abs(ratio_1)*100, tap_pos=np.sign(ratio_1),
            tap_side=tap_side, tap_neutral=0)
    # unused data of ppc: rateB, rateC


def _get_bus_pos(ppc, bus_names):
    try:
        return pd.Series(np.arange(ppc["bus"].shape[0], dtype=int), index=ppc["bus"][
            :, BUS_I]).loc[bus_names].values
    except:
        return pd.Series(np.arange(ppc["bus"].shape[0], dtype=int), index=ppc["bus"][
            :, BUS_I].astype(int)).loc[bus_names].values


def _gen_to_which(ppc, bus_pos=None, flattened=True):
    if bus_pos is None:
        bus_pos = _get_bus_pos(ppc, ppc["gen"][:, GEN_BUS])
    bus_type_df = pd.DataFrame({"bus_type": ppc["bus"][bus_pos, BUS_TYPE],
                                "bus": ppc["gen"][:, GEN_BUS]}).astype(int)
    bus_type_df = pd.concat([bus_type_df.loc[bus_type_df.bus_type == 3],
                            bus_type_df.loc[bus_type_df.bus_type == 2],
                            bus_type_df.loc[bus_type_df.bus_type == 1],
                            bus_type_df.loc[bus_type_df.bus_type == 4]
                            ])
    is_ext_grid = ((bus_type_df["bus_type"] == 3) & ~bus_type_df.duplicated(subset=[
        "bus"])).sort_index().values
    is_gen = ((bus_type_df["bus_type"] == 2) & ~bus_type_df.duplicated(subset=[
        "bus"])).sort_index().values
    is_sgen = ~(is_ext_grid | is_gen) & bus_type_df["bus_type"].sort_index().isin([3, 2, 1]).values
    return is_ext_grid, is_gen, is_sgen


def _branch_to_which(ppc, from_vn_kv=None, to_vn_kv=None, flattened=True):
    if from_vn_kv is None:
        from_bus = _get_bus_pos(ppc, ppc['branch'][:, F_BUS].real.astype(int))
        from_vn_kv = ppc['bus'][from_bus, BASE_KV]
    if to_vn_kv is None:
        to_bus = _get_bus_pos(ppc, ppc['branch'][:, T_BUS].real.astype(int))
        to_vn_kv = ppc['bus'][to_bus, BASE_KV]
    is_line = (from_vn_kv == to_vn_kv) & \
              ((ppc['branch'][:, TAP] == 0) | (ppc['branch'][:, TAP] == 1)) & \
              (ppc['branch'][:, SHIFT] == 0)
    to_vn_is_leq = to_vn_kv[~is_line] <= from_vn_kv[~is_line]
    return is_line, to_vn_is_leq


def _from_ppc_gencost(net, ppc, gen_lookup):
    # --- gencost -> create polynomial_cost, piecewise_cost
    if 'gencost' in ppc:
        if len(ppc['gencost'].shape) == 1:
            # reshape gencost if only one gencost is given -> no indexError
            ppc['gencost'] = ppc['gencost'].reshape((1, -1))
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


def _create_costs(net, ppc, gen_lookup, type, idx):
    if ppc['gencost'][idx, 0] == 1:
        if not len(ppc['gencost'][idx, COST:]) == 2*ppc['gencost'][idx, NCOST]:
            logger.error(f"In gencost line {idx}, the number n does not fit to the number of values")
            raise NotImplementedError
        pp.create_pwl_cost(net, gen_lookup.element.at[idx],
                           gen_lookup.element_type.at[idx],
                           ppc['gencost'][idx, 4:], type)
    elif ppc['gencost'][idx, 0] == 2:
        ncost = ppc['gencost'][idx, NCOST]
        if ncost == 1:
            cp2 = 0
            cp1 = 0
            cp0 = ppc['gencost'][idx, COST]
        elif ncost == 2:
            cp2 = 0
            cp1 = ppc['gencost'][idx, COST]
            cp0 = ppc['gencost'][idx, COST + 1]
        elif ncost == 3:
            cp2 = ppc['gencost'][idx, COST]
            cp1 = ppc['gencost'][idx, COST + 1]
            cp0 = ppc['gencost'][idx, COST + 2]
        elif ncost > 3:
            logger.warning("The pandapower poly_cost table only supports up to 2nd order " +
                           "polynomials. The ppc higher order polynomials cannot be converted.")
            cp2 = ppc['gencost'][idx, COST + ncost - 3]
            cp1 = ppc['gencost'][idx, COST + ncost - 2]
            cp0 = ppc['gencost'][idx, COST + ncost - 1]
        else:
            raise ValueError("'ncost' must be an positve integer but is " + str(ncost))
        pp.create_poly_cost(net, gen_lookup.element.at[idx], gen_lookup.element_type.at[idx],
                                  cp1_eur_per_mw=cp1, cp2_eur_per_mw2=cp2, cp0_eur=cp0)
    else:
        logger.info(f"Cost mode of gencost line {idx} is unknown.")


def _validate_diff_res(diff_res, max_diff_values):
    to_iterate = set(max_diff_values.keys()) & {'gen_q_mvar', 'branch_p_mw', 'branch_q_mvar',
                                                'gen_p_mw', 'bus_va_degree', 'bus_vm_pu'}
    if not len(to_iterate):
        logger.warning("There are no keys to validate.")
    val = True
    for i in to_iterate:
        elm = i.split("_")[0]
        sought = ["p", "q"] if elm != "bus" else ["vm", "va"]
        col = int(np.array([0, 1])[[j in i for j in sought]][0]) if elm != "branch" else \
            list(np.array([[0, 2], [1, 3]])[[j in i for j in sought]][0])
        val &= bool(np.max(abs(diff_res[elm][:, col])) < max_diff_values[i])
    return val


def _gen_bus_info(ppc, idx_gen):
    bus_name = int(ppc["gen"][idx_gen, GEN_BUS])
    # assumption: there is only one bus with this bus_name:
    idx_bus = int(np.where(ppc["bus"][:, BUS_I] == bus_name)[0][0])
    current_bus_type = int(ppc["bus"][idx_bus, 1])

    same_bus_gen = np.where(ppc["gen"][:, GEN_BUS] == ppc["gen"][idx_gen, GEN_BUS])[0].astype(int)
    same_bus_gen = same_bus_gen[np.where(ppc["gen"][same_bus_gen, GEN_STATUS] > 0)]
    first_same_bus = same_bus_gen[0] if len(same_bus_gen) else None

    return current_bus_type, idx_bus, first_same_bus


def validate_from_ppc(ppc_net, net, pf_type="runpp", max_diff_values={
    "bus_vm_pu": 1e-6, "bus_va_degree": 1e-5, "branch_p_mw": 1e-6, "branch_q_mvar": 1e-6,
        "gen_p_mw": 1e-6, "gen_q_mvar": 1e-6}, run=True):
    """
    This function validates the pypower case files to pandapower net structure conversion via a \
    comparison of loadflow calculation results. (Hence the opf cost conversion is not validated.)

    INPUT:

        **ppc_net** - The pypower case file, which must already contain the pypower powerflow
            results or pypower must be importable.

        **net** - The pandapower network.

    OPTIONAL:

        **pf_type** ("runpp", string) - Type of validated power flow. Possible are ("runpp",
            "rundcpp", "runopp", "rundcopp")

        **max_diff_values** - Dict of maximal allowed difference values. The keys must be
        'vm_pu', 'va_degree', 'p_branch_mw', 'q_branch_mvar', 'p_gen_mw' and 'q_gen_mvar' and
        the values floats.

        **run** (True, bool or list of two bools) - changing the value to False avoids trying to run
            (optimal) loadflows. Giving a list of two bools addresses first pypower and second
            pandapower.

    OUTPUT:

        **conversion_success** - conversion_success is returned as False if pypower or pandapower
        cannot calculate a powerflow or if the maximum difference values (max_diff_values )
        cannot be hold.

    EXAMPLE:

        import pandapower.converter as pc

        net = cv.from_ppc(ppc_net, f_hz=50)

        conversion_success = cv.validate_from_ppc(ppc_net, net)

    NOTE:

        The user has to take care that the loadflow results already are included in the provided \
        ppc_net or pypower is importable.
    """

    # check in case of optimal powerflow comparison whether cost information exist
    if "opp" in pf_type:
        if not (len(net.polynomial_cost) | len(net.piecewise_linear_cost)):
            if "gencost" in ppc_net:
                if not len(ppc_net["gencost"]):
                    logger.debug('ppc and pandapower net do not include cost information.')
                    return True
                else:
                    logger.error('The pandapower net does not include cost information.')
                    return False
            else:
                logger.debug('ppc and pandapower net do not include cost information.')
                return True

    # guarantee run parameter as list, for pypower and pandapower (optimal) powerflow run
    run = [run, run] if isinstance(run, bool) else run

    # --- check pypower powerflow success, if possible
    if pypower_import and run[0]:
        try:
            if pf_type == "runpp":
                ppc_net = runpf.runpf(ppc_net, ppopt)[0]
            elif pf_type == "rundcpp":
                ppc_net = rundcpf.rundcpf(ppc_net, ppopt)[0]
            elif pf_type == "runopp":
                ppc_net = runopf.runopf(ppc_net, ppopt)
            elif pf_type == "rundcopp":
                ppc_net = rundcopf.rundcopf(ppc_net, ppopt)
            else:
                raise ValueError(f"The pf_type {pf_type} is unknown")
        except:
            logger.debug("The pypower run did not work.")
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
    if run[1]:
        if pf_type == "runpp":
            try:
                pp.runpp(net, init="dc", calculate_voltage_angles=True, trafo_model="pi")
            except pp.LoadflowNotConverged:
                try:
                    pp.runpp(net, calculate_voltage_angles=True, init="flat", trafo_model="pi")
                except pp.LoadflowNotConverged:
                    try:
                        pp.runpp(net, trafo_model="pi", calculate_voltage_angles=False)
                        if "bus_va_degree" in max_diff_values.keys():
                            max_diff_values["bus_va_degree"] = 1e2 if max_diff_values[
                                "bus_va_degree"] < 1e2 else max_diff_values["bus_va_degree"]
                        logger.info("voltage_angles could be calculated.")
                    except pp.LoadflowNotConverged:
                        logger.error('The pandapower powerflow does not converge.')
        elif pf_type == "rundcpp":
            try:
                pp.rundcpp(net, trafo_model="pi")
            except pp.LoadflowNotConverged:
                logger.error('The pandapower dc powerflow does not converge.')
        elif pf_type == "runopp":
                try:
                    pp.runopp(net, init="flat", calculate_voltage_angles=True)
                except pp.OPFNotConverged:
                    try:
                        pp.runopp(net, init="pf", calculate_voltage_angles=True)
                    except (pp.OPFNotConverged, pp.LoadflowNotConverged, KeyError):
                        try:
                            pp.runopp(net, init="flat", calculate_voltage_angles=False)
                            logger.info("voltage_angles could be calculated.")
                            if "bus_va_degree" in max_diff_values.keys():
                                max_diff_values["bus_va_degree"] = 1e2 if max_diff_values[
                                    "bus_va_degree"] < 1e2 else max_diff_values["bus_va_degree"]
                        except pp.OPFNotConverged:
                            try:
                                pp.runopp(net, init="pf", calculate_voltage_angles=False)
                                if "bus_va_degree" in max_diff_values.keys():
                                    max_diff_values["bus_va_degree"] = 1e2 if max_diff_values[
                                        "bus_va_degree"] < 1e2 else max_diff_values["bus_va_degree"]
                                logger.info("voltage_angles could be calculated.")
                            except (pp.OPFNotConverged, pp.LoadflowNotConverged, KeyError):
                                logger.error('The pandapower optimal powerflow does not converge.')
        elif pf_type == "rundcopp":
            try:
                pp.rundcopp(net)
            except pp.LoadflowNotConverged:
                logger.error('The pandapower dc optimal powerflow does not converge.')
        else:
            raise ValueError("The pf_type %s is unknown" % pf_type)

    # --- prepare powerflow result comparison by reordering pp results as they are in ppc results
    if not ppc_success:
        return False
    if "opp" in pf_type:
        if not net.OPF_converged:
            return
    elif not net.converged:
        return False

    # --- store pypower powerflow results
    ppc_res = dict.fromkeys(ppc_elms)
    ppc_res["branch"] = ppc_net['branch'][:, 13:17]
    ppc_res["bus"] = ppc_net['bus'][:, 7:9]
    ppc_res["gen"] = ppc_net['gen'][:, 1:3]

    # --- pandapower bus result table
    pp_res = dict.fromkeys(ppc_elms)
    pp_res["bus"] = np.array(net.res_bus.sort_index()[['vm_pu', 'va_degree']])

    # --- pandapower gen result table
    pp_res["gen"] = np.zeros([1, 2])
    # consideration of parallel generators via storing how much generators have been considered
    # each node
    # if in ppc is only one gen -> numpy initially uses one dim array -> change to two dim array
    if len(ppc_net["gen"].shape) == 1:
        ppc_net["gen"] = np.array(ppc_net["gen"], ndmin=2)
    GENS = pd.DataFrame(ppc_net['gen'][:, [0]].astype(int))
    GEN_uniq = GENS.drop_duplicates()
    already_used_gen = pd.Series(np.zeros(GEN_uniq.shape[0]).astype(int),
                                 index=[int(v) for v in GEN_uniq.values])
    change_q_compare = []
    for i, j in GENS.iterrows():
        current_bus_type, current_bus_idx, first_same_bus_in_service_gen_idx, = _gen_bus_info(
            ppc_net, i)
        if current_bus_type == 3 and i == first_same_bus_in_service_gen_idx:
            pp_res["gen"] = np.append(pp_res["gen"], np.array(net.res_ext_grid[
                    net.ext_grid.bus == current_bus_idx][['p_mw', 'q_mvar']]).reshape((1, 2)), 0)
        elif current_bus_type == 2 and i == first_same_bus_in_service_gen_idx:
            pp_res["gen"] = np.append(pp_res["gen"], np.array(net.res_gen[
                    net.gen.bus == current_bus_idx][['p_mw', 'q_mvar']]).reshape((1, 2)), 0)
        else:
            pp_res["gen"] = np.append(pp_res["gen"], np.array(net.res_sgen[
                net.sgen.bus == current_bus_idx][['p_mw', 'q_mvar']])[
                already_used_gen.at[int(j)]].reshape((1, 2)), 0)
            already_used_gen.at[int(j)] += 1
            change_q_compare += [int(j)]
    pp_res["gen"] = pp_res["gen"][1:, :]  # delete initial zero row

    # --- pandapower branch result table
    pp_res["branch"] = np.zeros([1, 4])
    # consideration of parallel branches via storing how often branches were considered
    # each node-to-node-connection
    try:
        init1 = pd.concat([net.line.from_bus, net.line.to_bus], axis=1,
                          sort=True).drop_duplicates()
        init2 = pd.concat([net.trafo.hv_bus, net.trafo.lv_bus], axis=1,
                          sort=True).drop_duplicates()
    except TypeError:
        # legacy pandas < 0.21
        init1 = pd.concat([net.line.from_bus, net.line.to_bus], axis=1).drop_duplicates()
        init2 = pd.concat([net.trafo.hv_bus, net.trafo.lv_bus], axis=1).drop_duplicates()
    init1['hv_bus'] = np.nan
    init1['lv_bus'] = np.nan
    init2['from_bus'] = np.nan
    init2['to_bus'] = np.nan
    try:
        already_used_branches = pd.concat([init1, init2], axis=0, sort=True)
    except TypeError:
        # pandas < 0.21 legacy
        already_used_branches = pd.concat([init1, init2], axis=0)
    already_used_branches['number'] = np.zeros([already_used_branches.shape[0], 1]).astype(int)
    BRANCHES = pd.DataFrame(ppc_net['branch'][:, [0, 1, TAP, SHIFT]])
    for i in BRANCHES.index:
        from_bus = pp.get_element_index(net, 'bus', name=int(ppc_net['branch'][i, 0]))
        to_bus = pp.get_element_index(net, 'bus', name=int(ppc_net['branch'][i, 1]))
        from_vn_kv = ppc_net['bus'][from_bus, BASE_KV]
        to_vn_kv = ppc_net['bus'][to_bus, BASE_KV]
        ratio = BRANCHES[2].at[i]
        angle = BRANCHES[3].at[i]
        # from line results
        if (from_vn_kv == to_vn_kv) & ((ratio == 0) | (ratio == 1)) & (angle == 0):
            pp_res["branch"] = np.append(pp_res["branch"], np.array(net.res_line[
                (net.line.from_bus == from_bus) &
                (net.line.to_bus == to_bus)]
                [['p_from_mw', 'q_from_mvar', 'p_to_mw', 'q_to_mvar']])[
                int(already_used_branches.number.loc[
                   (already_used_branches.from_bus == from_bus) &
                   (already_used_branches.to_bus == to_bus)].values)].reshape(1, 4), 0)
            already_used_branches.number.loc[(already_used_branches.from_bus == from_bus) &
                                             (already_used_branches.to_bus == to_bus)] += 1
        # from trafo results
        else:
            if from_vn_kv >= to_vn_kv:
                pp_res["branch"] = np.append(pp_res["branch"], np.array(net.res_trafo[
                    (net.trafo.hv_bus == from_bus) &
                    (net.trafo.lv_bus == to_bus)]
                    [['p_hv_mw', 'q_hv_mvar', 'p_lv_mw', 'q_lv_mvar']])[
                    int(already_used_branches.number.loc[
                        (already_used_branches.hv_bus == from_bus) &
                        (already_used_branches.lv_bus == to_bus)].values)].reshape(1, 4), 0)
                already_used_branches.number.loc[(already_used_branches.hv_bus == from_bus) &
                                                 (already_used_branches.lv_bus == to_bus)] += 1
            else:  # switch hv-lv-connection of pypower connection buses
                pp_res["branch"] = np.append(pp_res["branch"], np.array(net.res_trafo[
                    (net.trafo.hv_bus == to_bus) &
                    (net.trafo.lv_bus == from_bus)]
                    [['p_lv_mw', 'q_lv_mvar', 'p_hv_mw', 'q_hv_mvar']])[
                    int(already_used_branches.number.loc[
                        (already_used_branches.hv_bus == to_bus) &
                        (already_used_branches.lv_bus == from_bus)].values)].reshape(1, 4), 0)
                already_used_branches.number.loc[
                    (already_used_branches.hv_bus == to_bus) &
                    (already_used_branches.lv_bus == from_bus)] += 1
    pp_res["branch"] = pp_res["branch"][1:, :]  # delete initial zero row

    # --- do the powerflow result comparison
    diff_res = dict.fromkeys(ppc_elms)
    diff_res["bus"] = ppc_res["bus"] - pp_res["bus"]
    diff_res["bus"][:, 1] -= diff_res["bus"][0, 1]  # remove va_degree offset
    diff_res["branch"] = ppc_res["branch"] - pp_res["branch"]
    diff_res["gen"] = ppc_res["gen"] - pp_res["gen"]
    # comparison of buses with several generator units only as q sum
    for i in GEN_uniq.loc[GEN_uniq[0].isin(change_q_compare)].index:
        next_is = GEN_uniq.index[GEN_uniq.index > i]
        if len(next_is) > 0:
            next_i = next_is[0]
        else:
            next_i = GENS.index[-1] + 1
        if (next_i - i) > 1:
            diff_res["gen"][i:next_i, 1] = sum(diff_res["gen"][i:next_i, 1])
    # logger info
    logger.debug("Maximum voltage magnitude difference between pypower and pandapower: "
                 "%.2e pu" % np.max(abs(diff_res["bus"][:, 0])))
    logger.debug("Maximum voltage angle difference between pypower and pandapower: "
                 "%.2e degree" % np.max(abs(diff_res["bus"][:, 1])))
    logger.debug("Maximum branch flow active power difference between pypower and pandapower: "
                 "%.2e MW" % np.max(abs(diff_res["branch"][:, [0, 2]])))
    logger.debug("Maximum branch flow reactive power difference between pypower and "
                 "pandapower: %.2e MVAr" % np.max(abs(diff_res["branch"][:, [1, 3]])))
    logger.debug("Maximum active power generation difference between pypower and pandapower: "
                 "%.2e MW" % np.max(abs(diff_res["gen"][:, 0])))
    logger.debug("Maximum reactive power generation difference between pypower and pandapower: "
                 "%.2e MVAr" % np.max(abs(diff_res["gen"][:, 1])))
    if _validate_diff_res(diff_res, {"bus_vm_pu": 1e-3, "bus_va_degree": 1e-3, "branch_p_mw": 1e-6,
                                     "branch_q_mvar": 1e-6}) and \
            (np.max(abs(diff_res["gen"])) > 1e-1).any():
        logger.debug("The active/reactive power generation difference possibly results "
                     "because of a pypower error. Please validate "
                     "the results via pypower loadflow.")  # this occurs e.g. at ppc case9
    # give a return
    if isinstance(max_diff_values, dict):
        return _validate_diff_res(diff_res, max_diff_values)
    else:
        logger.debug("'max_diff_values' must be a dict.")


if __name__ == "__main__":
    pass
