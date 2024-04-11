# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
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
from pandapower.pypower.idx_cost import MODEL, COST, NCOST
from pandapower.create import create_empty_network, create_buses, create_ext_grid, create_loads, \
    create_sgens, create_gens, create_lines_from_parameters, create_transformers_from_parameters, \
    create_shunts, create_ext_grid, create_pwl_costs, create_poly_costs
from pandapower.run import runpp

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)

ppc_elms = ["bus", "branch", "gen"]


def from_ppc(ppc, f_hz=50, validate_conversion=False, **kwargs):
    """
    This function converts pypower case files to pandapower net structure.

    INPUT:
        **ppc** (dict) -  The pypower case file.

        **f_hz** (int) - The frequency of the network, by default 50

        **validate_conversion** (bool) - If True, validate_from_ppc is run after conversion. For running the validation, the ppc must already contain the pypower powerflow results or pypower must be installed, by default False

        **kwargs** (dict) - keyword arguments for:
                            - validate_from_ppc if validate_conversion is True

                            - tap_side

                            - check_costs is passed as "check" to create_pwl_costs() and create_poly_costs()

    OUTPUT:
        **net** - ppc converted to pandapower net structure

    EXAMPLES:
        >>> import pandapower
        >>> from pandapower.test.converter.test_from_ppc import get_testgrids
        >>> ppc = get_testgrids('pypower_cases', 'case4gs.json')
        >>> net = pandapower.converter.from_ppc(ppc, f_hz=60)
    """
    # --- catch common failures
    if pd.Series(ppc['bus'][:, BASE_KV] <= 0).any():
        logger.info('There are false baseKV given in the pypower case file.')

    net = create_empty_network(f_hz=f_hz, sn_mva=ppc["baseMVA"])
    net._from_ppc_lookups = {}

    _from_ppc_bus(net, ppc)
    net._from_ppc_lookups["gen"] = _from_ppc_gen(net, ppc)
    net._from_ppc_lookups["branch"] = _from_ppc_branch(net, ppc, f_hz, **kwargs)
    _from_ppc_gencost(net, ppc, net._from_ppc_lookups["gen"], check=kwargs.get("check_costs", True))

    # areas are unconverted

    if validate_conversion:
        logger.setLevel(logging.DEBUG)
        if not validate_from_ppc(ppc, net, **kwargs):
            logger.error("Validation failed.")

    return net


def _from_ppc_bus(net, ppc):
    """ bus data -> create buses, sgen, load, shunt """
    # create buses
    idx_buses = create_buses(
        net, ppc['bus'].shape[0], name=ppc.get("bus_name", None),
        vn_kv=ppc['bus'][:, BASE_KV], type="b", zone=ppc['bus'][:, ZONE],
        in_service=(ppc['bus'][:, BUS_TYPE] != 4).astype(bool),
        max_vm_pu=ppc['bus'][:, VMAX], min_vm_pu=ppc['bus'][:, VMIN],
        index=ppc['bus'][:, BUS_I].astype(np.int64))

    # create loads
    is_load = (ppc['bus'][:, PD] > 0) | ((ppc['bus'][:, PD] == 0) & (ppc['bus'][:, QD] != 0))
    create_loads(net, idx_buses[is_load], p_mw=ppc['bus'][is_load, PD], q_mvar=ppc['bus'][
        is_load, QD], controllable=False)

    # create sgens
    is_sgen = ppc['bus'][:, PD] < 0
    create_sgens(net, idx_buses[is_sgen], p_mw=-ppc['bus'][is_sgen, PD], q_mvar=-ppc['bus'][
        is_sgen, QD], type="", controllable=False)

    # create shunts
    is_shunt = (ppc['bus'][:, GS] != 0) | (ppc['bus'][:, BS] != 0)
    create_shunts(net, idx_buses[is_shunt], p_mw=ppc['bus'][is_shunt, GS],
                     q_mvar=-ppc['bus'][is_shunt, BS])

    # unused data from ppc: VM, VA (partwise: in ext_grid), BUS_AREA


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

    gen_name = ppc.get("gen_name", np.array([None]*n_gen))

    # create ext_grid
    idx_eg = list()
    for i in np.arange(n_gen, dtype=np.int64)[is_ext_grid]:
        idx_eg.append(create_ext_grid(
            net, bus=net.bus.index[bus_pos[i]], vm_pu=vg_bus_lookup.at[bus_pos[i]],
            va_degree=ppc['bus'][bus_pos[i], VA],
            in_service=(ppc['gen'][i, GEN_STATUS] > 0).astype(bool),
            max_p_mw=ppc['gen'][i, PMAX], min_p_mw=ppc['gen'][i, PMIN],
            max_q_mvar=ppc['gen'][i, QMAX], min_q_mvar=ppc['gen'][i, QMIN],
            name=gen_name[i]))

    # create gen
    idx_gen = create_gens(
        net, buses=net.bus.index[bus_pos[is_gen]], vm_pu=vg_bus_lookup.loc[bus_pos[is_gen]].values,
        p_mw=ppc['gen'][is_gen, PG], sn_mva=ppc['gen'][is_gen, MBASE],
        in_service=(ppc['gen'][is_gen, GEN_STATUS] > 0), controllable=True,
        max_p_mw=ppc['gen'][is_gen, PMAX], min_p_mw=ppc['gen'][is_gen, PMIN],
        max_q_mvar=ppc['gen'][is_gen, QMAX], min_q_mvar=ppc['gen'][is_gen, QMIN],
        name=gen_name[is_gen])

    # create sgen
    idx_sgen = create_sgens(
        net, buses=net.bus.index[bus_pos[is_sgen]], p_mw=ppc['gen'][is_sgen, PG],
        q_mvar=ppc['gen'][is_sgen, QG], sn_mva=ppc['gen'][is_sgen, MBASE], type="",
        in_service=(ppc['gen'][is_sgen, GEN_STATUS] > 0),
        max_p_mw=ppc['gen'][is_sgen, PMAX], min_p_mw=ppc['gen'][is_sgen, PMIN],
        max_q_mvar=ppc['gen'][is_sgen, QMAX], min_q_mvar=ppc['gen'][is_sgen, QMIN],
        controllable=True, name=gen_name[is_sgen])

    neg_p_gens = np.arange(n_gen, dtype=np.int64)[(ppc['gen'][:, PG] < 0) & (is_gen | is_sgen)]
    neg_p_lim_false = np.arange(n_gen, dtype=np.int64)[ppc['gen'][:, PMIN] > ppc['gen'][:, PMAX]]
    neg_q_lim_false = np.arange(n_gen, dtype=np.int64)[ppc['gen'][:, QMIN] > ppc['gen'][:, QMAX]]
    if len(neg_p_gens):
        logger.info(f'These gen have PG < 0 and are not converted to ext_grid: {neg_p_gens}.')
    if len(neg_p_lim_false):
        logger.info(f'These gen have PMIN > PMAX: {neg_p_lim_false}.')
    if len(neg_q_lim_false):
        logger.info(f'These gen have QMIN > QMAX: {neg_q_lim_false}.')

    # unused data from ppc: Vg (partwise: in ext_grid and gen), mBase, Pc1, Pc2, Qc1min, Qc1max,
    # Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30,ramp_q, apf

    # gen_lookup
    gen_lookup = pd.DataFrame({"element": [-1]*n_gen, "element_type": [""]*n_gen})
    for is_, idx, et in zip([is_ext_grid, is_gen, is_sgen],
                            [idx_eg, idx_gen, idx_sgen],
                            ["ext_grid", "gen", "sgen"]):
        gen_lookup.loc[is_, "element"] = idx
        gen_lookup.loc[is_, "element_type"] = et
    return gen_lookup


def _from_ppc_branch(net, ppc, f_hz, **kwargs):
    """ branch data -> create line, trafo """
    n_bra = ppc["branch"].shape[0]

    # --- general_parameters
    baseMVA = ppc['baseMVA']  # MVA
    omega = pi * f_hz  # 1/s
    MAX_VAL = 99999.

    from_bus = _get_bus_pos(ppc, ppc['branch'][:, F_BUS].real.astype(np.int64))
    to_bus = _get_bus_pos(ppc, ppc['branch'][:, T_BUS].real.astype(np.int64))
    from_vn_kv = ppc['bus'][from_bus, BASE_KV]
    to_vn_kv = ppc['bus'][to_bus, BASE_KV]

    is_line, to_vn_is_leq = _branch_to_which(ppc, from_vn_kv=from_vn_kv, to_vn_kv=to_vn_kv)

    bra_name = ppc.get("branch_name", ppc.get("bra_name", np.array([None]*n_bra)))

    # --- create line
    Zni = ppc['bus'][to_bus, BASE_KV]**2/baseMVA  # ohm
    max_i_ka = ppc['branch'][:, 5]/ppc['bus'][to_bus, BASE_KV]/np.sqrt(3)
    i_is_zero = np.isclose(max_i_ka, 0)
    if np.any(i_is_zero):
        max_i_ka[i_is_zero] = MAX_VAL
        logger.debug("ppc branch rateA is zero -> Using MAX_VAL instead to calculate " +
                     "maximum branch flow")
    idx_line = create_lines_from_parameters(
        net, from_buses=net.bus.index[from_bus[is_line]], to_buses=net.bus.index[to_bus[is_line]],
        length_km=1, name=bra_name[is_line],
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
            tap_side_is_hv = np.array(tap_side == "hv")
        ratio_1 = ppc['branch'][~is_line, TAP].real
        ratio_is_zero = np.isclose(ratio_1, 0)
        ratio_1[~ratio_is_zero & ~tap_side_is_hv] **= -1
        ratio_1[~ratio_is_zero] -= 1
        i0_percent = -ppc['branch'][~is_line, BR_B].real * 100 * baseMVA / sn
        is_neg_i0_percent = i0_percent < 0
        if np.any(is_neg_i0_percent):
            logger.info(
                'Transformers always behave inductive consumpting but the susceptance of pypower '
                f'branches {np.arange(len(is_neg_i0_percent), dtype=np.int64)[is_neg_i0_percent]} '
                f'(hv_bus, lv_bus)=({hv_bus[is_neg_i0_percent]}, {hv_bus[is_neg_i0_percent]}) '
                'is positive.')
        vk_percent = np.sign(xk) * zk * sn * 100 / baseMVA
        vk_percent[~tap_side_is_hv] /= (1+ratio_1[~tap_side_is_hv])**2
        vkr_percent = rk * sn * 100 / baseMVA
        vkr_percent[~tap_side_is_hv] /= (1+ratio_1[~tap_side_is_hv])**2

        idx_trafo = create_transformers_from_parameters(
            net, hv_buses=net.bus.index[hv_bus], lv_buses=net.bus.index[lv_bus], sn_mva=sn,
            vn_hv_kv=vn_hv_kv, vn_lv_kv=vn_lv_kv, name=bra_name[~is_line],
            vk_percent=vk_percent, vkr_percent=vkr_percent,
            max_loading_percent=100, pfe_kw=0, i0_percent=i0_percent,
            shift_degree=ppc['branch'][~is_line, SHIFT].real,
            tap_step_percent=np.abs(ratio_1)*100, tap_pos=np.sign(ratio_1),
            tap_side=tap_side, tap_neutral=0)
    else:
        idx_trafo = []
    # unused data from ppc: rateB, rateC

    # branch_lookup: which branches are lines, and which ones are transformers
    branch_lookup = pd.DataFrame({"element": [-1] * n_bra, "element_type": [""] * n_bra})
    branch_lookup.loc[is_line, "element"] = idx_line
    branch_lookup.loc[is_line, "element_type"] = "line"
    branch_lookup.loc[~is_line, "element"] = idx_trafo
    branch_lookup.loc[~is_line, "element_type"] = "trafo"
    return branch_lookup


def _get_bus_pos(ppc, bus_names):
    try:
        return pd.Series(np.arange(ppc["bus"].shape[0], dtype=np.int64), index=ppc["bus"][
            :, BUS_I]).loc[bus_names].values
    except:
        return pd.Series(np.arange(ppc["bus"].shape[0], dtype=np.int64), index=ppc["bus"][
            :, BUS_I].astype(np.int64)).loc[bus_names].values


def _gen_to_which(ppc, bus_pos=None, flattened=True):
    if bus_pos is None:
        bus_pos = _get_bus_pos(ppc, ppc["gen"][:, GEN_BUS])
    bus_type_df = pd.DataFrame({"bus_type": ppc["bus"][bus_pos, BUS_TYPE],
                                "bus": ppc["gen"][:, GEN_BUS]}).astype(np.int64)
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
        from_bus = _get_bus_pos(ppc, ppc['branch'][:, F_BUS].real.astype(np.int64))
        from_vn_kv = ppc['bus'][from_bus, BASE_KV]
    if to_vn_kv is None:
        to_bus = _get_bus_pos(ppc, ppc['branch'][:, T_BUS].real.astype(np.int64))
        to_vn_kv = ppc['bus'][to_bus, BASE_KV]
    is_line = (from_vn_kv == to_vn_kv) & \
              ((ppc['branch'][:, TAP] == 0) | (ppc['branch'][:, TAP] == 1)) & \
              (ppc['branch'][:, SHIFT] == 0)
    to_vn_is_leq = to_vn_kv[~is_line] <= from_vn_kv[~is_line]
    return is_line, to_vn_is_leq


def _calc_pp_pwl_points(ppc_pwl_points):
    """
    Converts ppc pwl points which is (x1, y1, ..., xn, yn)
    into pandapower pwl points which is ((x1, x2, c12), ..., (x(n-1), xn, c(n-1)n))
    with c12 is the slope between the points x1 and x2.
    """

    def construct_list_of_list(row):
        arr = pts[row, ::2]
        arr = np.concatenate((arr[:1], np.repeat(arr[1:-1], 2), arr[-1:])).reshape((-1, 2))
        arr = np.c_[arr, c[row, :]]
        arr = arr[~np.isnan(arr[:, 2])]
        return arr.tolist()

    pts = ppc_pwl_points
    if not (pts.shape[1] % 2) == 0:
        raise ValueError("_calc_pp_pwl_points() expects ppc_pwl_points with shape[1] is "
                         f"multiple of 2. However, ppc_pwl_points.shape[1]={ppc_pwl_points}.")
    c = (pts[:, 3::2] - pts[:, 1:-2:2]) / (pts[:, 2::2] - pts[:, :-2:2])
    return [construct_list_of_list(row) for row in range(pts.shape[0])]


def _add_to_gencost(gencost, n_rows, model):
    if gencost.shape[0] == n_rows:
        return gencost
    else:
        assert n_rows > gencost.shape[0]
        to_add = np.zeros((n_rows-gencost.shape[0], gencost.shape[1]))
        to_add[:, MODEL] = model
        return np.concatenate((gencost, to_add))


def _add_pwls_to_gencost(gencost, n_rows):
    return _add_to_gencost(gencost, n_rows, 1)


def _add_polys_to_gencost(gencost, n_rows):
    return _add_to_gencost(gencost, n_rows, 2)


def _from_ppc_gencost(net, ppc, gen_lookup, check=True):
    # --- gencost -> create polynomial_cost, piecewise_cost

    if 'gencost' not in ppc:
        return

    if len(ppc['gencost'].shape) == 1:
        # reshape gencost if only one gencost is given -> no indexError
        ppc['gencost'] = ppc['gencost'].reshape((1, -1))

    # construct glu2, gc2consider, n_glu
    n_glu = gen_lookup.shape[0]
    glu2 = pd.concat([gen_lookup, gen_lookup])
    if ppc['gencost'].shape[0] > 2*n_glu:
        logger.warning(
            f"There are {ppc['gencost'].shape[0]} gencost rows. Since only {n_glu} gens are "
            f"created only the first {2*n_glu} rows in gencost are considered (for p and q).")
        gc2consider = ppc['gencost'][:2*n_glu, :]
    else:
        gc2consider = ppc['gencost']

    # check gencost type
    if not all(np.isclose(gc2consider[:, MODEL], 1) | np.isclose(gc2consider[:, MODEL], 2)):
        raise ValueError("ppc['gencost'][:, 0] must be in [1, 2].")

    # construct gc4pwl, gc4poly, is_p
    gc4pwl = _add_polys_to_gencost(gc2consider, glu2.shape[0])
    gc4poly = _add_pwls_to_gencost(gc2consider, glu2.shape[0])
    is_p = np.array([True]*n_glu + [False]*n_glu)

    # --- create pwl costs
    is_pwl = gc4pwl[:, MODEL] == 1
    for is_, power_type in zip([is_p, ~is_p], ["p", "q"]):
        is_i = is_pwl & is_
        if not sum(is_i):
            continue
        if not np.allclose(2*gc4pwl[is_i, NCOST], gc4pwl[is_i, COST:].shape[1]):
            raise ValueError("In pwl gencost, the number n does not fit to the number of values")
        pp_pwl_points = _calc_pp_pwl_points(gc4pwl[is_i, 4:])
        create_pwl_costs(net, glu2.element.values[is_i], glu2.element_type.values[is_i],
                         pp_pwl_points, power_type=power_type, check=check)

    # --- create poly costs
    is_poly = np.any(np.isclose(np.c_[gc4poly[:n_glu, MODEL], gc4poly[n_glu:, MODEL]], 2), axis=1)
    is_poly2 = np.concatenate((is_poly, is_poly))

    if sum(is_poly):
        ncost = gc4poly[:, NCOST]
        if any(is_poly2 & (ncost > 3)):
            logger.warning("The pandapower poly_cost table only supports up to 2nd order " +
                           "polynomials. The ppc higher order polynomials cannot be converted.")
            ncost[is_poly2 & (ncost > 3), NCOST] = 3
        is_cost1 = np.isclose(ncost, 1)
        is_cost2 = np.isclose(ncost, 2)
        is_cost3 = np.isclose(ncost, 3)
        if not any(is_poly2 & (~is_cost1 | ~is_cost2 | ~is_cost3)):
            raise ValueError("'ncost' must be an positve integers.")
        poly_c = {key: np.zeros((2*n_glu, )) for key in ["c0", "c1", "c2"]}
        poly_c["c0"][is_poly2 & is_cost1] = gc4poly[is_poly2 & is_cost1, COST]
        if any(is_cost2):
            poly_c["c0"][is_poly2 & is_cost2] = gc4poly[is_poly2 & is_cost2, COST+1]
            poly_c["c1"][is_poly2 & is_cost2] = gc4poly[is_poly2 & is_cost2, COST  ]
        if any(is_cost3):
            poly_c["c0"][is_poly2 & is_cost3] = gc4poly[is_poly2 & is_cost3, COST+2]
            poly_c["c1"][is_poly2 & is_cost3] = gc4poly[is_poly2 & is_cost3, COST+1]
            poly_c["c2"][is_poly2 & is_cost3] = gc4poly[is_poly2 & is_cost3, COST  ]

        create_poly_costs(
            net,
            gen_lookup.element.values[is_poly],
            gen_lookup.element_type.values[is_poly],
            cp0_eur =         poly_c["c0"][is_poly2 & is_p],
            cp1_eur_per_mw =  poly_c["c1"][is_poly2 & is_p],
            cp2_eur_per_mw2 = poly_c["c2"][is_poly2 & is_p],
            cq0_eur =           poly_c["c0"][is_poly2 & ~is_p],
            cq1_eur_per_mvar =  poly_c["c1"][is_poly2 & ~is_p],
            cq2_eur_per_mvar2 = poly_c["c2"][is_poly2 & ~is_p],
            check=check)


def _validate_diff_res(diff_res, max_diff_values):
    val = True
    for et_val in ['gen_q_mvar', 'branch_p_mw', 'branch_q_mvar', 'gen_p_mw', 'bus_va_degree',
                   'bus_vm_pu']:
        if max_diff_values[et_val] is not None:
            et = et_val.split("_")[0]
            log_key = et if et != "gen" else "gen_p" if "p" in et_val else "gen_q_sum_per_bus"
            i_col = _log_dict(log_key)[0]
            val &= bool(np.max(abs(diff_res[log_key][:, i_col])) < max_diff_values[et_val])
    return val


def _gen_q_per_bus_sum(q_array, ppc):
    return pd.DataFrame(
        np.c_[q_array, ppc["gen"][:, GEN_BUS].astype(np.int64)],
        columns=["q_mvar", "bus"]).groupby("bus").sum()


def _log_dict(key=None):
    log_dict = {
        "bus": [[0, 1], ["voltage magnitude", "voltage angle"], ["pu", "degree"]],
        "branch": [[[0, 2], [1, 3]], ["branch flow active power", "branch flow reactive power"],
                   ["MW", "Mvar"]],
        "gen_p": [[0], ["active power generation"], ["MW"]],
        "gen_q_sum_per_bus": [[0], ["reactive power generation sum per bus"], ["Mvar"]]}
    if key is None:
        return log_dict
    else:
        return log_dict[key]


def validate_from_ppc(ppc, net, max_diff_values={"bus_vm_pu": 1e-6, "bus_va_degree": 1e-5,
                                                 "branch_p_mw": 1e-6, "branch_q_mvar": 1e-6,
                                                 "gen_p_mw": 1e-6, "gen_q_mvar": 1e-6}):
    """
    This function validates the conversion of a pypower case file (ppc) to a pandapower net.
    It compares the power flow calculation results which must be provided within the ppc and the net.

    INPUT:
        **ppc** - dict

        **net** - pandapower.pandapowerNet

        **max_diff_values** - dict, optional by default { "bus_vm_pu": 1e-6, "bus_va_degree": 1e-5, "branch_p_mw": 1e-6,
            "branch_q_mvar": 1e-6, "gen_p_mw": 1e-6, "gen_q_mvar": 1e-6}

    OUTPUT:
    **pf_match** - Whether the power flow results matches.

    EXAMPLES:
        >>> import pandapower
        >>> from pandapower.test.converter.test_from_ppc import get_testgrids
        >>> ppc = get_testgrids('pypower_cases', 'case4gs.json')
        >>> net = pandapower.converter.from_ppc(ppc, f_hz=50)
        >>> pandapower.runpp(net)
        >>> pf_match = pandapower.converter.validate_from_ppc(ppc, net)
    """
    if "_from_ppc_lookups" not in net.keys() or \
            ("gen" not in net._from_ppc_lookups.keys() and len(ppc["gen"]) > 0) or \
            ("branch" not in net._from_ppc_lookups.keys() and len(ppc["branch"]) > 0):
        raise ValueError(
            "net._from_ppc_lookups must contain a lookup (dict of keys 'branch' and 'gen')")

    if net.res_bus.shape[0] == 0 and net.bus.shape[0] > 0:
        logger.debug("runpp() is performed by validate_from_ppc() since res_bus is empty.")
        runpp(net, calculate_voltage_angles=True, trafo_model="pi")

    # --- pypower powerflow results -> ppc_res -----------------------------------------------------
    ppc_res = dict.fromkeys(ppc_elms)
    ppc_res["bus"] = ppc['bus'][:, 7:9]
    ppc_res["branch"] = ppc['branch'][:, 13:17]
    ppc_res["gen"] = ppc['gen'][:, 1:3]
    ppc_res["gen_p"] = ppc_res["gen"][:, :1]
    ppc_res["gen_q_sum_per_bus"] = _gen_q_per_bus_sum(ppc_res["gen"][:, -1:], ppc)

    # --- pandapower powerflow results -> pp_res ---------------------------------------------------
    pp_res = dict.fromkeys(ppc_elms)

    # --- bus
    pp_res["bus"] = net.res_bus.loc[ppc["bus"][:, BUS_I].astype(np.int64), ['vm_pu', 'va_degree']].values

    # --- branch
    pp_res["branch"] = np.zeros(ppc_res["branch"].shape)
    from_to_buses = -np.ones((ppc_res["branch"].shape[0], 2), dtype=np.int64)
    for et in net._from_ppc_lookups["branch"].element_type.unique():
        if et == "line":
            from_to_cols = ["from_bus", "to_bus"]
            res_cols = ['p_from_mw', 'q_from_mvar', 'p_to_mw', 'q_to_mvar']
        elif et == "trafo":
            from_to_cols = ["hv_bus", "lv_bus"]
            res_cols = ['p_hv_mw', 'q_hv_mvar', 'p_lv_mw', 'q_lv_mvar']
        else:
            raise NotImplementedError(
                f"result columns for element type {et} are not implemented.")
        is_et = net._from_ppc_lookups["branch"].element_type == et
        pp_res["branch"][is_et] += net[f"res_{et}"].loc[
            net._from_ppc_lookups["branch"].element.loc[is_et], res_cols].values
        from_to_buses[is_et] = net[et].loc[
            net._from_ppc_lookups["branch"].element.loc[is_et], from_to_cols].values

    # switch direction as in ppc
    correct_from_to = np.all(from_to_buses == ppc["branch"][:, F_BUS:T_BUS+1].astype(np.int64), axis=1)
    switch_from_to = np.all(from_to_buses[:, ::-1] == ppc["branch"][:, F_BUS:T_BUS+1].astype(
        np.int64), axis=1)
    if not np.all(correct_from_to | switch_from_to):
        raise ValueError("ppc branch from and to buses don't fit to pandapower from and to + "
                        "hv and lv buses.")
    if np.any(switch_from_to):
        pp_res["branch"][switch_from_to, :] = pp_res["branch"][switch_from_to, :][:, [2, 3, 0, 1]]

    # --- gen
    pp_res["gen"] = np.zeros(ppc_res["gen"].shape)
    res_cols = ['p_mw', 'q_mvar']
    for et in net._from_ppc_lookups["gen"].element_type.unique():
        is_et = net._from_ppc_lookups["gen"].element_type == et
        pp_res["gen"][is_et] += net[f"res_{et}"].loc[
            net._from_ppc_lookups["gen"].element.loc[is_et], res_cols].values

    pp_res["gen_p"] = ppc_res["gen"][:, :1]
    pp_res["gen_q_sum_per_bus"] = _gen_q_per_bus_sum(pp_res["gen"][:, -1:], ppc)

    # --- log maximal differences the powerflow result comparison
    diff_res = dict()
    comp_keys = ["bus", "branch", "gen_p", "gen_q_sum_per_bus"]
    for comp_key in comp_keys:
        diff_res[comp_key] = ppc_res[comp_key] - pp_res[comp_key]
        if isinstance(diff_res[comp_key], pd.DataFrame):
            diff_res[comp_key] = diff_res[comp_key].values
        for i_col, var_str, unit in zip(*_log_dict(comp_key)):
            diff = diff_res[comp_key][:, i_col]
            logger.debug(f"Maximum {var_str} difference between pandapower and pypower: "
                         "%.2e %s" % (np.max(abs(diff)), unit))

    # --- do the powerflow result comparison -------------------------------------------------------
    pf_match = _validate_diff_res(diff_res, max_diff_values)

    # --- case of missmatch: result comparison with different max_diff (unwanted behaviour of
    # pypower possible)
    if not pf_match:
        other_max_diff = {
            "bus_vm_pu": 1e-3, "bus_va_degree": 1e-3, "branch_p_mw": 1e-6, "branch_q_mvar": 1e-6,
            "gen_p_mw": None, "gen_q_mvar": None}
        if _validate_diff_res(diff_res, other_max_diff) and \
                (np.max(abs(pp_res["gen"] - ppc_res["gen"])) > 1e-1).any():
            logger.debug("The active/reactive power generation difference possibly results "
                        "because of a pypower error. Please validate "
                        "the results via pypower loadflow.")  # this occurs e.g. at ppc case9

    return pf_match


if __name__ == "__main__":
    pass
