# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

__all__ = [
    "make_psdf",
    "run_psdf",
]

# Builds the DC PSDF matrix based on the DC PTDF
import logging
from copy import deepcopy
from math import pi

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
from numpy import arange, int64, isscalar, ones, r_, real
from numpy import flatnonzero as find
from scipy.sparse import csc_matrix, csr_matrix

from pandapower import pandapowerNet
from pandapower.pypower.idx_brch import F_BUS, T_BUS
from pandapower.pypower.idx_bus import BUS_TYPE, REF
from pandapower.pypower.make_b_dc import calc_b_from_branch
from pandapower.run import rundcpp
from pandapower.sensitivity.lodf import _init_lodf_pp_np, _lodf_pp_np_to_df, _lodf_ppci_to_pp
from pandapower.sensitivity.ptdf import _make_ptdf_ppci
from pandapower.sensitivity.utils import (
    BR_SIDE_MAPPING,
    ELE_IX_TYPE,
    _get_outage_branch_ix,
    branch_dict_to_ppci_branch_list,
)

logger = logging.getLogger(__name__)


def make_psdf(
    baseMVA: float,
    ptdf: npt.NDArray,
    bus: npt.NDArray,
    branch: npt.NDArray,
    using_sparse_solver: bool = False,
    branch_id: list[int] | None = None,
    reduced: bool = False,
    slack: int | npt.NDArray | None = None,
) -> npt.NDArray:
    """Build the DC PSDF matrix based on the DC PTDF.

    Returns the DC PSDF matrix. The matrix is
    C{nbr x nbr}, where C{nbr} is the number of branches. The DC PSDF is independent from the selected slack.
    Entry C{(l, k)} is the change of the active power flow (from side, in MW) on branch C{l} if the phase shift
    angle C{SHIFT} of branch C{k} is increased by 1 degree. C{SHIFT} has the same sign as the pandapower
    C{shift_degree}.
    To restrict the PSDF computation to a subset of branches, supply a list of ppci branch indices in C{branch_id}.
    If C{reduced==True}, the output is reduced to the branches given in C{branch_id}.
    @see: L{makeLODF}
    """
    if reduced and branch_id is None:
        raise ValueError("'reduced=True' is only valid if branch_id is not None")

    ## Select csc/csr B matrix
    sparse = csr_matrix if using_sparse_solver else csc_matrix

    ## use reference bus for slack by default
    if slack is None:
        slack = find(bus[:, BUS_TYPE] == REF)[0]

    ## set the slack bus to be used to compute initial PTDF
    if isscalar(slack):
        slack_bus = slack
    else:
        slack_bus = 0  ## use bus 1 for temp slack bus

    ## constants
    nb = bus.shape[0]  ## number of buses
    nl = branch.shape[0]  ## number of lines
    # noref = arange(1, nb)  ## use bus 1 for voltage angle reference
    noslack = find(arange(nb) != slack_bus)

    ## build connection matrix Cft = Cf - Ct for line and from - to buses
    f = real(branch[:, F_BUS]).astype(int64)  ## list of "from" buses
    t = real(branch[:, T_BUS]).astype(int64)  ## list of "to" buses
    i = r_[range(nl), range(nl)]  ## double set of row indices

    ## connection matrix
    Cft = sparse((r_[ones(nl), -ones(nl)], (i, r_[f, t])), (nl, nb))[:, noslack]

    b = calc_b_from_branch(branch, nl)

    if reduced:
        b = b[branch_id]
        Cft = Cft[branch_id, :]  # Zweige x Knoten

    Bd = sp.sparse.diags(b.real)

    PSDF = ptdf[:, noslack] * (Cft.T * Bd) - Bd
    PSDF = PSDF * (pi / 180 * baseMVA)
    return np.asarray(PSDF)


def _get_psdf_direct(
    net: pandapowerNet,
    phase_shift_branch_type: str | None,
    phase_shift_branch_ix: ELE_IX_TYPE | None = None,
    using_sparse_solver: bool = True,
    random_verify=False,
    branch_dict: dict[str, list[int] | None] | None = None,
    reduced=True,
):
    """Calculate the PSDF of a pp branch from the angle shift of 1 degree with pypower matrices."""
    if net.bus.shape[0] > 3000 and not using_sparse_solver:
        logger.warning("Calculating lodf for large network, switched to sparse solver!")
        using_sparse_solver = True

    # If branch_dict not None compute list of ppci branch indices and its branch type intervals as lookup
    branch_ppci_lookup: dict | None = None
    branch_id: list[int] | None = None
    if branch_dict is not None:
        branch_id, branch_ppci_lookup = branch_dict_to_ppci_branch_list(net=net, branch_dict=branch_dict)
    else:
        reduced = False

    ptdf_ppci, ppci = _make_ptdf_ppci(
        net, using_sparse_solver=using_sparse_solver, result_side=0, branch_id=branch_id, reduced=reduced
    )

    # Set super small value to 0 for better numerical stability
    ptdf_ppci[np.isclose(ptdf_ppci, 0, atol=1e-10)] = 0

    # Create psdf ppci with ptdf ppci

    psdf_ppci = make_psdf(
        ppci["baseMVA"],
        ptdf_ppci,
        ppci["bus"],
        ppci["branch"],
        using_sparse_solver=using_sparse_solver,
        branch_id=branch_id,
        reduced=reduced,
    )
    if branch_id is not None and not reduced:
        # all ppci branches that are not selected in branch_dict
        branch_id_complement = np.setdiff1d(np.arange(psdf_ppci.shape[0]), branch_id)
        psdf_ppci[:, branch_id_complement] = np.nan
        psdf_ppci[branch_id_complement, :] = np.nan

    # Checkout ppci lodf to pp level
    # lodf pp contains all data
    # Convert numpy array to pandas dataframe with the pandapower element index
    if reduced:
        psdf_pp_np = _lodf_ppci_to_pp(net, psdf_ppci, branch_ppci_lookup=branch_ppci_lookup)
        psdf = _lodf_pp_np_to_df(net, psdf_pp_np, branch_dict=branch_dict)
    else:
        psdf_pp_np = _lodf_ppci_to_pp(net, psdf_ppci)
        psdf = _lodf_pp_np_to_df(net, psdf_pp_np)

    # Select only required data points according to the outage_branch_type
    if phase_shift_branch_type is not None:
        outage_branch_ix = _get_outage_branch_ix(net, phase_shift_branch_type, phase_shift_branch_ix)
        if reduced:
            psdf = {key: value for key, value in psdf.items() if key[1] == phase_shift_branch_type}
        else:
            psdf = {
                key: value.loc[:, outage_branch_ix] for key, value in psdf.items() if key[1] == phase_shift_branch_type
            }

    return psdf


def _get_psdf_perturb(
    net: pandapowerNet,
    phase_shift_branch_type: str | None,
    phase_shift_branch_ix: ELE_IX_TYPE | None = None,
    distributed_slack=True,
    recycle: str | None = "lodf",
) -> dict[tuple[str, str], pd.DataFrame]:
    """Calculate the PSDF (MW per degree) of pp branches from the phase shift of a trafo with the perturb method.

    The shift_degree of each selected trafo is increased by 1 degree, one DC power flow is run and the change of the
    branch flows is stored.
    """
    if phase_shift_branch_type != "trafo":
        raise ValueError(
            "The perturb method is only available for phase_shift_branch_type='trafo' (shift_degree), "
            f"{phase_shift_branch_type!r} has no phase shift parameter in pandapower. Use perturb=False instead!"
        )

    # Using net_mod for the base case and net_mod_shift for the phase shift
    net_mod = deepcopy(net)
    rundcpp(net_mod, distributed_slack=distributed_slack, recycle=recycle)
    net_mod_shift = deepcopy(net_mod)

    phase_shift_branch_ix = _get_outage_branch_ix(net_mod, phase_shift_branch_type, phase_shift_branch_ix)

    # Init psdf array with NaN, Using Numpy array for better performance
    psdf_pp_np = _init_lodf_pp_np(net_mod, phase_shift_branch_type, phase_shift_branch_ix.shape[0])

    for ix, br_ix in enumerate(phase_shift_branch_ix):
        if not net[phase_shift_branch_type].at[br_ix, "in_service"]:
            # Skip out of service branch, its PSDF stays NaN
            continue

        # Increase the phase shift by 1 degree, run the power flow and reset the phase shift
        shift_degree = net_mod_shift.trafo.at[br_ix, "shift_degree"]
        net_mod_shift.trafo.at[br_ix, "shift_degree"] = shift_degree + 1.0
        rundcpp(net_mod_shift, distributed_slack=distributed_slack)
        net_mod_shift.trafo.at[br_ix, "shift_degree"] = shift_degree

        # calculate the PSDF factor: change of the branch flows per degree
        for br_type in ("line", "dcline", "trafo", "impedance"):
            if not net[br_type].empty:
                br_res_type = "p_" + BR_SIDE_MAPPING[br_type] + "_mw"
                psdf_pp_np[(br_type, phase_shift_branch_type)][:, ix] = (
                    net_mod_shift["res_" + br_type][br_res_type].to_numpy()
                    - net_mod["res_" + br_type][br_res_type].to_numpy()
                )

        if not net.trafo3w.empty:
            for side in ("hv", "mv", "lv"):
                br_res_type = "p_" + side + "_mw"
                psdf_pp_np[("trafo3w_" + side, phase_shift_branch_type)][:, ix] = (
                    net_mod_shift["res_trafo3w"][br_res_type].to_numpy()
                    - net_mod["res_trafo3w"][br_res_type].to_numpy()
                )

    psdf = _lodf_pp_np_to_df(
        net, psdf_pp_np, outage_branch_type=phase_shift_branch_type, outage_branch_ix=phase_shift_branch_ix
    )
    return psdf


def run_psdf(
    net: pandapowerNet,
    phase_shift_branch_type: str | None = "trafo",
    phase_shift_branch_ix: ELE_IX_TYPE | None = None,
    distributed_slack: bool = True,
    perturb: bool = False,
    recycle: str | None = None,
    using_sparse_solver: bool = True,
    branch_dict: dict[str, list[int] | None] | None = None,
    reduced: bool = True,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Calculate the PSDF of pp branches from the phase shift through a pp branch.

    The PSDF is calculated with pypower matrices or with the perturb method.

    The PSDF is defined as: (p_{side}_mw_new - p_{side}_mw_old) / 1 degree
    Side corresponds to the pandapower results definition, for PSDF calculation both
    sides give the same result, thus no side definition required.
    A positive PSDF means the flow increases if shift_degree increases.
    A phase shifting tap changer (tap_step_degree) on the hv side has the same sign; on the lv side pandapower applies
    the tap angle with the opposite sign, so there the flow change per tap step is -PSDF * tap_step_degree.

    :param net: A pandapower network
    :param phase_shift_branch_type: The name of the type of the phase shift branch. Default "trafo".
    :param phase_shift_branch_ix: The pandapower index of the phase shift branch (int/list/np.ndarray), if None then all branches
        will be used
    :param distributed_slack: Only passed to the DC power flows of the perturb method, where it currently has no
        effect, since rundcpp does not support a distributed slack. The PSDF does not depend on the slack, since a
        phase shift does not change the injections. The direct method logs a warning if this is True, so pass False
        with perturb=False.
    :param perturb: Set True to use the perturb version (brute-force): the shift_degree of each selected trafo is
        increased by 1 degree and a DC power flow is run. Faster for only a few elements on large networks, if a lot
        of elements required please set to False
    :param using_sparse_solver: Select whether sparse linear system should be used (more efficient for large network)
    :param branch_dict: dictionary with keys "line", "trafo", "impedance", "trafo3w"; if not None the computation is
        restricted to the branch indices given in the dict
    :param reduced: if True, the output is reduced to the branches given in branch_dict
    :return: {(goal_branch_type, phase_shift_branch_type):
        DataFrame(data=psdf, index=goal_branch_pp_index, columns=phase_shift_branch_ix)}
        with goal_branch_type in ("line", "trafo", "impedance", "trafo3w_{hv,mv,lv}")
        and phase_shift_branch_type in ("line", "trafo", "impedance")
    """
    # ToDo: check if distributed slack makes any difference

    if perturb:
        if phase_shift_branch_ix is None:
            logger.info("If a lot of branch required in psdf, please set perturb to False!")
        if recycle == "lodf" and distributed_slack == True:
            logger.warning("distributed_slack deactivated! recycling does not allow distributed slack")
        psdf = _get_psdf_perturb(
            net,
            phase_shift_branch_type=phase_shift_branch_type,
            phase_shift_branch_ix=phase_shift_branch_ix,
            distributed_slack=distributed_slack,
            recycle=recycle,
        )
    else:
        if distributed_slack:
            logger.warning("distributed_slack deactivated! Distirbuted slacks are used as Vref! Only Perturb Possible")
        psdf = _get_psdf_direct(
            net,
            phase_shift_branch_type=phase_shift_branch_type,
            phase_shift_branch_ix=phase_shift_branch_ix,
            using_sparse_solver=using_sparse_solver,
            branch_dict=branch_dict,
            reduced=reduced,
        )

    # Update psdf on net
    net._psdf = {"branch": {"table": phase_shift_branch_type, "element": None}}
    for (br_type, _), data in psdf.items():
        if net._psdf["branch"]["element"] is None:
            net._psdf["branch"]["element"] = data.columns.to_numpy(copy=True)
        net["psdf_" + br_type] = data
    return psdf
