# Builds the DC PSDF matrix based on the DC PTDF
import scipy as sp
from math import pi
from scipy.sparse import csr_matrix, csc_matrix

from pandapower.analysis.LODF import _LODF_ppci_to_pp, _LODF_pp_np_to_df
from pandapower.analysis.PTDF import _makePTDF_ppci
from pandapower.pypower.idx_brch import F_BUS, T_BUS
from pandapower.pypower.idx_bus import BUS_TYPE, REF
from pandapower.pypower.makeBdc import calc_b_from_branch
from numpy import ones, r_, real, int64, arange, flatnonzero as find, isscalar

from typing import Union, List, Dict, Tuple

import pandas as pd
import numpy as np

from pandapower import pandapowerNet
from pandapower.analysis.utils import branch_dict_to_ppci_branch_list, _get_outage_branch_ix, ELE_IX_TYPE

import logging
logger = logging.getLogger(__name__)


def makePSDF(baseMVA, PTDF, bus, branch, using_sparse_solver=False, branch_id=None, reduced=False, slack=None):
    """Builds the DC PSDF matrix based on the DC PTDF
    Returns the DC PSDF matrix . The matrix is
    C{nbr x nbr}, where C{nbr} is the number of branches. The DC PSDF is independent from the selected slack.
    To restrict the PSDF computation to a subset of branches, supply a list of ppci branch indices in C{branch_id}.
    If C{reduced==True}, the output is reduced to the branches given in C{branch_id}, otherwise the complement rows are set to NaN.
    @see: L{makeLODF}
    """
    if reduced and not branch_id:
        raise ValueError("'reduced=True' is only valid if branch_id is not None")

    ## Select csc/csr B matrix
    sparse = csr_matrix if using_sparse_solver else csc_matrix

    ## use reference bus for slack by default
    if slack is None:
        slack = find(bus[:, BUS_TYPE] == REF)
        slack = slack[0]

    ## set the slack bus to be used to compute initial PTDF
    if isscalar(slack):
        slack_bus = slack
    else:
        slack_bus = 0  ## use bus 1 for temp slack bus

    ## constants
    nb = bus.shape[0]  ## number of buses
    nl = branch.shape[0]  ## number of lines
    noref = arange(1, nb)  ## use bus 1 for voltage angle reference
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

    PSDF = Bd - PTDF[:, noslack] * (Cft.T * Bd)
    PSDF = PSDF * (pi / 180 * baseMVA)
    return PSDF


def _get_PSDF_direct(
    net,
    phase_shift_branch_type,
    phase_shift_branch_ix=None,
    using_sparse_solver=True,
    random_verify=False,
    branch_dict=None,
    reduced=True,
):
    """
    this function calculate PSDF of a pp branch from the angle shift of 1 degree
    with pypower matrix function.
    """
    if net.bus.shape[0] > 3000 and not using_sparse_solver:
        logger.warning("Calculating lodf for large network, switched to sparse solver!")

    # If branch_dict not None compute list of ppci branch indices and its branch type intervals as lookup
    branch_ppci_lookup = None
    branch_id = None
    if branch_dict is not None:
        branch_id, branch_ppci_lookup = branch_dict_to_ppci_branch_list(net=net, branch_dict=branch_dict)
    else:
        reduced = False

    ptdf_ppci, ppci = _makePTDF_ppci(
        net, using_sparse_solver=using_sparse_solver, result_side=0, branch_id=branch_id, reduced=reduced
    )

    # Set super small value to 0 for better numerical stability
    ptdf_ppci[np.isclose(ptdf_ppci, 0, atol=1e-10)] = 0

    # Create psdf ppci with ptdf ppci

    psdf_ppci = makePSDF(
        ppci["baseMVA"],
        ptdf_ppci,
        ppci["bus"],
        ppci["branch"],
        using_sparse_solver=using_sparse_solver,
        branch_id=branch_id,
        reduced=reduced,
    )

    # Checkout ppci lodf to pp level
    if reduced:
        psdf_pp_np = _LODF_ppci_to_pp(net, psdf_ppci, branch_ppci_lookup=branch_ppci_lookup)
    else:
        psdf_pp_np = _LODF_ppci_to_pp(net, psdf_ppci)

    # lodf pp contains all data
    # Convert numpy array to pandas dataframe with the pandapower element index
    if reduced:
        psdf = _LODF_pp_np_to_df(net, psdf_pp_np, branch_dict=branch_dict)
    else:
        psdf = _LODF_pp_np_to_df(net, psdf_pp_np)

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


def _get_PSDF_perturb(
    net: pandapowerNet,
    phase_shift_branch_type: str,
    phase_shift_branch_ix: ELE_IX_TYPE = None,
    distributed_slack=True,
    recycle="lodf",
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    this function calculates PSDF (ratio without unit) of branch to
    a pp branch with perturb method (brute-force)
    """
    raise NotImplementedError()


def run_PSDF(
    net: pandapowerNet,
    phase_shift_branch_type: Union[None, str],
    phase_shift_branch_ix: ELE_IX_TYPE = None,
    distributed_slack: bool = True,
    perturb: bool = False,
    recycle: Union[str, None] = None,
    using_sparse_solver: bool = True,
    branch_dict: Dict[str, Union[List[int], None]] = None,
    reduced: bool = True,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    this function is a wrapper of calculating PSDF of a pp branch from the phase shift through a pp branch
    with pypower matrix function or perturb function.

    The PSDF is defined as: (p_{side}_mw_new - p_{side}_mw_old) / 1 degree
    Side corresponds to the pandapower results definition, for PSDF calculation both
    sides give the same result, thus no side definition required

    :param net: A pandapower network
    :param phase_shift_branch_type: The name of the type of the phase shift branch ("line", "trafo", "impedance")
    :param phase_shift_branch_ix: The pandapower index of the phase shift branch (int/list/np.ndarray), if None then all branches
        will be used
    :param distributed_slack: Set True if p distribution amount distributed wished, or else slacks are
         only all voltage references! For non-perturb only False possible!!
    :param perturb: Set True to use the perturb version (brute-force) which is faster for calculating
        only a few elements on large networks, if a lot of elements required please set to False
    :param using_sparse_solver: Select whether sparse linear system should be used (more efficient for large network)
    :param branch_dict: dictionary with keys "line", "trafo", "impedance", "trafo3w"; if not None the computation is
        restricted to the branch indices given in the dict
    :param reduced: if True, the output is reduced to the branches given in branch_dict
    :return: {(goal_branch_type ("line", "trafo", "impedance", "trafo3w_{hv,mv,lv}"),
               phase_shift_branch_type (("line", "trafo", "impedance")):
        DataFrame(data=psdf, index=goal_branch_pp_index, columns=phase_shift_branch_ix)}
    """
    # ToDo: check if distributed slack makes any difference
    if perturb and phase_shift_branch_type is None:
        logger.info("If a lot of branch required in psdf, please set perturb to False!")

    if perturb:
        if recycle == "lodf" and distributed_slack == True:
            logger.warning("distributed_slack deactivated! recycling does not allow distributed slack")
        psdf = _get_PSDF_perturb(
            net,
            phase_shift_branch_type=phase_shift_branch_type,
            phase_shift_branch_ix=phase_shift_branch_ix,
            distributed_slack=distributed_slack,
            recycle=recycle,
        )
    else:
        if distributed_slack:
            logger.warning("distributed_slack deactivated! Distirbuted slacks are used as Vref! Only Perturb Possible")
        psdf = _get_PSDF_direct(
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


