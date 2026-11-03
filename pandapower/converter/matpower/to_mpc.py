# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

__all__ = ["to_mpc"]

import copy
import logging

import numpy as np
from scipy.io import savemat

from pandapower.converter.pypower import to_ppc

logger = logging.getLogger(__name__)


def to_mpc(net, filename=None, **kwargs):
    """
    This function converts a pandapower net to a matpower case files (.mat) version 2.
    Note: python is 0-based while Matlab is 1-based.

    Parameters:
        net: The pandapower net.
        filename (str, None): File path + name of the mat file which will be created. If None the mpc will only be
            returned

    Keyword Arguments:
        any: are passed to :func:`to_ppc`

    EXAMPLE:
        >>> from pandapower.converter.matpower import to_mpc
        >>> from pandapower.networks.power_system_test_cases import case9
        >>>
        >>> net = case9()
        >>> to_mpc(net, "case9.mat")
    """
    ppc = to_ppc(net, **kwargs)

    mpc = {}
    mpc["mpc"] = _ppc2mpc(ppc, trafo_branches=_trafo_branch_mask(net, ppc))
    if filename is not None:
        # savemat
        savemat(filename, mpc)

    return mpc


def _trafo_branch_mask(net, ppc):
    """
    Boolean mask of the branch rows of ppc that belong to transformer elements (trafo and
    trafo3w), based on the branch lookup that was created when the net was converted to ppc.

    The lookup refers to the rows of the full ppc, while the ppc that is converted here is
    the ppci returned by to_ppc, which only contains the in-service branches. The mask is
    therefore built in the coordinates of the full ppc first and then projected onto the
    rows of ppc with the same "branch_is" selection that created the ppci.
    """
    branch_is = ppc.get("internal", {}).get("branch_is")
    n_branch = len(branch_is) if branch_is is not None else ppc["branch"].shape[0]
    mask = np.zeros(n_branch, dtype=bool)
    try:
        branch_lookup = net["_pd2ppc_lookups"]["branch"]
    except (KeyError, TypeError):
        branch_lookup = {}
    for element in ("trafo", "trafo3w"):
        if element in branch_lookup:
            start, end = branch_lookup[element]
            mask[start:end] = True
    return mask[branch_is] if branch_is is not None else mask


def _ppc2mpc(ppc, trafo_branches=None):
    """
    Convert network in Pypower/Matpower format
    Convert 0-based python to 1-based Matlab

    **INPUT**:
        * net - The pandapower format network
        * filename - File path + name of the mat file which is created
    """

    # convert to matpower
    # Matlab is one-based, so all entries (buses, lines, gens) have to start with 1 instead of 0
    mpc = copy.deepcopy(ppc)
    if np.any(mpc["bus"][:, 0] == 0):
        mpc["bus"][:, 0] = mpc["bus"][:, 0] + 1
        mpc["gen"][:, 0] = mpc["gen"][:, 0] + 1
        mpc["branch"][:, 0:2] = mpc["branch"][:, 0:2] + 1
    # adjust for the matpower converter -> taps should be 0 when there is no transformer, but are 1.
    # Transformer branches keep their tap ratio, since in the matpower format a ratio of 1 denotes a
    # transformer without a tap changer (only a ratio of 0 denotes that there is no transformer).
    is_tap_1 = mpc["branch"][:, 8] == 1
    if trafo_branches is not None:
        is_tap_1 &= ~trafo_branches
    mpc["branch"][is_tap_1, 8] = 0
    # version is a string
    mpc["version"] = str(mpc["version"])
    # baseMVA has to be a float instead of int
    mpc["baseMVA"] = mpc["baseMVA"] * 1.0
    return mpc
