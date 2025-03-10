# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

# Builds the line outage distribution factor matrix.


from numpy import ones, diag, r_, arange
import numpy as np
from scipy.sparse import csr_matrix as sparse

from pandapower.pypower.idx_brch import F_BUS, T_BUS
from pandapower.auxiliary import version_check

try:
    from numba import jit
    version_check('numba')
except ImportError: # pragma: no cover
    from pandapower.pf.no_numba import jit



@jit(nopython=True)
def update_LODF_diag(LODF): # pragma: no cover
    for ix in range(LODF.shape[0]):
        # To preserve the data type of diagnol elments
        LODF[ix, ix] -= (LODF[ix, ix] + 1.)


def makeLODF(branch, PTDF):
    """Builds the line outage distribution factor matrix.

    Returns the DC line outage distribution factor matrix for a given PTDF.
    The matrix is C{nbr x nbr}, where C{nbr} is the number of branches.

    Example::
        H = makePTDF(baseMVA, bus, branch)
        LODF = makeLODF(branch, H)

    @see: L{makePTDF}

    @author: Ray Zimmerman (PSERC Cornell)
    """
    nl, nb = PTDF.shape
    f = np.real(branch[:, F_BUS])
    t = np.real(branch[:, T_BUS])
    Cft = sparse((r_[ones(nl), -ones(nl)],
                      (r_[f, t], r_[arange(nl), arange(nl)])), (nb, nl))

    H = PTDF * Cft
    h = diag(H, 0)
    # Avoid zero division error
    # Implies a N-1 contingency (No backup branch)
    den = (ones((nl, 1)) * h.T * -1 + 1.)

    # Silence warning caused by np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        LODF = (H / den)

    # LODF = LODF - diag(diag(LODF)) - eye(nl, nl)
    update_LODF_diag(LODF)

    return LODF


def makeOTDF(PTDF, LODF, outage_branches):
    """
    Compute the Outage Transfer Distribution Factors (OTDF) matrix.

    This function creates the OTDF matrix that relates bus power injections
    to branch flows for specified outage scenarios. It's essential that
    outage branches do not lead to isolated nodes or disconnected islands
    in the grid.

    The grid cannot have isolated nodes or disconnected islands. Use the
    pandapower.topology module to identify branches that, if outaged, would
    lead to isolated nodes (determine_stubs) or islands (find_graph_characteristics).

    The resulting matrix has a width equal to the number of nodes and a length
    equal to the number of outage branches multiplied by the total number of branches.
    The dot product of OTDF and the bus power vector in generation reference frame
    (positive for generation, negative for consumption - the opposite of res_bus.p_mw)
    yields an array with outage branch power flows for every outage scenario,
    facilitating the analysis of all outage scenarios under a DC
    power flow approximation.

    Parameters
    ----------
    PTDF : numpy.ndarray
        The Power Transfer Distribution Factor matrix, defining the sensitivity
        of branch flows to bus power injections.
    LODF : numpy.ndarray
        The Line Outage Distribution Factor matrix, describing how branch flows
        are affected by outages of other branches.
    outage_branches : list or numpy.ndarray
        Indices of branches for which outage scenarios are to be considered.

    Returns
    -------
    OTDF : numpy.ndarray
        The Outage Transfer Distribution Factor matrix. Rows correspond to
        outage scenarios, and columns correspond to branch flows.

    Examples
    --------
    >>> H = makePTDF(baseMVA, bus, branch)
    >>> LODF = makeLODF(branch, H)
    >>> outage_branches = [0, 2]  # Example branch indices for outage scenarios
    >>> OTDF = makeOTDF(H, LODF, outage_branches)
    >>> # To obtain a 2D array with the outage results:
    >>> outage_results = (OTDF @ Pbus).reshape(len(outage_branches), -1)

    Notes
    -----
    - The function assumes a DC power flow model.
    - Ensure that the specified outage branches do not lead to grid
      disconnection or isolated nodes.
    """
    OTDF = np.vstack([PTDF + LODF[:, [i]] @ PTDF[[i], :] for i in outage_branches])
    return OTDF


def outage_results_OTDF(OTDF, Pbus, outage_branches):
    """
    Calculate the branch power flows for each outage scenario based on the given
    Outage Transfer Distribution Factors (OTDF), bus power injections (Pbus), and
    specified outage branches.

    This function computes how branch flows are affected under N-1 contingency
    scenarios (i.e., for each branch outage specified). It uses the OTDF matrix and
    the bus power vector (Pbus) to determine the branch flows in each outage scenario.

    Pbus should represent the net power at each bus in the generation reference case.

    Parameters
    ----------
    OTDF : numpy.ndarray
        The Outage Transfer Distribution Factor matrix, which relates bus power
        injections to branch flows under specific outage scenarios. Its shape
        should be (num_outage_scenarios * num_branches, num_buses).
    Pbus : numpy.ndarray
        A vector representing the net power injections at each bus. Positive values
        for generation, negative for consumption. Its length should be equal to
        the total number of buses.
    outage_branches : numpy.ndarray
        An array of indices representing the branches that are outaged in each
        scenario. Its length should be equal to the number of outage scenarios.

    Returns
    -------
    numpy.ndarray
        A 2D array where each row corresponds to an outage scenario and each column
        represents the resulting power flow in a branch. The number of rows is equal
        to the number of outage scenarios, and the number of columns is equal to the
        number of branches.

    Examples
    --------
    >>> OTDF = np.array([...])  # example OTDF matrix
    >>> Pbus = np.array([...])  # example bus power vector
    >>> outage_branches = np.array([...])  # example outage branches
    >>> branch_flows = outage_results_OTDF(OTDF,Pbus,outage_branches)

    Notes
    -----
    The function assumes a linear relationship between bus power injections and
    branch flows, which is typical in DC power flow models.
    """
    # get branch flows as an array first:
    nminus1_otdf = (OTDF @ Pbus.reshape(-1, 1))
    # reshape to a 2D array with rows relating to outage scenarios and columns to
    # the resulting branch power flows
    nminus1_otdf = nminus1_otdf.reshape(outage_branches.shape[0], -1)
    return nminus1_otdf




