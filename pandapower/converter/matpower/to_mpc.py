# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import copy

import numpy as np
from scipy.io import savemat

from pandapower.auxiliary import _add_ppc_options
from pandapower.powerflow import reset_results, _pd2ppc

try:
    import pplog as logging
except:
    import logging

logger = logging.getLogger(__name__)

def to_mpc(net, filename=None, init="results", calculate_voltage_angles=False, trafo_model="t", mode = "pf"):
    """
    This function converts a pandapower net to a matpower case files (.mat) version 2.
    Note: python is 0-based while Matlab is 1-based.

    INPUT:

        **net** - The pandapower net.

    OPTIONAL:

        **filename** (None) - File path + name of the mat file which will be created. If None the mpc will only be returned

        **init** (str, "results") - initialization method of the loadflow
        For the conversion to a mpc, the following options can be chosen:

            - "flat"- flat start with voltage of 1.0pu and angle of 0° at all buses as initial solution
            - "results" - voltage vector of last loadflow from net.res_bus is copied to the mpc

        **calculate_voltage_angles** (bool, False) - copy the voltage angles from pandapower to the mpc

            If True, voltage angles are copied from pandapower to the mpc. In some cases with
            large differences in voltage angles (for example in case of transformers with high
            voltage shift), the difference between starting and end angle value is very large.
            In this case, the loadflow might be slow or it might not converge at all. That is why
            the possibility of neglecting the voltage angles of transformers and ext_grids is
            provided to allow and/or accelarate convergence for networks where calculation of
            voltage angles is not necessary.

            The default value is False because pandapower was developed for distribution networks.
            Please be aware that this parameter has to be set to True in meshed network for correct
            results!

        **trafo_model** (str, "t")  - transformer equivalent circuit model
        pandapower provides two equivalent circuit models for the transformer:

            - "t" - transformer is modelled as equivalent with the T-model. This is consistent with PowerFactory and is also more accurate than the PI-model. We recommend using this transformer model.
            - "pi" - transformer is modelled as equivalent PI-model. This is consistent with Sincal, but the method is questionable since the transformer is physically T-shaped. We therefore recommend the use of the T-model.

    EXAMPLE:

        import pandapower.converter as pc

        import pandapower.networks as pn

        net = pn.case9()

        pc.to_mpc(net)

    """
    # convert to matpower
    net["converged"] = False

    if not init == "results":
        reset_results(net)



    # select elements in service (time consuming, so we do it once)
    _get_std_options(net, init, calculate_voltage_angles, trafo_model)
    net["_options"]["mode"] = mode
    if mode == "opf":
        net["_options"]["copy_constraints_to_ppc"] = True
    # convert pandapower net to ppc
    ppc, ppci = _pd2ppc(net)

    # convert ppc to mpc
    if mode == "opf":
        ppc["gencost"] = ppci["gencost"]

    mpc = _ppc_to_mpc(ppc)
    if filename is not None:
        # savemat
        savemat(filename, mpc)

    return mpc


def _ppc_to_mpc(ppc):
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
    if len(np.where(mpc["bus"][:, 0] == 0)[0]):
        mpc["bus"][:, 0] = mpc["bus"][:, 0] + 1
        mpc["gen"][:, 0] = mpc["gen"][:, 0] + 1
        mpc["branch"][:, 0:2] = mpc["branch"][:, 0:2] + 1
    # adjust for the matpower converter -> taps should be 0 when there is no transformer, but are 1
    mpc["branch"][np.where(mpc["branch"][:, 8] == 1), 8] = 0
    # version is a string
    mpc["version"] = str(mpc["version"])
    return mpc


def _get_std_options(net, init, calculate_voltage_angles, trafo_model):
    mode = "pf"
    copy_constraints_to_ppc = False

    # init options
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=False,
                     mode=mode, copy_constraints_to_ppc=copy_constraints_to_ppc,
                     r_switch=0.0, init=init, enforce_q_lims=False,
                     recycle=None)