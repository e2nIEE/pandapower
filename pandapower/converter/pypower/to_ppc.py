# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

from pandapower.run import _pd2ppc, _select_is_elements, reset_results
import pplog

logger = pplog.getLogger(__name__)


def pp2ppc(net, init="results", calculate_voltage_angles=False, trafo_model="t"):
    """
     This function converts a pandapower net to a pypower case file.

    INPUT:

        **net** - The pandapower net.

    OPTIONAL:

        **init** (str, "results") - initialization method of the loadflow
        For the conversion to a ppc, the following options can be chosen:

            - "flat"- flat start with voltage of 1.0pu and angle of 0° at all buses as initial solution
            - "results" - voltage vector of last loadflow from net.res_bus is copied to the ppc

        **calculate_voltage_angles** (bool, False) - copy the voltage angles from pandapower to the ppc

            If True, voltage angles are copied from pandapower to the ppc. In some cases with
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
        Pandapower provides two equivalent circuit models for the transformer:

            - "t" - transformer is modelled as equivalent with the T-model. This is consistent with PowerFactory
                and is also more accurate than the PI-model. We recommend using this transformer model.
            - "pi" - transformer is modelled as equivalent PI-model. This is consistent with Sincal,
                but the method is questionable since the transformer is physically T-shaped. We therefore
                recommend the use of the T-model.

    OUTPUT:

        **ppc** - The Pypower casefile for usage with pypower

    EXAMPLE:

        import pandapower.converter as pc

        import pandapower.networks as pn

        net = pn.case9()

        ppc = pc.pp2ppc(net)

    """

    # convert to matpower
    net["converged"] = False
    if not init == "results":
        reset_results(net)

    # select elements in service (time consuming, so we do it once)
    is_elems = _select_is_elements(net)

    init_results = True if init == "results" else False
    ppc, ppci, bus_lookup = _pd2ppc(net, is_elems, calculate_voltage_angles, enforce_q_lims=False,
                                    trafo_model=trafo_model, init_results=init_results,
                                    copy_voltage_boundaries=True)
    ppc['branch'] = ppc['branch'].real
    ppc.pop('internal')

    return ppc
