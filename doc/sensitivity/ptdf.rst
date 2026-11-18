=============================================
Power Transfer Distribution Factors (PTDF)
=============================================

The PTDF describes how the active power flow on a branch changes if the active power injection at a bus is
increased. The additional power is balanced by the slack:

.. math::
   :nowrap:

   \begin{align*}
   PTDF_{l,n} = \frac{\Delta P_l}{\Delta P_n}
   \end{align*}

with :math:`\Delta P_l` the change of the active power flow on branch :math:`l` and :math:`\Delta P_n` the additional
injection at bus :math:`n` (an increase of generation or a decrease of load). The flow refers to the side selected with
``result_side`` (0: from/hv side, 1: to/lv side) as in the pandapower result tables.

The column of a slack bus is zero, since an injection at the slack does not change any flow. With several slacks,
the direct method distributes the additional power according to the ``slack_weight`` of the slack elements.

Running a PTDF calculation
==========================

.. autofunction:: pandapower.sensitivity.ptdf.run_ptdf

The result is a dictionary with one DataFrame per branch type (``"line"``, ``"trafo"``, ``"impedance"``,
``"trafo3w_hv"``, ``"trafo3w_mv"``, ``"trafo3w_lv"``). The rows are the branches and the columns the buses:

.. code:: python

    from pandapower.sensitivity.ptdf import run_ptdf
    from pandapower.networks import case30

    net = case30()
    ptdf = run_ptdf(net)

    # flow change on all lines if 10 MW more are injected at bus 9
    delta_p_line = ptdf["line"][9] * 10 #MW

.. _ppSensitivityPtdfProfile:

DC time series with PTDF
========================

With the PTDF, the branch flows of many injection scenarios are calculated with a single matrix product instead of one
power flow per time step. The profiles are given per element type as DataFrames with the time steps as index and the
element indices as columns. Elements without a profile keep their value from the net.

.. autofunction:: pandapower.sensitivity.ptdf.run_ptdf_dc_profile

.. code:: python

    import numpy as np
    import pandas as pd
    from pandapower.sensitivity.ptdf import run_ptdf_dc_profile

    scaling = 0.9 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, 96))
    profiles = {
        ("load", "p_mw"): pd.DataFrame(np.outer(scaling, net.load.p_mw.to_numpy()), columns=net.load.index)
    }
    res = run_ptdf_dc_profile(net, profiles)
    res[("res_line", "p_from_mw")]  # line flows per time step

``run_ptdf_dc_profile`` computes the PTDF in every call. If several profiles are calculated for the same grid, the PTDF
can be calculated once and passed with ``ptdf``:

.. code:: python

    ptdf = run_ptdf(net)
    res_day_1 = run_ptdf_dc_profile(net, profiles_day_1, ptdf=ptdf)
    res_day_2 = run_ptdf_dc_profile(net, profiles_day_2, ptdf=ptdf)

.. note::

    A precalculated PTDF must be calculated for the same grid state (topology, in-service status and slacks) and with
    the same ``result_side`` as the profile calculation, which is not checked. It must contain all buses with elements
    in the profiles.

.. note::

    ``run_ptdf_dc_profile`` only returns branch flows. For other results (e.g. bus voltage angles) or to calculate the
    profile with one DC power flow per time step (the perturb method), use the :ref:`time series module <timeseriesmodule>`
    with ``run_timeseries(net, run=rundcpp)``:

    .. code:: python

        from pandapower.control import ConstControl
        from pandapower.run import rundcpp
        from pandapower.timeseries import DFData, OutputWriter, run_timeseries

        load_profile = profiles[("load", "p_mw")]
        ConstControl(net, element="load", variable="p_mw", element_index=load_profile.columns,
                     data_source=DFData(load_profile), profile_name=load_profile.columns)
        ow = OutputWriter(net, time_steps=load_profile.index, output_path=None)
        ow.log_variable("res_bus", "va_degree")
        run_timeseries(net, time_steps=load_profile.index, run=rundcpp)
        ow.output["res_bus.va_degree"]

Verification
============

.. autofunction:: pandapower.sensitivity.ptdf.verify_ptdf
