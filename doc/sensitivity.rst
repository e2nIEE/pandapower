##########################
Sensitivities
##########################

.. _ppSensitivity:

The sensitivity module calculates linear sensitivities of the branch flows, so-called distribution factors. Once
calculated, they replace a power flow per scenario with a matrix product, which makes them useful for fast N-1
screening, time series and congestion studies.

.. list-table::
    :widths: 15 40 30
    :header-rows: 1

    * - Factor
      - Question it answers
      - Function
    * - PTDF
      - How does the flow on a branch change if 1 MW more is injected at a bus?
      - :func:`pandapower.sensitivity.ptdf.run_ptdf`
    * - LODF
      - How much of a branch's flow moves onto the other branches if it is switched off?
      - :func:`pandapower.sensitivity.lodf.run_lodf`
    * - PSDF
      - How do the branch flows change if the phase shift of a branch is increased by 1 degree?
      - :func:`pandapower.sensitivity.psdf.run_psdf`

All distribution factors are based on the :ref:`DC power flow <ppDCPF>`. They are therefore exact for
:func:`pandapower.run.rundcpp` and a linear approximation of the AC power flow: losses, reactive power and voltage
magnitudes are neglected.

A tutorial that introduces this feature with examples is available at
`Sensitivity analysis <https://github.com/e2nIEE/pandapower/tree/develop/tutorials/sensitivity_analysis.ipynb>`_.

.. toctree::
    :maxdepth: 2

    sensitivity/ptdf
    sensitivity/lodf
    sensitivity/psdf

Direct and perturb method
=========================

All distribution factors can be calculated with two methods:

- **direct** (``perturb=False``, default): the factors are calculated from the matrices of the internal ppc data
  structure. This is fast if many elements are required.
- **perturb** (``perturb=True``): brute force, one DC power flow per element (bus, outaged branch or phase shifting
  transformer). This is faster if only a few elements are required in a large grid and is useful to check results.

The direct PTDF distributes the slack according to the ``slack_weight`` of the slack elements (``ext_grid`` and
``gen`` with ``slack=True``), ``distributed_slack`` only changes the perturb method. LODF and PSDF do not depend on the
slack, so ``distributed_slack=False`` is passed for their direct method to avoid a warning.

Grid state
==========

The distribution factors are valid for the grid state at the time of the calculation (topology, in-service status and
slacks) and have to be recalculated if it changes. If no power flow was run on the net before, a DC power flow is run
to create the internal data structure. The results are also stored in the net, e.g. in ``net.ptdf_line``,
``net.lodf_line`` or ``net.psdf_line``.

Limitations
===========

- Bridge branches, whose outage splits the grid, have no meaningful LODF and are ``NaN``.
- dcline outages are only supported by the perturb method of the LODF. A dcline has a fixed active power setpoint, so its
  flow does not depend on the network state. If the setpoint is 0, its LODF is undefined and ``NaN``.
- The direct LODF and PSDF do not support dclines and three-winding transformers as outaged or phase shifting branches
  and return an empty dictionary for them. They are included as goal branches.
