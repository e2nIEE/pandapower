=============================================
Line Outage Distribution Factors (LODF)
=============================================

The LODF describes which share of the pre-outage flow of a branch moves onto another branch if the branch is switched
off:

.. math::
   :nowrap:

   \begin{align*}
   LODF_{l,k} = \frac{\Delta P_l}{P_k^{0}}
   \end{align*}

with :math:`\Delta P_l` the change of the active power flow on branch :math:`l` after the outage of branch
:math:`k` and :math:`P_k^{0}` the flow on branch :math:`k` before the outage. The diagonal is -1, since the outaged
branch loses its entire flow. The flow after the outage follows from:

.. math::
   :nowrap:

   \begin{align*}
   P_l^{N-1} = P_l^{0} + LODF_{l,k} \cdot P_k^{0}
   \end{align*}

Despite the name, LODFs are calculated for the outage of lines, transformers and impedances. In the DC power flow they
do not depend on the slack.

Running a LODF calculation
==========================

.. autofunction:: pandapower.sensitivity.lodf.run_lodf

The result is a dictionary with one DataFrame per combination of goal branch type and outage branch type, e.g.
``("line", "line")`` or ``("trafo", "line")``. The rows are the goal branches and the columns the outaged branches:

.. code:: python

    from pandapower.sensitivity.lodf import run_lodf
    from pandapower.networks import case30

    net = case30()
    lodf = run_lodf(net, outage_branch_type="line", outage_branch_ix=[0, 5, 10], distributed_slack=False)
    lodf[("line", "line")]

To get the LODF for the outage of several branch types, ``run_lodf`` is called once per outage branch type.

.. note::

    Bridge branches, whose outage splits the grid, have no meaningful LODF. The direct method returns ``NaN`` for their
    columns. If a single bridge branch is requested, a ``UserWarning`` is raised.

N-1 flows with LODF
===================

``run_lodf_dc_n1`` applies the LODF to the base case and returns the flows of all branches after each outage:

.. autofunction:: pandapower.sensitivity.lodf.run_lodf_dc_n1

.. code:: python

    from pandapower.sensitivity.lodf import run_lodf_dc_n1

    n1 = run_lodf_dc_n1(net, outage_branch_type="line", outage_branch_ix=[5, 10], distributed_slack=False)
    n1[("res_line", "p_from_mw")]  # one column per outage

Verification
============

.. autofunction:: pandapower.sensitivity.lodf.verify_lodf

.. autofunction:: pandapower.sensitivity.lodf.verify_dc_n1_with_lodf
