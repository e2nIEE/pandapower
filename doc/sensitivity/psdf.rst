=============================================
Phase Shift Distribution Factors (PSDF)
=============================================

The PSDF describes how the active power flows change if the phase shift angle of a branch is increased by 1 degree:

.. math::
   :nowrap:

   \begin{align*}
   PSDF_{l,k} = \frac{\Delta P_l}{\Delta \varphi_k} \quad \text{in MW/degree}
   \end{align*}

with :math:`\Delta P_l` the change of the active power flow on branch :math:`l` and :math:`\Delta \varphi_k` the change
of the phase shift angle of branch :math:`k`. A phase shift does not change the injections, so the PSDF does not depend
on the slack.

Running a PSDF calculation
==========================

.. autofunction:: pandapower.sensitivity.psdf.run_psdf

The result is a dictionary with one DataFrame per combination of goal branch type and phase shift branch type, e.g.
``("line", "trafo")``. The rows are the goal branches and the columns the phase shifting branches:

.. code:: python

    from pandapower.sensitivity.psdf import run_psdf
    from pandapower.networks import case14

    net = case14()
    psdf = run_psdf(net, phase_shift_branch_ix=[0], distributed_slack=False)

    # flow change on all lines if the shift_degree of trafo 0 is increased by 5 degree
    delta_p_line = 5 * psdf[("line", "trafo")][0]

By default, the PSDF is calculated for transformers, the only branch type with a phase shift parameter in pandapower.
With ``phase_shift_branch_type="line"`` or ``"impedance"``, the direct method gives the effect of a phase shifter in
series with these branches, e.g. to find suitable locations for a new phase shifter.

Sign convention
===============

The PSDF has the sign of the transformer parameter ``shift_degree``: a positive PSDF means that the flow increases if
``shift_degree`` increases.

Phase shifting tap changers (``tap_step_degree``) follow the same sign if the tap changer is on the high voltage side
(``tap_side="hv"``). On the low voltage side, pandapower applies the tap angle with the opposite sign, so the flow change
per tap step is :math:`-PSDF \cdot` ``tap_step_degree``.

.. note::

    The PSDF is the sensitivity to the phase shift angle, independent of how the angle is created. For ideal phase
    shifters (``tap_changer_type="Ideal"``), the flow change of a tap step is exactly the PSDF times the angle change of
    the step. For non-ideal phase shifters (``"Symmetrical"``, ``"Ratio"`` and ``"Tabular"``), a tap step also changes
    the voltage ratio, and the angle per step is not ``tap_step_degree`` but follows from the tap changer model. The
    PSDF times the angle change of the step then only approximates the flow change, since the changed ratio also
    changes the transformer susceptance. The error is small for quadrature boosters and can reach a few percent for
    asymmetrical phase shifters.

.. note::

    The direct method calculates the PSDF for lines, transformers and impedances. The perturb method (``perturb=True``)
    increases ``shift_degree`` by 1 degree and runs a DC power flow, so it is only available for transformers, since
    lines and impedances have no phase shift parameter in pandapower.

Matrix calculation
==================

The PSDF matrix is built from the PTDF matrix in the internal ppc data structure:

.. autofunction:: pandapower.sensitivity.psdf.make_psdf
