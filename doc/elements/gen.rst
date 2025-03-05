.. _gen:

=============
Generator
=============

.. note::
    A generator with positive active power represents a voltage controlled generator. If you want to model constant generation without voltage control, use the Static Generator element.

.. seealso::
    :ref:`Unit Systems and Conventions <conventions>`

    
Create Function
=====================

.. autofunction:: pandapower.create.create_gen

Input Parameters
=====================

*net.gen*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.1\linewidth}|p{0.25\linewidth}|p{0.4\linewidth}|
.. csv-table:: 
   :file: gen_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40

.. |br| raw:: html

   <br />
   
\*necessary for executing a power flow calculation |br| \*\*optimal power flow parameter |br| \*\*\*short-circuit calculation parameter

Generator Reactive Power Capability Curve Characteristics
============================

The generator reactive power capability curve characteristics serve as a reference for determining the reactive power
limits of a generator (Qmin and Qmax) as a function of its active power output.
The reactive power capability curve data can be imported into pandapower in a tabular format, populating
net.q_capability_curve_table. This table is either auto-generated via the CIM CGMES to pandapower converter,
provided this information is available in the Equipment (EQ) profile, or it can be manually defined by the user.

Q capability curve characteristic objects are then generated from net.q_capability_curve_table, populating
net.q_capability_curve_characteristic. The characteristics are either auto-generated via the CIM CGMES to pandapower
converter or they can be created by the user via the pandapower.control.util.create_q_capability_curve_characteristics_object
function, provided q_capability_curve_table is previously defined in the network case.

If the variable reactive_capability_curve in net.gen is set to True, it indicates that pairs of P vs Qmin/Qmax values
and the corresponding characteristic are defined in net.q_capability_curve_table and net.q_capability_curve_characteristic
respectively. This overrides the default reactive power limits of the generator when a power flow is executed
and the enforce_q_lims option is enabled.
The variable id_q_capability_curve_characteristic in net.gen establishes a link to the id_q_capability_curve column
in both net.q_capability_curve_table and net.q_capability_curve_characteristic, associating each generator with its
respective capability curve.

Below is an example of a q_capability_curve_table, populated for two sample generators.

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.55\linewidth}|p{0.15\linewidth}|p{0.55\linewidth}
.. csv-table::
   :file: gen_q_char_table.csv
   :delim: ,
   :widths: 10, 10, 55, 55, 55

The table below illustrates an example of a q_capability_curve_characteristic table populated for two generators.

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.55\linewidth}|p{0.55\linewidth}
.. csv-table::
   :file: gen_q_char_table_object.csv
   :delim: ,

.. note::
    - reactive_capability_curve has to be set to True, and id_q_capability_curve_characteristic and curve_style variables
      need to be populated in order to consider the reactive power limits of the corresponding characteristic.
    - Each generator supports only a single reactive_capability_curve.
    - In this version, only two types of generator reactive power capability characteristics are supported:
      1. constantYValue: The reactive power values are assumed constant until the next curve point and prior to the first curve point.
      2. straightLineYValues: The reactive power values are assumed to be a straight line between values.
    - Linear interpolation is employed to determine Qmin and Qmax based on the given active power dispatch for the above two curve types.

The function pandapower.control.util.q_capability_curve_table_diagnostic is available to perform sanity checks
on the generator reactive power capability curve table.

Electric Model
=================

Generators are modelled as PV-nodes in the power flow:

.. image:: gen.png
	:width: 12em
	:alt: alternate Text
	:align: center

Voltage magnitude and active power are defined by the input parameters in the generator table:

.. math::
   :nowrap:
   
   \begin{align*}
    P_{gen} &= p\_mw * scaling \\
    v_{bus} &= vm\_pu
   \end{align*}
    
Result Parameters
==========================
*net.res_gen*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.50\linewidth}|
.. csv-table:: 
   :file: gen_res.csv
   :delim: ;
   :widths: 10, 10, 50

The power flow returns reactive generator power and generator voltage angle:

.. math::
   :nowrap:

   \begin{align*}
    p\_mw &= P_{gen} \\
    q\_mvar &= Q_{gen} \\
    va\_degree &= \angle \underline{v}_{bus} \\
    vm\_degree &= |\underline{v}_{bus}|
   \end{align*}

.. note::
    If the power flow is run with the enforce_q_lims option and the generator reactive power exceeds / falls short of the maximum / minimum reactive power limit,
    the generator is converted to a static generator with the maximum / minimum reactive power as constant reactive power generation.
    The voltage at the generator bus is then no longer equal to the voltage set point defined in the parameter table.
