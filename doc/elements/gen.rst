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

Table of Generator Reactive Power Capability Curve Characteristics
----------------------------

The Table of Generator Reactive Power Capability Curve Characteristics (denoted as q_capability_curve_table) serves
as a reference for determining the reactive power limits of a generator, specifically Qmin and Qmax, as a function
of the active power output of the respective generator. This table is either auto-generated from version 3.1 onwards
via the CIM CGMES to pandapower converter, provided this information is available in the Equipment (EQ) profile,
or it can be manually defined by the user.

The variable id_q_capability_curve_table in net.gen establishes a link to the id_q_capability_curve
column in net.q_capability_curve_table, associating each generator with its respective capability curve.

If the variable curve_dependency_table in net.gen is set to True, it indicates that a corresponding characteristic
is defined in net.q_capability_curve_table. This overrides the default reactive power limits of the generator.

Below is an example of a q_capability_curve_table, populated for two sample generators.

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.55\linewidth}|p{0.15\linewidth}|p{0.55\linewidth}
.. csv-table::
   :file: gen_q_char_table.csv
   :delim: ,
   :widths: 10, 10, 55, 55, 55

.. note::
    - curve_dependency_table has to be set to True, and id_q_capability_curve_table and curve_style variables need to
      be populated in order to consider the corresponding q_capability_curve_table values.
    - Each generator supports only a single curve_dependency_table
    - In this version, only two types of generator reactive power capability characteristics are supported:
      "constantYValue" and "straightLineYValues".

The function pandapower.control.util.q_capability_curve_table_diagnostic is available to perform sanity checks
on the generator reactive power capability curve table. Additionally, the function
pandapower.control.util.auxiliary.create_q_capability_curve_characteristics_object can be utilized to automatically
generate Characteristic objects and populate the net.q_capability_curve_characteristic table based on the data
in the net.q_capability_curve_table.

Furthermore, an additional column, id_q_capability_curve_table, is created in net.gen to establish the reference
between the generator and its associated characteristics.

The table below illustrates an example of a q_capability_curve_characteristic table populated for generators.

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.55\linewidth}|p{0.55\linewidth}
.. csv-table::
   :file: gen_q_char_table_object.csv
   :delim: ,

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
    If the power flow is run with the enforce_qlims option and the generator reactive power exceeds / falls short of the maximum / minimum reactive power limit,
    the generator is converted to a static generator with the maximum / minimum reactive power as constant reactive power generation.
    The voltage at the generator bus is then no longer equal to the voltage set point defined in the parameter table.
