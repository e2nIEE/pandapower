==================
Static Generator
==================

.. note::

   Static generators should always have a positive p_mw value, since all power values are given in the generator convention. If you want to model constant power consumption, it is recommended to use a load element instead of a static generator with negative active power value.
   If you want to model a voltage controlled generator, use the generator element.


.. seealso::
    :ref:`Unit Systems and Conventions <conventions>`

Create Function
=====================

.. autofunction:: pandapower.create_sgen

.. autofunction:: pandapower.create_sgen_from_cosphi

Input Parameters
=====================

*net.sgen*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.40\linewidth}|
.. csv-table:: 
   :file: sgen_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40

.. |br| raw:: html

   <br />
   
\*necessary for executing a power flow calculation |br| \*\*optimal power flow parameter

   
Electric Model
=================

Static Generators are modelled as PQ-buses in the power flow calculation:

.. image:: pq.png
	:width: 8em
	:alt: alternate Text
	:align: center
    
The PQ-Values are calculated from the parameter table values as:

.. math::
   :nowrap:
   
   \begin{align*}
    P_{sgen} &= p\_mw \cdot scaling \\
    Q_{sgen} &= q\_mvar \cdot scaling \\
    \end{align*}


.. note::
    
    The apparent power value sn_mva is provided as additional information for usage in controller or other applications based on pandapower. It is not considered in the power flow!

Result Parameters
==========================
*net.res_sgen*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.50\linewidth}|
.. csv-table:: 
   :file: sgen_res.csv
   :delim: ;
   :widths: 10, 10, 50

The power values in the net.res_sgen table are equivalent to :math:`P_{sgen}` and :math:`Q_{sgen}`.

Table of Static Generator Reactive Power Capability Curve Characteristics
----------------------------

The Table of Static Generator Reactive Power Capability Curve Characteristics (referred to as q_capability_curve_table)
provides a reference framework for determining the reactive power limits (Qmin and Qmax) of static generators based on
their active power output. This table can be either automatically generated (from version 3.1 onwards) via the CIM CGMES
to pandapower converter, provided the relevant information is available in the Equipment (EQ) profile, or defined
manually by the user.

The variable id_q_capability_curve_table in net.sgen establishes a direct reference to the id_q_capability_curve column
in net.q_capability_curve_table, thereby associating each static generator with its corresponding capability curve.

When the variable curve_dependency_table in net.sgen is set to True, it signifies the presence of a corresponding
characteristic defined in net.q_capability_curve_table, which overrides the default reactive power limits of the static
generator.

Below is an example of a q_capability_curve_table populated for sample static generators.

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.55\linewidth}|p{0.15\linewidth}|p{0.55\linewidth}
.. csv-table::
   :file: sgen_q_char_table.csv
   :delim: ,
   :widths: 10, 10, 55, 55, 55

.. note::
    - curve_dependency_table has to be set to True, and id_q_capability_curve_table and curve_style variables need to
      be populated in order to consider the corresponding q_capability_curve_table values.
    - Each static generator supports only a single curve_dependency_table
    - In this version, only two types of static generator reactive power capability characteristics are supported:
      "constantYValue" and "straightLineYValues".

The function pandapower.control.util.q_capability_curve_table_diagnostic is available to perform sanity checks
on the static generator reactive power capability curve table. Additionally, the function
pandapower.control.util.auxiliary.create_q_capability_curve_characteristics_object can be utilized to automatically
static generate Characteristic objects and populate the net.q_capability_curve_characteristic table based on the data
in the net.q_capability_curve_table.

Furthermore, an additional column, id_q_capability_curve_table, is created in net.sgen to establish the reference
between the static generator and its associated characteristics.

The table below illustrates an example of a q_capability_curve_characteristic table populated for generators.

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.55\linewidth}|p{0.55\linewidth}
.. csv-table::
   :file: gen_q_char_table_object.csv
   :delim: ,