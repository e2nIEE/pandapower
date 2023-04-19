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
    If the power flow is run with the enforce_qlims option and the generator reactive power exceeds / underruns the maximum / minimum reactive power limit,
    the generator is converted to a static generator with the maximum / minimum reactive power as constant reactive power generation.
    The voltage at the generator bus is then no longer equal to the voltage set point defined in the parameter table.
