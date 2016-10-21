.. _gen:

=============
Generator
=============

.. seealso::
    :ref:`Create Generator<create_gen>`
    
**Parameters**

*net.gen*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.1\linewidth}|p{0.25\linewidth}|p{0.4\linewidth}|
.. csv-table:: 
   :file: gen_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40

*necessary for executing a loadflow calculation.

.. note::
    Active power should normally be negative to model a voltage controlled generator, since all power values are given in the load reference system. A generator with positive active power represents a voltage controlled machine.
    If you want to model constant generation without voltage control, use the Static Generator element.

**Loadflow Model**

Generators are modelled as PV-nodes in the loadflow:

.. image:: /pandapower/elements/gen/gen.png
	:width: 12em
	:alt: alternate Text
	:align: center

Voltage magnitude and active power are defined by the input parameters in the generator table:

.. math::
   :nowrap:
   
   \begin{align*}
    P_{gen} &= p\_kw * scaling \\
    v_{bus} &= vm\_pu
   \end{align*}
    
**Results**

*net.res_gen*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.50\linewidth}|
.. csv-table:: 
   :file: gen_res.csv
   :delim: ;
   :widths: 10, 10, 50

The loadflow returns reactive generator power and generator voltage angle:

.. math::
   :nowrap:

   \begin{align*}
    p\_kw &= P_{gen} \\
    q\_kvar &= Q_{gen} \\
    va\_degree &= \angle \underline{v}_{bus}
   \end{align*}

   
.. note::
    If the loadflow is run with the enforce_qlims option and the generator reactive power exceeds / underruns the maximum / minimum reactive power limit,
    the generator is converted to a static generator with the maximum / minimum reactive power as constant reactive power generation.
    The voltage at the generator bus is then no longer equal to the voltage set point defined in the parameter table.