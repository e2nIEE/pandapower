==================
Static Generator
==================

.. seealso::
    :ref:`Unit Systems and Conventions <conventions>`

Create Function
=====================

.. autofunction:: pandapower.create_sgen

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
    P_{sgen} &= p\_kw \cdot scaling \\
    Q_{sgen} &= q\_kvar \cdot scaling \\
    \end{align*}

.. note::

   Static generators should always have a negative p_kw value, since all power values are given in the consumer system. If you want to model constant power consumption, please use the load element instead of a static generator with positive active power value.
   If you want to model a voltage controlled generator, use the generator element.

.. note::
    
    The apparent power value sn_kva is provided as additional information for usage in controller or other applications based on panadapower. It is not considered in the power flow!

Result Parameters
==========================
*net.res_sgen*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.50\linewidth}|
.. csv-table:: 
   :file: sgen_res.csv
   :delim: ;
   :widths: 10, 10, 50

The power values in the net.res_sgen table are equivalent to :math:`P_{sgen}` and :math:`Q_{sgen}`.