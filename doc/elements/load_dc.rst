.. _load_dc:

=============
Load DC
=============

.. note::

   DC Loads, as their AC counterparts, should always have a positive p_mw value, since all power values are given in the consumer system. If you want to model constant generation, use a source_dc instead of a negative load.

.. seealso::
    :ref:`Unit Systems and Conventions <conventions>`

Create Function
=====================

.. autofunction:: pandapower.create_load_dc

Input Parameters
=====================

*net.load_dc*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.40\linewidth}|

.. csv-table::
   :file: load_dc_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40

\*necessary for executing a power flow calculation.

.. note::
    
    The type is provided as additional information for usage in controller or other applications based on pandapower. It is not considered in the power flow! Together with source_dc one could build a pure multi terminal DC systen. But currently such pure system will not run. A small AC system is still needed, but no connection between both systems is needed.


Electric Model
=================

Loads are modelled as P-buses in the power flow calculation:

.. image:: load.png
	:width: 8em
	:alt: alternate Text
	:align: center


What part of the load is considered constant with constant power:
The load power values are then defines as:

.. math::
   :nowrap:
   
   \begin{align*}
    P_{load} =&  p\_mw \cdot scaling \cdot (p_{const_p} + i_{const_p} \cdot V ) \\
    \end{align*}


Result Parameters
==========================    
*net.res_load_dc*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.45\linewidth}|
.. csv-table:: 
   :file: load_dc_res.csv
   :delim: ;
   :widths: 10, 10, 45
   
The power values in the net.res_load_dc table are equivalent to :math:`P_{load}`.

