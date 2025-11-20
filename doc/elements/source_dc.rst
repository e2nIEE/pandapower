.. _source_dc:

==================
Source DC
==================

.. note::

   Sources should always have a positive p_mw value, since all power values are given in the generator convention. If you want to model constant power consumption, it is recommended to use a load element instead of a source with negative active power value.

.. seealso::
    :ref:`Unit Systems and Conventions <conventions>`

Create Function
=====================

.. autofunction:: pandapower.create_source_dc

Input Parameters
=====================

*net.sgen*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.40\linewidth}|
.. csv-table:: 
   :file: source_dc_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40

.. |br| raw:: html

   <br />
   
\*necessary for executing a power flow calculation

Electric Model
=================

DC sources are modelled as P-buses in the power flow calculation:

.. image:: pq.png
	:width: 8em
	:alt: alternate Text
	:align: center
    
The P-Values are calculated from the parameter table values as:

.. math::
   :nowrap:
   
   \begin{align*}
    P_{sgen} &= p\_mw \cdot scaling \\
    \end{align*}

.. note::
    
    Other values are provided as additional information for usage in controller or other applications based on pandapower. It is not considered in the power flow!

Result Parameters
==========================
*net.res_source_dc*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.50\linewidth}|
.. csv-table:: 
   :file: res_source_dc.csv
   :delim: ;
   :widths: 10, 10, 50

The power values in the net.res_source_dc table are equivalent to :math:`P_{sgen}`.