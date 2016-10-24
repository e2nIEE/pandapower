=============
Load
=============

.. seealso::
    :ref:`Unit Systems and Conventions <conventions>`

Create Function
=====================

.. autofunction:: pandapower.create_load

Parameters
=============

*net.load*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.40\linewidth}|

.. csv-table:: 
   :file: load_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40

*necessary for executing a loadflow calculation.

   
Loadflow Model
=================

Loads are modelled as PQ-buses in the loadflow calculation:

.. image:: /elements/load/pq.png
	:width: 8em
	:alt: alternate Text
	:align: center

The PQ-Values are calculated from the parameter table values as:

.. math::
   :nowrap:
   
   \begin{align*}
    P_{load} = p\_kw \cdot scaling \\
    Q_{load} = q\_kvar \cdot scaling \\
    \end{align*}

.. note::

   Loads should always have a positive p_kw value, since all power values are given in the consumer system. If you want to model constant generation, use a Static Generator (sgen element) instead of a negative load.

.. note::
    
    The apparent power value sn_kva is provided as additional information for usage in controller or other applications based on panadapower. It is not considered in the loadflow!  
 
Result Parameters
==================    
*net.res_load*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.45\linewidth}|
.. csv-table:: 
   :file: load_res.csv
   :delim: ;
   :widths: 10, 10, 45
   
The power values in the net.res_load table are equivalent to :math:`P_{load}` and :math:`Q_{load}`.

