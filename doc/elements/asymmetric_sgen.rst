.. -asymmetric_sgen

============================
Asymmetric Static Generator
============================

.. note::

   Static generators should always have a positive p_mw value, since all power values are given in the generator convention. If you want to model constant power consumption, it is recommended to use a load element instead of a static generator with negative active power value.

.. seealso::
    :ref:`Unit Systems and Conventions <conventions>`

Create Function
=====================

.. autofunction:: pandapower.create.create_asymmetric_sgen


Table Structure
=====================

*net.asymmetric_sgen*

.. csv-table::
    :file: table_structures/asymmetric_sgen.csv
    :header-rows: 1
    :delim: ,

   
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

Result Table
==========================
*net.asymmetric_sgen*

.. csv-table::
    :file: table_structures/res_asymmetric_sgen.csv
    :header-rows: 1
    :delim: ,

The power values in the net.res_sgen table are equivalent to :math:`P_{sgen}` and :math:`Q_{sgen}`.
