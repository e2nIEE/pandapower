.. _ward:

=============
Ward
=============

.. seealso::
    :ref:`Unit Systems and Conventions <conventions>`

Create Function
=====================

.. autofunction:: pandapower.create.create_ward
.. autofunction:: pandapower.create.create_wards

Table Structure
=========================

*net.ward*

.. csv-table::
    :file: table_structures/ward.csv
    :header-rows: 1
    :delim: ,
    :widths: 10, 10, 15, 40, 2, 2

   
Electric Model
=================

.. image:: ward.png
    :width: 15em
    :align: center

The ward equivalent is a combination of a constant apparent power consumption and a constant impedance load. The constant apparent power is given by:

.. math::
   :nowrap:
   
   \begin{align*}
   P_{const} &= ps\_mw\\
   Q_{const} &= qs\_mvar\\
   \end{align*}
    
The shunt admittance part of the ward equivalent is calculated as described :ref:`here<shunt>`:

.. math::
   :nowrap:
   
   \begin{align*}
   \underline{y}_{shunt} &= \frac{pz\_mw + j \cdot qz\_mvar}{S_{N}}
   \end{align*}

Result Table
==========================
*net.res_ward*

.. csv-table::
    :file: table_structures/res_ward.csv
    :header-rows: 1
    :delim: ,
    :widths: 10, 10, 50, 2, 2


.. math::
   :nowrap:
   
   \begin{align*}
   vm\_pu &= v_{bus} \\
   p\_mw &= P_{const} + Re(\frac{\underline{V}_{bus}^2}{\underline{Y}_{shunt}}) \\
   q\_mvar &= Q_{const} + Im(\frac{\underline{V}_{bus}^2}{\underline{Y}_{shunt}})
   \end{align*}
