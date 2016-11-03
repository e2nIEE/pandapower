=============
External Grid
=============

.. seealso::
    :ref:`Unit Systems and Conventions <conventions>`
    
Create Function
=====================

.. autofunction:: pandapower.create_ext_grid


Input Parameters
=============================

*net.ext_grid*

.. tabularcolumns:: |p{0.15\linewidth}|p{0.10\linewidth}|p{0.15\linewidth}|p{0.40\linewidth}|
.. csv-table:: 
   :file: ext_grid_par.csv
   :delim: ;
   :widths: 15, 10, 15, 40

\*necessary for executing a power flow calculation.

*Optional Parameters*:

.. tabularcolumns:: |p{0.15\linewidth}|p{0.10\linewidth}|p{0.15\linewidth}|p{0.40\linewidth}|
.. csv-table:: 
   :file: ext_grid_opt.csv
   :delim: ;
   :widths: 15, 10, 15, 40

   
Electric Model
=================

The external grid is modelled as a voltage source in the power flow calculation, which means the node the grid is connected to is treated as a slack node:

.. image:: ext_grid.png
	:width: 12em
	:alt: alternate Text
	:align: center

with:
    
.. math::
   :nowrap:
   
   \begin{align*}
    \underline{v}_{bus} &= vm\_pu \cdot e^{j \cdot \theta} \\
   \theta &= shift\_degree \cdot \frac{\pi}{180}
   \end{align*}

Result Parameters
==========================    
*net.res_ext_grid*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.1\linewidth}|p{0.50\linewidth}|
.. csv-table:: 
   :file: ext_grid_res.csv
   :delim: ;
   :widths: 10, 10, 50

Active and reactive power feed-in / consumption at the slack node is a result of the power flow:
   
.. math::
   :nowrap:
   
   \begin{align*}
    p\_kw &= P_{eg} \\
    q\_kvar &= Q_{eg}
    \end{align*}
    
.. note::

   All power values are given in the consumer system, therefore p_kw is positive if the external grid is absorbing power and negative if it is supplying power.
   
Optimal Power Flow Parameters
=============================

An external grid is not considered as a flexibility for the optimal power flow. The voltage setpoint can not be changed by the optimal power flow. Still, costs can be defined for an external grid:

.. tabularcolumns:: |l{0.10\linewidth}|l{0.10\linewidth}|l{0.40\linewidth}|
.. csv-table:: 
    :file: ext_grid_opf.csv
    :delim: ;
    :widths: 10, 10, 40
