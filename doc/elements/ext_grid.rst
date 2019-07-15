﻿=============
External Grid
=============

.. note::

   Power values of external grids are given in the generator system, therefore p_mw is negative if the external grid is absorbing power and positive if it is supplying/generating power.

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

.. |br| raw:: html

   <br />
   
\*necessary for executing a power flow calculation |br| \*\*optimal power flow parameter |br| \*\*\*short-circuit calculation parameter
   
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
    p\_mw &= P_{eg} \\
    q\_mvar &= Q_{eg}
    \end{align*}