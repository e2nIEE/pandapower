=============
Motor
=============

Create Function
=====================

.. autofunction:: pandapower.create_motor

Input Parameters
=====================

*net.motor*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.40\linewidth}|

.. csv-table::
   :file: motor_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40

\*necessary for executing a power flow calculation.

Electric Model
=================

.. math::
   :nowrap:
   
    \begin{align*}
    P_{motor, n} =& pn\_mech\_mw / (efficiency\_percent/100) \\
    P_{motor} =& P_{motor, n} * (loading\_percent / 100) * scaling \\
    S_{motor} =& P_{motor} / cos\_phi \\
    Q_{motor} =& \sqrt{S_{motor}^2 - P_{motor}^2} 
    \end{align*}


Result Parameters
==========================    
*net.res_motor*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.45\linewidth}|
.. csv-table:: 
   :file: motor_res.csv
   :delim: ;
   :widths: 10, 10, 45
   
The power values in the net.res_motor table are equivalent to :math:`P_{motor}` and :math:`Q_{motor}`.

