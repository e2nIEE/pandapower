=============
Motor
=============

Create Function
=====================

.. autofunction:: pandapower.create.create_motor

Input Parameters
=====================

*net.motor*

.. csv-table::
    :file: table_structures/motor.csv
    :header-rows: 1
    :delim: ,

\*necessary for executing a power flow calculation.

Electric Model
=================

.. math::
    P_{motor, n} &= \frac{pn\_mech\_mw}{\mathit{efficiency}\_percent/100} \\
    P_{motor} &= P_{motor, n} * (loading\_percent / 100) * scaling \\
    S_{motor} &= \frac{P_{motor}}{cos\_phi} \\
    Q_{motor} &= \sqrt{S_{motor}^2 - P_{motor}^2}


Result Parameters
==========================    
*net.res_motor*

.. csv-table::
    :file: table_structures/res_motor.csv
    :header-rows: 1
    :delim: ,
   
The power values in the net.res_motor table are equivalent to :math:`P_{motor}` and :math:`Q_{motor}`.
