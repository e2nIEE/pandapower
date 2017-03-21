===================
Correction Factors
===================

.. _c:

Voltage Corection Factor c
====================================

The voltage correction factors :math:`c_{min}` for minimum and :math:`c_{max}` for maximum short-circuit currents are applied in calculating
short-circuit impedances for some elements (transformer, ext_grid) as well as for the equivalent voltage source for the calculation of the
initials short-circuit current :math:`I''_k`. 

It is defined for each bus depending on the voltage level.
In the low voltage level, there is an additional distinction between networks with a tolerance of 6% vs. a tolerance of 10% for :math:`c_{max}`:

.. |cmin| replace:: :math:`c_{min}`
.. |cmax| replace:: :math:`c_{max}`

+--------------+---------------+--------+--------+
|Voltage Level                 | |cmin| | |cmax| |
+==============+===============+========+========+
|              | Tolerance 6%  |        |  1.05  |
|< 1 kV        +---------------+  0.95  +--------+
|              | Tolerance 10% |        |        |
+--------------+---------------+--------+  1.10  +
|> 1 kV                        |  1.00  |        |
+--------------+---------------+--------+--------+

.. _kappa:

Peak Factor :math:`\kappa`
============================

The factor :math:`\kappa` is used for calculation of the peak short-circuit current :math:`i_p`, thermal equivalent short-circuit 
current :math:`I_{th}` and unsymmetrical short-circuit currents.

In radial networks, :math:`\kappa` is given as:

.. math::

    \kappa = 1.02 + 0.98 e^{-\frac{3}{R/X}}
    
where :math:`R/X` is the R/X ratio of the equivalent short-circuit impedance :math:`Z_k` at the fault location.

In meshed networks, the standard defines three possibilities for the definition of  :math:`\kappa`:

    a ) Uniform Ratio R/X 
    b ) Ratio R/X at short-circuit location
    c ) Equivalent frequency 

pandapower implements version b), in which the factor :math:`\kappa` is given as:

.. math::

    \kappa = [1.02 + 0.98 e^{-\frac{3}{R/X}}] \cdot 1.15

   
while being limited with :math:`\kappa_{min} < \kappa < \kappa_{max}` depending on the voltage level:

.. |kmin| replace:: :math:`\kappa_{min}`
.. |kmax| replace:: :math:`\kappa_{max}`

+-------------+--------+--------+
|Voltage Level| |kmin| | |kmax| |
+=============+========+========+
| < 1 kV      |        | 1.8    |
+-------------+  1.0   +--------+
| > 1 kV      |        | 2.0    |
+-------------+--------+--------+


.. _mn:

Thermal Factors m and n
========================
The factors m and n are necessary for the calculation of the thermal equivalent short-circuit current :math:`I_{th}`.

pandapower currently only implements short-circuit currents far from synchronous generators, where:

.. math::

    n = 1

and m is given as:
   
.. math::

    m = \frac{1}{2 \cdot f \cdot T_k \cdot ln(\kappa - 1)} [e^{4 \cdot f \cdot T_k \cdot ln(\kappa - 1)} - 1]
    
where :math:`\kappa` is defined as above and :math:`T_k` is the duration of the short-circuit current that can be defined as a parameter when
running the short-circuit calculation. 

    
    
