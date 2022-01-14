Peak Short-Circuit Current
===============================

Current Calculation
---------------------------

The peak short-circuit current is calculated as:

.. math::

    \begin{bmatrix}
    i_{p, 1}  \\
    \vdots  \\
    i_{p, n}  \\
    \end{bmatrix}
    = \sqrt{2} \left(
    \begin{bmatrix}
    \kappa_{1}  \\
    \vdots  \\
    \kappa_{1}   \\
    \end{bmatrix}
    \begin{bmatrix}
    \underline{I}''_{kI, 1} \\
    \vdots  \\
    \underline{I}''_{kI, n} \\
    \end{bmatrix} +
    \begin{bmatrix}
    \underline{I}''_{kII, 1} \\
    \vdots  \\
    \underline{I}''_{kII, n} \\
    \end{bmatrix} \right)

where :math:`\kappa` is the peak factor.

.. _kappa:

Peak Factor :math:`\kappa`
-------------------------------

In radial networks, :math:`\kappa` is given as:

.. math::

    \kappa = 1.02 + 0.98 e^{-{3}{R/X}}
    
where :math:`R/X` is the R/X ratio of the equivalent short-circuit impedance :math:`Z_k` at the fault location.

In meshed networks, the standard defines three possibilities for the calculation of :math:`\kappa`:

- Method A: Uniform Ratio R/X
- Method B: R/X ratio at short-circuit location
- Method C: Equivalent frequency 

The user can chose between Methods B and C when running a short circuit calculation. Method C yields the most accurate results according to the standard and is therefore the default option.
Method A is only suited for estimated manual calculations with low accuracy and therefore not implemented in pandapower.

**Method C: Equivalent frequency**

For method C, the same formula for :math:`\kappa` is used as for radial grids. The R/X value that is inserter is however not the 


**Method B: R/X Ratio at short-circuit location**

For method B, :math:`\kappa` is given as:

.. math::

    \kappa = [1.02 + 0.98 e^{-{3}{R/X}}] \cdot 1.15

   
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
