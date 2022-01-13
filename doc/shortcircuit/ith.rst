Thermal Short-Circuit Current
==================================

Current Calculation
---------------------------

The equivalent thermal current is calculated as:

.. math::

    \begin{bmatrix}
    \underline{I}_{th, 1} \\
    \vdots  \\
    \underline{I}_{th, n} \\
    \end{bmatrix} =   
    \begin{bmatrix}
    \sqrt{m_1 + n_1} \\
    \vdots  \\
    \sqrt{m_n + n_n} \\
    \end{bmatrix}
    \begin{bmatrix}
    \underline{I}''_{k, 1} \\
    \vdots  \\
    \underline{I}''_{k, n} \\
    \end{bmatrix}

where m and n represent the dc and ac part of the thermal load.

Correction Factors m and n
----------------------------

For short-circuit currents far from synchronous generators, the factors are given as:

.. math::

    n = 1
    m = \frac{1}{2 \cdot f \cdot T_k \cdot ln(\kappa - 1)} [e^{4 \cdot f \cdot T_k \cdot ln(\kappa - 1)} - 1]
    
where :math:`\kappa` is the peak factor defined :ref:`here <kappa>` and :math:`T_k` is the duration of the short-circuit current that can be defined as a parameter when
running the short-circuit calculation. 
