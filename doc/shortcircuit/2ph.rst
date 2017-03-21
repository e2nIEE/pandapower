Unsymmetric Two-Phase Current
==================================
    
Initial Short-Circuit Current
---------------------------------
The two-phase initial short-circuit current is calculated in the same way the three-phase current is calculated, only with
a source voltage of :math:`c \cdot \sqrt{2} \cdot V_N` instead of :math:`\frac{c \cdot V_N}{\sqrt{3}}`:

.. math::
   
    \begin{bmatrix}
    \underline{I}''_{k2, 1} \\
    \vdots  \\
    \underline{I}''_{k2, m} \\
    \end{bmatrix}
    = 
    \begin{bmatrix}
    \frac{c_1 \cdot \sqrt{2} \cdot V_{N, 1}}{Z_{11}  + Z_{fault}}  \\
    \vdots  \\
    \frac{c_n \cdot \sqrt{2} \cdot V_{N, n}}{Z_{nn} + Z_{fault}} 
    \end{bmatrix}


Peak Short-Circuit Current
---------------------------------

The peak short-circuit current is calculated as:

.. math::

    \begin{bmatrix}
    i_{p2, 1}  \\
    \vdots  \\
    i_{p2, n}  \\
    \end{bmatrix}
    = \sqrt{2}
    \begin{bmatrix}
    \kappa_{1}  \\
    \vdots  \\
    \kappa_{1}   \\
    \end{bmatrix}
    \begin{bmatrix}
    \underline{I}''_{k2, 1} \\
    \vdots  \\
    \underline{I}''_{k2, n} \\
    \end{bmatrix}

where the factor :math:`\kappa` is calculated  for each bus as defined :ref:`here <kappa>`.
    
Thermal Short-Circuit Current
---------------------------------

The equivalent 

.. math::

    \begin{bmatrix}
    \underline{I}_{th2, 1} \\
    \vdots  \\
    \underline{I}_{th2, n} \\
    \end{bmatrix} =   
    \begin{bmatrix}
    \sqrt{m_1 + n_1} \\
    \vdots  \\
    \sqrt{m_n + n_n} \\
    \end{bmatrix}
    \begin{bmatrix}
    \underline{I}''_{k2, 1} \\
    \vdots  \\
    \underline{I}''_{k2, n} \\
    \end{bmatrix}
    
where the factors m and n are calculated for each bus as defined :ref:`here <mn>`.
