Symmetric Three-Phase Current
==================================
    
Initial Short-Circuit Current
---------------------------------

The general ohmic network equation is given as: 

.. math::
   
    \begin{bmatrix}
    \underline{Y}_{11} & \dots & \dots & \underline{Y}_{n1} \\
    \vdots & \ddots & & \vdots \\
    \vdots & & \ddots & \vdots \\
    \underline{Y}_{1n} & \dots & \dots & \underline{Y}_{nn}
    \end{bmatrix}
    \begin{bmatrix}
    \underline{V}_{1} \\
    \vdots  \\
    \vdots  \\
    \underline{V}_{n}
    \end{bmatrix}   
    = 
    \begin{bmatrix}
    \underline{I}_{1} \\
    \vdots  \\
    \vdots  \\
    \underline{I}_{n}
    \end{bmatrix}

For the short-circuit calculation with the equivalent voltage source, two assumptions are made:

1. All operational currents are neglected
2. The voltage at the fault bus is assumed to be :math:`\frac{c \cdot \underline{V}_{N}}{\sqrt{3}}`

where :math:`V_N` is the nominal voltage at the fault bus and c is the :ref:`voltage correction factor <c>`.
    
For the calculation of a short-circuit at bus m, this yields the following equations:

.. math::
   
   \begin{bmatrix}
    \underline{Y}_{11} & \dots & \dots & \underline{Y}_{n1} \\
    \vdots & \ddots & & \vdots \\
    \vdots & & \ddots & \vdots \\
    \underline{Y}_{1n} & \dots & \dots & \underline{Y}_{nn}
    \end{bmatrix}
    \begin{bmatrix}
    \underline{V}_{1}  \\
    \vdots  \\
    \frac{c_m \cdot \underline{V}_{N, m}}{\sqrt{3}}  \\
    \vdots  \\
    \underline{V}_{n}
    \end{bmatrix}  
    = 
    \begin{bmatrix}
    0 \\
    \vdots  \\
    \underline{I}''_{k, m} \\
    \vdots  \\
    0 
    \end{bmatrix}

where :math:`\underline{I}''_{k, m}` is the short-circuit current at bus m and all other bus currents are assumed to be zero.
The voltages at all non-fault buses and the current at the fault bus are unknown. To solve for :math:`\underline{I}''_{k, m}` , 
we multipliy with the inverted nodal point admittance matrix (impedance matrix):
    
.. math::
   
    \begin{bmatrix}
    \underline{V}_{1}  \\
    \vdots  \\
    \frac{c_k \cdot \underline{V}_{N, k}}{\sqrt{3}}  \\
    \vdots  \\
    \underline{V}_{n}
    \end{bmatrix}  
    = 
    \begin{bmatrix}
    \underline{Z}_{11} & \dots & \dots & \underline{Z}_{n1} \\
    \vdots & \ddots & & \vdots \\
    \vdots & & \ddots & \vdots \\
    \underline{Z}_{1n} & \dots & \dots & \underline{Z}_{nn}
    \end{bmatrix}
    \begin{bmatrix}
    0 \\
    \vdots  \\
    \underline{I}''_{k, m} \\
    \vdots  \\
    0 
    \end{bmatrix}

The short-circuit current for bus m is now given as:

.. math::
   
   I''_{k, m} = \frac{c \cdot V_{N, m}}{\sqrt{3} \cdot Z_{mm}}

To calculate the vector of the short-circuit currents at all buses, the equation can be expanded as follows:

.. math::
   
    \begin{bmatrix}
    \underline{V}_{1}  \\
    \vdots  \\
    \frac{c_k \cdot \underline{V}_{N, k}}{\sqrt{3}}  \\
    \vdots  \\
    \underline{V}_{n}
    \end{bmatrix}  
    = 
    \begin{bmatrix}
    \underline{Z}_{11} & \dots & \dots & \underline{Z}_{n1} \\
    \vdots & \ddots & & \vdots \\
    \vdots & & \ddots & \vdots \\
    \underline{Z}_{1n} & \dots & \dots & \underline{Z}_{nn}
    \end{bmatrix}
    \begin{bmatrix}
    \underline{I}_{k, 0} & \dots & \dots & 0 \\
    \vdots & \ddots & & \vdots \\
    \vdots & & \ddots & \vdots \\
    0 & \dots & \dots & \underline{I}_{k, n}
    \end{bmatrix}

which yields:
    
.. math::
   
    \begin{bmatrix}
    \underline{I}''_{k, 0} \\
    \vdots  \\
    \underline{I}''_{k, m} \\
    \end{bmatrix}
    = 
    \begin{bmatrix}
    \frac{c_1 \cdot \underline{V}_{N, 1}}{\sqrt{3} \cdot Z_{11}}  \\
    \vdots  \\
    \frac{c_n \cdot \underline{V}_{N, n}}{\sqrt{3} \cdot Z_{nn}} \\
    \end{bmatrix}

In that way, all short-circuit currents can be calculated at once with one inversion of the nodal point admittance matrix.

In case a fault impedance is specified, it is added to the diagonal of the impedance matrix. The short-circuit currents
at all buses are then calculated as:

.. math::
   
    \begin{bmatrix}
    \underline{I}''_{k, 1} \\
    \vdots  \\
    \underline{I}''_{k, m} \\
    \end{bmatrix}
    = 
    \begin{bmatrix}
    \frac{c_1 \cdot \underline{V}_{N, 1}}{\sqrt{3} \cdot (Z_{11}  + Z_{fault})}  \\
    \vdots  \\
    \frac{c_n \cdot \underline{V}_{N, n}}{\sqrt{3} \cdot (Z_{nn} + Z_{fault})} 
    \end{bmatrix}


Peak Short-Circuit Current
---------------------------------

The peak short-circuit current is calculated as:

.. math::

    \begin{bmatrix}
    i_{p, 1}  \\
    \vdots  \\
    i_{p, n}  \\
    \end{bmatrix}
    = \sqrt{2}
    \begin{bmatrix}
    \kappa_{1}  \\
    \vdots  \\
    \kappa_{1}   \\
    \end{bmatrix}
    \begin{bmatrix}
    \underline{I}''_{k, 1} \\
    \vdots  \\
    \underline{I}''_{k, n} \\
    \end{bmatrix}

where the factor :math:`\kappa` is calculated  for each bus as defined :ref:`here <kappa>`.
    
Thermal Short-Circuit Current
---------------------------------

The equivalent 

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
    
where the factors m and n are calculated for each bus as defined :ref:`here <mn>`.
