Initial Short-Circuit Current
==================================

The general ohmic network equation is given as: 
    
The SC is calculated in two steps:
    - calculate the SC contribution :math:`I''_{kI}` of all voltage source elements
    - calculate the SC contribution :math:`I''_{kII}` of all current source elements
    
These two currents are then combined into the total initial SC current  :math:`I''_{k} = I''_{kI} + I''_{kII}`.

.. _c:

Equivalent Voltage Source
-----------------------------

For the short-circuit calculation with the equivalent voltage source, all voltage sources are replaced by one equivalent voltage source :math:`V_Q` at the fault location.
The voltage magnitude at the fault bus is assumed to be:

.. math::

    V_Q =
    \left\{
    \begin{array}{@{}ll@{}}
      \frac{c \cdot \underline{V}_{N}}{\sqrt{3}} & \text{for three phase short circuit currents} \\
      \frac{c \cdot \underline{V}_{N}}{2} & \text{for two phase short circuit currents}
    \end{array}\right.
     
where :math:`V_N` is the nominal voltage at the fault bus and c is the voltage correction factor, which accounts for operational deviations from the nominal voltage in the network.

The voltage correction factors :math:`c_{min}` for minimum and :math:`c_{max}` for maximum short-circuit currents are defined for each bus depending on the voltage level.
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


Voltage Source Contribution 
-----------------------------

To calculate the contribution of all voltage source elements, the following assumptions are made:

1. Operational currents at all buses are neglected
2. All current source elements are neglected
3. The voltage at the fault bus is equal to :math:`V_Q`
   
For the calculation of a short-circuit at bus :math:`j`, this yields the following network equations:

.. math::
   
   \begin{bmatrix}
    \underline{Y}_{11} & \dots & \dots & \underline{Y}_{n1} \\[0.3em]
    \vdots & \ddots & & \vdots \\[0.3em]
    \vdots & & \ddots & \vdots \\[0.3em]
    \underline{Y}_{1n} & \dots & \dots & \underline{Y}_{nn}
    \end{bmatrix}
    \begin{bmatrix}
    \underline{V}_{1}  \\
    \vdots  \\
    V_{Qj}  \\
    \vdots  \\
    \underline{V}_{n}
    \end{bmatrix}  
    = 
    \begin{bmatrix}
    0 \\
    \vdots  \\
    \underline{I}''_{kIj} \\
    \vdots  \\
    0 
    \end{bmatrix}

where :math:`\underline{I}''_{kIj}` is the voltage source contribution of the short-circuit current at bus :math:`j`.
The voltages at all non-fault buses and the current at the fault bus are unknown. To solve for :math:`\underline{I}''_{kIj}` , 
we multipliy with the inverted nodal point admittance matrix (impedance matrix):
    
.. math::
   
    \begin{bmatrix}
    \underline{V}_{1}  \\
    \vdots  \\[0.4em]
    V_{Qj}  \\[0.4em]
    \vdots  \\
    \underline{V}_{n}
    \end{bmatrix}  
    = 
    \begin{bmatrix}
    \underline{Z}_{11} & \dots & \dots & \dots & \underline{Z}_{n1} \\
    \vdots & \ddots &  & & \vdots \\
    \vdots & & \underline{Z}_{jj} & & \vdots \\
    \vdots & & & \ddots & \vdots \\
    \underline{Z}_{1n} & \dots & \dots & \dots & \underline{Z}_{nn}
    \end{bmatrix}
    \begin{bmatrix}
    0 \\
    \vdots  \\[0.25em]
    \underline{I}''_{kIj} \\[0.25em]
    \vdots  \\
    0 
    \end{bmatrix}

The short-circuit current for bus m is now given as:

.. math::
   
   I''_{kIj} = \frac{V_{Qj}}{Z_{jj}}

To calculate the vector of the short-circuit currents at all buses, the equation can be expanded as follows:

.. math::
   
    \begin{bmatrix}
    \underline{V}_{Q1} & \dots & \underline{V}_{n1} \\[0.4em]
    \vdots & \ddots & \vdots \\[0.4em]
    \underline{V}_{1n} & \dots & \underline{V}_{Qn}
    \end{bmatrix}  
    = 
    \begin{bmatrix}
    \underline{Z}_{11} & \dots & \underline{Z}_{n1} \\[0.8em]
    \vdots & \ddots & \vdots \\[0.8em]
    \underline{Z}_{1n} & \dots & \underline{Z}_{nn}
    \end{bmatrix}
    \begin{bmatrix}
    \underline{I}''_{kI1} & \dots & 0 \\[0.8em]
    \vdots & \ddots & \vdots \\[0.8em]
    0 & \dots & \underline{I}''_{kIn}
    \end{bmatrix}

which yields:
    
.. math::
   
    \begin{bmatrix}
    I''_{kI1} \\[0.25em]
    \vdots  \\[0.25em]
    I''_{kIn} \\
    \end{bmatrix}
    = 
    \begin{bmatrix}
    \frac{V_{Q1}}{Z_{11}}  \\
    \vdots  \\
    \frac{V_{Qn}}{Z_{nn}} 
    \end{bmatrix}

In that way, all short-circuit currents can be calculated at once with one inversion of the nodal point admittance matrix.

In case a fault impedance is specified, it is added to the diagonal of the impedance matrix. The short-circuit currents
at all buses are then calculated as:

.. math::
     
    \begin{bmatrix}
    I''_{kI1} \\[0.25em]
    \vdots  \\[0.25em]
    I''_{kIn} \\
    \end{bmatrix}
    = 
    \begin{bmatrix}
    \frac{V_{Q1}}{Z_{11} + Z_{fault}}  \\
    \vdots  \\
    \frac{V_{Qn}}{Z_{nn} + Z_{fault}} 
    \end{bmatrix}
    

Current Source Contribution
-----------------------------

To calculate the current source component of the SC current, all voltage sources are short circuited and only current sources are considered. The bus currents are then given as:

.. math::
    \begin{bmatrix}
    I_1 \\[0.2em]
    \vdots  \\[0.2em]
    I_m \\[0.2em]
    \vdots  \\
    I_n
    \end{bmatrix}
    =
    \begin{bmatrix}
    0 \\[0.2em]
    \vdots  \\[0.2em]
    \underline{I}''_{kIIj} \\[0.2em]
    \vdots  \\
    0
    \end{bmatrix}
    -
    \begin{bmatrix}
    I''_{kC1} \\[0.2em]
    \vdots  \\[0.2em]
    \underline{I}''_{kCj} \\[0.2em]
    \vdots  \\
    I''_{kCn}
    \end{bmatrix}
    =
    \begin{bmatrix}
    -I''_{kC1} \\[0.2em]
    \vdots  \\[0.2em]
    \underline{I}''_{kIIj} - \underline{I}''_{kCj} \\[0.2em]
    \vdots  \\
    -I''_{kCn}
    \end{bmatrix}

where :math:`I''_{kC}` are the SC currents that are fed in by converter element at each bus and :math:`\underline{I}''_{kIIj}` is the contribution of converter elements at the fault bus :math:`j`.
With the voltage at the fault bus known to be zero, the network equations are given as:

.. math::

    \begin{bmatrix}
    \underline{V}_{1}  \\
    \vdots  \\[0.4em]
    0  \\[0.4em]
    \vdots  \\
    \underline{V}_{n}
    \end{bmatrix}  
    = 
    \begin{bmatrix}
    \underline{Z}_{11} & \dots & \dots & \dots & \underline{Z}_{n1} \\
    \vdots & \ddots &  & & \vdots \\
    \vdots & & {Z}_{jj} & & \vdots \\
    \vdots & & & \ddots & \vdots \\
    \underline{Z}_{1n} & \dots & \dots & \dots & \underline{Z}_{nn}
    \end{bmatrix}
    \begin{bmatrix}
    -I''_{kC1} \\[0.2em]
    \vdots  \\[0.2em]
    \underline{I}''_{kIIj} - \underline{I}''_{kCj} \\[0.2em]
    \vdots  \\
    -I''_{kCn}
    \end{bmatrix}

From which row :math:`j` of the equation yields:

.. math::

    0 = \underline{Z}_{jj} \cdot \underline{I}''_{kIIj} - \sum_{m=1}^{n}{\underline{Z}_{jm} \cdot \underline{I}_{kCj}}

which can be converted into:
    
.. math::
 
    \underline{I}''_{kIIj} = \frac{1}{\underline{Z}_{jj}} \cdot \sum_{m=1}^{n}{\underline{Z}_{jm} \cdot \underline{I}_{kC, m}}

To calculate all SC currents for faults at each bus simultaneously, this can be generalized into the following matrix equation:

.. math::

    \begin{bmatrix}
    \underline{I}''_{kII1} \\[0.5em]
    \vdots  \\[0.5em]
    \vdots  \\[0.5em]
    \underline{I}''_{kIIn}
    \end{bmatrix} = 
    \begin{bmatrix}
    \underline{Z}_{11} & \dots & \dots & \underline{Z}_{n1} \\[0.3em]
    \vdots & \ddots & & \vdots \\[0.3em]
    \vdots & & \ddots & \vdots \\[0.3em]
    \underline{Z}_{1n} & \dots & \dots & \underline{Z}_{nn}
    \end{bmatrix}
    \begin{bmatrix}
    \frac{I''_{kC1}}{\underline{Z}_{11}} \\[0.25em]
    \vdots  \\
    \vdots  \\[0.25em]
    \frac{I''_{kCn}}{\underline{Z}_{nn}}
    \end{bmatrix}
