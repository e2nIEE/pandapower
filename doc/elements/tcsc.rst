.. _tcsc:

=============
Thyristor-Controlled Series Capacitor (TCSC)
=============

We implement the FACTS devices based on the PhD Thesis of Ara Panosyan, PhD.

.. seealso::
    :ref:`Unit Systems and Conventions <conventions>`

Create Function
=====================

.. autofunction:: pandapower.create_tcsc

Input Parameters
=====================

*net.tcsc*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.4\linewidth}|
.. csv-table:: 
   :file: tcsc_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40

\*necessary for executing a power flow calculation.

   
Electric Model
=================


.. image:: tcsc.png
	:width: 12em
	:alt: alternate Text
	:align: center

The series impedance :math:`X_{TCSC}` of the TCSC element is calculated equivalently to SVC, according to the following equation:
   
.. math::
   :nowrap:
   
   \begin{align*}
   X_{TCSC} &= \frac{\pi X_L}{2 (\pi - \alpha) + \sin{(2\alpha)} + \frac{\pi X_L}{X_{Cvar}}}
   \end{align*}

The term :math:`X_L` stands for the reactance of the reactor (x_l_ohm) and the term :math:`X_{Cvar}` stands for the
total capacitance (x_cvar_ohm). The thyristor firing angle :math:`\alpha` is the state variable that on the one hand
defines the impedance of the element, and at the same time is the result of the Newton-Raphson calculation.
The admittance :math:`Y_{TCSC}` equals :math:`-1j \frac{1}{X_{TCSC}}`.

The power flow through the TCSC element is described by the following equation:

.. math::
   :nowrap:

    \begin{align*}
         \begin{bmatrix}
        \underline{S}_{TCSC_i} \\
        \underline{S}_{TCSC_j}
        \end{bmatrix}
        =
        \begin{bmatrix}
        \underline{U}_i \\
        \underline{U}_j
        \end{bmatrix}
        \begin{bmatrix}
        \underline{Y}_{TCSC} & -\underline{Y}_{TCSC}\\
        -\underline{Y}_{TCSC} & \underline{Y}_{TCSC}
        \end{bmatrix}^*
         \begin{bmatrix}
        \underline{U}_i \\
        \underline{U}_j
        \end{bmatrix}^*
    \end{align*}

Result Parameters
==========================
*net.res_tcsc*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.40\linewidth}|
.. csv-table:: 
   :file: svc_res.csv
   :delim: ;
   :widths: 10, 10, 40
