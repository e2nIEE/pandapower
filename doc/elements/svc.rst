.. _svc:

=============
Static Var Compensator (SVC)
=============

We implement the FACTS devices based on the PhD Thesis of Ara Panosyan, PhD.

.. seealso::
    :ref:`Unit Systems and Conventions <conventions>`

Create Function
=====================

.. autofunction:: pandapower.create_svc

Input Parameters
=====================

*net.svc*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.4\linewidth}|
.. csv-table:: 
   :file: svc_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40

\*necessary for executing a power flow calculation.

   
Electric Model
=================


.. image:: svc.png
	:width: 12em
	:alt: alternate Text
	:align: center

The shunt impedance :math:`X_{SVC}` of the SVC element is calculated according to the following equation:
   
.. math::
   :nowrap:
   
   \begin{align*}
   X_{SVC} &= \frac{\pi X_L}{2 (\pi - \alpha) + \sin{(2\alpha)} + \frac{\pi X_L}{X_{Cvar}}}
   \end{align*}

The term :math:`X_L` stands for the reactance of the reactor (x_l_ohm) and the term :math:`X_{Cvar}` stands for the
total capacitance (x_cvar_ohm). The thyristor firing angle :math:`\alpha` is the state variable that on the one hand
defines the impedance of the element, and at the same time is the result of the Newton-Raphson calculation.

The reactive power consumption of the SVC element is calculated with:

.. math::
   :nowrap:
   
   \begin{align*}
   Q_{SVC} = \frac{V^2}{X_{SVC}}
   \end{align*}
   
Where V is the complex voltage observed at the connection bus of the SVC element.
The reference values for the per unit system as defined in :ref:`Unit Systems and Conventions<conventions>`.

Result Parameters
==========================
*net.res_svc*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.40\linewidth}|
.. csv-table:: 
   :file: svc_res.csv
   :delim: ;
   :widths: 10, 10, 40
