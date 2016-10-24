=============
Extended Ward
=============

.. seealso::
    :ref:`Unit Systems and Conventions <conventions>`

Create Function
=====================

.. autofunction:: pandapower.create_xward

Input Parameters
=================

*net.xward*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.1\linewidth}|p{0.15\linewidth}|p{0.55\linewidth}|
.. csv-table:: 
   :file: xward_par.csv
   :delim: ;
   :widths: 10, 10, 15, 55

*necessary for executing a loadflow calculation.

   
Loadflow Model
=================

The extended ward equivalent is a :ref:`ward equivalent<ward>`: with additional PV-node with an internal resistance.

.. image:: /pandapower/elements/xward/xward.png
	:width: 25em
	:align: center

The constant apparent power is given by:

.. math::
   :nowrap:
   
   \begin{align*}
   P_{const} &= ps\_kw\\
   Q_{const} &= qs\_kvar\\
   \end{align*}
    
The shunt admittance part of the extended ward equivalent is calculated as described :ref:`here<shunt>`:

.. math::
   :nowrap:
   
   \begin{align*}
   \underline{y}_{shunt} &= \frac{pz\_kw + j \cdot qz\_kvar}{S_{N}}
   \end{align*}

The internal resistance is defined as:

.. math::
   :nowrap:
   
   \begin{align*}
   \underline{z}_{int} &= r\_pu + j \cdot x\_pu
   \end{align*}
   
The internal voltage source is modelled as a PV-node (:ref:`generator<gen>`) with:

.. math::
   :nowrap:
   
   \begin{align*}
   p\_kw &= 0 \\
   vm\_pu &= vm\_pu
   \end{align*}

Result Parameters
==================
*net.res_xward*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.1\linewidth}|p{0.50\linewidth}|
.. csv-table:: 
   :file: xward_res.csv
   :delim: ;
   :widths: 10, 10, 50

   

.. math::
   :nowrap:
   
   \begin{align*}
   vm\_pu &= v_{bus} \\
   p\_kw &= P_{const} + Re(\frac{\underline{V}_{bus}^2}{\underline{Y}_{shunt}}) + Re(\underline{I}_{int} \cdot \underline{V}_{bus}) \\
   q_kvar &= Q_{const} + Im(\frac{\underline{V}_{bus}^2}{\underline{Y}_{shunt}} + Im(\underline{I}_{int} \cdot \underline{V}_{bus}) )
   \end{align*}