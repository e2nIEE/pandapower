=============
Impedance
=============

.. seealso::
    :ref:`Unit Systems and Conventions <conventions>`

Create Function
=====================

.. autofunction:: pandapower.create_impedance

Parameters
=============

*net.impedance*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.15\linewidth}|p{0.4\linewidth}|
.. csv-table:: 
   :file: impedance_par.csv
   :delim: ;
   :widths: 10, 10, 15, 40

*necessary for executing a loadflow calculation.

   
Loadflow Model
=================

The impedance is modelled as a simple longitudinal per unit impedance:

.. image:: /pandapower/elements/impedance/impedance.png
	:width: 25em
	:alt: alternate Text
	:align: center

The per unit values given in the parameter table are assumed to be relative to the rated voltage of from and to bus as well as to the apparent power given in the table.
The per unit values are therefore transformed into the network per unit system:

.. math::
   :nowrap:

   \begin{align*}
    \underline{z}_{impedance} &= r\_pu + j \cdot x_{pu} \\
    \underline{z} &= \underline{z}_{impedance} \frac{S_{N}}{sn\_kva}
    \end{align*}

with :math:`S_{N} = 1 \ MVA` (see :ref:`Unit Systems and Conventions<conventions>`). 


Result Parameters
==================
*net.res_impedance*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.1\linewidth}|p{0.55\linewidth}|
.. csv-table:: 
   :file: impedance_res.csv
   :delim: ;
   :widths: 10, 10, 55

.. math::
   :nowrap:
   
   \begin{align*}
    i\_from\_ka &= i_{from}\\
    i\_to\_ka &= i_{to}\\
    p\_from\_kw &= Re(\underline{v}_{from} \cdot \underline{i}_{from}) \\    
    q\_from\_kvar &= Im(\underline{v}_{from} \cdot \underline{i}_{from}) \\
    p\_to\_kw &= Re(\underline{v}_{to} \cdot \underline{i}_{to}) \\
    q\_to\_kvar &= Im(\underline{v}_{to} \cdot \underline{i}_{to}) \\
	pl\_kw &= p\_from\_kw + p\_to\_kw \\
	ql\_kvar &= q\_from\_kvar + q\_to\_kvar \\
    \end{align*}