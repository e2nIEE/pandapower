.. _shunt:

=============
Shunt
=============

.. seealso::
    :ref:`Unit Systems and Conventions <conventions>`

Create Function
=====================

.. autofunction:: pandapower.create.create_shunt

.. autofunction:: pandapower.create.create_shunt_as_capacitor


Input Parameters
=====================

*net.shunt*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.4\linewidth}|
.. csv-table:: 
   :file: shunt_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40

\*necessary for executing a power flow calculation.

   
Electric Model
=================


.. image:: shunt.png
	:width: 12em
	:alt: alternate Text
	:align: center

The power values are given at :math:`v = 1` pu and are scaled linearly with the number of steps:
   
.. math::
   :nowrap:
   
   \begin{align*}
   \underline{S}_{shunt, ref} &= (p\_mw + j \cdot q\_mvar) \cdot step
   \end{align*}
   
Since :math:`\underline{S}_{shunt, ref}` is the apparent power at the nominal voltage, we know that:

.. math::
   :nowrap:
   
   \begin{align*}
   \underline{Y}_{shunt} = \frac{\underline{S}_{shunt, ref}}{vn\_kv^2}
   \end{align*}
   
Converting to the per unit system results in:

.. math::
   :nowrap:
   
   \begin{align*}
   \underline{y}_{shunt} &= \frac{\underline{S}_{shunt, ref}}{V_{N}^2} \cdot Z_{N}\\
                         &= \frac{\underline{S}_{shunt, ref}}{V_{N}^2} \cdot \frac{V_{N}^2}{S_{N}} \\
                         &= \frac{S_{shunt, ref}}{S_{N}}
   \end{align*}

with the reference values for the per unit system as defined in :ref:`Unit Systems and Conventions<conventions>`.
   
Result Parameters
==========================
*net.res_shunt*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.40\linewidth}|
.. csv-table:: 
   :file: shunt_res.csv
   :delim: ;
   :widths: 10, 10, 40

.. math::
   :nowrap:
   
   \begin{align*}
    p\_mw &= Re(\underline{v}_{bus} \cdot \underline{i}_{shunt}) \\    
    q\_mvar &= Im(\underline{v}_{bus} \cdot \underline{i}_{shunt}) \\    
    vm\_pu &= v_{bus}
    \end{align*}
