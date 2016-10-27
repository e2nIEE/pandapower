.. _shunt:

=============
Shunt
=============

.. seealso::
    :ref:`Unit Systems and Conventions <conventions>`

Create Function
=====================

.. autofunction:: pandapower.create_shunt

Input Parameters
=====================

*net.shunt*

.. tabularcolumns:: |p{0.10\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.4\linewidth}|
.. csv-table:: 
   :file: shunt_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40

\*necessary for executing a power flow calculation.

   
Loadflow Model
=================


.. image:: /elements/shunt/shunt.png
	:width: 12em
	:alt: alternate Text
	:align: center

The power values are given at :math:`v = 1 pu` or :math:`V = V_{N}`:
   
.. math::
   :nowrap:
   
   \begin{align*}
   \underline{S}_{shunt, ref} &= p\_kw + j \cdot q\_kvar
   \end{align*}
   
Since :math:`\underline{S}_{shunt, ref}` is the apparent power at the nominal voltage, we know that:

.. math::
   :nowrap:
   
   \begin{align*}
   \underline{S}_{shunt, ref} &= \frac{\underline{Y}_{shunt}}{V_{N}^2} \\
   \underline{Y}_{shunt} &= \frac{\underline{S}_{shunt, ref}}{V_{N}^2}
   \end{align*}
   
Converting to the per unit system results in:

.. math::
   :nowrap:
   
   \begin{align*}
   \underline{y}_{shunt} &= \frac{\underline{S}_{shunt, ref}}{V_{N}^2} \cdot Z_{N}\\
                         &= \frac{\underline{S}_{shunt, ref}}{V_{N}^2} \cdot \frac{V_{N}^2}{S_{N}} \\
                         &= \frac{S_{shunt, ref}}{S_{N}}
   \end{align*}

with :math:`S_{N} = 1 \ MVA` (see :ref:`Unit Systems and Conventions<conventions>`). 

   
   
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
    p\_kw &= Re(\underline{v}_{bus} \cdot \underline{i}_{shunt}) \\    
    q\_kvar &= Im(\underline{v}_{bus} \cdot \underline{i}_{shunt}) \\    
    vm\_pu &= v_{bus}
    \end{align*}