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

Shunt characteristic table
============================

A shunt characteristic table (shunt_characteristic_table) can be used to adjust the shunt parameters
(q_mvar, p_mw) according to the selected step position. This lookup table is created automatically
from version 3.0 onwards through the CIM CGMES to pandapower converter (if this information is available in the EQ
profile), or the user may define this table manually. The id_characteristic_table variable in net.shunt references
the id_characteristic column in net.shunt_characteristic_table per shunt.

If the shunt_dependency_table variable in net.shunt is set to True, this indicates that there is a corresponding
characteristic available in net.shunt_characteristic_table, which overwrites the default shunt parameters
q_mvar and p_mw.

The below table provides an example shunt_characteristic_table, populated for two shunt elements.

.. tabularcolumns:: |p{0.10\linewidth}|p{0.55\linewidth}|p{0.15\linewidth}|p{0.20\linewidth}|p{0.20\linewidth}
.. csv-table::
   :file: shunt_char_table.csv
   :delim: ,
   :widths: 10, 20, 15, 20, 20

.. note::
    shunt_dependency_table has to be set to True and the id_characteristic_table variable needs to be populated in order to consider the corresponding shunt_characteristic_table values.

The function pandapower.control.shunt_characteristic_table_diagnostic can be used for sanity checks.
The function pandapower.control.create_shunt_characteristic_object can be used to automatically create
SplineCharacteristic objects and populate the net.shunt_characteristic_spline table according to the
net.shunt_characteristic_table table. An additional column id_characteristic_spline is also created in net.shunt
to set up the reference to the spline characteristics.

The below table provides an example shunt_characteristic_spline table, populated for two shunt elements.

.. tabularcolumns:: |p{0.10\linewidth}|p{0.55\linewidth}|p{0.55\linewidth}|p{0.55\linewidth}
.. csv-table::
   :file: shunt_char_spline.csv
   :delim: ,
   :widths: 10, 20, 30, 30

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
