=============
Line
=============


.. |br| raw:: html

    <br>
    
.. seealso::

    :ref:`Unit Systems and Conventions <conventions>` |br|
    :ref:`Standard Type Libraries <std_types>`
    
Create Function
=====================

.. _create_line:

Lines can be either created from the standard type library (create_line) or with custom values (create_line_from_parameters).

.. autofunction:: pandapower.create_line

.. autofunction:: pandapower.create_line_from_parameters

Input Parameters
=============================

*net.line*

.. tabularcolumns:: |p{0.15\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.40\linewidth}|
.. csv-table:: 
   :file: line_par.csv
   :delim: ;
   :widths: 15, 10, 25, 40

\*necessary for executing a power flow calculation.

.. note::

    Defining a line with length zero leads to a division by zero in the power flow and is therefore not allowed. Lines with a very low impedance might lead to convergence problems in the power flow
    for the same reason. If you want to directly connect two buses, please use the switch element instead of a line with a small impedance!

*net.line_geodata*

.. tabularcolumns:: |l|l|l|
.. csv-table:: 
   :file: line_geo.csv
   :delim: ;
   :widths: 10, 10, 55

   
Electric Model
=================

Lines are modelled with the :math:`\pi`-equivalent circuit:

.. image:: line.png
	:width: 25em
	:alt: alternate Text
	:align: center

 
    
The elements in the equivalent circuit are calculated from the parameters in the net.line dataframe as:

.. math::
   :nowrap:

   \begin{align*}
    \underline{Z} &= (r\_ohm\_per\_km + j \cdot x\_ohm\_per\_km) \cdot \frac{length\_km}{parallel}  \\
    \underline{Y}&= j \cdot 2 \pi f \cdot c\_nf\_per\_km \cdot 1 \cdot 10^9 \cdot length\_km \cdot parallel
   \end{align*}
    
The power system frequency :math:`f` is defined when creating an empty network, the default value is :math:`f = 50 Hz`.

The parameters are then transformed in the per unit system:

.. math::
   :nowrap:

   \begin{align*}
    Z_{N} &= \frac{V_{N}^2}{S_{N}} \\
    \underline{z} &= \frac{\underline{Z}}{Z_{N}} \\
    \underline{y} &= \underline{Y} \cdot Z_{N} \\
    \end{align*}

Where :math:`S_{N} = 1 \ MVA` (see :ref:`Unit Systems and Conventions<conventions>`) and :math:`U_{N}` is the nominal voltage at the from bus.

.. note::
    pandapower assumes that nominal voltage of from bus and to bus are equal, which means pandapower does not support lines that connect different voltage levels.
    If you want to connect different voltage levels, either use a transformer or an impedance element.
    
Result Parameters
==========================
   
*net.res_line*

.. tabularcolumns:: |p{0.15\linewidth}|p{0.10\linewidth}|p{0.55\linewidth}|
.. csv-table:: 
   :file: line_res.csv
   :delim: ;
   :widths: 15, 10, 55
   
The power flow results in the net.res_line table are defined as:

.. math::
   :nowrap:
   
   \begin{align*}
    p\_from\_kw &= Re(\underline{v}_{from} \cdot \underline{i}^*_{from}) \\    
    q\_from\_kvar &= Im(\underline{v}_{from} \cdot \underline{i}^*_{from}) \\
    p\_to\_kw &= Re(\underline{v}_{to} \cdot \underline{i}^*_{to}) \\
    q\_to\_kvar &= Im(\underline{v}_{to} \cdot \underline{i}^*_{to}) \\
	pl\_kw &= p\_from\_kw + p\_to\_kw \\
	ql\_kvar &= q\_from\_kvar + q\_to\_kvar \\
    i\_from\_ka &= i_{from} \\
    i\_to\_ka &= i_{to} \\
    i\_ka &= max(i_{from}, i_{to}) \\
    loading\_percent &= \frac{i\_ka}{imax\_ka \cdot df \cdot parallel} \cdot 100 
    \end{align*}
    

*net.res_line_est*

The state estimation results are put into *net.res_line_est* with the same definition as in *net.res_line*.

.. tabularcolumns:: |p{0.15\linewidth}|p{0.10\linewidth}|p{0.55\linewidth}|
.. csv-table:: 
   :file: line_res.csv
   :delim: ;
   :widths: 15, 10, 55
   

Optimal Power Flow Parameters
=============================

The line loading constraint is an upper constraint for loading_percent.

.. tabularcolumns:: |l|l|l|
.. csv-table:: 
   :file: line_opf.csv
   :delim: ;
   :widths: 10, 10, 40