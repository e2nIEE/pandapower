===========================
Three Winding Transformer
===========================

.. |br| raw:: html

    <br>
    
.. seealso::

    :ref:`Unit Systems and Conventions <conventions>` |br|
    :ref:`Standard Type Libraries <std_types>`

Create Function
=====================

.. _create_trafo3w:


.. autofunction:: pandapower.create_transformer3w

.. autofunction:: pandapower.create_transformer3w_from_parameters

.. note::
    All short circuit voltages are given relative to the maximum apparent power
    flow. For example vsc_hv_percent is the short circuit voltage from the high to
    the medium level, it is given relative to the minimum of the rated apparent
    power in high and medium level: min(sn_hv_kva, sn_mv_kva). This is consistent
    with most commercial network calculation software (e.g. PowerFactory).
    Some tools (like PSS Sincal) however define all short ciruit voltages relative
    to the overall rated apparent power of the transformer:
    max(sn_hv_kva, sn_mv_kva, sn_lv_kva). You might have to convert the
    values depending on how the short-circuit voltages are defined.

Input Parameters
=================

*net.trafo3w*

.. tabularcolumns:: |p{0.16\linewidth}|p{0.08\linewidth}|p{0.25\linewidth}|p{0.39\linewidth}|
.. csv-table:: 
   :file: trafo3w_par.csv
   :delim: ;
   :widths: 15, 10, 25, 40

*necessary for executing a loadflow calculation.

   
Loadflow Model
=================

Three Winding Transformers are modelled by three two-winding transformers:

.. image:: /pandapower/elements/trafo3w/trafo3w.png
	:width: 25em
	:alt: alternate Text
	:align: center

The parameters of the three transformers are defined as follows:

.. tabularcolumns:: |p{0.15\linewidth}|p{0.15\linewidth}|p{0.15\linewidth}|p{0.15\linewidth}|
.. csv-table:: 
   :file: trafo3w_conversion.csv
   :delim: ;
   :widths: 10, 15, 15, 15

The definition of the two winding transformer parameter can be found :ref:`here<trafo>`.
    
To calculate the short-circuit voltages :math:`v_{k, t1..t3}` and :math:`v_{r, t1..t3}`, first all short-circuit voltages are converted to the high voltage level:

.. math::
   :nowrap:

   \begin{align*}
    v'_{k, h} &= vsc\_hv\_percent  \\
    v'_{k, m} &= vsc\_mv\_percent \cdot \frac{sn\_hv\_kva}{sn\_mv\_kva} \\
    v'_{k, l} &= vsc\_lv\_percent \cdot \frac{sn\_hv\_kva}{sn\_lv\_kva}
    \end{align*}


    
    
The short-circuit voltages of the three transformers are then calculated as follows:

.. math::
   :nowrap:

   \begin{align*}
    v'_{k, t1} &= \frac{1}{2} (v'_{k, h} + v'_{k, l} - v'_{k, m}) \\
    v'_{k, t2} &= \frac{1}{2} (v'_{k, m} + v'_{k, h} - v'_{k, l}) \\
    v'_{k, t3} &= \frac{1}{2} (v'_{k, m} + v'_{k, l} - v'_{k, h})
    \end{align*}
    
Since these voltages are given relative to the high voltage side, they have to be transformed back to the voltage level of each transformer:

.. math::
   :nowrap:

   \begin{align*}
    v_{k, t1} &= v'_{k, t1} \\
    v_{k, t2} &= v'_{k, t2} \cdot \frac{sn\_mv\_kva}{sn\_hv\_kva} \\
    v_{k, t3} &= v'_{k, t3} \cdot \frac{sn\_lv\_kva}{sn\_hv\_kva}
    \end{align*}

The real part of the short-circuit voltage is calculated in the same way.

.. note::
    All short circuit voltages are given relative to the maximum apparent power
    flow. For example vsc_hv_percent is the short circuit voltage from the high to
    the medium level, it is given relative to the minimum of the rated apparent
    power in high and medium level: min(sn_hv_kva, sn_mv_kva). This is consistent
    with most commercial network calculation software (e.g. PowerFactory).
    Some tools (like PSS Sincal) however define all short circuit voltages relative
    to the overall rated apparent power of the transformer:
    max(sn_hv_kva, sn_mv_kva, sn_lv_kva). You might have to convert the
    values depending on how the short-circuit voltages are defined.

The tap changer adapts the nominal voltages of the transformer in the equivalent to the 2W-Model:

.. tabularcolumns:: |p{0.2\linewidth}|p{0.15\linewidth}|p{0.15\linewidth}|p{0.15\linewidth}|
.. csv-table:: 
   :file: trafo3w_tap.csv
   :delim: ;
   :widths: 20, 15, 15, 15

with 

.. math::
   :nowrap:
   
   \begin{align*}
    n_{tap} = 1 + (tp\_pos - tp\_mid) \cdot \frac{tp\_st\_percent}{100}
    \end{align*}
   
.. seealso::
    `MVA METHOD FOR 3-WINDING TRANSFORMER <https://pangonilo.com/index.php?sdmon=files/MVA_Method_3-Winding_Transformer.pdf>`_


    

Result Parameters
==================
**net.res_trafo3w**

.. tabularcolumns:: |p{0.15\linewidth}|p{0.1\linewidth}|p{0.60\linewidth}|
.. csv-table:: 
   :file: trafo3w_res.csv
   :delim: ;
   :widths: 15, 10, 60

.. math::
   :nowrap:
   
   \begin{align*}
    p\_hv\_kw &= Re(\underline{v}_{hv} \cdot \underline{i}_{hv}) \\    
    q\_hv\_kvar &= Im(\underline{v}_{hv} \cdot \underline{i}_{hv}) \\
    p\_mv\_kw &= Re(\underline{v}_{mv} \cdot \underline{i}_{mv}) \\    
    q\_mv\_kvar &= Im(\underline{v}_{mv} \cdot \underline{i}_{mv}) \\
    p\_lv\_kw &= Re(\underline{v}_{lv} \cdot \underline{i}_{lv}) \\
    q\_lv\_kvar &= Im(\underline{v}_{lv} \cdot \underline{i}_{lv}) \\
	pl\_kw &= p\_hv\_kw + p\_lv\_kw \\
	ql\_kvar &= q\_hv\_kvar + q\_lv\_kvar \\
    i\_hv\_ka &= i_{hv} \\
    i\_mv\_ka &= i_{mv} \\
    i\_lv\_ka &= i_{lv}
    \end{align*}
    
The definition of the transformer loading depends on the trafo_loading parameter of the loadflow.

For trafo_loading="current", the loading is calculated as:

.. math::
   :nowrap:
   
   \begin{align*}  
    loading\_percent &= max(\frac{i_{hv} \cdot vn\_hv\_kv}{sn\_hv\_kva}, \frac{i_{mv} \cdot vn\_mv\_kv}{sn\_mv\_kva}, \frac{i_{lv} \cdot vn\_lv\_kv}{sn\_lv\_kva})  \cdot 100
   \end{align*}
    

For trafo_loading="power", the loading is defined as:
    
.. math::
   :nowrap:
   
   \begin{align*}  
    loading\_percent &= max( \frac{i_{hv} \cdot v_{hv}}{sn\_hv\_kva}, \frac{i_{mv} \cdot v_{mv}}{sn\_mv\_kva}, \frac{i_{lv} \cdot v_{lv}}{sn\_lv\_kva}) \cdot 100
    \end{align*}
