=====================
Run a Power Flow
=====================
.. _ppLoadflow:

.. note::

    If you are interested in the pypower casefile that pandapower is using for power flow, you can find it in net["_mpc_last_cycle"].
    However all necessary informations are written into the pandpower format net, so the pandapower user should not usually have to deal with pypower.

AC Power Flow
=====================

.. autofunction:: pandapower.runpp

.. warning::
    Neglecting voltage angles is only valid in radial networks! pandapower was developed for distribution networks, which is why omitting the voltage angles is the default. 
    However be aware that voltage angle differences in networks with multiple galvanically coupled external grids lead to balancing power flows between slack nodes.  
    That is why voltage angles always have to be considered in meshed network, such as in the sub-transmission level!


DC Power flow
=====================
.. _ppDCPF:

.. warning::
    To run an AC power flow with DC power flow initialization, use the AC power flow with init="dc".

.. autofunction:: pandapower.rundcpp


Optimal Power Flow
=====================
.. _ppOPF:


**pandapower optimal power flow**

The pandapower optimal power flow is the tool for optimizing your grid state. It is reading constraints and costs from the pandapower tables and is processing it into the pypower case file that is handed over to the pypower OPF.
Hitherto we only have a limited set of constraints and cost function options, that is fitting our special use cases. In the future, this module will be developed further in order to allow opf calculations with more felxible cost functions. 

You can find the documentation of the pypower OPF under http:/www.pserc.cornell.edu/matpower/manual.pdf
In theory, the pandapower OPF can do exactly the same, it just requires an extension of the pandapower <-> pypower interface


.. note::


    The pandapower OPF is a recent development that is still growing. If you have further requirements you are very welcome to design your own cost functions or use your own solvers. 
    For coordination, please contact friederike.meier@uni-kassel.de
    Also we like to discuss future use cases in our weekly meeting.


**Input constraints**

.. tabularcolumns:: |p{0.12\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.30\linewidth}|
.. csv-table:: 
   :file: opf_constraints.csv
   :delim: ;
   :widths: 10, 10, 25

**Input Costs**

The costs for power can be written into the pandapower tables. They will be used to create the pypower gencost array in the future.

.. tabularcolumns:: |p{0.12\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.30\linewidth}|
.. csv-table:: 
   :file: opf_cost.csv
   :delim: ;
   :widths: 10, 10, 25

**Available cost functions**

At the moment we only have the following cost function:

.. math::
		max\{P_G\} \hspace{1cm}\text{subject to:}\hspace{1cm} U_{min} <\ U_B\ &< U_{max}\\
		P_{min} <\ P_G\ &< P_{max}\\
		Q_{min} <\ Q_G\ &< Q_{max}\\
		I_{Branch} &<\ I_{max}
   
**OPF Output**

The result of the OPF are written to the "official" pandapower result tables such as res_bus, res_sgen, res_gen etc. 
So you find the optimal setting values of the generator in their respective result tables.
Also you can see the resulting voltage and loading values in res_line, res_trafo and res_bus.




