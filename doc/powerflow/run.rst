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
Hitherto we have two cost function options, that are fitting our special use cases. In the future, this module will be developed further in order to allow opf calculations with more flexible cost functions. 

You can find the documentation of the pypower OPF under http:/www.pserc.cornell.edu/matpower/manual.pdf
In theory, the pandapower OPF can do exactly the same, it just requires an extension of the pandapower <-> pypower interface

**Available cost functions**

*Linear costs*

.. math::
		min & \sum_{i  \ \epsilon \ gen }{P_{g,i} * w_{g,i }} \\ 
        & subject   \ to \\
        & P_{max,i} <= P_{g,i} < P_{min,i}, i  \ \epsilon \ gen   \\
        & U_{min,j} <= Q_{g,i} < Q_{min,i}, j  \ \epsilon \ bus   \\
        & Q_{max,i} <= Q_{g,i} < Q_{min,i}, i  \ \epsilon \ gen   \\
        & S_{k} < S_{max,k}, k \ \epsilon \ trafo  \\
        & I_{l} < I_{max,l}, l \ \epsilon \ line
        
        
        
Where :math:`gen` contains all generators and controllable static generators. The weighted costs :math:`w_{g,i}` can be defined in the pandapower Generator and Static generator tables, see :ref:`elements`. 


*Linear costs with loss minimization"


.. math::
		min & \sum_{i  \ \epsilon \ gen }{P_{g,i} * w_{g,i }} + \sum_{l  \ \epsilon \ line }{P_{loss,l}} \\ 
        & subject   \ to \\
        & P_{max,i} <= P_{g,i} < P_{min,i}, i  \ \epsilon \ gen   \\
        & U_{min,j} <= Q_{g,i} < Q_{min,i}, j  \ \epsilon \ bus   \\
        & Q_{max,i} <= Q_{g,i} < Q_{min,i}, i  \ \epsilon \ gen   \\
        & S_{k} < S_{max,k}, k \ \epsilon \ trafo  \\
        & I_{l} < I_{max,l}, l \ \epsilon \ line
   
**OPF Output**

The result of the OPF are written to the "official" pandapower result tables such as res_bus, res_sgen, res_gen etc. 
So you find the optimal setting values of the generator in their respective result tables.
Also you can see the resulting voltage and loading values in res_line, res_trafo and res_bus.




**OPF caveats**

The costs are not respected for uncontrollable Static Generators. 
If a Generator should not be respected as a flexibility, the Power limits can be set to the actual power value (+/- a certain inaccuracy) to take away the flexibility for the OPF.
