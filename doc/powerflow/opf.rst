.. _opf:

Optimal Power Flow
=====================

The pandapower optimal power flow is a tool for optimizing the grid state. It offers two cost function options, that are fitting special use cases. 
In addition to the equality constraints in the problem formulations below, the full set of nonlinear real and reactive power balance equations is always considered in the pandapower optimal power flow.
To understand the usage, the OPF tutorial is recommended (see :ref:`tutorial`).

pandapower uses PYPOWER to solve the optimal power flow problem:

.. image:: /pics/pandapower_optimal_powerflow.png
		:width: 40em
		:alt: alternate Text
		:align: center

.. autofunction:: pandapower.runopp

.. note::

    If you are interested in the pypower casefile that pandapower is using for power flow, you can find it in net["_ppc_opf"].
    However all necessary informations are written into the pandpower format net, so the pandapower user should not usually have to deal with pypower.

    

**Generator Flexibilities**

The active and reactive power generation of generators and static generators can be defined as a flexibility for the OPF.

.. tabularcolumns:: |p{0.40\linewidth}|p{0.4\linewidth}|
.. csv-table:: 
   :file: opf_flexibility.csv
   :delim: ;

If you do want to have flexible active but fixed reactive power, you can set the maximum and minimum reactive power constraints to the same value 
(might cause convergence problems, in which case relax the constraint a little bit). If you want to set active and reactive power without flexibility for a specific generator,
set the controllable flag (net.gen.controllable / net.sgen.controllable) to False .
   
.. note::
    Only (static) generators with controllable = True are considered as flexible. If controllable == False, the (static) generators is considered fixed as in a regular power flow.
    
   
**Network Constraints**

Constraints can be defined for bus voltages, line and transformer loading.

.. tabularcolumns:: |p{0.40\linewidth}|p{0.4\linewidth}|
.. csv-table:: 
   :file: opf_constraints.csv
   :delim: ;
        

**Cost Functions**

The default cost function summs up the weighted generator costs:

.. math::
		min & \sum_{i  \ \epsilon \ gen }{P_{g,i} * w_{g,i }} \\

The generator costs are defined in the "cost_per_km" column in net.gen, net.sgen and net.ext_grid respectively.
If no costs are specified, the costs default to 1, which results in a minimization of overall power injection.

By specifying objectivetype='linear_minloss' in the OPF, the cost function is evaluated as a summation of the weighted generator costs and the sum of the line losses:

.. math::
		min & \sum_{i  \ \epsilon \ gen }{P_{g,i} * w_{g,i }} + \lambda \sum_{l  \ \epsilon \ line }{P_{loss,l}} \\ 
     
The weighting factor :math:`\lambda` can be specified as the keyword argument lambda_opf for runopp().
   
.. note::

    The pandapower optimal power flow is a recent development. In the future, this module will be developed further in order to allow more flexible opf calculations.
