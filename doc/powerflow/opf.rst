.. _opf:

Optimal Power Flow
=====================

Pandapower provides an interface for AC and DC optimal power flow calculations. In the following, it is presented how the optimisation problem can be formulated with the pandapower data format.

.. note::

    We highly recommend the tutorials for the usage of the optimal power flow.

Optimisation problem
---------------------

The equation describes the basic formulation of the optimal power flow problem. The pandapower optimal power flow can be constrained by either, AC and DC loadflow equations. The branch constraints represent the maximum apparent power loading of transformers and the maximum line current loadings. The bus constraints can contain maximum and minimum voltage magnitude and angle. For the external grid, generators, loads, DC lines and static generators, the maximum and minimum active resp. reactive power can be considered as operational constraints for the optimal power flow. The constraints are defined element wise in the respective element tables.

.. math::
		& min & \sum_{i  \ \epsilon \ gen, sgen, load, extgrid }{P_{i} * f_{i}(P_i)} \\
        & subject \ to \\
        & & Loadflow \ equations \\
        & & branch \ constraints  \\
        & & bus \ constraints \\
        & & operational \ power \ constraints \\
        
        
**Generator Flexibilities / Operational power constraints**

The active and reactive power generation of generators, loads, dc lines and static generators can be defined as a flexibility for the OPF.

.. tabularcolumns:: |p{0.40\linewidth}|p{0.4\linewidth}|
.. csv-table:: 
   :file: opf_flexibility.csv
   :delim: ;

.. note::
	Defining operational constraints is indispensable for the OPF, it will not start if contraints are not defined. 

**Network Constraints**

The network constraints contain constraints for bus voltages and branch flows:

.. tabularcolumns:: |p{0.40\linewidth}|p{0.4\linewidth}|
.. csv-table:: 
   :file: opf_constraints.csv
   :delim: ;
           
The defaults are 100% loading for branch elements and +-0.1 p.u. for bus voltages.
		   
Cost functions
---------------

The cost function is specified element wise and is organized in tables as well, which makes the parametrization user friendly. There are two options formulating a cost function for each element: 
A piecewise linear function with $n$ data points.

.. math::
        f_{pwl}(p) = f_{\alpha} +(p-p_{\alpha}) \frac{f_{\alpha + 1}-f_{\alpha}}{p_{\alpha + 1}-p_{\alpha}}  \ , \ (p_{\alpha},f_{\alpha}) \ =\begin{cases}
                                                          (p_{0},f_{0}) \ , \ & p_{0} < p <p_{1}) \\
                                                          ...\\
                                                          (p_{n-1},f_{n-1}) \ , & \ p_{n-1} < p <p_{n})
                                                          \end{cases} \\  \\
        f_{pwl}(q) = f_{1} +(q-q_{1}) \frac{f_{2}-f_{1}}{q_{2}-q_{1}}  
                                                        
Piecewise linear cost functions can be specified using create_piecewise_linear_costs():


.. autofunction:: pandapower.create_piecewise_linear_cost


The other option is to formulate a n-polynomial cost function:

.. math::
        f_{pol}(p) = c_n p^n + ... + c_1 p + c_0 \\
        f_{pol}(q) = c_2 q^2 + c_1 q + c_0

Polynomial cost functions can be speciefied using create_polynomial_cost():
        
.. autofunction:: pandapower.create_polynomial_cost 

.. note::
	Please note, that polynomial costs for reactive power can only be quadratic, linear or constant.
	Piecewise linear cost funcions for reactive power are not working at the moment with 2 segments or more.
	Loads can only have 2 data points in their piecewise linear cost function for active power.

Active and reactive power costs are calculted seperately. The costs of all types are summed up to determine the overall costs for a grid state.

Parametrisation of the calculation
-----------------------------------

The internal solver uses the interior point method. By default, the initial state is the center of the operational constraints. Another option would be to initialize the optimisation with a valid loadflow solution. For optimiation of a timeseries, this warm start possibilty could imply a significant speedup. 
This is not yet provided in the actual version, but could be an useful extension in the future.
Another parametrisation for the AC OPF is, if voltage angles should be considered, which is the same option than for the loadflow calculation with pandapower.runpp: 

.. autofunction:: pandapower.runopp

References:
      - "On the Computation and Application of Multi-period
        Security-Constrained Optimal Power Flow for Real-time
        Electricity Market Operations", Cornell University, May 2007.
      - H. Wang, C. E. Murillo-Sanchez, R. D. Zimmerman, R. J. Thomas,
        "On Computational Issues of Market-Based Optimal Power Flow",
        IEEE Transactions on Power Systems, Vol. 22, No. 3, Aug. 2007,
        pp. 1185-1193.
      - R. D. Zimmerman, C. E. Murillo-Sánchez, and R. J. Thomas, "MATPOWER: Steady-State 
        Operations, Planning and Analysis Tools for Power Systems Research and Education," 
        Power Systems, IEEE Transactions on, vol. 26, no. 1, pp. 12-19, Feb. 2011.
      
      