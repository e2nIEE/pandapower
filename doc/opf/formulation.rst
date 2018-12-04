.. _opf:

Optimisation problem
======================

The equation describes the basic formulation of the optimal power flow problem. The pandapower optimal power flow can be constrained by either, AC and DC loadflow equations. The branch constraints represent the maximum apparent power loading of transformers and the maximum line current loadings. The bus constraints can contain maximum and minimum voltage magnitude and angle. For the external grid, generators, loads, DC lines and static generators, the maximum and minimum active resp. reactive power can be considered as operational constraints for the optimal power flow. The constraints are defined element wise in the respective element tables.

.. math::
		& min & \sum_{i  \ \epsilon \ gen, sgen, load, extgrid }{f_{i}(P_i)} \\
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

Visualization of cost functions
--------------------------------

**Minimizing Generation**

The most common optimization goal is the minimization of the overall generator feed in. The according cost function would be formulated like this:

.. code:: python
	
	pp.create_polynomial_cost(net, 0, 'sgen', np.array([-1, 0]))
	pp.create_polynomial_cost(net, 0, 'gen', np.array([-1, 0]))
	pp.create_polynomial_cost(net, 0, 'ext_grid', np.array([-1, 0]))
	pp.create_piecewise_linear_cost(net, 0, "sgen", np.array([[net.sgen.min_p_kw.at[0], 1000], [0, 0]]))
	pp.create_piecewise_linear_cost(net, 0, "gen", np.array([[net.gen.min_p_kw.at[0], 1000], [0, 0]]))
	pp.create_piecewise_linear_cost(net, 0, "ext_grid", np.array([[-1e9, 1e9], [1e9, -1e9]]))

	
	
It is a straight with a negative slope, so that it has the highest cost value at p_min_kw and is zero when the feed in is zero:

.. image:: /pics/opf/minimizegeneration.png
		:width: 20em
		:alt: alternate Text
		:align: center

		
**Maximizing generation**

This cost function may be used, when the curtailment of renewables should be minimized, which at the same time means that the feed in of those renewables should be maximized. This can be realized by the following cost function definitions:
		
.. code:: python
	
	pp.create_polynomial_cost(net, 0, 'sgen', np.array([1, 0]))
	pp.create_polynomial_cost(net, 0, 'gen', np.array([1, 0]))
	pp.create_piecewise_linear_cost(net, 0, "sgen", np.array([[net.sgen.min_p_kw.at[0], -1000], [0, 0]]))
	pp.create_piecewise_linear_cost(net, 0, "gen", np.array([[net.gen.min_p_kw.at[0], -1000], [0, 0]]))
	pp.create_piecewise_linear_cost(net, 0, "ext_grid", np.array([[-1e9, -1e9], [1e9, 1e9]]))

	
It is a straight with a positive slope, so that the cost is zero at p_min_kw and is at its maximum when the generation equals zero.
		
		
.. image:: /pics/opf/maximizegeneration.png
		:width: 20em
		:alt: alternate Text
		:align: center


**Maximize load**

In case that the load should be maximized, the cost function could be defined like this:
		
.. code:: python
	
	pp.create_polynomial_cost(net, 0, 'load', np.array([-1, 0]))
	pp.create_polynomial_cost(net, 0, 'storage', np.array([-1, 0]))
	pp.create_piecewise_linear_cost(net, 0, "sgen", np.array([[0, 0], [net.load.max_p_kw.at[0], -1000]]))
	pp.create_piecewise_linear_cost(net, 0, "gen", np.array([[net.storage.min_p_kw.at[0], 1000], [net.storage.max_p_kw.at[0], -1000]]))

	
	
		
.. image:: /pics/opf/maximizeload.png
		:width: 20em
		:alt: alternate Text
		:align: center
		
**Minimizing load** 

In case that the load should be minimized, the cost function could be defined like this:
		
.. code:: python
	
	pp.create_polynomial_cost(net, 0, 'load', np.array([1, 0]))
	pp.create_polynomial_cost(net, 0, 'storage', np.array([1, 0]))
	pp.create_piecewise_linear_cost(net, 0, "sgen", np.array([[0, 0], [net.load.max_p_kw.at[0], 1000]]))
	pp.create_piecewise_linear_cost(net, 0, "gen", np.array([[net.storage.min_p_kw.at[0], -1000], [net.storage.max_p_kw.at[0], 1000]]))


.. image:: /pics/opf/minimizeload.png
		:width: 20em
		:alt: alternate Text
		:align: center

**DC line behaviour**

Please note, that the costs of the DC line transmission are always related to the power at the from_bus! 

		
You can always check your Optimization result by comparing your result (From res_sgen, res_load etc.).      
      