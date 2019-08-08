.. _opf:

Optimisation problem
======================

The equation describes the basic formulation of the optimal power flow (OPF) problem.
The pandapower optimal power flow can be constrained by either AC or DC loadflow equations.
The branch constraints represent the maximum apparent power loading of transformers and the maximum line current loadings.
The bus constraints can contain maximum and minimum voltage magnitude and angle.
For the external grid, generators, loads, DC lines and static generators, the maximum and minimum active resp. reactive power can be considered as operational constraints for the optimal power flow.
The constraints are defined element wise in the respective element tables.

.. math::
		& min & \sum_{i  \ \epsilon \ gen, sgen, load, ext\_grid}{f_{i}(P_i)} \\
        & subject \ to \\
        & & loadflow \ equations \\
        & & branch \ constraints  \\
        & & bus \ constraints \\
        & & operational \ power \ constraints \\


**Generator flexibilities / Operational power constraints**

The active and reactive power generation of generators, loads, dc lines and static generators can be defined as a flexibility for the OPF.

.. tabularcolumns:: |p{0.40\linewidth}|p{0.4\linewidth}|
.. csv-table::
   :file: opf_flexibility.csv
   :delim: ;

.. note::
	Defining operational constraints is indispensable for the OPF, it will not start if constraints are not defined.

**Network constraints**

The network constraints contain constraints for bus voltages and branch flows:

.. tabularcolumns:: |p{0.40\linewidth}|p{0.4\linewidth}|
.. csv-table::
   :file: opf_constraints.csv
   :delim: ;

The defaults are unconstraint branch loadings and :math:`\pm 1.0 pu` for bus voltages.

Cost functions
---------------

The cost function is specified element wise and is organized in tables as well, which makes the parametrization user friendly. There are two options formulating a cost function for each element:
A piecewise linear function with :math:`n` data points.

.. math::
        f_{pwl}(p) = f_{\alpha} +(p-p_{\alpha}) \frac{f_{\alpha + 1}-f_{\alpha}}{p_{\alpha + 1}-p_{\alpha}}  \ , \ (p_{\alpha},f_{\alpha}) \ =\begin{cases}
                                                          (p_{0},f_{0}) \ , \ & p_{0} < p <p_{1}) \\
                                                          ...\\
                                                          (p_{n-1},f_{n-1}) \ , & \ p_{n-1} < p <p_{n})
                                                          \end{cases} \\  \\
        f_{pwl}(q) = f_{1} +(q-q_{1}) \frac{f_{2}-f_{1}}{q_{2}-q_{1}}

Piecewise linear cost functions can be specified using create_pwl_costs():


.. autofunction:: pandapower.create_pwl_cost


The other option is to formulate a n-polynomial cost function:

.. math::
        f_{pol}(p) = c_n p^n + ... + c_1 p + c_0 \\
        f_{pol}(q) = c_2 q^2 + c_1 q + c_0

Polynomial cost functions can be specified using create_poly_cost():

.. autofunction:: pandapower.create_poly_cost

.. note::
	Please note, that polynomial costs for reactive power can only be quadratic, linear or constant.
	Piecewise linear cost funcions for reactive power are not working at the moment with 2 segments or more.
	Loads can only have 2 data points in their piecewise linear cost function for active power.

Active and reactive power costs are calculted separately. The costs of all types are summed up to determine the overall costs for a grid state.

Visualization of cost functions
--------------------------------

**Minimizing generation**

The most common optimization goal is the minimization of the overall generator feed in. The according cost function would be formulated like this:

.. code:: python

	pp.create_poly_cost(net, 0, 'sgen', cp1_eur_per_mw=1)
	pp.create_poly_cost(net, 0, 'gen', cp1_eur_per_mw=1)
	pp.create_pwl_cost(net, 0, "sgen", [[net.sgen.min_p_mw.at[0], net.sgen.max_p_mw.at[0], 1]])
	pp.create_pwl_cost(net, 0, "gen", [[net.gen.min_p_mw.at[0], net.gen.max_p_mw.at[0], 1]])



It is a straight with a negative slope, so that it has the highest cost value at p_min_mw and is zero when the feed in is zero:

.. image:: /pics/opf/minimizeload.png
		:width: 20em
		:alt: alternate Text
		:align: center


**Maximizing generation**

This cost function may be used, when the curtailment of renewables should be minimized, which at the same time means that the feed in of those renewables should be maximized. This can be realized by the following cost function definitions:

.. code:: python

	pp.create_poly_cost(net, 0, 'sgen', cp1_eur_per_mw=-1)
	pp.create_poly_cost(net, 0, 'gen', cp1_eur_per_mw=-1)
	pp.create_pwl_cost(net, 0, "sgen", [[net.sgen.min_p_mw.at[0], net.sgen.max_p_mw.at[0], -1]])
	pp.create_pwl_cost(net, 0, "gen", [[net.gen.min_p_mw.at[0], net.gen.max_p_mw.at[0], -1]])


It is a straight with a positive slope, so that the cost is zero at p_min_mw and is at its maximum when the generation equals zero.


.. image:: /pics/opf/maximizeload.png
		:width: 20em
		:alt: alternate Text
		:align: center


**Maximize load**

In case that the load should be maximized, the cost function could be defined like this:

.. code:: python

	pp.create_poly_cost(net, 0, 'load', cp1_eur_per_mw=-1)
	pp.create_poly_cost(net, 0, 'storage', cp1_eur_per_mw=-1)
	pp.create_pwl_cost(net, 0, "load", [[net.load.min_p_mw.at[0], net.load.max_p_mw.at[0], -1]])
	pp.create_pwl_cost(net, 0, "storage", [[net.storage.min_p_mw.at[0], net.storage.max_p_mw.at[0], -1]])




.. image:: /pics/opf/maximizeload.png
		:width: 20em
		:alt: alternate Text
		:align: center

**Minimizing load**

In case that the load should be minimized, the cost function could be defined like this:

.. code:: python

	pp.create_poly_cost(net, 0, 'load', cp1_eur_per_mw=1)
	pp.create_poly_cost(net, 0, 'storage', cp1_eur_per_mw=1)
	pp.create_pwl_cost(net, 0, "load", [[net.load.min_p_mw.at[0], net.load.max_p_mw.at[0], 1]])
	pp.create_pwl_cost(net, 0, "storage", [[net.storage.min_p_mw.at[0], net.storage.max_p_mw.at[0], 1]])


.. image:: /pics/opf/minimizeload.png
		:width: 20em
		:alt: alternate Text
		:align: center

**DC line behaviour**

Please note, that the costs of the DC line transmission are always related to the power at the from_bus!


You can always check your optimization result by comparing your result (From res_sgen, res_load etc.).
