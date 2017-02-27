.. _opf:

Optimal Power Flow
=====================

Pandapower provides an interface for AC and DC optimal power flow calculations. In the following, it is presented how the optimisation problem can be formulated with the pandapower data format and how it is processed into the pypower internal format.


Optimisation problem
---------------------

Equation \refhere describes the basic formulation of the optimal power flow problem. The pandapower optimal power flow can be constrained by both, AC and DC loadflow equations. The branch constraints represent the maximum apparent power loading of transformers and the maximum line current loadings. The bus constraints can contain maximum and minimum voltage magnitude and angle. For the external grid, generators, loads and static generators, the maximum and minimum active resp. reactive power can be considered as operational constraints for the optimal power flow. The constraints are defined element wise in the respective element tables.

.. math::
		min & \sum_{i  \ \epsilon \ gen, sgen, load, ext_grid }{P_{i} * f_{i}(P_i)} \\
        subject to \\
        Loadflow equations \\
        branch constraints  \\
        bus constraints \\
        operational power constraints \\
        
        
**Generator Flexibilities**

The active and reactive power generation of generators and static generators can be defined as a flexibility for the OPF.

.. tabularcolumns:: |p{0.40\linewidth}|p{0.4\linewidth}|
.. csv-table:: 
   :file: opf_flexibility.csv
   :delim: ;

**Network Constraints**

Constraints can be defined for bus voltages, line and transformer loading.

.. tabularcolumns:: |p{0.40\linewidth}|p{0.4\linewidth}|
.. csv-table:: 
   :file: opf_constraints.csv
   :delim: ;
           

Cost functions
---------------

The cost function is specified element wise and is organized in tables as well, which makes the parametrization user friendly. There are two options formulating a cost function for each element: A piecewise linear function (see equation \refhere) covering the power range of the respective element or a n-polynomial cost function (see equation \refhere). Reactive and reactive power costs are calculted seperately. 

.. math::
        f_{pwl}(P_i) = ..... \\
        f_{pwl}(Q_i) = .....
        

.. math::
        f_{pol}(P_i) = ..... \\
        f_{pol}(Q_i) = .....
        
The costs are summed up to determine the overall costs for a grid state.

Parametrisation of the calculation
-----------------------------------

As mentioned above, the user can choose between a DC and an AC optimal power flow. The internal solver uses the interior point method.\ref By default, the initial state is the center of the operational constraints. Another option is to initialize the optimisation with a valid loadflow solution. For optimiation of a timeseries, this warm start possibilty can imply a significant speedup.

.. image:: /pics/pandapower_optimal_powerflow.png
		:width: 40em
		:alt: alternate Text
		:align: center


Bibs:

    PhD dissertation:
      - "On the Computation and Application of Multi-period
        Security-Constrained Optimal Power Flow for Real-time
        Electricity Market Operations", Cornell University, May 2007.

    See also:
      - H. Wang, C. E. Murillo-Sanchez, R. D. Zimmerman, R. J. Thomas,
        "On Computational Issues of Market-Based Optimal Power Flow",
        IEEE Transactions on Power Systems, Vol. 22, No. 3, Aug. 2007,
        pp. 1185-1193.