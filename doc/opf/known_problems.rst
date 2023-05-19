
Known Problems
===========================


pypower OPF
-----------------------------------

As previously mentioned in the documentation, :py:meth:`pandapower.runopp` does not have the best convergence properties.
You are welcome to use the function but, please, understand that the support of the pandapower developers for the function made usable from pypower is minimal.


OPFs with pandapower networks
--------------------------------------------

This documentation presents the origins of the networks provided by pandapower.
The majority of the networks was not created for optimization studies.
Consequently and in contrast to power flow analyses, optimal power flows do not converge on all networks.
To give pandapower users a guidance which networks are favorable if optimization should be performed with a provided network, the following table show which data must be adjusted to enable OPFs. Unfortunately, this does not mean, that :py:meth:`pandapower.runopp` converges, but at least the data are valid and ready to br optimized.

.. tabularcolumns:: |p{0.15\linewidth}|p{0.25\linewidth}|p{0.15\linewidth}|p{0.15\linewidth}|p{0.15\linewidth}|
.. csv-table::
   :file: opf_ready_nets.csv
   :delim: ;
   :widths: 15, 25, 15, 15, 15

