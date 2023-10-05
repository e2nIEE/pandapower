#####################
Contingency analysis
#####################

We can define N-1 cases to be analysed as contingencies. This means that indices of net.line, net.trafo, net.trafo3w can be defined as contingencies, which are switched off one at a time. The power system is analyzed with power flow calculations, and the min/max values among all the N-1 cases are obtained for relevant variables.

A tutorial that introduces this feature with an example is available at `Contingency analysis <https://github.com/e2nIEE/pandapower/tree/develop/tutorials/contingency_analysis.ipynb>`_

.. autofunction:: pandapower.contingency.run_contingency

.. autofunction:: pandapower.contingency.run_contingency_ls2g

.. autofunction:: pandapower.contingency.get_element_limits

.. autofunction:: pandapower.contingency.check_elements_within_limits

.. autofunction:: pandapower.contingency.report_contingency_results
