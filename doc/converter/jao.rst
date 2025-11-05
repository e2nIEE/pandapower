JAO Static Grid Model Converter Function
========================================

The ``from_jao`` function allows users to convert the Static Grid Model provided by JAO (Joint Allocation Office) into a pandapower network by reading and processing the provided Excel and HTML files.

The table below shows the mapping of the JAO columns to pandapower parameters necessary for executing a balanced power flow calculation.

.. tabularcolumns:: |p{0.15\linewidth}|p{0.15\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.35\linewidth}|
.. csv-table:: JAO - pandapower mapping
   :file: jao-converter.csv
   :delim: ,
   :quote: "
   :header-rows: 1
   :widths: 15, 15, 10, 25, 35
Function Overview
-----------------

.. autofunction:: pandapower.converter.jao.from_jao
