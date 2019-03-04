======================
Basic Standard Types
======================

Every pandapower network comes with a default set of standard types. 

.. note ::
    The pandapower standard types are compatible with 50 Hz systems, please be aware that the standard type values might not be realistic for 60 Hz (or other) power systems.

Lines
--------

.. tabularcolumns:: |l|l|l|l|l|l|l|l|
.. csv-table:: 
   :file: linetypes.csv
   :delim: ;
   :widths: 60, 15, 15, 15, 15, 15, 15, 15

.. note ::
    To add the optional column "alpha" to net.line that is used if power flow is calculated for a different line temperature than 20 °C, use the function pp.add_temperature_coefficient()

Transformers
-----------------

.. tabularcolumns:: |l|l|l|l|l|l|l|l|l|l|l|l|l|l|l|l|
.. csv-table:: 
   :file: trafotypes.csv
   :delim: ;
   :widths: 60, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15

Three Winding Transformers
--------------------------------

.. tabularcolumns:: |l|l|l|l|l|l|l|l|l|l|l||l|l|l|l|l|l|l|l|l|l|l|
.. csv-table:: 
   :file: trafo3wtypes.csv
   :delim: ;
   :widths: 60, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15

