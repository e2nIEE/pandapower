======================
Basic Standard Types
======================

Every pandapower network comes with a default set of standard types. 

.. note ::
    The pandapower standard types are compatible with 50 Hz systems, please be aware that the standard type values might not be realistic for 60 Hz (or other) power systems.

Lines
-----

.. csv-table::
    :file: tables/line_std_types.csv
    :header-rows: 1
    :delim: ;

.. note ::
    To add the optional column "alpha" to net.line that is used if power flow is calculated for a different line temperature than 20 °C, use the function pp.add_temperature_coefficient()

DC Lines
--------

.. csv-table::
    :file: tables/line_dc_std_types.csv
    :header-rows: 1
    :delim: ;

.. note ::
    To add the optional column "alpha" to net.line_dc that is used if power flow is calculated for a different line temperature than 20 °C, use the function pp.add_temperature_coefficient()

Transformers
------------

.. csv-table:: 
    :file: tables/trafo_std_types.csv
    :header-rows: 1
    :delim: ;

Three Winding Transformers
--------------------------

.. csv-table:: 
    :file: tables/trafo3w_std_types.csv
    :header-rows: 1
    :delim: ;

Fuse
----

.. csv-table::
    :file: tables/fuse_std_types.csv
    :header-rows: 1
    :delim: ;
