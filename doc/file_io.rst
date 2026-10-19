.. _file_io:

========================
Save and Load Networks
========================

.. tabularcolumns:: |l|l|l|l|
.. csv-table:: 
   :file: fileIO.csv
   :delim: ;
   :widths: 20, 20, 20, 20



pickle
-----------

.. autofunction:: pandapower.to_pickle

.. autofunction:: pandapower.from_pickle


Excel
-----------

Using excel is highly discouraged as to_excel uses pandas to_excel which is a lossy conversion.
Loading from excel may result in a different network.

.. autofunction:: pandapower.to_excel

.. autofunction:: pandapower.from_excel


Json
-----------

.. autofunction:: pandapower.to_json

.. autofunction:: pandapower.from_json

SQL
-----------

.. autofunction:: pandapower.sql_io.to_sqlite

.. autofunction:: pandapower.sql_io.from_sqlite

PostgreSQL
-----------

.. autofunction:: pandapower.sql_io.to_postgresql

.. autofunction:: pandapower.sql_io.from_postgresql

.. autofunction:: pandapower.sql_io.delete_postgresql_net