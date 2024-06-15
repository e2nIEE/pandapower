====================================
Transforming network geodata formats
====================================

Geodata in pandapower can be stored in different formats.


.. autofunction:: pandapower.plotting.geo.convert_gis_to_geodata

.. autofunction:: pandapower.plotting.geo.convert_geodata_to_gis

.. autofunction:: pandapower.plotting.geo.convert_gis_to_geojson

.. autofunction:: pandapower.plotting.geo.convert_epsg_bus_geodata

.. autofunction:: pandapower.plotting.geo.convert_crs

All bus and lines of a network can be dumped to geojson,
including all of their properties from `bus`,`res_bus`, `line` and `res_line` tables.

.. autofunction:: pandapower.plotting.geo.dump_to_geojson