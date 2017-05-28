============================================================
Transforming network geodata from any projection to lat/long
============================================================

In case network geodata are not in The World Geodetic System (WGS84), that is latitude/longitude format, but in some of
the map-projections, it may be converted to lat/long by providing name of the projection
(in the form ``'epsg:<projection_number>'`` according to `spatialreference <http://spatialreference.org/ref/epsg/>`_).
A sample of converting geodata from mv_oberrhein network can be found in the tutorial.

.. autofunction:: pandapower.plotting.plotly.geo_data_to_latlong