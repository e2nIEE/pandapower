Proposed Changes
====

The main change should be done to the format used for storing geodata.
As per issue #709 ascheidl proposed to store geojson strings in `net.bus.geo` & `net.line.geo`.
This is already done in the pandaplan version and would make data interoperable between a number of software.

We would need some additional information in a geojson feature before being able to use it for all components in a
network.
`net.bus.geo` and `net.line.geo` should only contain a GeoJSON geometry object. **Not** a feature.
When exporting from pandapower as geojson a feature should be created containing some foreign keys in the features
`properties` object. This is done in compliance with the
`GeoJSON specification <https://datatracker.ietf.org/doc/html/rfc7946#section-3.2>`_.

Example:

.. literalinclude:: format.json
    :language: json

The ids set for the features are generated from a combination of `pp_type` and `pp_index`. This should make them
globally unique.


Changes required in pandapower
----

functions that require updating:

* plotting.geo.dump_to_geojson
* plotting.simple_plot
* plotting.plotting_toolbox
* pandapower.create_bus (should also allow for LineString as bus geometry)
* pandapower.create_line

not reliant on the geodata is `plotting.to_html` its graph is generated without regard for present geodata.

All users should be forced to update their geodata using the conversion functions present in `pandapower.plotting.geo`.
This should be done so that any functions for working with the old geodata formats can be dropped in a future release.
A conversion function for converting geojson to the old geodata format should **not** be provided.
Converting from geojson to geopandas.geodataframe is trivial. This can be done by any software requiring a geodataframe.
Any geodataframe should not be written into the pandapowerNet.