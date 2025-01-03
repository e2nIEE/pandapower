# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os

from typing_extensions import deprecated

from pandapower.plotting import geo

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)


def _on_map_test(x, y):
    """
    checks if bus_geodata can be located on a map using geopy
    """
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut
        geolocator = Nominatim(user_agent="pandapower_user_mapboxplot")

    except ImportError:
        # if geopy is not available there will be no geo-coordinates check
        # therefore if geo-coordinates are not real and user sets on_map=True, an empty map will be plot!
        raise ImportError(
            'Geo-coordinates check cannot be performed because geopy package not available \n\t--> '
            'if geo-coordinates are not in lat/lon format an empty plot may appear...'
        )
    try:
        location = geolocator.reverse(f"{x}, {y}", language='en-US')
    except GeocoderTimedOut:
        logger.error("Existing net geodata cannot be geo-located: possible reason: geo-data not in lat/long ->"
                     "try geo_data_to_latlong(net, projection) to transform geodata to lat/long!")
    else:
        if location.address is None:
            return False
    return True


@deprecated('geo_data_to_latlong is deprecated and will be removed shortly, use pandapower.geo.convert_crs instead')
def geo_data_to_latlong(net, projection):
    """
    Transforms network's geodata (in `net.bus_geodata` and `net.line_geodata`) from specified projection to lat/long (WGS84).

    INPUT:
        **net** (pandapowerNet) - The pandapower network

        **projection** (String) - projection from which geodata are transformed to lat/long. some examples

                - "epsg:31467" - 3-degree Gauss-Kruger zone 3
                - "epsg:2032" - NAD27(CGQ77) / UTM zone 18N
                - "epsg:2190" - Azores Oriental 1940 / UTM zone 26N
    """
    geo.convert_crs(net, epsg_in=projection.split(':')[1], epsg_out=4326)


def set_mapbox_token(token):
    from pandapower.__init__ import pp_dir
    path = os.path.join(pp_dir, "plotting", "plotly")
    filename = os.path.join(path, 'mapbox_token.txt')
    with open(filename, "w") as mapbox_file:
        mapbox_file.write(token)


def _get_mapbox_token():
    from pandapower.__init__ import pp_dir
    path = os.path.join(pp_dir, "plotting", "plotly")
    filename = os.path.join(path, 'mapbox_token.txt')
    with open(filename, "r") as mapbox_file:
        return mapbox_file.read()
