# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import os
from typing_extensions import deprecated
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
        logger.error("Existing net geodata cannot be geo-located: possible reason: geo-data not in wgs84 ->"
                     "try convert_crs(net, epsg_in=projection) to transform geodata to wgs84!")
    else:
        if location.address is None:
            return False
    return True


@deprecated("A token is not required for maplibre. Call to set_mapbox_token can be removed.")
def set_mapbox_token(token):
    from pandapower.__init__ import pp_dir
    path = os.path.join(pp_dir, "plotting", "plotly")
    filename = os.path.join(path, 'mapbox_token.txt')
    with open(filename, "w") as mapbox_file:
        mapbox_file.write(token)


@deprecated("A token is not required for maplibre. Call to _get_mapbox_token can be removed.")
def _get_mapbox_token():
    from pandapower.__init__ import pp_dir
    path = os.path.join(pp_dir, "plotting", "plotly")
    filename = os.path.join(path, 'mapbox_token.txt')
    try:
        with open(filename, "r") as mapbox_file:
            token = mapbox_file.read()
    except FileNotFoundError:
        token = "no_token"
    return token
