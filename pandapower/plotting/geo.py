try:
    from fiona.crs import from_epsg
    from shapely.geometry import Point, LineString
    from geopandas import GeoDataFrame, GeoSeries
    from pyproj import Proj, transform
except ImportError:
    pass


def convert_geodata_to_gis(net, epsg=31467, bus_geodata=True, line_geodata=True):
    if bus_geodata:
        g = net.bus_geodata
        geo = [Point(x, y) for x, y in g[["x", "y"]].values]
        net.bus_geodata = GeoDataFrame(g, crs=from_epsg(epsg), geometry=geo, index=g.index)
    if line_geodata:
        l = net.line_geodata
        geo = GeoSeries([LineString(x) for x in net.line_geodata.coords.values],
                        index=net.line_geodata.index, crs=from_epsg(epsg))
        net.line_geodata = GeoDataFrame(l, crs=from_epsg(epsg), geometry=geo, index=l.index)
    net["gis_epsg_code"] = epsg


def convert_gis_to_geodata(net, bus_geodata=True, line_geodata=True):
    if bus_geodata:
        net.bus_geodata["x"] = [x.x for x in net.bus_geodata.geometry]
        net.bus_geodata["y"] = [x.y for x in net.bus_geodata.geometry]
    if line_geodata:
        net.line_geodata["coords"] = net.line_geodata.geometry.apply(lambda x: list(x.coords))


def get_collection_sizes(net, bus_size=1.0, ext_grid_size=1.0, trafo_size=1.0, load_size=1.0,
                         sgen_size=1.0, switch_size=2.0, switch_distance=1.0):
    """
    Calculates the size for most collection types according to the distance between min and max
    geocoord so that the collections fit the plot nicely

    # Comment: This is implemented because if you would choose a fixed values
    # (e.g. bus_size = 0.2), the size
    # could be to small for large networks and vice versa
    INPUT

    net - pp net
    bus_size (float)
    ext_grid_size (float)
    trafo_size (float)
    load_size (float)
    sgen_size (float)
    switch_size (float)
    switch_distance (float)

    Returns

    sizes (dict) - containing all scaled sizes in a dict
    """

    mean_distance_between_buses = sum((net['bus_geodata'].max() - net[
        'bus_geodata'].min()).dropna() / 200)

    sizes = {
        "bus": bus_size * mean_distance_between_buses,
        "ext_grid": ext_grid_size * mean_distance_between_buses * 1.5,
        "switch": switch_size * mean_distance_between_buses * 1,
        "switch_distance": switch_distance * mean_distance_between_buses * 2,
        "load": load_size * mean_distance_between_buses,
        "sgen": sgen_size * mean_distance_between_buses,
        "trafo": trafo_size * mean_distance_between_buses
    }
    return sizes


def convert_epgs_bus_geodata(net, epsg_in=4326, epsg_out=31467):
    """
    Converts bus geodata in net from epsg_in to epsg_out

    Parameters
    ----------
    net
    epsg_in - 4326 = WGS 84
    epsg_out - 31467 = Gauss-Krüger Zone 3
    """
    inProj = Proj(init='epsg:%i' % epsg_in)
    outProj = Proj(init='epsg:%i' % epsg_out)
    x1, y1 = net.bus_geodata.loc[:, "x"].values, net.bus_geodata.loc[:, "y"].values
    net.bus_geodata.loc[:, "x"], net.bus_geodata.loc[:, "y"] = transform(inProj, outProj, x1, y1)
    return net
