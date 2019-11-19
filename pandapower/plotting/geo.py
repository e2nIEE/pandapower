try:
    from fiona.crs import from_epsg
    from shapely.geometry import Point, LineString
    from geopandas import GeoDataFrame, GeoSeries
    from pyproj import Proj, transform
except ImportError:
    pass


def convert_geodata_to_gis(net, epsg=31467, bus_geodata=True, line_geodata=True):
    if bus_geodata:
        bus_geo = net.bus_geodata
        geo = [Point(x, y) for x, y in bus_geo[["x", "y"]].values]
        net.bus_geodata = GeoDataFrame(bus_geo, crs=from_epsg(epsg), geometry=geo,
                                       index=bus_geo.index)
    if line_geodata:
        line_geo = net.line_geodata
        geo = GeoSeries([LineString(x) for x in net.line_geodata.coords.values],
                        index=net.line_geodata.index, crs=from_epsg(epsg))
        net.line_geodata = GeoDataFrame(line_geo, crs=from_epsg(epsg), geometry=geo,
                                        index=line_geo.index)
    net["gis_epsg_code"] = epsg


def convert_gis_to_geodata(net, bus_geodata=True, line_geodata=True):
    if bus_geodata:
        net.bus_geodata["x"] = [x.x for x in net.bus_geodata.geometry]
        net.bus_geodata["y"] = [x.y for x in net.bus_geodata.geometry]
    if line_geodata:
        net.line_geodata["coords"] = net.line_geodata.geometry.apply(lambda x: list(x.coords))


def convert_epgs_bus_geodata(net, epsg_in=4326, epsg_out=31467):
    """
    Converts bus geodata in net from epsg_in to epsg_out

    :param net: The pandapower network
    :type net: pandapowerNet
    :param epsg_in: current epsg projection
    :type epsg_in: int, default 4326 (= WGS84)
    :param epsg_out: epsg projection to be transformed to
    :type epsg_out: int, default 31467 (= Gauss-Krüger Zone 3)
    :return: net - the given pandapower network (no copy!)
    """
    in_proj = Proj(init='epsg:%i' % epsg_in)
    out_proj = Proj(init='epsg:%i' % epsg_out)
    x1, y1 = net.bus_geodata.loc[:, "x"].values, net.bus_geodata.loc[:, "y"].values
    net.bus_geodata.loc[:, "x"], net.bus_geodata.loc[:, "y"] = transform(in_proj, out_proj, x1, y1)
    return net
