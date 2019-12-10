try:
    from fiona.crs import from_epsg
    from shapely.geometry import Point, LineString
    from geopandas import GeoDataFrame, GeoSeries
    from pyproj import Proj, transform
except ImportError:
    pass


def convert_geodata_to_gis(net, epsg=31467, node_geodata=True, branch_geodata=True,
                           name_node_geodata='bus', name_branch_geodata='line'):
    name_node = name_node_geodata
    name_branch = name_branch_geodata
    if node_geodata:
        node_geo = net[name_node + '_geodata']
        geo = [Point(x, y) for x, y in node_geo[["x", "y"]].values]
        net[name_node + '_geodata'] = GeoDataFrame(node_geo, crs=from_epsg(epsg), geometry=geo,
                                       index=node_geo.index)
    if branch_geodata:
        branch_geo = net[name_branch + '_geodata']
        geo = GeoSeries([LineString(x) for x in net[name_branch + '_geodata'].coords.values],
                        index=net[name_branch + '_geodata'].index, crs=from_epsg(epsg))
        net[name_branch + '_geodata'] = GeoDataFrame(branch_geo, crs=from_epsg(epsg), geometry=geo,
                                        index=branch_geo.index)
    net["gis_epsg_code"] = epsg


def convert_gis_to_geodata(net, node_geodata=True, branch_geodata=True, name_node_geodata='bus',
                           name_branch_geodata='line'):
    name_node = name_node_geodata
    name_branch = name_branch_geodata
    if node_geodata:
        net[name_node + '_geodata']["x"] = [x.x for x in net[name_node + '_geodata'].geometry]
        net[name_node + '_geodata']["y"] = [x.y for x in net[name_node + '_geodata'].geometry]
    if branch_geodata:
        net[name_branch +'_geodata']["coords"] = \
            net[name_branch +'_geodata'].geometry.apply(lambda x: list(x.coords))


def convert_epgs_bus_geodata(net, epsg_in=4326, epsg_out=31467, name_node_geodata='bus'):
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
    name_node = name_node_geodata
    in_proj = Proj(init='epsg:%i' % epsg_in)
    out_proj = Proj(init='epsg:%i' % epsg_out)
    x1, y1 = net[name_node + '_geodata'].loc[:, "x"].values, \
             net[name_node + '_geodata'].loc[:, "y"].values
    net[name_node + '_geodata'].loc[:, "x"], net[name_node + '_geodata'].loc[:, "y"] = \
        transform(in_proj, out_proj, x1, y1)
    return net
