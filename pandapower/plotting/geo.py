try:
    from fiona.crs import from_epsg
    from shapely.geometry import Point, LineString
    from geopandas import GeoDataFrame, GeoSeries
except:
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