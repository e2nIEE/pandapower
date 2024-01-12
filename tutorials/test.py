if __name__ == '__main__':
    import pandapower as pp
    import geojson
    import pandas as pd
    import pandapower.plotting.geo as geo

    net = pp.from_json(r"C:\temp\NetzeBW\net_for_plot.json")
    bus_geo = pd.read_csv(r"C:\git\netzebw_visualisierung\src\analyse\bus_geodata.csv", index_col=0)
    line_geo = pd.read_csv(r"C:\git\netzebw_visualisierung\src\analyse\line_geodata.csv", index_col=0)
    net.bus_geodata = bus_geo
    net.line_geodata.coords = line_geo.coords.apply(lambda row: geojson.loads(row))
    geo.convert_geodata_to_geojson(net, delete=True)
    g = geo.dump_to_geojson(net, nodes=True, branches=True)

    print('finished.')