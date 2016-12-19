import matplotlib.pyplot as plt
import pandapower.plotting as plot
try:
    import pplog as logging
except:
    import logging

try:
    import seaborn
    colors = seaborn.color_palette()
except:
    colors = ["b", "g", "r", "c", "y"]

logger = logging.getLogger(__name__)


def simple_plot(net=None, respect_switches=False, line_width=1.0, bus_size=None, ext_grid_size=None,
                bus_color=colors[0], line_color='grey', trafo_color='g', ext_grid_color='y'):
    if net is None:
        import pandapower.networks as nw
        logger.warning("No Pandapower network provided -> Plotting mv_oberrhein")
        net = nw.mv_oberrhein()

    # create geocoord if none are available
    if len(net.line_geodata) == 0 and len(net.bus_geodata) == 0:
        logger.warning("No or insufficient geodata available --> Creating artificial coordinates." +
                       " This may take some time")
        plot.create_generic_coordinates(net, respect_switches=respect_switches)

    if bus_size or ext_grid_size is None:
        # if either bus_size or ext_grid size is None -> calc size from distance between min and
        # max geocoord
        mean_distance_between_buses = sum((net['bus_geodata'].max() - net[
            'bus_geodata'].min()) / 200)
        # set the bus / ext_grid sizes accordingly
        # Comment: This is implemented because if you would choose a fixed values
        # (e.g. bus_size = 0.2), the size
        # could be to small for large networks and vice versa
        if bus_size is None:
            bus_size = mean_distance_between_buses
        if ext_grid_size is None:
            ext_grid_size = mean_distance_between_buses * 1.5

    # if bus geodata is available, but no line geodata
    if len(net.line_geodata) == 0:
        use_line_geodata = False
    else:
        use_line_geodata = True

    # create bus collections ti plot
    bc = plot.create_bus_collection(net, net.bus.index, size=bus_size, color=bus_color, zorder=10)
    lc = plot.create_line_collection(net, net.line.index, color=line_color, linewidths=line_width,
                                     use_line_geodata=use_line_geodata)
    sc = plot.create_bus_collection(net, net.ext_grid.bus.values, patch_type="rect",
                                    size=ext_grid_size, color=ext_grid_color, zorder=11)
    # create trafo collection if trafo is available
    if len(net.trafo):
        tc = plot.create_trafo_collection(net, net.trafo.index, color=trafo_color)
    else:
        tc = None
    plot.draw_collections([lc, bc, tc, sc])
    plt.show()

if __name__ == "__main__":
    simple_plot()
