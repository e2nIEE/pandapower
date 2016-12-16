import matplotlib.pyplot as plt
import pandapower.plotting as plot

try:
    import seaborn
    colors = seaborn.color_palette()
except:
    colors = ["b", "g", "r", "c", "y"]

def simple_plot(net=None, respect_switches=False):
    if net is None:
        import pandapower.networks as nw
        print("No Pandapower network provided -> Plotting mv_oberrhein")
        net = nw.mv_oberrhein()

    try:
        # try with available geodata "bus_geodata" in net and "line_geodata" in net
        lc = plot.create_line_collection(net, net.line.index, color="grey", zorder=1)  # create lines
        bc = plot.create_bus_collection(net, net.bus.index, size=80, color=colors[0], zorder=2)  # create buses
        plot.draw_collections([lc, bc], figsize=(8, 6))  # plot lines and buses
    except:
        # delete the geocoordinates
        if "bus_geodata" in net:
            del net.bus_geodata
        if "line_geodata" in net:
            del net.line_geodata
        print("No or insufficient geodata available --> Creating artificial coordinates. This may take some time")
        plot.create_generic_coordinates(net, respect_switches=respect_switches)
        bc = plot.create_bus_collection(net, net.bus.index, size=.2, color=colors[0], zorder=10)
        tc = plot.create_trafo_collection(net, net.trafo.index, color="g")
        lcd = plot.create_line_collection(net, net.line.index, color="grey", linewidths=0.5, use_line_geodata=False)
        sc = plot.create_bus_collection(net, net.ext_grid.bus.values, patch_type="rect", size=.5, color="y", zorder=11)
        plot.draw_collections([lcd, bc, tc, sc], figsize=(8, 6))

    plt.show()

if __name__ == "__main__":
    simple_plot()