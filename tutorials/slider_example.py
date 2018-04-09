# -*- coding: utf-8 -*-

import sys
from PyQt4.uic import loadUiType
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

Ui_MainWindow, QMainWindow = loadUiType('slider_example.ui')

import pandapower.plotting as plot
import pandapower as pp
import matplotlib.pyplot as plt

class SliderWidget(QMainWindow, Ui_MainWindow):
    def __init__(self, net):
        super(SliderWidget, self).__init__()
        self.setupUi(self)
        self.net = net
        self.net.line_geodata.drop(set(net.line_geodata.index) - set(net.line.index), inplace=True)
        cmap, norm = plot.cmap_continous([(0.97, "blue"), (1.0, "green"), (1.03, "red")])
        self.bc = plot.create_bus_collection(net, size=90, zorder=2, cmap=cmap, norm=norm, picker=True,
                                             infofunc=lambda x: "This is bus %s"%net.bus.name.at[x])
        cmap, norm = plot.cmap_continous([(20, "green"), (50, "yellow"), (60, "red")])
        self.lc = plot.create_line_collection(net, zorder=1, cmap=cmap, norm=norm, linewidths=2,
                                              infofunc=lambda x: "This is line %s"%net.line.name.at[x])
        self.fig, self.ax = plt.subplots()
        plot.draw_collections([self.bc, self.lc], ax=self.ax)
        plt.close()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.mpl_connect('pick_event', self.pick_event)
        self.gridLayout.addWidget(self.canvas)
        self.canvas.draw()
        self.LoadSlider.valueChanged.connect(self.slider_changed)
        self.SgenSlider.valueChanged.connect(self.slider_changed)
        self.setWindowTitle("PyQt with pandapower Demo")

    def slider_changed(self):
        self.net.load.scaling = self.LoadSlider.value() / 100.
        self.net.sgen.scaling = self.SgenSlider.value() / 100.
        pp.runpp(self.net)
        self.ax.collections[0].set_array(self.net.res_bus.vm_pu.values)
        self.ax.collections[1].set_array(self.net.res_line.loading_percent.values)
        self.canvas.draw()

    def pick_event(self, event):
        idx = event.ind[0]
        collection = event.artist
        self.info = QLabel()
        self.info.setText(collection.info[idx])
        self.info.show()

def main(net):
   app = QApplication(sys.argv)
   ex = SliderWidget(net)
   ex.show()
   sys.exit(app.exec_())

if __name__ == '__main__':
    from pandapower.networks import mv_oberrhein
    net = mv_oberrhein()
    main(net)