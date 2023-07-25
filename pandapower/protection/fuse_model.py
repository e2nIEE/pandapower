import copy
import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pandapower as pp
from pandapower import load_std_type, create_std_type, available_std_types
from pandapower.control import plot_characteristic
from pandapower.protection.protection_devices.fuse import Fuse
from pandapower.protection.run_protection import calculate_protection_times
import pandapower.shortcircuit as sc
from scipy.optimize import curve_fit

'''The goal of the fuse model is to be able to create an analytical equation describing the time-current
characteristics of a fuse given its rated current and manufacturer. Our first attempt will analyze data from the
Siemens NH fuse series to create our first model.  '''

def plot_normalized_curves():
    # This function will normalize all of the Siemens NH fuses available in the std library.
    #  The goal is to gain some more intuition/insight
    net = pp.create_empty_network()
    fuse_lib = available_std_types(net, element="fuse")

    # remove fuses that are not Siemens
    for k in fuse_lib.index:
        if not 'Siemens NH' in k:
            fuse_lib = fuse_lib.drop(k)
    print(fuse_lib)

    plt.figure(1)
    for k in fuse_lib.index:
        x = fuse_lib.x_avg.at[k]
        y = fuse_lib.t_avg.at[k]
        plt.loglog(x, y, 'x-')
    plt.xlim(1, 100000)
    plt.ylim(0.001, 100000)
    plt.grid(True, which="both", ls="-")
    plt.title('Raw Data of Siemens Fuses')

    plt.figure(2)
    for k in fuse_lib.index:
        x = fuse_lib.x_avg.at[k] / fuse_lib.i_rated_a.at[k]
        y = fuse_lib.t_avg.at[k]
        plt.loglog(x, y, 'g.')
    plt.xlim(1, 100)
    plt.ylim(0.001, 100000)
    plt.grid(True, which="both", ls="-")
    plt.title('Normalized Data of Siemens Fuses by Inverse')

    '''plt.figure(3)
    for k in fuse_lib.index:
        x = np.square(fuse_lib.x_avg.at[k]) / np.square(fuse_lib.i_rated_a.at[k])
        y = fuse_lib.t_avg.at[k]
        plt.loglog(x, y, 'x')
    plt.xlim(1, 10000)
    plt.ylim(0.001, 100000)
    plt.grid(True, which="both", ls="-")
    plt.title('Normalized Data of Siemens Fuses by Inverse Squared')'''

    # fit data to model 1
    xdata = []
    ydata = []
    for k in fuse_lib.index:
        xdata.extend(fuse_lib.x_avg.at[k] / fuse_lib.i_rated_a.at[k])
        ydata.extend(fuse_lib.t_avg.at[k])


    popt, pcov = curve_fit(model1, xdata, ydata)
    xpoints = np.logspace(np.log10(1.2), np.log10(55))
    plt.plot(xpoints, model1(xpoints, *popt), 'r-')

    plt.show()


def model1(x, a, alpha):
    return a * pow(x, alpha)

if __name__ == "__main__":
    plot_normalized_curves()

