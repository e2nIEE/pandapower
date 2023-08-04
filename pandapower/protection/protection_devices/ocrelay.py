import copy

from pandapower import std_type_exists, load_std_type

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False

import numpy as np
from pandapower.control.util.characteristic import LogSplineCharacteristic
from pandapower.protection.basic_protection_device import ProtectionDevice
from pandapower.auxiliary import soft_dependency_error, ensure_iterability
import pandapower.shortcircuit as sc
from pandapower.protection.oc_relay_model import time_grading
from pandapower.protection.utility_functions import create_sc_bus, bus_path_multiple_ext_bus, get_line_path,get_line_idx,get_bus_idx,\
                                                    parallel_lines, plot_tripped_grid, create_I_t_plot


class OCRelay(ProtectionDevice):

    def __init__(self, net, switch_index, oc_relay_type, time_settings, overload_factor=1.2, ct_current_factor=1.25,
                 safety_factor=1, inverse_overload_factor=1.2, pickup_current_manual=None, in_service=True,
                 overwrite=True, sc_fraction=0.95, curve_type='standard_inverse', **kwargs):
        super().__init__(net, in_service=in_service, overwrite=overwrite, **kwargs)
        self.switch_index = switch_index
        self.oc_relay_type = oc_relay_type
        self.time_settings = time_settings
        self.overload_factor = overload_factor
        self.ct_current_factor = ct_current_factor
        self.safety_factor = safety_factor
        self.inverse_overload_factor = inverse_overload_factor
        self.pickup_current_manual = pickup_current_manual
        self.sc_fraction = sc_fraction
        self.curve_type = curve_type

        self.activation_parameter = "i_ka"
        self.tripped = False
        self.I_g = None
        self.I_gg = None
        self.I_s = None
        self.time_grading = None
        self.t_g = None
        self.t_gg = None
        self.create_protection_function(net=net)

    def create_protection_function(self, net):
        # this function is called when the OCRelay is instantiated to determine the parameters for the
        # protection function
        net_sc = copy.deepcopy(net)

        if (net.switch.closed.at[self.switch_index] == True) and (net.switch.et.at[self.switch_index] == 'l'):
            line_idx = net.switch.element.at[self.switch_index]
            net_sc = create_sc_bus(net_sc, line_idx, self.sc_fraction)
            bus_idx = max(net_sc.bus.index)

        elif (net.switch.closed.at[self.switch_index] == True) and (net.switch.et.at[self.switch_index] == 't'):
            bus_idx = net.switch.bus.at[self.switch_index]
        else:
            raise ValueError("OC Relay must be connected to line or transformer element")

        sc.calc_sc(net_sc, bus=bus_idx, branch_results=True)

        if self.oc_relay_type == 'DTOC':
            if self.pickup_current_manual is None:
                self.I_g = net_sc.line.max_i_ka.at[line_idx] * self.overload_factor * self.ct_current_factor
                self.I_gg = net_sc.res_line_sc.ikss_ka.at[line_idx] * self.safety_factor
            else:
                self.I_g = self.pickup_current_manual.I_g.at[self.switch_index]
                self.I_gg = self.pickup_current_manual.I_g.at[self.switch_index]

        if self.oc_relay_type == 'IDMT':
            if self.pickup_current_manual is None:
                self.I_s = net_sc.line.max_i_ka.at[line_idx] * self.inverse_overload_factor
            else:
                self.I_s = self.pickup_current_manual.I_s.at[self.switch_index]
            self._select_k_alpha()

        if self.oc_relay_type == 'IDTOC':
            if self.pickup_current_manual is None:
                self.I_g = net_sc.line.max_i_ka.at[line_idx] * self.overload_factor * self.ct_current_factor
                self.I_gg = net_sc.res_line_sc.ikss_ka.at[line_idx] * self.safety_factor
                self.I_s = net_sc.line.max_i_ka.at[line_idx] * self.inverse_overload_factor
            else:
                self.I_g = self.pickup_current_manual.I_g.at[self.switch_index]
                self.I_gg = self.pickup_current_manual.I_g.at[self.switch_index]
                self.I_s = self.pickup_current_manual.I_s.at[self.switch_index]
            self._select_k_alpha()

        self.time_grading = time_grading(net, self.time_settings)
        self.t_g = self.time_grading.t_g
        self.t_gg = self.time_grading.t_gg

    def _select_k_alpha(self):
        if self.curve_type == 'standard_inverse':
            self.k = 0.140
            self.alpha = 0.02
        elif self.curve_type == 'very_inverse':
            self.k = 13.5
            self.alpha = 1
        elif self.curve_type == 'extremely_inverse':
            self.k = 80
            self.alpha = 2
        elif self.curve_type == 'long_inverse':
            self.k = 120
            self.alpha = 1

    def reset_device(self):
        self.tripped = False

    def has_tripped(self):
        return self.tripped

    def status_to_net(self, net):
        # update self.tripped status to net
        net.switch.closed.at[self.switch_index] = not self.tripped

    def protection_function(self, net, scenario):
        # compute protection time in net under short-circuit or operating conditions
        if scenario == "sc":
            i_ka = net.res_switch_sc.ikss_ka.at[self.switch_index]
        elif scenario == "pp":
            i_ka = net.res_switch.i_ka.at[self.switch_index]
        else:
            raise ValueError("scenario must be either sc or pp")

        if self.oc_relay_type == 'DTOC':
            if i_ka > self.I_gg:
                self.tripped = True
                act_time_s = self.t_gg.at[self.switch_index]
            elif i_ka > self.I_g:
                self.tripped = True
                act_time_s = self.t_g.at[self.switch_index]
            else:
                self.tripped = False
                act_time_s = np.inf

        if self.oc_relay_type == 'IDMT':
            if i_ka > self.I_s:
                self.tripped = True
                act_time_s = (self.t_gg.at[self.switch_index] * self.k) / (((i_ka/self.I_s)**self.alpha)-1)\
                              + self.t_g.at[self.switch_index]
            else:
                self.tripped = False
                act_time_s = np.inf

        if self.oc_relay_type == 'IDTOC':
            if i_ka > self.I_gg:
                self.tripped = True
                act_time_s = self.t_gg.at[self.switch_index]
            elif i_ka > self.I_g:
                self.tripped = True
                act_time_s = self.t_g.at[self.switch_index]
            elif i_ka > self.I_s:
                self.tripped = True
                act_time_s = (self.t_gg.at[self.switch_index] * self.k) / (((i_ka/self.I_s)**self.alpha)-1)\
                              + self.t_g.at[self.switch_index]
            else:
                self.tripped = False
                act_time_s = np.inf

        protection_result = {"switch_id": self.switch_index,
                             "protection_type": self.__class__.__name__,
                             "trip_melt": self.has_tripped(),
                             "activation_parameter": self.activation_parameter,
                             "activation_parameter_value": i_ka,
                             "trip_melt_time_s": act_time_s}
        return protection_result

    def plot_protection_characteristic(self, net, num=35, xlabel="I [A]", ylabel="time [s]",
                                       title="Time-Current Characteristic of OC Relay"):
        xmin = 10
        xmax = 100000
        ymin = 0.001
        ymax = 10000

        if self.oc_relay_type == 'DTOC':
            plt.loglog(0, 0)
            plt.step([self.I_g, self.I_gg, xmax], [ymax, self.t_g, self.t_gg])

        elif self.oc_relay_type == 'IDMT':
            x = np.logspace(np.log10(self.I_s+0.001), np.log10(xmax))
            plt.loglog(x, (self.t_gg.at[self.switch_index] * self.k) / (((x/self.I_s)**self.alpha)-1)
                       + self.t_g.at[self.switch_index])
        else:
            raise ValueError('Plot only implemented for DTOC and IDMT OCRelay')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.title(title)
        plt.grid(True, which="both", ls="-")

    def __str__(self):
        s = 'Protection Device: %s \nType: %s \nName: %s' % (self.__class__.__name__, self.oc_relay_type, self.name)
        self.characteristic_index = 1
        return s