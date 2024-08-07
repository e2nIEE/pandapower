import copy
import numpy as np
from pandapower.protection.basic_protection_device import ProtectionDevice
import pandapower.shortcircuit as sc
from pandapower.protection.oc_relay_model import time_grading
from pandapower.protection.utility_functions import create_sc_bus

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False


class OCRelay(ProtectionDevice):
    """
    OC Relay used in circuit protection

    INPUT:
        **net** (attrdict) - pandapower net

        **switch_index** (int) - index of the switch that the relay acts upon

        **oc_relay_type** (str) - string specifying the type of over-current protection. Must be either "DTOC", "IDMT",
        or "IDTOC".

        **time_settings** (list of DataFrame) - If given as a list, the time grading will be calculated based on
        topological grid search, and manual tripping time can be provided as a dataframe by respecting the column names.

                For DTOC:
                time_settings =[t>>, t>, t_diff] or Dataframe columns as 'switch_id', 't_gg', 't_g'

                - t>> (t_gg): instantaneous tripping time in seconds
                - t> (t_g):  primary backup tripping time in seconds,
                - t_diff: time grading delay difference in seconds


                For IDMT:
                time_settings =[tms, t_grade] or Dataframe columns as 'switch_id', 'tms', 't_grade'

                - tms: time multiplier settings in seconds
                - t_grade:  time grading delay difference in seconds

                For IDTOC:
                time_settings =[t>>, t>, t_diff, tms,t_grade] or Dataframe columns as 'switch_id', 't_gg', 't_g','tms',
                 't_grade'

                - t>> (t_gg): instantaneous tripping time in seconds
                - t> (t_g):  primary backup tripping time in seconds,
                - t_diff: time grading delay difference in seconds
                - tms: time multiplier settings in seconds
                - t_grade:  time grading delay difference in seconds

        **overload_factor** - (float, 1.25)- Allowable overloading on the line used to calculate the pick-up current

        **ct_current_factor** -(float, 1.2) - Current multiplication factor to calculate the pick-up current

        **safety_factor** -(float, 1) - Safety limit for the instantaneous pick-up current

        **inverse_overload_factor** -(float, 1.2)- Allowable inverse overloading to define the pick-up current in IDMT
        relay

        **pickup_current_manual** - (DataFrame, None) - User-defined relay trip currents can be given as a dataframe.

                DTOC: Dataframe with columns as 'switch_id', 'I_gg', 'I_g'

                IDMT: Dataframe with columns as 'switch_id', 'I_s'

                IDTOC: Dataframe with columns as 'switch_id', 'I_gg', 'I_g', 'I_s'

        **in_service** (bool, True) - specify whether relay is in service

        **overwrite** (bool, True) - specify whether this oc relay should overwrite existing protection devices acting
        on the switch

        **sc_fraction** (float, 0.95) - Maximum possible extent to which the short circuit can be created on the line

        **curve_type** (str, 'standard_inverse') - specify the type of curve used in IDMT and IDTOC.

            Curve type can be:
                 - 'standard_inverse'
                 - 'very_inverse',
                 - 'extremely_inverse',
                 - 'long_inverse'
    """

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
        self.kwargs = kwargs

        self.activation_parameter = "i_ka"
        self.tripped = False
        self.I_g = None
        self.I_gg = None
        self.I_s = None
        self.time_grading = None
        self.t_g = None
        self.t_gg = None
        self.t_grade = None
        self.tms = None
        self.create_protection_function(net=net)

    def create_protection_function(self, net):
        # this function is called when the OCRelay is instantiated to determine the parameters for the
        # protection function
        net_sc = copy.deepcopy(net)

        if (net.switch.closed.at[self.switch_index] == True) and (net.switch.et.at[self.switch_index] == 'l'):
            line_idx = int(net.switch.element.at[self.switch_index])
            net_sc = create_sc_bus(net_sc, line_idx, self.sc_fraction)
            bus_idx = max(net_sc.bus.index)

        elif (net.switch.closed.at[self.switch_index] == True) and (net.switch.et.at[self.switch_index] == "t" or "b"):
            bus_idx = net.switch.bus.at[self.switch_index]
            line_idx = None
        else:
            raise ValueError("OC Relay must be connected to line or transformer element")

        sc.calc_sc(net_sc, bus=bus_idx, branch_results=True)

        if self.oc_relay_type == 'DTOC':
            if self.pickup_current_manual is None:
                self.I_g = float(net_sc.line.max_i_ka.iloc[line_idx]) * self.overload_factor * self.ct_current_factor
                self.I_gg = float(net_sc.res_line_sc.ikss_ka.iloc[line_idx]) * self.safety_factor
            else:
                self.I_g = float(self.pickup_current_manual.I_g.iloc[self.switch_index])
                self.I_gg = float(self.pickup_current_manual.I_g.iloc[self.switch_index])
            self.time_grading = time_grading(net, self.time_settings)
            self.t_g = float(self.time_grading.t_g[self.switch_index])
            self.t_gg = float(self.time_grading.t_gg[self.switch_index])

        if self.oc_relay_type == 'IDMT':
            if self.pickup_current_manual is None:
                self.I_s = float(net_sc.line.max_i_ka.iloc[line_idx]) * self.inverse_overload_factor
            else:
                self.I_s = float(self.pickup_current_manual.I_s.iloc[self.switch_index])
            self._select_k_alpha()
            self.time_grading = time_grading(net, self.time_settings)
            self.t_grade = float(self.time_grading.t_g[self.switch_index])
            self.tms = float(self.time_grading.t_gg[self.switch_index])

        if self.oc_relay_type == 'IDTOC':
            if self.pickup_current_manual is None:
                self.I_g = float(net_sc.line.max_i_ka.iloc[line_idx]) * self.overload_factor * self.ct_current_factor
                self.I_gg = float(net_sc.res_line_sc.ikss_ka.iloc[line_idx]) * self.safety_factor
                self.I_s = float(net_sc.line.max_i_ka.iloc[line_idx]) * self.inverse_overload_factor
            else:
                self.I_g = float(self.pickup_current_manual.I_g.iloc[self.switch_index])
                self.I_gg = float(self.pickup_current_manual.I_g.iloc[self.switch_index])
                self.I_s = float(self.pickup_current_manual.I_s.iloc[self.switch_index])
            self._select_k_alpha()
            # calculate time grading first for DTOC part to obtain t_g and t_gg
            self.time_grading = time_grading(net, [self.time_settings[0], self.time_settings[1], self.time_settings[2]])
            self.t_g = float(self.time_grading.t_g[self.switch_index])
            self.t_gg = float(self.time_grading.t_gg[self.switch_index])
            # calculate time grading again for IDMT part to obtain t_grade and tms
            self.time_grading = time_grading(net, [self.time_settings[3], self.time_settings[4]])
            self.tms = float(self.time_grading.t_gg[self.switch_index])
            self.t_grade = float(self.time_grading.t_g[self.switch_index])

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
        net.switch.at[self.switch_index, "closed"] = not self.tripped

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
                act_time_s = self.t_gg
            elif i_ka > self.I_g:
                self.tripped = True
                act_time_s = self.t_g
            else:
                self.tripped = False
                act_time_s = np.inf

        if self.oc_relay_type == 'IDMT':
            if i_ka > self.I_s:
                self.tripped = True
                act_time_s = (self.tms * self.k) / (((i_ka/self.I_s)**self.alpha)-1) + self.t_grade
            else:
                self.tripped = False
                act_time_s = np.inf

        if self.oc_relay_type == 'IDTOC':
            if i_ka > self.I_gg:
                self.tripped = True
                act_time_s = self.t_gg
            elif i_ka > self.I_g:
                self.tripped = True
                act_time_s = self.t_g
            elif i_ka > self.I_s:
                self.tripped = True
                act_time_s = (self.tms * self.k) / (((i_ka / self.I_s)**self.alpha)-1) + self.t_grade
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

    def plot_protection_characteristic(self, net, num=60, xlabel="I [A]", ylabel="time [s]", xmin=10, xmax=10000,
                                       ymin=0.01, ymax=10000, title="Time-Current Characteristic of OC Relay "):

        if self.oc_relay_type == 'DTOC':
            plt.loglog(0, 0)
            plt.step(np.array([self.I_g*1000, self.I_gg*1000, xmax]), np.array([ymax, self.t_g, self.t_gg]))

        elif self.oc_relay_type == 'IDMT':
            x = np.logspace(np.log10((1000*self.I_s)+0.001), np.log10(xmax), 60)
            plt.loglog(x, (self.tms * self.k) / (((x/(1000*self.I_s))**self.alpha)-1) + self.t_grade)

        elif self.oc_relay_type == 'IDTOC':
            x = np.logspace(np.log10((1000 * self.I_s) + 0.001), np.log10(1000 * self.I_g), num=num)
            plt.loglog(x, (self.tms * self.k) / (((x / (1000 * self.I_s)) ** self.alpha) - 1) + self.t_grade)
            plt.step(np.array([self.I_g * 1000, self.I_gg * 1000, xmax]), np.array([
                (self.tms * self.k) / (((self.I_g / self.I_s) ** self.alpha) - 1) + self.t_grade, self.t_g, self.t_gg]))
        else:
            raise ValueError('oc_relay_type must be DTOC, IDMT, or IDTOC')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.title(title + str(self.switch_index))
        plt.grid(True, which="both", ls="-")

    def __str__(self):
        s = 'Protection Device: %s \nType: %s \nName: %s' % (self.__class__.__name__, self.oc_relay_type, self.name)
        self.characteristic_index = 1
        return s
