try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False

import numpy as np
from pandapower.control.util.characteristic import LogSplineCharacteristic
from pandapower.protection.basic_protection_device import ProtectionDevice
from pandapower.auxiliary import soft_dependency_error, ensure_iterability


class Fuse(ProtectionDevice):
    def __init__(self, net, rated_i_a, switch_index, fuse_type = "none", characteristic_index=None, in_service=True, overwrite=False,
                 **kwargs):
        super().__init__(net, in_service=in_service, overwrite=overwrite, **kwargs)
        self.rated_i_a = rated_i_a
        self.switch_index = switch_index
        self.fuse_type = fuse_type
        self.characteristic_index = characteristic_index
        self.tripped = False
        self.i_start_a = None
        self.i_stop_a = None
        self.activation_parameter = "i_ka"

    def create_characteristic(self, net, x_values, y_values, interpolator_kind="Pchip", fill_value="extrapolate", **kwargs):
        c = LogSplineCharacteristic(net, x_values=x_values, y_values=y_values, interpolator_kind=interpolator_kind, **kwargs)
        self.characteristic_index = c.index
        self.i_start_a = min(x_values)
        self.i_stop_a = max(x_values)
        # todo: for fuses with multiple characteristics either a) create a separate class or
        #  b) adjust the code to use a list of indices for characteristic_index by default and then
        #  add the c.index value to the list

    def reset_device(self):
        self.tripped = False

    def has_tripped(self):
        return self.tripped

    def protection_function(self, net):  # separate into protection_time and protection_decision?
        # trips switch in net accordingly, returns dictionary protection_result
        i_ka = net.res_switch_sc.ikss_ka.at[self.switch_index]
        c = net.characteristic.at[self.characteristic_index, "object"]
        if i_ka*1000 < self.i_start_a:
            self.tripped = False
            net.switch.at[self.switch_index, 'closed'] = not self.has_tripped()
            act_time_s = np.inf
        elif i_ka*1000 <= self.i_stop_a:
            self.tripped = True
            net.switch.at[self.switch_index, 'closed'] = not self.has_tripped()
            act_time_s = c(i_ka*1000)
        else:
            self.tripped = True
            net.switch.at[self.switch_index, 'closed'] = not self.has_tripped()
            act_time_s = 0

        protection_result = {"switch_id": self.switch_index,
                             "prot_type": self.__class__.__name__,
                             "trip_melt": self.has_tripped(),
                             "act_param": self.activation_parameter,
                             "act_param_val": i_ka,
                             "trip_melt_time_s": act_time_s}

        return protection_result

    def plot_protection_characteristic(self, net, num=35, xlabel="I [kA]", ylabel="time [s]",
                                       title="Time-Current Characteristic of Fuse"):
        # plots protection characteristic for fuse
        if not MATPLOTLIB_INSTALLED:
            soft_dependency_error(str(sys._getframe().f_code.co_name) + "()", "matplotlib")
        start = min(net.characteristic.at[self.characteristic_index, "object"].x_vals)
        stop = max(net.characteristic.at[self.characteristic_index, "object"].x_vals)
        c = net.characteristic.at[self.characteristic_index, "object"]
        x = np.logspace(start, stop, num)
        y = c(x)
        plt.loglog(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(10**np.floor(start), 10**np.ceil(stop))
        plt.ylim(10**np.floor(min(net.characteristic.at[self.characteristic_index, "object"].y_vals)), 10**np.floor(max(net.characteristic.at[self.characteristic_index, "object"].y_vals)))
        plt.title(title)
        plt.grid(True, which="both", ls="-")
        pass

    # def __repr__(): display Fuse + name instead of Fuse


