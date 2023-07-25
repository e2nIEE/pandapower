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


class Fuse(ProtectionDevice):
    def __init__(self, net, switch_index, fuse_type="none", rated_i_a=0, characteristic_index=None, in_service=True,
                 overwrite=False, curve_select=0, z_ohm=0.0001, name=None, **kwargs):
        super().__init__(net, in_service=in_service, overwrite=overwrite, **kwargs)
        self.switch_index = switch_index
        self.fuse_type = fuse_type
        self.in_service = in_service
        self.name = name
        # create protection characteristic curve from std type
        if std_type_exists(net, fuse_type, element="fuse"):
            fuse_data = load_std_type(net=net, name=fuse_type, element="fuse")
            self.rated_i_a = fuse_data['i_rated_a']
            if not np.isnan(fuse_data["t_avg"]).any():
                self.create_characteristic(net, fuse_data["x_avg"], fuse_data["t_avg"])
            elif not np.isnan(fuse_data["t_min"]).any() and curve_select == 0:
                self.create_characteristic(net, fuse_data["x_min"], fuse_data["t_min"])
            elif not np.isnan(fuse_data["t_total"]).any() and curve_select == 1:
                self.create_characteristic(net, fuse_data["x_total"], fuse_data["t_total"])
            else:
                raise ValueError("curve_select must equal 0 or 1")
        else:
            self.rated_i_a = rated_i_a
            self.characteristic_index = characteristic_index
            self.i_start_a = None
            self.i_stop_a = None

        self.activation_parameter = "i_ka"
        self.tripped = False
        self.z_ohm = z_ohm
        net.switch.at[self.switch_index, 'z_ohm'] = self.z_ohm

    def create_characteristic(self, net, x_values, y_values, interpolator_kind="Pchip", **kwargs):
        c = LogSplineCharacteristic(net, x_values=x_values, y_values=y_values, interpolator_kind=interpolator_kind,
                                    **kwargs)
        self.characteristic_index = c.index
        self.i_start_a = min(x_values)
        self.i_stop_a = max(x_values)

    def reset_device(self):
        self.tripped = False

    def has_tripped(self):
        return self.tripped

    def protection_function(self, net, scenario="sc"):
        # compute protection time in net under short-circuit or operating conditions
        if scenario == "sc":
            i_ka = net.res_switch_sc.ikss_ka.at[self.switch_index]
        elif scenario == "op":
            i_ka = net.res_switch.i_ka.at[self.switch_index]
        else:
            raise ValueError("scenario must be either sc or op")
        c = net.characteristic.at[self.characteristic_index, "object"]
        if i_ka * 1000 < self.i_start_a:
            self.tripped = False
            act_time_s = np.inf
        elif i_ka * 1000 <= self.i_stop_a:
            self.tripped = True
            act_time_s = c(i_ka * 1000)
        else:
            self.tripped = True
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
        plt.xlim(10 ** np.floor(start), 10 ** np.ceil(stop))
        plt.ylim(10 ** np.floor(min(net.characteristic.at[self.characteristic_index, "object"].y_vals)),
                 10 ** np.floor(max(net.characteristic.at[self.characteristic_index, "object"].y_vals)))
        plt.title(title)
        plt.grid(True, which="both", ls="-")

    def __str__(self):  # display Fuse + name instead of Fuse
        s = 'Protection Device: %s \nType: %s \nName: %s' % (self.__class__.__name__, self.fuse_type, self.name)
        self.characteristic_index = 1
        return s
