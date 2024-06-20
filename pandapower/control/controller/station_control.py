import numbers
import numpy as np

import pandapower as pp

from pandapower.control.basic_controller import Controller
from pandapower.auxiliary import _detect_read_write_flag, read_from_net, write_to_net


class BinarySearchControl(Controller):
    def __init__(self, net, output_element, output_variable, output_element_index, output_values_distribution,
                 input_element, input_variable, input_element_index,
                 set_point, tol=0.001, in_service=True, order=0, level=0, drop_same_existing_ctrl=False,
                 matching_params=None, **kwargs):
        super().__init__(net, in_service=in_service, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, **kwargs)
        self.input_element = input_element
        self.input_element_index = input_element_index
        self.output_element = output_element
        self.output_element_index = output_element_index
        # normalize the values distribution:
        self.output_values_distribution = np.array(output_values_distribution, dtype=np.float64) / np.sum(
            output_values_distribution)
        self.set_point = set_point
        self.tol = tol
        self.applied = False
        self.output_values = None
        self.output_values_old = None
        self.diff = None
        self.diff_old = None
        self.write_flag, self.output_variable = _detect_read_write_flag(net, output_element, output_element_index,
                                                                        output_variable)
        self.read_flag, self.input_variable = _detect_read_write_flag(net, input_element, input_element_index,
                                                                      input_variable)

    def initialize_control(self, net):
        self.output_values = read_from_net(net, self.output_element, self.output_element_index, self.output_variable,
                                           self.write_flag)

    def is_converged(self, net):
        """
        Actual implementation of the convergence criteria: If controller is applied, it can stop
        """
        # todo: if output_element or input_element not in_service, return True
        # read input values
        input_values = read_from_net(net, self.input_element, self.input_element_index, self.input_variable,
                                     self.read_flag)
        # read previous set values
        # compare old and new set values
        self.diff_old = self.diff
        self.diff = self.set_point - input_values
        converged = np.all(np.abs(self.diff) < self.tol)

        return converged

    def control_step(self, net):
        if self.output_values_old is None:
            self.output_values_old, self.output_values = self.output_values, self.output_values + 1e-3
        else:
            step_diff = self.diff - self.diff_old
            x = self.output_values - self.diff * (self.output_values - self.output_values_old) / np.where(
                step_diff == 0, 1e-6, step_diff)
            x = x * self.output_values_distribution if isinstance(x, numbers.Number) else sum(
                x) * self.output_values_distribution
            self.output_values_old, self.output_values = self.output_values, x

        # write new set values
        write_to_net(net, self.output_element, self.output_element_index, self.output_variable, self.output_values,
                     self.write_flag)

    def __str__(self):
        return super().__str__() + " [%s.%s.%s.%s]" % (
        self.input_element, self.input_variable, self.output_element, self.output_variable)


class DroopControl(Controller):
    def __init__(self, net, q_droop_mvar, bus_idx, vm_set_pu, controller_idx, tol=1e-6, in_service=True,
                 order=-1, level=0, drop_same_existing_ctrl=False, matching_params=None, **kwargs):
        super().__init__(net, in_service=in_service, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, **kwargs)
        self.q_droop_mvar = q_droop_mvar
        self.bus_idx = bus_idx
        self.vm_set_pu = vm_set_pu
        self.controller_idx = controller_idx
        self.tol=tol
        self.applied = False
        self.read_flag, self.input_variable = _detect_read_write_flag(net, "res_bus", bus_idx, "vm_pu")

        self.q_set_mvar = None
        self.q_set_old_mvar = None
        self.delta = None

    def is_converged(self, net):
        return self.delta is not None and np.all(np.abs(self.delta) < self.tol)

    def control_step(self, net):
        vm_pu = read_from_net(net, "res_bus", self.bus_idx, "vm_pu", self.read_flag)
        self.q_set_old_mvar, self.q_set_mvar = self.q_set_mvar, (vm_pu - self.vm_set_pu) * self.q_droop_mvar
        net.controller.at[self.controller_idx, "object"].set_point = self.q_set_mvar
        if self.q_set_old_mvar is not None:
            self.delta = self.q_set_mvar - self.q_set_old_mvar