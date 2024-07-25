import numbers
import numpy as np

import pandapower as pp

from pandapower.control.basic_controller import Controller
from pandapower.auxiliary import _detect_read_write_flag, read_from_net, write_to_net


class BinarySearchControl(Controller):
    def __init__(self, net, ctrl_in_service, output_element, output_variable, output_element_index, output_element_in_service,
                 output_values_distribution, input_element, input_variable, input_element_index,
                 set_point, tol=0.001, in_service=True, order=0, level=0, drop_same_existing_ctrl=False,
                 matching_params=None, **kwargs):
        super().__init__(net, in_service=in_service, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, **kwargs)
        self.in_service = ctrl_in_service
        self.input_element = input_element
        # for element in input_element:
        #    self.input_element.append(element)
        self.input_element_index = []
        if isinstance(input_element_index, list):
            for element in input_element_index:
                self.input_element_index.append(element)
        else:
            self.input_element_index.append(input_element_index)
        self.output_element = output_element
        self.output_element_index = output_element_index
        self.output_element_in_service = output_element_in_service
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
        self.read_flag = []
        self.input_variable = []
        self.input_element_in_service = []
        for i in range(len(self.input_element_index)):
            match self.input_element:
                case "res_line":
                    self.input_element_in_service.append(net.line.in_service[self.input_element_index[i]])
                case "res_trafo":
                    self.input_element_in_service.append(net.trafo.in_service[self.input_element_index[i]])
                case "res_switch":
                    #print("switch")
                    #print([net[input_element].pf_in_service[self.input_element_index[i]]])
                    self.input_element_in_service.append(net[self.input_element].pf_in_service[self.input_element_index[i]])
            if isinstance(input_variable, list):
                read_flag_temp, input_variable_temp = _detect_read_write_flag(net, self.input_element,
                                                                              self.input_element_index[i],
                                                                              input_variable[i])
            else:
                read_flag_temp, input_variable_temp = _detect_read_write_flag(net, self.input_element,
                                                                              self.input_element_index[i],
                                                                              input_variable)
            self.read_flag.append(read_flag_temp)
            self.input_variable.append(input_variable_temp)
        # self.read_flag, self.input_variable = _detect_read_write_flag(net, input_element, input_element_index,
        #                                                              input_variable[0])

    def initialize_control(self, net):
        self.output_values = read_from_net(net, self.output_element, self.output_element_index, self.output_variable,
                                           self.write_flag)

    def is_converged(self, net):
        """
        Actual implementation of the convergence criteria: If controller is applied, it can stop
        """
        # if controller not in_service, return True
        self.in_service = net.controller.in_service[self.index]
        if not self.in_service:
            return True
        # if output_element or input_element not in_service, return True
        self.input_element_in_service.clear()
        self.output_element_in_service.clear()
        for i in range(len(self.input_element_index)):
            match self.input_element:
                case "res_line":
                    self.input_element_in_service.append(net.line.in_service[self.input_element_index[i]])
                case "res_trafo":
                    self.input_element_in_service.append(net.trafo.in_service[self.input_element_index[i]])
                case "res_switch":
                    self.input_element_in_service.append(True)#net[self.input_element].pf_in_service[self.input_element_index[i]])
        for i in range(len(self.output_element_index)):
            match self.output_element:
                case "gen":
                    self.output_element_in_service.append(net.gen.in_service[self.output_element_index[i]])
                case "sgen":
                    self.output_element_in_service.append(net.sgen.in_service[self.output_element_index[i]])
        # check if at least one input and one output element is in_service
        if not (any(self.input_element_in_service) and any(self.output_element_in_service)):
            converged = True
            # all(self.input_element_in_service) and not all(self.output_element_in_service)):
            print("Either Input or Output Element not in Service!")
            return converged
        # read input values
        input_values = []
        for i in range(len(self.input_element_index)):
            input_values.append(read_from_net(net, self.input_element, self.input_element_index[i],
                                              self.input_variable[i], self.read_flag[i]))
        # read previous set values
        # compare old and new set values
        self.diff_old = self.diff
        self.diff = self.set_point - sum(input_values)
        converged = np.all(np.abs(self.diff) < self.tol)

        return converged

    def control_step(self, net):
        if not self.in_service:
            return
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
        #print(self.input_element)
        #print(self.output_element)
        #print(self.output_element_index)
        write_to_net(net, self.output_element, self.output_element_index, self.output_variable, self.output_values,
                     self.write_flag)

    def __str__(self):
        return super().__str__() + " [%s.%s.%s.%s]" % (
            self.input_element, self.input_variable, self.output_element, self.output_variable)


class DroopControl(Controller):
    def __init__(self, net, q_droop_mvar, bus_idx, vm_set_pu, controller_idx, voltage_ctrl, tol=1e-6, in_service=True,
                 order=-1, level=0, drop_same_existing_ctrl=False, matching_params=None, vm_set_lb=None, vm_set_ub=None,
                 **kwargs):
        super().__init__(net, in_service=in_service, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, **kwargs)
        self.q_droop_mvar = q_droop_mvar
        self.bus_idx = bus_idx
        self.vm_pu = None
        self.vm_pu_old = self.vm_pu
        self.vm_set_pu = vm_set_pu
        self.lb_voltage = vm_set_lb
        self.ub_voltage = vm_set_ub
        self.controller_idx = controller_idx
        self.voltage_ctrl = voltage_ctrl
        self.tol = tol
        self.applied = False
        self.read_flag, self.input_variable = _detect_read_write_flag(net, "res_bus", bus_idx, "vm_pu")

        self.q_set_mvar = None
        self.q_set_old_mvar = None
        self.delta = None

    def is_converged(self, net):
        return self.delta is not None and np.all(np.abs(self.delta) < self.tol)

    def control_step(self, net):
        if self.index == 28:
            print("#########################################")
        self.vm_pu_old = self.vm_pu
        self.vm_pu = read_from_net(net, "res_bus", self.bus_idx, "vm_pu", self.read_flag)
        delta = 0
        if not self.voltage_control:
            if self.vm_pu_old is not None:
                delta = self.vm_pu - self.vm_pu_old
            if self.lb_voltage is not None and self.ub_voltage is not None:
                match self.vm_pu:
                    case self.vm_pu if self.vm_pu > self.ub_voltage:
                        self.q_set_old_mvar, self.q_set_mvar = (
                            self.q_set_mvar, -(self.ub_voltage - self.vm_pu) * self.q_droop_mvar)
                    case  self.vm_pu if self.vm_pu < self.lb_voltage:
                        self.q_set_old_mvar, self.q_set_mvar = (
                            self.q_set_mvar, -(self.lb_voltage - self.vm_pu) * self.q_droop_mvar)
            else:
                self.q_set_old_mvar, self.q_set_mvar = self.q_set_mvar, (self.vm_pu - self.vm_set_pu) * self.q_droop_mvar
            net.controller.at[self.controller_idx, "object"].set_point = self.q_set_mvar
            if self.q_set_old_mvar is not None:
                self.delta = self.q_set_mvar - self.q_set_old_mvar

            if self.index == 28:
                print("#########################################")
                print("Regler: ", self.index)
                print("Setpoint: ", self.vm_set_pu)
                print("Spannung pre: ", self.vm_pu_old)
                print("Spannung post: ", self.vm_pu)
                print("Delta Q: ", self.delta)
                print("Delta U: ", delta)
                print("Q setpoint: ", self.q_set_mvar)
                print("#########################################")
        else:
            print("Voltage Control")

