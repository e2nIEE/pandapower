import numbers
import numpy as np

from pandapower.control.basic_controller import Controller
from pandapower.auxiliary import _detect_read_write_flag, read_from_net, write_to_net


class BinarySearchControl(Controller):
    """
        The Binary search control is a controller which is used to reach a given set point . It can be used for
        reactive power control or voltage control. in case of voltage control, the input parameter voltage_ctrl must be
        set to true. Input and output elements and indexes can be lists. Input elements can be transformers, switches,
        lines or busses (only in case of voltage control). in case of voltage control, a bus_index must be present,
        where the voltage will be controlled. Output elements are sgens, where active and reactive power can be set. The
        output value distribution describes the distribution of reactive power provision between multiple
        output_elements and must sum up to 1.

        INPUT:
            **self**

            **net** - A pandapower grid

            **ctrl_in_service** - Whether the controller is in service or not.

            **output_element** - Output element of the controller. Takes a string value "gen" or "sgen", with
            reactive power control, currently only "sgen" is possible.

            **output_variable** - Output variable of that element, normally "q_mvar".

            **output_element_index** - Index of output element in e.g. "net.sgen".

            **output_element_in_service** - Whether output elements are in service or not.

            **output_values_distribution** - Distribution of reactive power provision.

            **input_element** - Measurement location, can be a transformers, switches, lines or busses (only with
            voltage_ctrl), indicated by string value "res_trafo", "res_switch", "res_line" or "res_bus". In case of
            "res_switch", an additional small impedance is introduced in the switch.

            **input_variable** - Variable which is used to take the measurement from. Indicated by string value.

            **input_element_index** - Element of input element in net.

            **set_point** - Set point of the controller, can be a reactive power provision or a voltage set point. In
            case of voltage set point, voltage control must be set to true, bus_idx must be set to measurement bus and
            input_element must be "res_bus". Can be overwritten by a droop controller chained with the binary search
            control.

            **voltage_ctrl** - Whether the controller is used for voltage control or not.

            **bus_idx=None** - Bus index which is used for voltage control.

            **tol=0.001** - Tolerance criteria of controller convergence.
       """
    def __init__(self, net, ctrl_in_service, output_element, output_variable, output_element_index,
                 output_element_in_service, output_values_distribution, input_element, input_variable,
                 input_element_index, set_point, voltage_ctrl, bus_idx=None, tol=0.001, in_service=True, order=0, level=0,
                 drop_same_existing_ctrl=False, matching_params=None, **kwargs):
        super().__init__(net, in_service=in_service, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, **kwargs)
        self.in_service = ctrl_in_service
        self.input_element = input_element
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
        self.voltage_ctrl = voltage_ctrl
        self.bus_idx = bus_idx
        self.tol = tol
        self.applied = False
        self.output_values = None
        self.output_values_old = None
        self.diff = None
        self.diff_old = None
        self.converged = False
        self.overwrite_covergence = False
        self.write_flag, self.output_variable = _detect_read_write_flag(net, output_element, output_element_index,
                                                                        output_variable)
        self.read_flag = []
        self.input_variable = []
        self.input_element_in_service = []
        counter = 0
        for input_index in self.input_element_index:
            if self.input_element == "res_line":
                self.input_element_in_service.append(net.line.in_service[input_index])
            elif self.input_element == "res_trafo":
                self.input_element_in_service.append(net.trafo.in_service[input_index])
            elif self.input_element == "res_switch":
                self.input_element_in_service.append(
                    net[self.input_element].pf_in_service[input_index])
            elif self.input_element == "res_bus":
                self.input_element_in_service.append(net.bus.in_service[input_index])

            if isinstance(input_variable, list):
                read_flag_temp, input_variable_temp = _detect_read_write_flag(net, self.input_element,
                                                                              input_index,
                                                                              input_variable[counter])
            else:
                read_flag_temp, input_variable_temp = _detect_read_write_flag(net, self.input_element,
                                                                              input_index,
                                                                              input_variable)
            self.read_flag.append(read_flag_temp)
            self.input_variable.append(input_variable_temp)
            counter += 1

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
        self.input_element_in_service.clear()
        self.output_element_in_service.clear()
        for input_index in self.input_element_index:
            if self.input_element == "res_line":
                self.input_element_in_service.append(net.line.in_service[input_index])
            elif self.input_element == "res_trafo":
                self.input_element_in_service.append(net.trafo.in_service[input_index])
            elif self.input_element == "res_switch":
                self.input_element_in_service.append(net.switch.closed[input_index])
            elif self.input_element == "res_bus":
                self.input_element_in_service.append(net.bus.in_service[input_index])
        for output_index in self.output_element_index:
            if self.output_element == "gen":
                self.output_element_in_service.append(net.gen.in_service[output_index])
            elif self.output_element == "sgen":
                self.output_element_in_service.append(net.sgen.in_service[output_index])
            elif self.output_element == "shunt":
                self.output_element_in_service.append(net.shunt.in_service[output_index])
        # check if at least one input and one output element is in_service
        if not (any(self.input_element_in_service) and any(self.output_element_in_service)):
            self.converged = True
            return self.converged

        # read input values
        input_values = []
        counter = 0
        for input_index in self.input_element_index:
            input_values.append(read_from_net(net, self.input_element, input_index,
                                              self.input_variable[counter], self.read_flag[counter]))
            counter += 1
        # read previous set values
        # compare old and new set values
        if not self.voltage_ctrl or self.bus_idx is None:
            self.diff_old = self.diff
            self.diff = self.set_point - sum(input_values)
            self.converged = np.all(np.abs(self.diff) < self.tol)
        else:
            self.diff_old = self.diff
            self.diff = self.set_point - net.res_bus.vm_pu.at[self.bus_idx]
            self.converged = np.all(np.abs(self.diff) < self.tol)

        if self.overwrite_covergence:
            self.overwrite_covergence = False
            return False
        else:
            return self.converged

    def control_step(self, net):
        self._binarysearchcontrol_step(net)

    def _binarysearchcontrol_step(self, net):
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
        write_to_net(net, self.output_element, self.output_element_index, self.output_variable, self.output_values,
                     self.write_flag)

    def __str__(self):
        return super().__str__() + " [%s.%s.%s.%s]" % (
            self.input_element, self.input_variable, self.output_element, self.output_variable)


class DroopControl(Controller):
    """
            The droop controller is used in case of a droop based control. It can operate either as a Q(U) controller or
            as a U(Q) controller and is used in addition to a binary search controller (bsc). The linked binary search
            controller is specified using the controller index, which refers to the linked bsc. The droop controller
            behaves in a similar way to the station controllers presented in the Power Factory Tech Ref, although not
            all possible settings from Power Factory are yet available.

            INPUT:
                **self**

                **net** - A pandapower grid.

                **q_droop_var** - Droop Value in Mvar/p.u.

                **bus_idx** - Bus index in case of voltage control.

                **vm_set_pu** - Voltage set point in case of voltage control.

                **controller_idx** - Index of linked Binary< search control (if present).

                **voltage_ctrl** - Whether the controller is used for voltage control or not.

                **bus_idx=None** - Bus index which is used for voltage control.

                **tol=1e-6** - Tolerance criteria of controller convergence.

                **vm_set_lb=None** - Lower band border of dead band

                **vm_set_ub=None** - Upper band border of dead band
           """
    def __init__(self, net, q_droop_mvar, bus_idx, vm_set_pu, controller_idx, voltage_ctrl, tol=1e-6, in_service=True,
                 order=-1, level=0, drop_same_existing_ctrl=False, matching_params=None, vm_set_lb=None, vm_set_ub=None,
                 **kwargs):
        super().__init__(net, in_service=in_service, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, **kwargs)
        # TODO: implement maximum and minimum of droop control
        self.q_droop_mvar = q_droop_mvar
        self.bus_idx = bus_idx
        self.vm_pu = None
        self.vm_pu_old = self.vm_pu
        self.vm_set_pu = vm_set_pu
        self.vm_set_pu_new = None
        self.lb_voltage = vm_set_lb
        self.ub_voltage = vm_set_ub
        self.controller_idx = controller_idx
        self.voltage_ctrl = voltage_ctrl
        self.tol = tol
        self.applied = False
        self.read_flag, self.input_variable = _detect_read_write_flag(net, "res_bus", bus_idx, "vm_pu")
        self.q_set_mvar_bsc = None
        self.q_set_mvar = None
        self.q_set_old_mvar = None
        self.diff = None
        self.converged = False

    def is_converged(self, net):
        if self.voltage_ctrl:
            self.diff = (net.controller.at[self.controller_idx, "object"].set_point -
                         read_from_net(net, "res_bus", self.bus_idx, "vm_pu", self.read_flag))
        else:
            counter = 0
            input_values = []
            for input_index in net.controller.at[self.controller_idx, "object"].input_element_index:
                input_values.append(
                    read_from_net(net, net.controller.at[self.controller_idx, "object"].input_element, input_index,
                                  net.controller.at[self.controller_idx, "object"].input_variable[counter],
                                  net.controller.at[self.controller_idx, "object"].read_flag[counter]))
                counter += 1
            self.diff = ((net.controller.at[self.controller_idx, "object"].set_point - sum(input_values)))
        # bigger differences with switches as input elements, increase tolerance
        #if net.controller.at[self.controller_idx, "object"].input_element == "res_switch":
        #    self.tol = 0.2
        if self.bus_idx is None:
            self.converged = np.all(np.abs(self.diff) < self.tol)
        else:
            if np.all(np.abs(self.diff) < self.tol):
                self.converged = net.controller.at[self.controller_idx, "object"].converged
            elif net.controller.at[self.controller_idx, "object"].diff_old is not None:
                net.controller.at[self.controller_idx, "object"].overwrite_covergence = True

        return self.converged

    def control_step(self, net):
        self._droopcontrol_step(net)

    def _droopcontrol_step(self, net):
        self.vm_pu_old = self.vm_pu
        self.vm_pu = read_from_net(net, "res_bus", self.bus_idx, "vm_pu", self.read_flag)
        if not self.voltage_ctrl:
            if self.q_set_mvar_bsc is None:
                self.q_set_mvar_bsc = net.controller.at[self.controller_idx, "object"].set_point
            if self.lb_voltage is not None and self.ub_voltage is not None:
                if self.vm_pu > self.ub_voltage:
                    self.q_set_old_mvar, self.q_set_mvar = (
                        self.q_set_mvar, self.q_set_mvar_bsc + (self.ub_voltage - self.vm_pu) * self.q_droop_mvar)
                elif self.vm_pu < self.lb_voltage:
                    self.q_set_old_mvar, self.q_set_mvar = (
                        self.q_set_mvar, self.q_set_mvar_bsc + (self.lb_voltage - self.vm_pu) * self.q_droop_mvar)
                else:
                    self.q_set_old_mvar, self.q_set_mvar = (self.q_set_mvar, self.q_set_mvar_bsc)
            else:
                self.q_set_old_mvar, self.q_set_mvar = (
                    self.q_set_mvar, self.q_set_mvar - (self.vm_set_pu - self.vm_pu) * self.q_droop_mvar)

            if self.q_set_old_mvar is not None:
                self.diff = self.q_set_mvar - self.q_set_old_mvar
            if self.q_set_mvar is not None:
                net.controller.at[self.controller_idx, "object"].set_point = self.q_set_mvar

        else:
            input_element = net.controller.at[self.controller_idx, "object"].input_element
            input_element_index = net.controller.at[self.controller_idx, "object"].input_element_index
            input_variable = net.controller.at[self.controller_idx, "object"].input_variable
            read_flag = net.controller.at[self.controller_idx, "object"].read_flag
            input_values = []
            counter = 0
            for input_index in input_element_index:
                input_values.append(read_from_net(net, input_element, input_index,
                                                  input_variable[counter], read_flag[counter]))
            self.vm_set_pu_new = self.vm_set_pu + sum(input_values) / self.q_droop_mvar
            net.controller.at[self.controller_idx, "object"].set_point = self.vm_set_pu_new
