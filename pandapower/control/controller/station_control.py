import numbers
from winreg import error

import numpy as np
from numpy.f2py.auxfuncs import throw_error
from win32pdh import counter_status_error
import matplotlib.pyplot as plt #todo weg damit

from pandapower.control.basic_controller import Controller
from pandapower.auxiliary import _detect_read_write_flag, read_from_net, write_to_net
try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

class BinarySearchControl(Controller):
    """
        The Binary search control is a controller which is used to reach a given set point . It can be used for
        reactive power control, voltage control, cosinus(phi) or tangens(phi) control. The control modus can be set via
        the modus parameter. Input and output elements and indexes can be lists. Input elements can be transformers,
        switches, lines or buses (only in case of voltage control). in case of voltage control, a bus_index must be
        present, where the voltage will be controlled. Output elements are sgens, where active and reactive power can
        be set. The output value distribution describes the distribution of reactive power provision between multiple
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
            V_ctrl), indicated by string value "res_trafo", "res_switch", "res_line" or "res_bus". In case of
            "res_switch", an additional small impedance is introduced in the switch.

            **input_variable** - Variable which is used to take the measurement from. Indicated by string value.

            **input_element_index** - Element of input element in net.

            **set_point** - Set point of the controller, can be a reactive power provision or a voltage set point. In
            case of voltage set point, modus must be V_ctrl, bus_idx must be set to measurement bus and
            input_element must be "res_bus". Can be overwritten by a droop controller chained with the binary search
            control.

            **modus** - Enables the selection of the available control modi by taking one of the strings: Q_ctrl, V_ctrl,
            PF_ctrl or tan(phi)_ctrl. Formerly called Voltage_ctrl

            **bus_idx=None** - Bus index which is used for voltage control.

            **tol=0.001** - Tolerance criteria of controller convergence.
       """
    def __init__(self, net, ctrl_in_service, output_element, output_variable, output_element_index,
                 output_element_in_service, output_values_distribution, input_element, input_variable,
                 input_element_index, set_point, modus= None, bus_idx=None, tol=0.001, in_service=True, order=0, level=0,
                 drop_same_existing_ctrl=False, matching_params=None, **kwargs):
        super().__init__(net, in_service=in_service, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, **kwargs)
        self.counter_deprecation_message=False
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
        self.input_variable_p = []
        self.input_element_in_service = []
        counter = 0

        if modus == True: #Only functions written out!?!
            self.modus = "V_ctrl"
        elif modus == False: #Only functions written out!?!
            self.modus = "Q_ctrl"
            logger.error("Ambivalent type, using Q_ctrl from available types 'Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan(phi)_ctrl'\n")
        elif modus == "PF_ctrl_cap": # -1 for capacitive, 1 for inductive systems
            self.modus = "PF_ctrl"
            self.reactance= -1
        elif modus == "PF_ctrl_ind":
            self.modus = "PF_ctrl"
            self.reactance = 1 #todo bei droop einfach inductiv? oder Kapazitiv?
        elif modus == "PF_ctrl":
            self.modus = modus
            self.reactance = -1

        else:
            self.modus = modus
        if self.modus == 'PF_ctrl':
            if abs(self.set_point) >1:
                raise UserWarning('Set point out of range ([-1,1]')

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
            if self.modus == "PF_ctrl" or self.modus=='tan(phi)_ctrl':
                if isinstance(input_variable, list):
                    if input_variable[counter]=="q_from_mvar":
                        input_variable_p= "p_from_mw"
                    elif input_variable[counter]=="q_to_mvar":
                        input_variable_p= 'p_to_mw'
                    elif input_variable[counter]=='q_hv_mvar':
                        input_variable_p = 'p_hv_mw'
                    elif input_variable[counter]=='q_lv_mvar':
                        input_variable_p='p_lv_mw'
                    else:
                        logger.error('incorrect input variable: ', input_variable[counter],'\n')
                        return
                    read_flag_temp, input_variable_temp_p = _detect_read_write_flag(net, self.input_element,
                                                                                  input_index,
                                                                                  input_variable_p)
                else:
                    input_variable_p = input_variable.replace('q', 'p').replace('var','w') #replace string or if statements?

                    read_flag_temp, input_variable_temp_p = _detect_read_write_flag(net, self.input_element,
                                                                                  input_index,
                                                                                  input_variable_p)
                self.read_flag.append(read_flag_temp)
                self.input_variable_p.append(input_variable_temp_p)
            self.read_flag.append(read_flag_temp)
            self.input_variable.append(input_variable_temp)
            counter += 1

    def __getattr__(self, name):
        if name == "modus":
            try:
                if not self.counter_deprecation_message:
                    logger.error("The 'voltage_ctrl' attribute is deprecated and will be removed in future versions.\n"
                            "Please use 'modus' ('Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan(phi)_ctrl') instead.\n")
                else:
                    self.counter_deprecation_message = True
                return self.voltage_ctrl

            except AttributeError:
                self.counter_deprecation_message = True
                logger.error("The 'voltage_ctrl' attribute is deprecated and will be removed in future versions.\n"
                      "Please use 'modus' ('Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan(phi)_ctrl') instead.\n")
                return self.voltage_ctrl
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

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
        input_values = [] #reactive q
        p_input_values = [] #active p
        counter = 0
        for input_index in self.input_element_index:
            input_values.append(read_from_net(net, self.input_element, input_index,
                                              self.input_variable[counter], self.read_flag[counter]))
            if self.modus == "PF_ctrl" or self.modus == 'tan(phi)_ctrl':
                p_input_values.append(read_from_net(net,self.input_element, input_index,
                                                self.input_variable_p[counter], self.read_flag[counter]))

            counter += 1
        # read previous set values
        # compare old and new set values
        #Q_ctrl, V_ctrl, PF_ctrl or tan(phi)_ctrl
        if self.modus == "Q_ctrl" or (self.modus=='V_ctrl' and self.bus_idx is None):
            self.diff_old = self.diff
            self.diff = self.set_point - sum(input_values)
            self.converged = np.all(np.abs(self.diff) < self.tol)

        elif self.modus == "PF_ctrl":#capacitive => reactance = -1, inductive => reactance = 1
            self.diff_old = self.diff
            q_set = self.reactance * sum(p_input_values) * (np.tan(np.arccos(self.set_point)))
            self.diff = q_set - sum(input_values)
            self.converged = np.all(np.abs(self.diff)<self.tol)

        elif self.modus == "tan(phi)_ctrl":
            self.diff_old = self.diff
            q_set = sum(p_input_values) * self.set_point
            self.diff = q_set - sum(input_values)
            self.converged = np.all(np.abs(self.diff) < self.tol)
        else:
            if self.modus != "V_ctrl":
                logger.error("No Controller Modus specified, using V_ctrl.\n"
                      "Please specify 'modus' ('Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan(phi)_ctrl')\n")
                #self.modus = 'V_ctrl'
            self.diff_old = self.diff
            self.diff = self.set_point - net.res_bus.vm_pu.at[self.bus_idx] #error when old import? no bus_idx
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
            The droop controller is used in case of a droop based control. It can operate either as a Q(U) controller,
            as a U(Q) controller, as a cosphi(P) controller or as a cosphi(U) controller and is used in addition to
            a binary search controller (bsc). The linked binary search controller is specified using the
            controller index, which refers to the linked bsc. The droop controller behaves in a similar way to the
            station controllers presented in the Power Factory Tech Ref, although not
            all possible settings from Power Factory are yet available.

            INPUT:
                **self**

                **net** - A pandapower grid.

                **q_droop_var** - Droop Value in Mvar/p.u.

                **bus_idx** - Bus index in case of voltage control.

                **controller_idx** - Index of linked Binary< search control (if present).

                **modus** - takes string: Q_ctrl, V_ctrl, PF_ctrl or tan(phi)_ctrl. Select droop variety of PF_ctrl by
                choosing 'PF_ctrl_P' for P-Characteristic or 'PF_ctrl_U' for U-Characteristic. Formerly called
                voltage_ctrl.

                **vm_set_pu=None** - Voltage set point in case of voltage control.

                **PF_overexcited=None** - Static overexcited limit for Phi in case of PF_ctrl.

                **PF_underexcited=None** - Static underexcited limit for Phi in case of PF_ctrl.

                **bus_idx=None** - Bus index which is used for voltage control.

                **tol=1e-6** - Tolerance criteria of controller convergence.

                **vm_set_lb=None** - Lower band border of dead band; The Power[W] or Voltage[pu] at which Phi is static and underexcited
                (inductive) in case of PF_ctrl

                **vm_set_ub=None** - Upper band border of dead band; The Power[W] or Voltage[pu] at which Phi is static and overexcited
                (capacitive) in case of PF_ctrl
           """
    def __init__(self, net, q_droop_mvar,  controller_idx, modus, vm_set_pu= None, PF_overexcited=None, PF_underexcited=None,bus_idx=None, tol=1e-6, in_service=True,
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
        self.tol = tol
        self.applied = False
        self.read_flag, self.input_variable = _detect_read_write_flag(net, "res_bus", bus_idx, "vm_pu")
        self.q_set_mvar_bsc = None
        self.q_set_mvar = None
        self.q_set_old_mvar = None
        self.diff = None
        self.converged = False
        self.counter_deprecation_message = False
        #self.output_values_old = None
        self.pf_over = PF_overexcited
        self.pf_under = PF_underexcited

        if modus == True: #Only functions when written out!?!
            self.modus = "V_ctrl"
        elif modus == False: #Only functions when written out!?!
            self.modus = "Q_ctrl"
            logger.error("Ambivalent type, using Q_ctrl from available types 'Q_ctrl', 'V_ctrl' or 'PF_ctrl'\n")
        elif modus == "PF_ctrl_cap" or modus == "PF_ctrl_ind" or modus == 'PF_ctrl' or modus == 'PF_ctrl_P':
            if modus != 'PF_ctrl_P':
                logger.warning("Power Factor Droop Control: Modus is ambivalent, using 'PF_ctrl_P' from available modi: 'PF_ctrl_P' and 'PF_ctrl_U'\n")
            self.modus = 'PF_ctrl'
            self.p_cosphi = True
        elif modus == 'PF_ctrl_U':
            self.modus = 'PF_ctrl'
            self.p_cosphi = False
        else:
            if modus == 'Q_ctrl' or modus == "V_ctrl":
                if vm_set_pu is None and modus == 'V_ctrl':
                    raise UserWarning(f'vm_set_pu must be a number, not {type(vm_set_pu)}')
                self.modus = modus
            else:
                raise UserWarning('Droop Control Modus not decipherable')


        if self.modus == 'PF_ctrl':
            if self.lb_voltage is None or self.ub_voltage is None:
                raise UserWarning('Input error, vm_set_lb and vm_set_ub must be a number')
            if self.lb_voltage < 0 or self.ub_voltage < 0:
                if self.p_cosphi:
                    raise UserWarning('P_Maximum (vm_set_ub) and P_Minimun (vm_set_lb) must be >= 0 W\n')
                elif not self.p_cosphi:
                    raise UserWarning('U_Maximum (vm_set_ub) and U_Minimun (vm_set_lb) must be >= 0 pu\n')
                else:
                    raise UserWarning(f'Something wrong with the entered values {self.lb_voltage, self.ub_voltage}')
            if  1 < self.pf_over < 0 or 1 < self.pf_under < 0:
                raise UserWarning('Power Factor limtits PF_overexcited and PF_underexcited must be between 0 and 1')
            if self.lb_voltage == self.ub_voltage:
                if self.p_cosphi:
                    raise UserWarning('P_Maximum and P_Minimum may not be the same value')
                elif not self.p_cosphi:
                    raise UserWarning('U_Maximum and U_Minimum may not be the same value')
                else:
                    raise UserWarning(f'Something wrong with the entered values {self.lb_voltage, self.ub_voltage}')

    def __getattr__(self, name):
        if name == "modus":
            try:
                if not self.counter_deprecation_message:
                    logger.error("The 'voltage_ctrl' attribute is deprecated and will be removed in future versions.\n"
                            "Please use 'modus' ('Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan(phi)_ctrl') instead.\n")
                else:
                    self.counter_deprecation_message = True
                return self.voltage_ctrl

            except AttributeError:
                self.counter_deprecation_message = True
                logger.error("The 'voltage_ctrl' attribute is deprecated and will be removed in future versions.\n"
                      "Please use 'modus' ('Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan(phi)_ctrl') instead.\n")
                return self.voltage_ctrl
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def is_converged(self, net):
        if self.modus == 'V_ctrl':
            self.diff = (net.controller.at[self.controller_idx, "object"].set_point -
                         read_from_net(net, "res_bus", self.bus_idx, "vm_pu", self.read_flag))
        elif self.modus == 'PF_ctrl':
            if self.q_set_old_mvar is not None and self.q_set_mvar:
                self.diff = self.q_set_mvar - self.q_set_old_mvar
            else:
                counter = 0
                input_values = []
                p_input_values = []
                for input_index in net.controller.at[self.controller_idx, "object"].input_element_index:
                    input_values.append(
                        read_from_net(net, net.controller.at[self.controller_idx, "object"].input_element, input_index,
                                      net.controller.at[self.controller_idx, "object"].input_variable[counter],
                                      net.controller.at[self.controller_idx, "object"].read_flag[counter]))
                    p_input_values.append(
                        read_from_net(net, net.controller.at[self.controller_idx, "object"].input_element, input_index,
                                      net.controller.at[self.controller_idx, "object"].input_variable_p[counter],
                                      net.controller.at[self.controller_idx, "object"].read_flag[counter]))
                    counter += 1

                q_set = net.controller.at[self.controller_idx, "object"].reactance * sum(p_input_values) * (
                    np.tan(np.arccos(net.controller.at[self.controller_idx, "object"].set_point)))
                self.diff = q_set - sum(input_values)

        elif self.modus == 'tan(phi)_ctrl':
            raise UserWarning('No droop option for tan(phi) controller')
        else:
            if self.modus != 'Q_ctrl':
                logger.error('No specified modus in droop controller, using Q_ctrl')
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

        if self.modus != 'PF_ctrl' or self.p_cosphi == False:
            self.vm_pu = read_from_net(net, "res_bus", self.bus_idx, "vm_pu", self.read_flag)
            self.vm_pu_old = self.vm_pu

        if self.modus=='Q_ctrl':
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
        elif self.modus == 'PF_ctrl':
            counter = 0
            input_values = []
            p_input_values = [] #P_values if p_cosphi, U_values if not
            if self.p_cosphi:
                for input_index in net.controller.at[self.controller_idx, "object"].input_element_index:
                    input_values.append(
                        read_from_net(net, net.controller.at[self.controller_idx, "object"].input_element, input_index,
                                      net.controller.at[self.controller_idx, "object"].input_variable[counter],
                                      net.controller.at[self.controller_idx, "object"].read_flag[counter]))

                    p_input_values.append(
                        read_from_net(net, net.controller.at[self.controller_idx, "object"].input_element, input_index,
                                      net.controller.at[self.controller_idx, "object"].input_variable_p[counter],
                                      net.controller.at[self.controller_idx, "object"].read_flag[counter]))
                    counter += 1
            elif not self.p_cosphi:
                for input_index in net.controller.at[self.controller_idx, "object"].input_element_index:
                    input_values.append(
                        read_from_net(net, net.controller.at[self.controller_idx, "object"].input_element, input_index,
                                      net.controller.at[self.controller_idx, "object"].input_variable[counter],
                                      net.controller.at[self.controller_idx, "object"].read_flag[counter]))
                    counter += 1
                p_input_values.append(self.vm_pu)


            else:
                raise UserWarning('wrong modus, should not happen')
            if self.lb_voltage > self.ub_voltage: #phi overexcited > phi underexcited
                if self.lb_voltage <= sum(p_input_values)/len(p_input_values) or sum(p_input_values)/len(p_input_values) <= -self.lb_voltage:#underexcited limit(-1)
                    pf_cosphi = self.pf_under
                    net.controller.at[self.controller_idx, "object"].reactance = -1
                elif -self.ub_voltage <= sum(p_input_values)/len(p_input_values) <= self.ub_voltage:# overexcited limit (1)
                    pf_cosphi = self.pf_over
                    net.controller.at[self.controller_idx, "object"].reactance = 11
                else: #droop
                    m = ((1-self.pf_under) + (1-self.pf_over)) / (self.ub_voltage - self.lb_voltage)
                    b = (1-self.pf_over) - m * self.ub_voltage
                    if sum(p_input_values)/len(p_input_values) >= 0:
                        droop_set_point = m * sum(p_input_values)/len(p_input_values) + b
                    else:#f(x)=f(-x) for p<0
                        droop_set_point = m * -sum(p_input_values)/len(p_input_values) + b
                    if droop_set_point < 0: #reactance from droop_set_point
                        net.controller.at[self.controller_idx, "object"].reactance = -1
                    else:
                        net.controller.at[self.controller_idx, "object"].reactance = 1
                    pf_cosphi = (1-abs(droop_set_point)) #pass on new setpoint

                    """x_val = np.linspace(self.ub_voltage,self.lb_voltage, 100)
                    x_val2 = np.linspace(-self.lb_voltage, -self.ub_voltage, 100)
                    fun2 = m*-x_val2+b
                    fun = m * x_val + b
                    plt.plot(x_val, fun)
                    plt.plot(x_val2, fun2)
                    plt.scatter(sum(p_input_values)/len(p_input_values), droop_set_point, marker='o', color='red')
                    plt.title(pf_cosphi)
                    plt.show()"""

            elif self.lb_voltage < self.ub_voltage: #phi overexcited < phi underexcited
                if -self.ub_voltage >= sum(p_input_values)/len(p_input_values) or sum(p_input_values)/len(p_input_values) >= self.ub_voltage:#overexcited limit (1)
                    pf_cosphi = self.pf_over
                    net.controller.at[self.controller_idx, "object"].reactance = 1
                elif -self.lb_voltage <= sum(p_input_values)/len(p_input_values) <= self.lb_voltage:# underexcited limit(-1)
                    pf_cosphi = self.pf_under
                    net.controller.at[self.controller_idx, "object"].reactance = -1
                else:#droop
                    m = ((1-self.pf_under)+ (1-self.pf_over)) / (self.ub_voltage - self.lb_voltage)
                    b = -(1-self.pf_under) - m * self.lb_voltage
                    if sum(p_input_values)/len(p_input_values) >= 0:
                        droop_set_point = (m * sum(p_input_values)/len(p_input_values) + b)
                    else: #f(x) = f(-x) for p<0
                        droop_set_point = (m * -sum(p_input_values)/len(p_input_values) + b)
                    if droop_set_point >= 0:
                        net.controller.at[self.controller_idx, "object"].reactance = 1
                    else: #reactance from droop_set_point
                        net.controller.at[self.controller_idx, "object"].reactance = -1

                    pf_cosphi = (1-abs(droop_set_point)) #pass on new setpoint

                    """x_val = np.linspace(self.lb_voltage, self.ub_voltage,100)
                    fun = m*x_val+b
                    x_val2 = np.linspace(-self.ub_voltage, -self.lb_voltage, 100)
                    fun2 = m * -x_val2 + b
                    plt.plot(x_val, fun)
                    plt.plot(x_val2, fun2)
                    plt.scatter(sum(p_input_values)/len(p_input_values), droop_set_point, marker = 'o', color = 'red')
                    plt.title(pf_cosphi)
                    plt.show()"""#todo remove including the import
            else:
                raise UserWarning(f'error with limits {self.lb_voltage, self.ub_voltage}')

            self.q_set_old_mvar, self.q_set_mvar = self.q_set_mvar, pf_cosphi


        elif self.modus == 'tan(phi)_ctrl':
            pass  # tanphi_ctrl doesnt have droop control
        else:
            if self.modus != "V_ctrl":
                logger.error("No Droop Controller Modus specified, using V_ctrl.\n"
                             "Please specify 'modus' ('Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan(phi)_ctrl')\n")
            if self.q_set_mvar is not None:
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
