import numpy as np
from collections.abc import Sequence
import logging

from pandapower.control.basic_controller import Controller
from pandapower.auxiliary import _detect_read_write_flag, read_from_net, write_to_net

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class BinarySearchControl(Controller):
    """
        The Binary search control is a controller which is used to reach a given set point. It can be used for
        reactive power control, voltage control, cosines(phi) or tangens(phi) control. The control modus can be set via
        the control_modus parameter. Input and output elements and indexes can be lists. Input elements can be transformers,
        switches, lines or buses (only in case of voltage control). in case of voltage control, the controlled bus must be
        given to input_element_index. Output elements are sgens, where active and reactive power can be set. The
        output value distribution describes the distribution of reactive power provision between multiple
        output_elements and must sum up to 1.

        INPUT:
            **self**

            **net** - A pandapower grid

            **ctrl_in_service** - Whether the controller is in service or not.

            **output_element** - Output element of the controller. Takes a string value "gen" or "sgen", with
            reactive power control, currently only "sgen" is possible.

            **output_variable** - Output variable of that element, normally "q_mvar".

            **output_element_index** - Index or list of indices of the output element(s) in net.

            **output_element_in_service** - List indicating whether each output element is in service.

            **output_values_distribution** - Distribution of reactive power provision among output elements.

            **input_element** - Measurement location, can be a transformer, switches or lines. Must be a bus for
            V_ctrl. Indicated by string value "res_trafo", "res_switch", "res_line" or "res_bus". In case of
            "res_switch": an additional small impedance is introduced in the switch.

            **input_variable** - Variable which is used to take the measurement from. Indicated by string value. Must
            be 'vm_pu' for 'V_ctrl'.

            **input_inverted** - List of Booleans that indicates if the measurement of the input elements must be inverted. Required
            when importing from PowerFactory.

            **gen_q_response** - List of +/- 1 that indicates the Q gen response of the measurement location. Used in
            order to invert the droop value of the controller.

            **input_element_index** - Index or list of indices of the input element(s) in net.

            **set_point** - Set point of the controller, can be a reactive power provision or a voltage set point. In
            case of voltage set point, control_modus must be V_ctrl, input_element_index must be a bus (input_variable must be
            'vm_pu' input_element must be 'res_bus'). Can be overwritten by a droop controller chained with the binary
            search control. If 'V_ctrl' and automated bus selection (input_element_index == 'auto'), set_point will be
            the search criteria in kV for the controlled bus (V_bus >= V_set_point).

            **output_values_distribution** - Distribution of reactive power provision.

            **control_modus=None** - Enables the selection of the available control modi by taking one of the strings: Q_ctrl, V_ctrl,
            PF_ctrl (PF_ctrl_ind or PF_ctrl_cap for reactance of PF_ctrl) or tan(phi)_ctrl. Formerly called Voltage_ctrl

            **output_min_q_mvar** - Minimum reactive power limits for each output element.

            **output_max_q_mvar** - Maximum reactive power limits for each output element.

            **tol=0.001** - Tolerance for controller convergence.

            **in_service=True** - Whether the controller itself is in service.

            **order=0** - Execution order of the controller.

            **level=0** - Execution level of the controller.

            **drop_same_existing_ctrl=False** - Whether to drop existing controllers with the same parameters.

            **matching_params=None** - Parameters for matching controllers.

            **name=""** - Name of the controller.

            **kwargs** - Additional keyword arguments.

   """

    def __init__(self, net, ctrl_in_service:bool, output_element, output_variable, output_element_index,
                 output_element_in_service, input_element, input_variable,
                 input_element_index, set_point:float, output_values_distribution,
                 control_modus:str = None, name = "", input_inverted=None, gen_q_response:list=None, tol=0.001, order=0, level=0,
                 drop_same_existing_ctrl=False, matching_params=None, **kwargs):
        super().__init__(net, in_service=ctrl_in_service, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params)
        # write kwargs in self
        for key, value in kwargs.items():
            setattr(self, key, value)
        ###allocating variables
        self.name = name #name of controller, not unambiguous
        self.set_point = set_point
        self.tol = tol #tolerance
        self.input_sign = []#direction of Q at element
        self.input_variable = [] #unit of controlled element Q
        self.input_variable_p = [] #unit of controlled element P
        self.input_element_in_service = []
        self.input_element_index = []  # for boundaries
        self.in_service = ctrl_in_service
        self.input_element = input_element #point to be controlled
        self.output_values = None
        self.output_values_old = None
        self.output_element = output_element #typically sgens, output of Q
        self.output_values_distribution = np.array(output_values_distribution, dtype=np.float64) / np.sum(
            output_values_distribution)
        self.diff = None
        self.diff_old = None
        self.converged = False  # criteria for success of controller
        self.redistribute_values = None  # Values to save for redistributed gens
        self.counter_warning = False  # only one message that only one active output element
        self.read_flag = []  # type of read value
        self.write_flag, self.output_variable = _detect_read_write_flag(net, output_element, output_element_index,
                                                                        output_variable)
        ###catching errors in variables, allocating
        if input_inverted is None: input_inverted = []#for robustness
        if gen_q_response is None: gen_q_response = []#robustness and legacy
        if isinstance(output_element_index, list) or isinstance(output_element_index, np.ndarray):
            self.output_element_index = [int(item) for item in output_element_index]
        else:
            self.output_element_index = []
            self.output_element_index.append(output_element_index)
            self.output_element_index = self.output_element_index
        if isinstance(output_element_in_service, bool):
            self.output_element_in_service = [output_element_in_service]
        else:
            self.output_element_in_service = output_element_in_service
        if isinstance(input_element_index, list) or isinstance(input_element_index, np.ndarray):
            for element in input_element_index:
                self.input_element_index.append(element)
        else:
            self.input_element_index.append(input_element_index)
        if self.tol is None: #old order
            self.tol = 0.001

        ###Q direction at element
        n = len(self.input_element_index)
        if input_inverted is None or (isinstance(input_inverted, Sequence) and len(input_inverted) == 0):
            # empty, then set all entries to 1
            self.input_sign = [1] * n
        elif isinstance(input_inverted, bool):
            # single bool, then set all entries to desired value +/-1
            self.input_sign = ([-1] if input_inverted else [1]) * n
        else:
            inv_list = list(input_inverted)[:n]
            if len(inv_list) < n:
                inv_list += [False] * (n - len(inv_list))
            self.input_sign = [-1 if inv else 1 for inv in inv_list]
        self.output_element = output_element
        self.output_element_index = output_element_index
        self.output_element_in_service = output_element_in_service

        # normalize the values distribution:
        self._normalize_distribution_in_service(initial_pf_distribution=output_values_distribution)

        self._update_min_max_q_mvar(net)

        self.output_adjustable = np.array([False if not distribution else service
                                            for distribution, service in zip(np.atleast_1d(self.output_values_distribution),
                                                np.atleast_1d(self.output_element_in_service))], dtype=np.bool)

        n = len(np.atleast_1d(self.output_element_index))
        if gen_q_response is None or (isinstance(gen_q_response, Sequence) and len(gen_q_response) == 0):
            # empty, then set all entries to 1
            self.gen_q_response = [1] * n
        else:
            if len(gen_q_response) < n:
                gen_q_response += [1] * (n - len(gen_q_response))  # missing entries with +1
            self.gen_q_response = gen_q_response
        ###finding correct control_modus, catching deprecated voltage_ctrl argument###todo unambiguous control_modus also with droop
        if control_modus is None: #catching old attribute voltage_ctrl
            if hasattr(self, 'voltage_ctrl'):
                control_modus = self.voltage_ctrl
                if not hasattr(self, '_deprecation_warned'):#only one message that voltage ctrl is deprecated
                    logger.warning(
                        f"'voltage_ctrl' in Controller {self.index} is deprecated. "
                        "Use 'control_modus' ('Q_ctrl', 'V_ctrl', etc.) instead.")
                    self._deprecation_warned = True
        if type(control_modus) == bool and control_modus == True: #Only functions written out!?!
            self.control_modus = "V_ctrl"
            logger.warning(f"Deprecated Controller control_modus for Controller {self.index}, using 'V_ctrl' from available"
                         f" types 'Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan(phi)_ctrl'\n")
        elif type(control_modus) == bool and control_modus == False: #Only functions written out!?!
            self.control_modus = "Q_ctrl"
            logger.warning(f"Deprecated Controller control_modus for Controller {self.index}, using Q_ctrl from available"
                         f" types 'Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan(phi)_ctrl'\n")
        elif control_modus == "PF_ctrl_cap": # -1 for capacitive, 1 for inductive systems
            self.control_modus = "PF_ctrl"
            self.reactance = -1
        elif control_modus == "PF_ctrl_ind":
            self.control_modus = "PF_ctrl"
            self.reactance = 1
        elif control_modus == "PF_ctrl":
            logger.warning(f"Ambivalent reactive power flow direction for Controller {self.index}, using capacitive direction.\n")
            self.control_modus = control_modus
            self.reactance = -1
        else:
            if control_modus == "tan(phi)_ctrl" or control_modus == "V_ctrl":
                self.control_modus = control_modus
            else:
                if control_modus != 'Q_ctrl':
                    logger.warning(f"Control_modus {control_modus} not recognized, using 'Q_ctrl' from available"
                             f" types 'Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan(phi)_ctrl'\n")
                self.control_modus = 'Q_ctrl'

        if self.control_modus == 'PF_ctrl': #checking cos(phi) limits
            if abs(self.set_point) > 1:
                raise UserWarning(f'Power Factor Controller {self.index}: Set point out of range ([-1,1]')
        ###adding input elements###
        counter = 0
        for input_index in np.atleast_1d(self.input_element_index):
            if self.input_element == "res_line":
                self.input_element_in_service.append(net.line.in_service[input_index])
            elif self.input_element == "res_trafo":
                self.input_element_in_service.append(net.trafo.in_service[input_index])
            elif self.input_element == "res_switch":
                self.input_element_in_service.append(
                    net[self.input_element].pf_in_service[input_index])
            elif self.input_element == "res_impedance":
                self.input_element_in_service.append(net.impedance.in_service[input_index])
            elif self.input_element == "res_bus":
                self.input_element_in_service.append(net.bus.in_service[input_index])
            elif self.input_element == "res_gen":
                self.input_element_in_service.append(ctrl_in_service)

            if isinstance(input_variable, list): #get Q variable and read flags for input elements
                read_flag_temp, input_variable_temp = _detect_read_write_flag(net, self.input_element, input_index,
                                                                              input_variable[counter])
            else:
                read_flag_temp, input_variable_temp = _detect_read_write_flag(net, self.input_element,
                                                                              input_index,
                                                                              input_variable)
            ###get p variables for input elements for Phi controller
            if self.control_modus == "PF_ctrl" or self.control_modus== 'tan(phi)_ctrl':
                if isinstance(input_variable, list):
                    input_variable_p = input_variable[counter].replace('q', 'p').replace('var','w')
                    read_flag_temp_p, input_variable_temp_p = _detect_read_write_flag(net, self.input_element,input_index,
                                                                                  input_variable_p)
                else:
                    input_variable_p = input_variable.replace('q', 'p').replace('var', 'w')
                    read_flag_temp_p, input_variable_temp_p = _detect_read_write_flag(net, self.input_element,
                                                                                      input_index,
                                                                                      input_variable_p)
                self.input_variable_p.append(input_variable_temp_p) #read flag p not necessary, flag same as Q variables
            self.read_flag.append(read_flag_temp)
            self.input_variable.append(input_variable_temp)
            counter += 1

    def __str__(self):
        return super().__str__() + " [%s.%s.%s.%s]" % (
            self.input_element, self.input_variable, self.output_element, self.output_variable)

    def __getattr__(self, name):
        if name == "control_modus":
            if not hasattr(self, '_deprecation_warned'):
                logger.warning(
                    f"'voltage_ctrl' in Controller {self.index} is deprecated. "
                    "Use 'control_modus' ('Q_ctrl', 'V_ctrl', etc.) instead."
                )
                self._deprecation_warned = True#only one message that voltage ctrl is deprecated
            return self.voltage_ctrl
        if name == 'bus_idx':
            if not hasattr(self, '_deprecation_warned_bus_idx'):
                logger.warning(
                    f"Variable 'bus_idx' in Binary Search Control {self.index} for control_modus V_ctrl is deprecated. "
                    f"Give index of controlled bus to input_element_index. Input_variable must be 'vm_pu' and"
                    f" input_element 'res_bus'"
                )
                self._deprecation_warned_bus_idx = True#only one warning about bus_idx deprecation
            return self.input_element_index
        raise AttributeError(f"{self.__class__.__name__!r} has no attribute {name!r}")


    def initialize_control(self, net):
        output_element_index = np.atleast_1d(self.output_element_index)[0] if self.write_flag == 'single_index' else \
                self.output_element_index #ruggedize for single index
        self.output_values = read_from_net(net, self.output_element, output_element_index, self.output_variable,
                                           self.write_flag)
        self.output_values_old = None
        self.output_adjustable = np.array([False if not distribution else service
                                            for distribution, service in zip(np.atleast_1d(self.output_values_distribution),
                                                                            np.atleast_1d(self.output_element_in_service))],
                                            dtype=np.bool)

    def is_converged(self, net):
        """
        Actual implementation of the convergence criteria: If controller is applied, it can stop
        """
        # if controller not in_service, return True
        self.in_service = net.controller.in_service[self.index]
        if not self.in_service:
            self.converged = True
            return self.converged
        ###updating input & output elements in service lists
        self.input_element_in_service = list(np.atleast_1d(self.input_element_in_service)).clear()
        self.output_element_in_service = list(np.atleast_1d(self.output_element_in_service)).clear()
        self.input_element_in_service = []
        self.output_element_in_service = []
        for input_index in np.atleast_1d(self.input_element_index):
            if self.input_element == "res_line":
                self.input_element_in_service.append(net.line.in_service[input_index])
            elif self.input_element == "res_trafo":
                self.input_element_in_service.append(net.trafo.in_service[input_index])
            elif self.input_element == "res_trafo3w":
                self.input_element_in_service.append(net.trafo3w.in_service[input_index])
            elif self.input_element == "res_switch":
                self.input_element_in_service.append(net.switch.closed[input_index])
            elif self.input_element == "res_impedance":
                self.input_element_in_service.append(net.impedance.in_service[input_index])
            elif self.input_element == "res_bus":
                self.input_element_in_service.append(net.bus.in_service[input_index])
            elif self.input_element == "res_gen":
                self.input_element_in_service.append(net.gen.in_service[input_index])
        for output_index in np.atleast_1d(self.output_element_index):
            if self.output_element == "gen":
                self.output_element_in_service.append(net.gen.in_service[output_index])
            elif self.output_element == "sgen":
                self.output_element_in_service.append(net.sgen.in_service[output_index])
            elif self.output_element == "shunt":
                self.output_element_in_service.append(net.shunt.in_service[output_index])

        # check if at least one input and one output element is in_service
        if not (any(self.input_element_in_service) and any(self.output_element_in_service)):
            logger.warning("Input and/or output elements for controller %i out of service, putting controller "
                           "out of service" % self.index)
            self.converged = True
            net.controller.loc[self.index, "in_service"] = False
            self.in_service = False
            return self.converged
        # if only one output element is in service
        if sum(self.output_element_in_service) <= 1 and not getattr(self, 'counter_warning', False):
            self.counter_warning = True
            if len(self.output_element_in_service) <= 1:
                logger.warning(
                    f'Reactive Power Distribution for one output element cannot be modified. The active {self.output_element}'
                    f' at index {str(np.array(self.output_element_index))}'
                    f' will provide 100% of the reactive power in Controller {self.index}.\n')
            else:
                logger.warning(
                    f'Reactive Power Distribution for one output element cannot be modified. The active '
                    f'{self.output_element[np.array(self.output_element_in_service)]} at index '
                    f'{self.output_element_index[np.array(self.output_element_in_service)]} will provide 100% of the'
                    f' reactive power in Controller {self.index}.\n')

        # read input values
        input_values = [] #reactive power q
        p_input_values = [] #active power p for power factor controllers
        counter = 0
        if self.input_element != 'res_bus':
            for input_index in self.input_element_index:
                if self.input_element_in_service[counter]: # input element not in service
                    input_values.append(read_from_net(net, self.input_element, input_index,
                                                      self.input_variable[counter], self.read_flag[counter]))
                    if self.control_modus == "PF_ctrl" or self.control_modus == 'tan(phi)_ctrl':
                        p_input_values.append(read_from_net(net,self.input_element, input_index,
                                                        self.input_variable_p[counter], self.read_flag[counter]))
                counter += 1
        # compare old and new set values
        if self.control_modus == "Q_ctrl" or (self.control_modus == 'V_ctrl' and self.input_element_index is None):
            if self.control_modus == 'V_ctrl':
                logger.warning('Missing attribute self.input_element_index, defaulting to Q_ctrl\n')
                self.control_modus = 'Q_ctrl'
            self.diff_old = self.diff
            if not any(self.output_adjustable):
                logging.info('All stations controlled by %s reached reactive power limits.' %self.name)
                self.converged = True
                return self.converged
            else:
                # adapt output adjustable depending on in_service
                self.output_adjustable = np.array([in_service and adjustable for in_service, adjustable
                                                   in zip(self.output_element_in_service, self.output_adjustable)], dtype=np.bool)

                # normalize the values distribution
                self._normalize_distribution_in_service()

            self.diff = self.set_point - sum(input_values)
            self.converged = np.all(np.abs(self.diff) < self.tol)

        elif str(self.control_modus).startswith("PF_ctrl"):#capacitive => reactance = -1, inductive => reactance = 1
            if self.control_modus == 'PF_ctrl_ind':
                self.control_modus = 'PF_ctrl'
                self.reactance = 1
            elif self.control_modus == 'PF_ctrl_cap':
                self.control_modus = 'PF_ctrl'
                self.reactance = -1

            self.diff_old = self.diff
            if not any(self.output_adjustable):
                logging.info('PF_ctrl: All stations controlled by %s reached reactive power limits.' %self.name)
                self.converged = True
                return self.converged
            else:
                # adapt output adjustable depending on in_service
                self.output_adjustable = np.array([in_service and adjustable for in_service, adjustable
                                                   in zip(self.output_element_in_service, self.output_adjustable)], dtype=np.bool)

                # normalize the values distribution
                self._normalize_distribution_in_service()

            q_set = self.reactance * sum(p_input_values)/len(p_input_values) * (np.tan(np.arccos(self.set_point)))
            self.diff = q_set - sum(input_values)/len(input_values)
            self.converged = np.all(np.abs(self.diff)<self.tol)

        elif self.control_modus == "tan(phi)_ctrl":
            self.diff_old = self.diff
            if not any(self.output_adjustable):
                logging.info('tan(phi)_ctrl: All stations controlled by %s reached reactive power limits.' %self.name)
                self.converged = True
                return self.converged
            else:
                # adapt output adjustable depending on in_service
                self.output_adjustable = np.array([in_service and adjustable for in_service, adjustable
                                                   in zip(self.output_element_in_service, self.output_adjustable)], dtype=np.bool)

                # normalize the values distribution
                self._normalize_distribution_in_service()

            q_set = sum(p_input_values)/len(p_input_values) * self.set_point
            self.diff = q_set - sum(input_values)/len(input_values)
            self.converged = np.all(np.abs(self.diff) < self.tol)
        else:
            ###catching deprecated modi from old imports
            if type(self.control_modus) == bool and self.control_modus == True and self.input_element_index is not None:
                self.control_modus = "V_ctrl"  # catching old implementation
                logger.warning(
                    f"Deprecated Control Modus in Controller {self.index}, using V_ctrl from available types\n")
            elif (type(self.control_modus) == bool and self.control_modus == False) or (type(self.control_modus) == bool and self.control_modus == True
                                                                                        and self.input_element_index is None):
                if self.control_modus is True:
                    logger.warning(f'Deprecated Control Modus in Controller {self.index}, attempted to use "V_ctrl" but '
                                   f'missing attribute input_element_index, defaulting to Q_ctrl\n')
                else:
                    logger.warning(
                        f"Deprecated Control Modus in Controller {self.index}, using Q_ctrl from available types\n")
                self.control_modus = "Q_ctrl"

            if self.control_modus == "V_ctrl":
                if self.input_element != 'res_bus':# and not any(getattr(net.controller.at[x, 'object'], 'controller_idx', False) ==
                            #self.index for x in net.controller.index):#no droop, disable for legacy, see below
                    if hasattr(self, 'bus_idx') and getattr(self, 'bus_idx') is not None:  # legacy
                        self.diff_old = self.diff
                        if not any(self.output_adjustable):
                            logging.info(
                                'Q_ctrl: All stations controlled by %s reached reactive power limits.' % self.name)
                            self.converged = True
                            return self.converged
                        else:
                            # adapt output adjustable depending on in_service
                            self.output_adjustable = np.array([in_service and adjustable for in_service, adjustable
                                                               in zip(self.output_element_in_service,
                                                                      self.output_adjustable)], dtype=np.bool)

                            # normalize the values distribution
                            self._normalize_distribution_in_service()

                        self.diff = self.set_point - net.res_bus.vm_pu.at[self.bus_idx]
                        self.converged = np.all(np.abs(self.diff) < self.tol)
                    else:
                        logger.warning(f"'input_element' must be 'res_bus' for V_ctrl not {self.input_element}, correcting.")
                        self.input_element = 'res_bus'
                        if np.atleast_1d(self.input_variable)[0] != 'vm_pu':
                            logger.warning(f"'input_variable' must be 'vm_pu' for V_ctrl not {self.input_variable}, correcting ")
                            self.input_variable = 'vm_pu'
                        self.diff_old = self.diff  # V_ctrl
                        if not any(self.output_adjustable):
                            logging.info(
                                'V_ctrl: All stations controlled by %s reached reactive power limits.' % self.name)
                            self.converged = True
                            return self.converged
                        else:
                            # adapt output adjustable depending on in_service
                            self.output_adjustable = np.array([in_service and adjustable for in_service, adjustable
                                                               in zip(self.output_element_in_service, self.output_adjustable)], dtype=np.bool)

                            # normalize the values distribution
                            self._normalize_distribution_in_service()

                        self.diff = self.set_point - net.res_bus.vm_pu.at[np.atleast_1d(self.input_element_index)[0]]
                        self.converged = np.all(np.abs(self.diff) < self.tol)
                else:
                    self.diff_old = self.diff  # V_ctrl
                    if not any(self.output_adjustable):
                        logging.info('V_ctrl: All stations controlled by %s reached reactive power limits.' % self.name)
                        self.converged = True
                        return self.converged
                    else:
                        # adapt output adjustable depending on in_service
                        self.output_adjustable = np.array([in_service and adjustable for in_service, adjustable
                                                           in zip(self.output_element_in_service, self.output_adjustable)], dtype=np.bool)

                        # normalize the values distribution
                        self._normalize_distribution_in_service()

                    self.diff = self.set_point - net.res_bus.vm_pu.at[np.atleast_1d(self.input_element_index)[0]]
                    self.converged = np.all(np.abs(self.diff) < self.tol)
            else:
                if self.control_modus != 'Q_ctrl':
                    logger.warning(f"No Controller Modus specified for Controller {self.index}, using Q_ctrl.\n"
                                   "Please specify 'control_modus' ('Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan(phi)_ctrl')\n")
                    self.control_modus = 'Q_ctrl'
                self.diff_old = self.diff  # Q_ctrl
                if not any(self.output_adjustable):
                    logging.info('Q_ctrl: All stations controlled by %s reached reactive power limits.' % self.name)
                    self.converged = True
                    return self.converged
                else:
                    # adapt output adjustable depending on in_service
                    self.output_adjustable = np.array([in_service and adjustable for in_service, adjustable
                                                       in zip(self.output_element_in_service, self.output_adjustable)], dtype=np.bool)

                    # normalize the values distribution
                    self._normalize_distribution_in_service()

                self.diff = self.set_point - sum(input_values)
                self.converged = np.all(np.abs(self.diff) < self.tol)
        if self.converged and (
                    any(np.atleast_1d(getattr(net.controller.at[x, 'object'], 'controller_idx', None)) == self.index
                    and not net.controller.object[x].converged for x in net.controller.index)):
            self.converged = False
        return self.converged

    def control_step(self, net):
        self._binary_search_control_step(net)

    def _binary_search_control_step(self, net):
        if not self.in_service:
            return
        if self.output_values_old is None:  # first step
            # is ok that values are set for all stations even though they are out of service or not adjustable --> following step will correct this
            self.output_values_old, self.output_values = (
                np.atleast_1d(self.output_values)[self.output_element_in_service],
                np.atleast_1d(self.output_values)[self.output_element_in_service] + 1e-3)
            positions_not_adjustable = [i for i, val in enumerate(self.output_adjustable) if not val]
            for i in positions_not_adjustable:
                if self.output_values_distribution[i]==0 or not self.output_element_in_service[i] :
                    self.output_values[i] = 0
                else:
                    continue
        else:#second step
            step_diff = self.diff - self.diff_old
            x = self.output_values - self.diff * (self.output_values - self.output_values_old) / np.where(
                step_diff == 0, 1e-6, step_diff)  #converging

            rel_cap = 2
            cap = rel_cap * (np.abs(self.output_values) + 1e-6) + 50  # add epsilon to avoid zero; absolute cap +50 MVAr

            delta = x - self.output_values
            delta = np.clip(delta, -cap, +cap)

            x = self.output_values + delta

            if not all(self.output_adjustable) and net._options['enforce_q_lims']:
                positions_adjustable = [i for i, val in enumerate(self.output_adjustable) if val]  # gives which is/are adjustable
                positions_not_adjustable = [i for i, val in enumerate(self.output_adjustable) if not val]  # can be one or multiple ## gives which is/are not adjustable anymore

                sum_adjustable = sum(x) - sum(self.output_values[positions_not_adjustable])  # stations that are still adjustablke, rest of the power must be achieved
                x[positions_adjustable] = sum_adjustable * self.output_values_distribution[positions_adjustable]

                for i in positions_not_adjustable:
                    if self.output_element_in_service[i]:
                        x[i] = self.output_values[i]  # reset value to q_limit
                    else:
                        x[i] = 0  # reset value to 0 because station is out of service

            else:
                x = sum(x) * self.output_values_distribution

            if self.output_adjustable is not None and net._options['enforce_q_lims']:  # none if output element is a shunt
                if isinstance(x, np.ndarray) and len(x)>1:
                    self._update_min_max_q_mvar(net)

                    # check if x is a list, multiple assets in station controller

                    # check if a limit is reached, consider element in service
                    reached_min_qmvar = [val <= min_val and in_service
                                         for val, min_val, in_service
                                         in zip(x, self.output_min_q_mvar, self.output_element_in_service)]
                    reached_max_qmvar = [val >= max_val and in_service
                                         for val, max_val, in_service
                                         in zip(x, self.output_max_q_mvar, self.output_element_in_service)]

                    if any(reached_max_qmvar):
                        positions = [i for i, val in enumerate(reached_max_qmvar) if val is np.True_]  # can be one or multiple
                        reached_index = [self.output_element_index[i] for i in positions]
                        logging.info('Station(s) controlled by %s reached the maximum reactive power limit: %s'
                              % (self.name, ', '.join(net[self.output_element].loc[reached_index].name.tolist())))
                        self.output_adjustable[positions] = False
                        sum_old = sum(x)
                        max_q_mvar_limit = self.output_max_q_mvar[positions]

                        # adapt distribution and x
                        self.output_values_distribution[positions] = 0
                        if np.all(self.output_values_distribution == 0):
                            # all stations reached limit, prevent for division with 0 resulting in nan array
                            pass
                        else:
                            self.output_values_distribution /= sum(self.output_values_distribution)
                        x = (sum_old-sum(max_q_mvar_limit))*self.output_values_distribution
                        x[positions] = max_q_mvar_limit # reset to limit

                    elif any(reached_min_qmvar):
                        positions = [i for i, val in enumerate(reached_min_qmvar) if val is np.True_]
                        reached_index = [self.output_element_index[i] for i in positions]
                        logging.info('Station(s) controlled by %s reached the minimum reactive power limit: %s'
                              % (self.name, ', '.join(net[self.output_element].loc[reached_index].name.tolist())))
                        self.output_adjustable[positions] = False
                        sum_old = sum(x)
                        min_q_mvar_limit = self.output_min_q_mvar[positions]

                        # adapt distribution and x
                        self.output_values_distribution[positions] = 0
                        if np.all(self.output_values_distribution == 0):
                            # all stations reached limit, prevent for division with 0 resulting in nan array
                            pass
                        else:
                            self.output_values_distribution /= sum(self.output_values_distribution)

                        x = (sum_old-sum(min_q_mvar_limit))*self.output_values_distribution
                        x[positions] = min_q_mvar_limit # reset to limit

                    self.output_values_old, self.output_values = self.output_values, x
                else:
                    # check when x is a single value (only one adjustable machine)
                    # check if limit is reached
                    self._update_min_max_q_mvar(net)

                    reached_min_qmvar = x<self.output_min_q_mvar
                    reached_max_qmvar = x>self.output_max_q_mvar

                    if reached_min_qmvar or reached_max_qmvar:
                        logging.info('Station %s controlled by %s reached a reactive power limit.' % (self.output_element_index, self.name))
                        self.output_adjustable = np.array([False], dtype=np.bool)
                        if reached_min_qmvar:
                            self.output_values_old, self.output_values = self.output_values, self.output_min_q_mvar
                        elif reached_max_qmvar:
                            self.output_values_old, self.output_values = self.output_values, self.output_max_q_mvar
                    else:
                        self.output_values_old, self.output_values = self.output_values, x
            else:
                self.output_values_old, self.output_values = self.output_values, x

            ### write new set of Q values to output elements###
        output_element_index = (list(np.atleast_1d(self.output_element_index)[self.output_element_in_service])[0] if self.write_flag
            == 'single_index' else list(np.array(self.output_element_index)[self.output_element_in_service])) #ruggedizing code
        output_values = (list(self.output_values)[0] if self.write_flag
            == 'single_index' else list(self.output_values))  # ruggedizing code
        write_to_net(net, self.output_element, output_element_index, self.output_variable, output_values, self.write_flag)

    def _normalize_distribution_in_service(self, initial_pf_distribution=None):
        # normalize distribution depending on in service of stations
        if initial_pf_distribution is None:
            distribution = self.output_values_distribution
        else:
            distribution = initial_pf_distribution

        # normalize the values distribution
        # set output_values_distribution to 0, if station is not in service
        self.output_values_distribution = [0 if not in_service else value
               for in_service, value in zip(np.atleast_1d(self.output_element_in_service), np.atleast_1d(distribution))]
        total = np.sum(self.output_values_distribution)
        if total > 0:  # To avoid division by zero
            self.output_values_distribution = np.array(self.output_values_distribution, dtype=np.float64) / total
        else:
            self.output_values_distribution = np.zeros_like(self.output_values_distribution, dtype=np.float64)

    def _update_min_max_q_mvar(self, net):
        if 'min_q_mvar' in net[self.output_element].columns:
            self.output_min_q_mvar = np.nan_to_num(np.atleast_1d(net[self.output_element].loc[self.output_element_index, 'min_q_mvar']), nan=-np.inf)
        else:
            self.output_min_q_mvar = np.array([-np.inf]*len(self.output_element_index), dtype=np.float64)
        if 'max_q_mvar' in net[self.output_element].columns:
            self.output_max_q_mvar = np.nan_to_num(np.atleast_1d(net[self.output_element].loc[self.output_element_index, 'max_q_mvar']), nan=np.inf)
        else:
            self.output_max_q_mvar = np.array([np.inf]*len(self.output_element_index), dtype=np.float64)

    def finalize_control(self, net):
        pass
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

        **vm_set_pu_bsc** - Initial voltage set point in case of voltage control.

        **controller_idx** - Index of linked Binary< search control (if present).

        **voltage_ctrl** - Whether the controller is used for voltage control or not.

        **bus_idx=None** - Bus index which is used for voltage control.

        **q_set_mvar_bsc** - Initial voltage set point in case of no voltage control.

        **tol=1e-6** - Tolerance criteria of controller convergence.

        **vm_set_lb=None** - Lower band border of dead band

        **vm_set_ub=None** - Upper band border of dead band
       """

    def __init__(self, net, q_droop_mvar, controller_idx, voltage_ctrl, bus_idx=None, tol=1e-6,
                 q_set_mvar_bsc=None, in_service=True, order=-1, level=0, name="", drop_same_existing_ctrl=False,
                 matching_params=None, vm_set_pu_bsc=None, vm_set_lb=None, vm_set_ub=None, **kwargs):
        super().__init__(net, in_service=in_service, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params)
        # TODO: implement maximum and minimum of droop control
        self.name = name
        # write kwargs in self
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.name = name
        self.q_droop_mvar = q_droop_mvar
        self.bus_idx = bus_idx
        self.vm_pu = None
        self.vm_pu_old = self.vm_pu
        value = vm_set_pu_bsc if vm_set_pu_bsc is not None else kwargs.get('vm_set_pu')
        if voltage_ctrl and value is None:
            raise ValueError("vm_set_pu_bsc missing in input variables of droop controller!")
        else:
            self.vm_set_pu_bsc = value
        self.vm_set_pu_new = None
        self.lb_voltage = vm_set_lb
        self.ub_voltage = vm_set_ub
        self.controller_idx = controller_idx
        self.voltage_ctrl = voltage_ctrl
        self.tol = tol
        self.applied = False
        self.read_flag, self.input_variable = _detect_read_write_flag(net, "res_bus", bus_idx, "vm_pu")
        self.q_set_mvar_bsc = q_set_mvar_bsc
        self.q_set_mvar = None
        self.q_set_old_mvar = None
        self.diff = None
        self.converged = False


    def is_converged(self, net):
        if (not net.controller.at[self.controller_idx, "object"].in_service or
                net.controller.at[self.controller_idx, "object"].converged):
            self.converged = True
            return self.converged
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
            input_sign = np.asarray(net.controller.at[self.controller_idx, "object"].input_sign)
            input_values = (input_sign * np.asarray(input_values)).tolist()
            self.diff = (net.controller.at[self.controller_idx, "object"].set_point - sum(input_values))
        self.converged = np.all(np.abs(self.diff) < self.tol)
        return self.converged

    def control_step(self, net):
        self._droop_control_step(net)

    def _droop_control_step(self, net):
        self.vm_pu_old = self.vm_pu
        self.vm_pu = read_from_net(net, "res_bus", self.bus_idx, "vm_pu", self.read_flag)
        if not self.voltage_ctrl:
            if self.q_set_mvar_bsc is None:
                self.q_set_mvar_bsc = net.controller.at[self.controller_idx, "object"].set_point
            if hasattr(net.controller.object[self.controller_idx], 'gen_q_response'):
                gen_q_response = net.controller.object[self.controller_idx].gen_q_response[0]
            else: gen_q_response = None #todo gen_q_response fällt weg
            if gen_q_response is None: gen_q_response = 1 #legacy and robustness
            if self.lb_voltage is not None and self.ub_voltage is not None:
                if self.vm_pu > self.ub_voltage:
                    self.q_set_old_mvar, self.q_set_mvar = (self.q_set_mvar, self.q_set_mvar_bsc +
                                                            gen_q_response * (self.ub_voltage - self.vm_pu) * self.q_droop_mvar)
                elif self.vm_pu < self.lb_voltage:
                    self.q_set_old_mvar, self.q_set_mvar = (self.q_set_mvar, self.q_set_mvar_bsc +
                                                            gen_q_response * (self.lb_voltage - self.vm_pu) * self.q_droop_mvar)
                else:
                    self.q_set_old_mvar, self.q_set_mvar = (self.q_set_mvar, self.q_set_mvar_bsc)
            else:
                self.q_set_old_mvar, self.q_set_mvar = (
                    self.q_set_mvar, self.q_set_mvar + net.controller.object[self.controller_idx].gen_q_response[0] * (
                                self.q_set_mvar_bsc - self.vm_pu) * self.q_droop_mvar)

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
            input_values = (
                        net.controller.at[self.controller_idx, "object"].input_sign * np.asarray(input_values)).tolist()
            self.vm_set_pu_new = self.vm_set_pu_bsc + sum(
                input_values) / self.q_droop_mvar  # net.controller.at[self.controller_idx, "object"].gen_Q_response[0] *
            net.controller.at[self.controller_idx, "object"].set_point = self.vm_set_pu_new


class VDroopControl_local(Controller):
    """
    The VDroopControl_local is used in case of a local droop based voltage control. It is used in addition to
    a binary search controller (bsc). The linked binary search controller is specified using the controller index,
    which refers to the linked bsc.

    INPUT:
        **self**

        **net** - A pandapower grid.

        **q_droop_var** - Droop Value in Mvar/p.u.

        **vm_set_pu_bsc** - Inital voltage set point.

        **controller_idx** - Index of linked Binary< search control (if present).

        **tol=1e-6** - Tolerance criteria of controller convergence.

        **vm_set_lb=None** - Lower band border of dead band

        **vm_set_ub=None** - Upper band border of dead band
       """

    def __init__(self, net, q_droop_mvar, controller_idx, bus_idx, tol=1e-6, in_service=True, order=-1, level=0,
                 name="", drop_same_existing_ctrl=False, matching_params=None, q_set_mvar=None, vm_set_pu_bsc=None,
                 vm_set_lb=None, vm_set_ub=None, **kwargs):
        super().__init__(net, in_service=in_service, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params)
        # TODO: implement maximum and minimum of droop control
        # write kwargs in self
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.name = name
        self.q_droop_mvar = q_droop_mvar
        self.vm_pu = None
        self.vm_pu_old = self.vm_pu
        value = vm_set_pu_bsc if vm_set_pu_bsc is not None else kwargs.get('vm_set_pu')
        self.vm_set_pu_bsc = value
        self.vm_set_pu_new = None
        self.q_set_mvar = q_set_mvar
        self.lb_voltage = vm_set_lb
        self.ub_voltage = vm_set_ub
        self.controller_idx = controller_idx
        self.bus_idx = bus_idx
        self.tol = tol
        self.applied = False
        gen_idx = net.controller.at[self.controller_idx, "object"].input_element_index[0]
        self.read_flag, self.input_variable = _detect_read_write_flag(net, "res_bus", bus_idx, "vm_pu")
        self.diff = None
        self.converged = False

    def is_converged(self, net):
        if (not net.controller.at[self.controller_idx, "object"].in_service or
                net.controller.at[self.controller_idx, "object"].converged):
            self.converged = True
            return self.converged

        self.diff = (net.controller.at[self.controller_idx, "object"].set_point -
                     read_from_net(net, "res_bus", self.bus_idx, "vm_pu", self.read_flag))
        self.converged = np.all(np.abs(self.diff) < self.tol)
        return self.converged

    def control_step(self, net):
        self._Vdroopcontrol_step(net)

    def _Vdroopcontrol_step(self, net):
        self.vm_pu_old = self.vm_pu
        self.vm_pu = read_from_net(net, "res_bus", self.bus_idx, "vm_pu", self.read_flag)

        input_element = net.controller.at[self.controller_idx, "object"].input_element
        input_element_index = net.controller.at[self.controller_idx, "object"].input_element_index
        input_variable = net.controller.at[self.controller_idx, "object"].input_variable
        read_flag = net.controller.at[self.controller_idx, "object"].read_flag
        input_values = []
        counter = 0
        for input_index in input_element_index:
            input_values.append(read_from_net(net, input_element, input_index,
                                              input_variable[counter], read_flag[counter]))
        input_values = (net.controller.at[self.controller_idx, "object"].input_sign * np.asarray(input_values)).tolist()
        self.vm_set_pu_new = self.vm_set_pu_bsc - (sum(
            input_values) - self.q_set_mvar) / self.q_droop_mvar
        net.controller.at[self.controller_idx, "object"].set_point = self.vm_set_pu_new
