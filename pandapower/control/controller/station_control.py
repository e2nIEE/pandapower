import numpy as np
import numbers
from cmath import isnan
import numpy as np
from enum import Enum
from collections.abc import Sequence
from scipy.optimize import minimize
from pandapower import create_gen, create_sgen
import pandas as pd
from pandapower.control.basic_controller import Controller
from pandapower.auxiliary import _detect_read_write_flag, read_from_net, write_to_net
from pandapower.control.util.auxiliary import get_min_max_q_mvar_from_characteristics_object
import logging
import pandapower.topology as top
import networkx as nx

logger = logging.getLogger(__name__)


class BinarySearchControl(Controller):
    """
    The Binary search control is a controller that adjusts output values in order to reach a given set point.
    It can be used for reactive power control, voltage control, cosines(phi) or tangens(phi) control. The control modus
    can be set via the control_modus parameter. Input and output elements and indexes can be lists. Input elements can
    be transformers, switches, lines or buses (only in voltage control). the controlled bus must be
    given to input_element_index. Output elements are sgens, where active and reactive power can be set.
    Distribution_method takes a string and selects the type of reactive power distribution.
    Output_distribution_value describes the distribution of reactive power provision between multiple
    output_elements and will be normalized to 100 % (1).

    Parameters
    ----------
    self : BinarySearchControl
    net : pandapowerNet
        A pandapower grid.
    ctrl_in_service : bool
        Whether the controller is in service.
    output_element : str
        Output element type: ``"gen"`` or ``"sgen"``.
        For reactive power control, currently only ``"sgen"`` is supported.
    output_variable : str or list of str
        Output variable of the element (e.g., ``"q_mvar"``).
    output_element_index : int or list of int
        Index or list of indices of the output element(s) in net (e.g. ``"net.sgen"``).
    output_element_in_service : bool or list of bool
        Indicates whether each output element is in service.
    distribution_method : str -> ControlModusEnum
        Takes string to select one of the different available reactive power distribution
        methods: 'rel_P' -Q is relative to dispatched Power, 'rel_rated_S' -Q is relative to the rated apparent power S, currently
        using the sgen attribute 'sn_mva', 'set_Q' -set individual reactive power for each output element,
        'max_Q' -maximized reactive power reserve for the output elements, 'rel_V_pu' -Q is relative to the voltage
        limits of the output element.
    output_values_distribution : int, float or list of float
        The values of the Q distribution, only applicable if distribution_method = 'set_Q' or rel_V_pu.
        For 'set_Q': list of floats - Distribution of reactive power provision among output elements (must sum to 1).
        For 'rel_V_pu': list of lists - Must be a list containing lists
        [Target Voltage, minimal allowed Voltage, maximal allowed Voltage] for each output element.
    input_element : str
        Measurement location, can be a transformer, switches or lines. Must be a bus for
        ``"V_ctrl"``. Indicated by string value ``"res_trafo"``, ``"res_switch"``, ``"res_line"`` or ``"res_bus"``.
        In case of ``"res_switch"``, an additional small impedance is introduced in the switch.
    input_variable : str or list of string
        Variable which is used to take the measurement from. Indicated by string value. Must
        be ``"vm_pu"`` for ``"V_ctrl"``.
    input_inverted : bool or list of bool
        Indicates whether the measurement of each input element must be inverted.
        Required when importing from PowerFactory.
    input_element_index : int or list of int
        Element of input element in net.
    control_modus : str -> ControlModusEnum:
        Enables the selection of the available control modi by taking one of the strings: ``"Q_ctrl"``, ``"V_ctrl"``,
        ``"PF_ctrl_ind"`` or ``"PF_ctrl_cap"`` for power factor control with reactance or ``"tan_phi_ctrl"``.
        Formerly called Voltage_ctrl.
    set_point : float
        Set point of the controller, can be a reactive power provision, a power factor or a voltage set point. In
        case of voltage set point, control_modus must be "V_ctrl", input_element_index must be a bus (input_variable must
        be "vm_pu" input_element must be "res_bus"). Can be overwritten by a droop controller chained with the binary
        search control. If "V_ctrl" and automated bus selection (input_element_index = "auto"), set_point will be
        the search criteria in kV for the controlled bus (V_bus >= V_set_point).
    output_min_q_mvar : float or list of float
        Minimum Q limits for each output element. Considered when runpp is
        executed with enforce_q_lims=True.
    output_max_q_mvar : float or list of float
        Maximum Q limits for each output element. Considered when runpp is
        executed with enforce_q_lims=True.
    tol : float, optional
        Tolerance for controller convergence. Default is 0.001.
    in_service : bool, optional
        Whether the controller is in service. Default is True.
    order : int, optional
        Execution order of the controller.
    level : int, optional
        Execution level of the controller.
    drop_same_existing_ctrl : bool, optional
        Whether to drop existing controllers with the same parameters.
    matching_params : dict, optional
        Parameters used to match controllers.
    name : str, optional
        Name of the controller.
    kwargs : dict, optional
        Additional keyword arguments.

    """
    def __init__(self, net, ctrl_in_service:bool, output_element, output_variable, output_element_index,
                 output_element_in_service, input_element, input_variable, input_element_index, set_point:float,
                 distribution_method:str = None, output_values_distribution = None, control_modus:str=None, name="",
                 input_inverted=None, tol=0.001, in_service=True, order=0, level=0, drop_same_existing_ctrl=False,
                 matching_params=None, **kwargs):
        super().__init__(net, in_service=in_service, order=order, level=level,
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
        self.output_values_distribution = output_values_distribution
        self.max_q_mvar = [] #limits of output element Q
        self.min_q_mvar = []
        self.diff = None
        self.diff_old = None
        self.converged = False  # criteria for success of controller
        self.redistribute_values = None  # Values to save for redistributed gens
        self.applied_distribution = False #true if distribution is applied at least once, no convergence if False
        self.counter_warning = False  # only one message that only one active output element
        self.read_flag = []  # type of read value
        self.write_flag, self.output_variable = _detect_read_write_flag(net, output_element, output_element_index,
                                                                        output_variable)
        ###catching errors in variables, allocating
        if input_inverted is None: input_inverted = []#for robustness
        if isinstance(output_element_index, list) or isinstance(output_element_index, np.ndarray):
            self.output_element_index = [int(item) for item in output_element_index]
        else:
            self.output_element_index = []
            self.output_element_index.append(output_element_index)
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
        try:
            self.distribution_method = ControlModusEnum(distribution_method)
        except ValueError:
            logger.warning(f"Control_modus {getattr(self, 'distribution_method', None)} not recognized,"
                       f" using 'rel_P' from available types 'rel_P', 'max_Q', 'set_Q', 'rel_V_pu' or 'rel_rated_S'\n")
            if self.output_values_distribution is not None:
                self.distribution_method = ControlModusEnum.set_Q
            else:
                self.distribution_method = ControlModusEnum.rel_P

        ###Q direction at element
        n = len(self.input_element_index)
        if input_inverted is None or (isinstance(input_inverted, Sequence) and len(input_inverted) == 0):
            # empty, then set all entries to 1
            self.input_sign = [1] * n
        elif isinstance(input_inverted, bool):
            # single bool, then set all entries to desired value +/-1
            self.input_sign = ([-1] if input_inverted else [1]) * n
        else:
            inv_list = list(np.atleast_1d(input_inverted))[:n]
            if len(inv_list) < n:
                inv_list += [False] * (n - len(inv_list))
            self.input_sign = [-1 if inv else 1 for inv in inv_list]
        self.output_element = output_element
        self.output_element_index = output_element_index
        self.output_element_in_service = output_element_in_service
        if self.tol is None: #old order
            self.tol = 0.001
        ###allocating distribution method and distribution values
        if self.distribution_method == ControlModusEnum.rel_V_pu:
            self.bus_idx_dist = []  # initializing bus idx
            output_values_distribution = np.array(self.output_values_distribution)  # forming limit arrays
            if output_values_distribution.ndim == 1:  # one controlled sgen
                try:
                    self.v_set_point_pu = np.array(output_values_distribution)[0]
                    self.v_min_pu = np.minimum(np.array(output_values_distribution)[1],
                                               np.array(output_values_distribution)[2])
                    self.v_max_pu = np.maximum(np.array(output_values_distribution)[1], np.array(output_values_distribution)[2])
                except IndexError: #insufficient values in array
                    logger.warning(f"Insufficient values in distribution rel_V_pu {self.output_values_distribution} In "
                                   f"Controller {self.index}. Using set point 1 pu and min/max 0.9/1.1 pu\n")
                    equal_array = [1, 0.9, 1.1]
                    self.output_values_distribution = np.tile(equal_array, (len(np.array(self.output_element_in_service)), 1))[0]
                    output_values_distribution = np.array(self.output_values_distribution)  # forming limit arrays
                    self.v_set_point_pu = output_values_distribution[0]
                    self.v_min_pu = output_values_distribution[1]
                    self.v_max_pu = output_values_distribution[2]

            elif output_values_distribution.ndim >= 2: #more than one controlled sgen
                try:#insufficient values in arrays
                    self.v_set_point_pu = np.array(output_values_distribution)[:, 0]
                    self.v_min_pu = np.minimum(np.array(output_values_distribution)[:, 1], np.array(output_values_distribution)[:, 2])
                    self.v_max_pu = np.maximum(np.array(output_values_distribution)[:, 1], np.array(output_values_distribution)[:, 2])
                except IndexError:
                    logger.warning(f"Insufficient values in distribution rel_V_pu {self.output_values_distribution} In "
                                   f"Controller {self.index}. Using set point 1 pu and min/max 0.9/1.1 pu\n")
                    equal_array = [1, 0.9, 1.1]
                    self.output_values_distribution = np.full(len(np.array(self.output_element_in_service)), equal_array)
                    output_values_distribution = np.array(self.output_values_distribution)  # forming limit arrays
                    self.v_set_point_pu = output_values_distribution[:, 0]
                    self.v_min_pu = output_values_distribution[:, 1]
                    self.v_max_pu = output_values_distribution[:, 2]
            else:
                self.output_values_distribution = None
        # normalize the values distribution:
        self._normalize_distribution_in_service(initial_pf_distribution=output_values_distribution)
        if self.distribution_method == ControlModusEnum.rel_V_pu:
            self.output_adjustable = np.array([
                                service if distribution is None else (False if not distribution else service)
                                for distribution, service in zip(np.atleast_1d(np.atleast_2d(
                                    self.output_values_distribution)[0][0]), np.atleast_1d(self.output_element_in_service))],
                                dtype=np.bool)
        else: #rel_V_pu has arrays as output_values_distribution
            self.output_adjustable = np.array([False if not distribution else service
                                            for distribution, service in zip(list(np.atleast_1d(self.output_values_distribution)),
                                                list(np.atleast_1d(self.output_element_in_service)))], dtype=bool)
        ###finding correct control_modus, catching deprecated voltage_ctrl argument###
        if control_modus is None: #catching old attribute voltage_ctrl
            if hasattr(self, 'voltage_ctrl'):
                control_modus = self.voltage_ctrl
                if not hasattr(self, '_deprecation_warned'):#only one message that voltage ctrl is deprecated
                    logger.warning(
                        f"'voltage_ctrl' in Controller {self.index} is deprecated. "
                        "Use 'control_modus' ('Q_ctrl', 'V_ctrl', etc.) instead.")
                    self._deprecation_warned = True
        if isinstance(control_modus,bool) and control_modus == True: #Only functions written out!?!
            self.control_modus = ControlModusEnum.v_ctrl
            logger.warning(f"Deprecated Controller control_modus for Controller {self.index}, using 'V_ctrl' from available"
                         f" types 'Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan_phi_ctrl'\n")
        elif isinstance(control_modus, bool) and control_modus == False: #Only functions written out!?!
            self.control_modus = ControlModusEnum.q_ctrl
            logger.warning(f"Deprecated Controller control_modus for Controller {self.index}, using Q_ctrl from available"
                         f" types 'Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan_phi_ctrl'\n")
        else:
            try:
                self.control_modus = ControlModusEnum(control_modus)
            except ValueError:
                logger.warning(f"Control_modus {control_modus} not recognized, using 'Q_ctrl' from available"
                               f" types 'Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan_phi_ctrl'\n")
                self.control_modus = ControlModusEnum.q_ctrl
        if self.control_modus in ControlModusEnum.pf_modes():  # checking cos(phi) limits
            if self.control_modus == ControlModusEnum.PF_ctrl_cap: #-1 for capacitive, 1 for inductive systems
                self.reactance= -1
            else:
                if control_modus == ControlModusEnum.PF_ctrl:
                    logger.warning(
                        f"Ambivalent reactive power flow direction for Controller {self.index}, using inductive direction.\n")
                    self.control_modus = ControlModusEnum.PF_ctrl_ind
                self.reactance = 1

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
            if self.control_modus in ControlModusEnum.pf_modes() or self.control_modus== ControlModusEnum.tan_phi_ctrl:
                if isinstance(input_variable, list):
                    input_variable_p = input_variable[counter].replace('q', 'p').replace('var','w')
                    _, input_variable_temp_p = _detect_read_write_flag(net, self.input_element,input_index,
                                                                                  input_variable_p)
                else:
                    input_variable_p = input_variable.replace('q', 'p').replace('var', 'w')
                    _, input_variable_temp_p = _detect_read_write_flag(net, self.input_element,
                                                                                      input_index,
                                                                                      input_variable_p)
                self.input_variable_p.append(input_variable_temp_p) #read flag p not necessary, flag same as Q variables
            self.read_flag.append(read_flag_temp)
            self.input_variable.append(input_variable_temp)
            counter += 1

        ###reading Q limits###
        for output_index in np.atleast_1d(self.output_element_index):
            try:
                min_q = read_from_net(net, self.output_element, output_index, 'min_q_mvar', 'single_index')
                assert(np.isnan(min_q) == False) # error if nan
            except Exception as e:
                logger.error(e)
                logger.warning(
                    f'Output element {self.output_element} at index {output_index} is missing required attribute min_q_mvar'
                    f' for Controller {self.index}. Using -20 as lower limit\n')
                min_q = -20
            try:
                max_q = read_from_net(net, self.output_element, output_index, 'max_q_mvar', 'single_index')
                assert(np.isnan(max_q) == False)#error if nan
            except Exception as e:
                logger.error(e)
                logger.warning(
                    f'Output element {self.output_element} at index {output_index} is missing required attribute max_q_mvar'
                    f' for Controller {self.index}. Using 20 as upper limit\n')
                max_q = 20
            self.max_q_mvar.append(max(min_q, max_q)) #if min > max, switch
            self.min_q_mvar.append(min(min_q, max_q))

        #normalize the values distribution:
        self._normalize_distribution_in_service(initial_pf_distribution=output_values_distribution)
        self._update_min_max_q_mvar(net)
        ###directions of q and inverted index
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
        try:
            self.distribution_method = ControlModusEnum(getattr(self, 'distribution_method', None))
        except ValueError:
            logger.warning(f"Control_modus {getattr(self, 'distribution_method', None)} not recognized,"
                       f" using 'rel_P' from available types 'rel_P', 'max_Q', 'set_Q', 'rel_V_pu' or 'rel_rated_S'\n")
            if self.output_values_distribution is not None:
                self.distribution_method = ControlModusEnum.set_Q
            else:
                self.distribution_method = ControlModusEnum.rel_P
        #reread output elements
        output_element_index = np.atleast_1d(self.output_element_index)[0] if self.write_flag == 'single_index' else \
                self.output_element_index #ruggedize for single index
        self.output_values = read_from_net(net, self.output_element, output_element_index, self.output_variable,
                                           self.write_flag)
        self.output_values_old = None
        if self.distribution_method == ControlModusEnum.rel_V_pu:
            self.output_adjustable = np.array([
                                service if distribution is None else (False if not distribution else service)
                                for distribution, service in zip(np.atleast_1d(np.atleast_2d(
                                    self.output_values_distribution)[0][0]), np.atleast_1d(self.output_element_in_service))],
                                dtype=np.bool)
        else: #rel_V_pu has arrays as output_values_distribution
            self.output_adjustable = np.array([
                service if distribution is None else (False if not distribution else service)
                for distribution, service in zip(
                    list(np.atleast_1d(self.output_values_distribution)),
                    list(np.atleast_1d(self.output_element_in_service))
                )
            ], dtype=bool)

    def is_converged(self, net):
        """
        Actual implementation of the convergence criteria: If controller is applied, it can stop
        """
        # if controller not in_service, return True
        self.in_service = net.controller.in_service[self.index]
        if not self.in_service:
            self.converged = True
            return self.converged
        ###legacy before ControlModusEnum
        if isinstance(self.control_modus,bool) and self.control_modus == True: #Only functions written out!?!
            self.control_modus = ControlModusEnum.v_ctrl
            logger.warning(f"Deprecated Controller control_modus for Controller {self.index}, using 'V_ctrl' from available"
                         f" types 'Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan_phi_ctrl'\n")
        elif isinstance(self.control_modus, bool) and self.control_modus == False: #Only functions written out!?!
            self.control_modus = ControlModusEnum.q_ctrl
            logger.warning(f"Deprecated Controller control_modus for Controller {self.index}, using Q_ctrl from available"
                         f" types 'Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan_phi_ctrl'\n")
        else:
            try:
                self.control_modus = ControlModusEnum(self.control_modus)
            except ValueError:
                logger.warning(f"Control_modus {self.control_modus} not recognized, using 'Q_ctrl' from available"
                               f" types 'Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan_phi_ctrl'\n")
                self.control_modus = ControlModusEnum.q_ctrl
        if isinstance(self.control_modus, str):
            try:
                self.control_modus = ControlModusEnum(self.control_modus)
            except ValueError:
                logger.warning(f"Control_modus {self.control_modus} not recognized, using 'Q_ctrl' from available"
                               f" types 'Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan_phi_ctrl'\n")
                self.control_modus = ControlModusEnum.q_ctrl
        if isinstance(self.distribution_method, str):
            try:
                self.distribution_method = ControlModusEnum(self.distribution_method)
            except ValueError:
                logger.warning(f"Control_modus {getattr(self, 'distribution_method', None)} not recognized,"
                       f" using 'rel_P' from available types 'rel_P', 'max_Q', 'set_Q', 'rel_V_pu' or 'rel_rated_S'\n")
                if self.output_values_distribution is not None:
                    self.distribution_method = ControlModusEnum.set_Q
                else:
                    self.distribution_method = ControlModusEnum.rel_P
        ###updating input & output elements in service lists
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
        if self.input_element != 'res_bus' and self.input_element != "res_gen":  # and not any(getattr(net.controller.at[x, 'object'], 'controller_idx', False) ==
            for input_index in self.input_element_index:
                if self.input_element_in_service[counter]: # input element not in service
                    input_values.append(read_from_net(net, self.input_element, input_index,
                                                      self.input_variable[counter], self.read_flag[counter]))
                    if self.control_modus in ControlModusEnum.pf_modes() or self.control_modus == ControlModusEnum.tan_phi_ctrl:
                        p_input_values.append(read_from_net(net,self.input_element, input_index,
                                                        self.input_variable_p[counter], self.read_flag[counter]))
                counter += 1
            input_values = (self.input_sign * np.asarray(input_values)).tolist()
        if self.control_modus in  ControlModusEnum.pf_modes() or self.control_modus == ControlModusEnum.tan_phi_ctrl:
            p_input_values = (self.input_sign * np.asarray(p_input_values)).tolist()
        ###reading Q limits in case of skipped initialization###
        if not hasattr(self, 'min_q_mvar') or not hasattr(self, 'max_q_mvar'):
            self.max_q_mvar = []  # limits of output element Q
            self.min_q_mvar = []
            for output_index in self.output_element_index:
                try:
                    min_q = read_from_net(net, self.output_element, output_index, 'min_q_mvar', 'single_index')
                    assert(np.isnan(min_q) == False) # error if nan
                except Exception as e:
                    logger.error(e)
                    logger.warning(
                        f'Output element {self.output_element} at index {output_index} is missing required attribute min_q_mvar'
                        f' for Controller {self.index}. Using -20 as lower limit\n')
                    min_q = -20
                try:
                    max_q = read_from_net(net, self.output_element, output_index, 'max_q_mvar', 'single_index')
                    assert(np.isnan(max_q) == False)#error if nan
                except Exception as e:
                    logger.error(e)
                    logger.warning(
                        f'Output element {self.output_element} at index {output_index} is missing required attribute max_q_mvar'
                        f' for Controller {self.index}. Using 20 as upper limit\n')
                    max_q = 20
                self.max_q_mvar.append(max(min_q, max_q)) #if min > max, switch
                self.min_q_mvar.append(min(min_q, max_q))

        # read previously set values
        # compare old and new set values
        if self.control_modus in ControlModusEnum.q_modes() or (self.control_modus in ControlModusEnum.v_modes()
                        and self.control_modus in ControlModusEnum.droop_modes() and self.bus_idx is None):
            if self.control_modus in ControlModusEnum.v_modes():
                logger.warning('Missing attribute self.input_element_index, defaulting to Q_ctrl\n')
                self.control_modus = ControlModusEnum.q_ctrl
            self.diff_old = self.diff
            if self.diff is None: #first step for assured bsc_ctrl_step
                self.diff = 1
            else:
                # adapt output adjustable depending on in_service
                self.output_adjustable = np.array([in_service and adjustable for in_service, adjustable in zip(
                    self.output_element_in_service, self.output_adjustable)], dtype=bool)
                # normalize the values distribution
                self._normalize_distribution_in_service()
                self.diff = self.set_point - sum(input_values)
            self.converged = np.all(np.abs(self.diff) < self.tol)
        elif self.control_modus in ControlModusEnum.pf_modes():
            if self.control_modus == ControlModusEnum.PF_ctrl_ind:#capacitive => reactance = -1, inductive => reactance = 1
                self.reactance = 1
            elif self.control_modus == ControlModusEnum.PF_ctrl_cap:
                self.reactance = -1
            elif self.control_modus == ControlModusEnum.PF_ctrl:
                if self.reactance == 1:
                    self.control_modus = ControlModusEnum.PF_ctrl_ind
                else:
                    self.control_modus = ControlModusEnum.PF_ctrl_cap #self.reactance == -1:
            self.diff_old = self.diff
            if self.diff is None: #first step for assured bsc_ctrl_step
                self.diff = 1
            else:
                if -0.012 < self.set_point < 0.012:
                    min_q = -0.012
                    max_q = -min_q
                    self.set_point = np.where((self.set_point >= 0) & (self.set_point <= max_q), max_q, self.set_point)
                    self.set_point = np.where((self.set_point >= min_q) & (self.set_point < 0), min_q, self.set_point)
                    logger.warning(f"Power factor calculation with set_point 0 not possible with BSC {self.index}.\n"
                                   f"Maximizing Q output by clipping set_point to {self.set_point}\n")
                q_set = self.reactance * sum(p_input_values)/len(p_input_values) * (np.tan(np.arccos(self.set_point)))
                self.diff = q_set - sum(input_values)/len(input_values)
            self.converged = np.all(np.abs(self.diff)<self.tol)

        elif self.control_modus == ControlModusEnum.tan_phi_ctrl:
            self.diff_old = self.diff
            if self.diff is None: #first step for assured bsc_ctrl_step
                self.diff = 1
            else:
                q_set = sum(p_input_values)/len(p_input_values) * self.set_point
                self.diff = q_set - sum(input_values)/len(input_values)
            self.converged = np.all(np.abs(self.diff) < self.tol)
        else:
            ###catching deprecated modi from old imports
            if isinstance(self.control_modus, bool) and self.control_modus == True and self.input_element_index is not None:
                self.control_modus = ControlModusEnum.v_ctrl  # catching old implementation
                logger.warning(
                    f"Deprecated Control Modus in Controller {self.index}, using V_ctrl from available types\n")
            elif (isinstance(self.control_modus, bool) and self.control_modus == False) or (isinstance(self.control_modus, bool) and self.control_modus == True
                                                                                        and self.input_element_index is None):
                if self.control_modus is True:
                    logger.warning(f'Deprecated Control Modus in Controller {self.index}, attempted to use "V_ctrl" but '
                                   f'missing attribute input_element_index, defaulting to Q_ctrl\n')
                else:
                    logger.warning(
                        f"Deprecated Control Modus in Controller {self.index}, using Q_ctrl from available types\n")
                self.control_modus = ControlModusEnum.q_ctrl

            if self.control_modus in ControlModusEnum.v_modes():
                self.diff_old = self.diff #V_ctrl
                if self.diff is None:  # first step for assured bsc_ctrl_step
                    self.diff = 1
                else:
                    if self.control_modus not in ControlModusEnum.droop_modes():
                        self.diff = self.set_point - net.res_bus.vm_pu.at[np.atleast_1d(self.input_element_index)[0]]
                    else:
                        self.diff = self.set_point - net.res_bus.vm_pu.at[np.atleast_1d(self.bus_idx)[0]]
                self.converged = np.all(np.abs(self.diff) < self.tol)
            else:
                if self.control_modus not in ControlModusEnum.q_modes():
                    logger.warning(f"No Controller Modus specified for Controller {self.index}, using Q_ctrl.\n"
                                   "Please specify 'control_modus' ('Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan(phi)_ctrl')\n")
                    self.control_modus = ControlModusEnum.q_ctrl
                self.diff_old = self.diff #Q_ctrl
                if self.diff is None:  # first step for assured bsc_ctrl_step
                    self.diff = 1
                else:
                    self.diff = self.set_point - sum(input_values)
                self.converged = np.all(np.abs(self.diff) < self.tol)
        ### check hard limits
        if net._options['enforce_q_lims']:
            if not any(self.output_adjustable):
                logging.info(f'All stations controlled by {self.name} with modus {self.control_modus} reached reactive power limits.')
                self.converged = True
                return self.converged
            else:
                # adapt output adjustable depending on in_service
                self.output_adjustable = np.array([in_service and adjustable for in_service, adjustable in zip(
                    self.output_element_in_service, self.output_adjustable
                )], dtype=bool)

                # normalize the values distribution
                self._normalize_distribution_in_service()
        ### check soft limits after convergence###
        if self.converged and not net._options["enforce_q_lims"]: #check overshot of gens when not enforcing q_lims
            if self.distribution_method == ControlModusEnum.rel_V_pu:
                vm_pu = read_from_net(net, "res_bus", self.bus_idx_dist, "vm_pu", 'auto')
                v_max_pu = np.atleast_1d(self.v_max_pu)[self.output_element_in_service]
                v_min_pu = np.atleast_1d(self.v_min_pu)[self.output_element_in_service]
                for i in range(len(vm_pu)):
                    if vm_pu[i] > v_max_pu[i]:
                        logger.warning(f'Controller {self.index}: Generator {self.output_element} {self.output_element_index[i]}'
                                       f' exceeded maximum Voltage at bus {self.bus_idx_dist[i]}: {vm_pu[i]} > {v_max_pu[i]}\n')
                    elif vm_pu[i] < v_min_pu[i]:
                        logger.warning(f'Controller {self.index}: Generator {self.output_element} {self.output_element_index[i]}'
                            f' exceeded maximum Voltage at bus {self.bus_idx_dist[i]}: {vm_pu[i]} < {v_min_pu[i]}\n')
            if len(self.min_q_mvar) == len(self.max_q_mvar) == len(self.output_element_in_service):

                exceed_limit_min = np.flatnonzero(np.atleast_1d(self.output_values)[np.atleast_1d(self.output_element_in_service)]
                                        < np.atleast_1d(self.min_q_mvar)[np.atleast_1d(self.output_element_in_service)])
                exceed_limit_max = np.flatnonzero(np.atleast_1d(self.output_values)[np.atleast_1d(self.output_element_in_service)]
                                        > np.atleast_1d(self.max_q_mvar)[np.atleast_1d(self.output_element_in_service)])
                for i in exceed_limit_max:
                    logger.warning(f'Controller {self.index} converged but the Reactive Power Output for Element '
                f'{self.output_element}: {self.output_element_index[i]} exceeds upper limits: {self.output_values[i]} > {self.max_q_mvar[i]}\n')
                for i in exceed_limit_min:
                    logger.warning(f'Controller {self.index} converged but the Reactive Power Output for Element '
                   f'{self.output_element}: {self.output_element_index[i]} falls short of lower limit: {self.output_values[i]} < {self.min_q_mvar[i]}\n')
            else:
                logger.warning(f'Mismatching number of minimum and maximum limits of the output elements in Controller {self.index}.'
                                           f'Possible exceedance of output element {self.output_element}'
                               f' {str(np.array(self.output_element_index))} limits\n')
        if self.converged and net.controller['object'].apply(
                lambda obj: getattr(obj, 'controller_idx', None) == self.index and not getattr(obj, 'converged', True)).any()\
                or getattr(self, 'applied_distribution', False) is False: #force appliance of distribution
            self.converged = False
        return self.converged

    def control_step(self, net):
        self._binary_search_control_step(net)

    def _binary_search_control_step(self, net):
        from pandapower import runpp #to avoid circular imports, import here
        generators_not_at_limit = None
        if not self.in_service: #redundant
            return
        ### Distribution corrections, no warnings due to q_limit incompatibility###
        if getattr(self, 'output_values_distribution', None) is not None: #catch errors
            if (self.distribution_method == ControlModusEnum.rel_P or
                                                    self.distribution_method == ControlModusEnum.rel_rated_S or
                                                    self.distribution_method == ControlModusEnum.max_Q):
                self.output_values_distribution, output_distribution_values_in_service = None, None
            elif self.distribution_method == ControlModusEnum.imported or self.distribution_method == ControlModusEnum.set_Q:
                if len(self.output_values_distribution) < len(np.array(self.output_element_in_service)):#check if enough values
                    equal_val = 1 / (len(np.array(self.output_element_in_service) - len(self.output_values_distribution)))
                    logger.warning(
                        f'Mismatched lengths of output elements {self.output_element} and output_values_distribution'
                        f'{len(np.array(self.output_element_in_service))} > {len(self.output_values_distribution)}'
                        f' in Controller {self.index}.\n Appending values {equal_val} \n')
                    self.output_values_distribution = (np.append(self.output_values_distribution, [equal_val] *
                                                                 (len(self.output_element_in_service) - len(self.output_values_distribution))))
                output_element_in_service = np.array(self.output_element_in_service)#ruggedizing code for wrong inputs
                output_element_in_service.resize((len(np.array(self.output_values_distribution)),), refcheck=False)
                output_distribution_values_in_service = (np.array(self.output_values_distribution)
                [np.array(output_element_in_service)]) ###only distributing between active output elements
            elif self.distribution_method == ControlModusEnum.rel_V_pu:
                if ((np.array(self.output_values_distribution).ndim > 1 and any(len(element) != 3 for element in self.output_values_distribution))
                        or (np.array(self.output_values_distribution).ndim == 1 and len(self.output_values_distribution) != 3)):
                    logger.warning(f"Insufficient values in distribution rel_V_pu {self.output_values_distribution} In "
                               f"Controller {self.index}. Using set point 1 pu and min/max 0.9/1.1 pu\n")
                    equal_array = [1, 0.9, 1.1]
                    self.output_values_distribution = self.output_values_distribution = np.tile(equal_array,
                                                                                                (len(np.array(self.output_element_in_service)), 1))[0]
                    output_distribution_values = np.array(self.output_values_distribution)  # forming limit arrays
                    self.v_set_point_pu = np.atleast_2d(output_distribution_values)[:, 0]
                    self.v_min_pu = np.atleast_2d(output_distribution_values)[:, 1]
                    self.v_max_pu = np.atleast_2d(output_distribution_values)[:, 2]
                output_distribution_values_in_service = None
            else: output_distribution_values_in_service, self.output_values_distribution = None, None
        elif getattr(self, 'output_values_distribution', None) is None:
            if self.distribution_method == ControlModusEnum.set_Q or self.distribution_method == ControlModusEnum.imported:
                if self.distribution_method == ControlModusEnum.set_Q:
                    logger.warning(f'Reactive Power Distribution method "set_Q" needs values given to output_values_distribution '
                         f'in Controller {self.index}. Distributing the reactive power equally between all available output elements.\n')
                else:#self.distribution_method == 'imported'
                    logger.warning(f"Something went wrong while importing output distribution values in Controller"
                         f" {self.index}. Distributing the reactive power equally between all available output elements.\n")
                equal = 1 / sum(np.array(self.output_element_in_service))
                self.output_values_distribution = np.full(len(np.array(self.output_element_in_service)), equal)
                output_distribution_values_in_service = self.output_values_distribution[np.array(self.output_element_in_service)]  #only distributing between active output elements
            elif self.distribution_method == ControlModusEnum.rel_V_pu:
                logger.warning(f"Missing values for output distribution values 'rel_V_pu in Controller {self.index}. "
                             f"Using set point 1 pu and min/max 0.9/1.1 pu\n ")
                equal_array = [1, 0.9, 1.1]
                self.output_values_distribution = self.output_values_distribution = np.tile(equal_array,
                                                                                            (len(np.array(self.output_element_in_service)), 1))[0]#new vals
                output_distribution_values = np.atleast_2d(self.output_values_distribution)  # forming limit arrays
                self.v_set_point_pu = output_distribution_values[:, 0]
                self.v_min_pu = output_distribution_values[:, 1]
                self.v_max_pu = output_distribution_values[:, 2]
                output_distribution_values_in_service = None
            else: self.output_values_distribution, output_distribution_values_in_service = None, None#rel_rated_S and rel_P, max_Q
        else: raise UserWarning(f"Output_values_distribution in Controller {self.index} is {self.output_values_distribution}")

        ###calculate output values###
        if self.output_values_old is None:  # first step
            # is ok that values are set for all stations even though they are out of service or not adjustable --> following step will correct this
            self.output_values_old, self.output_values = (
                np.atleast_1d(self.output_values)[self.output_element_in_service],
                np.atleast_1d(self.output_values)[self.output_element_in_service] + 1e-3)
            positions_not_adjustable = [i for i, val in enumerate(self.output_adjustable) if not val]
            for i in positions_not_adjustable:
                if np.atleast_1d(self.output_values_distribution)[i]==0 or not self.output_element_in_service[i] :
                    self.output_values[i] = 0
                else:
                    continue
        else:#second step
            self.applied_distribution = True
            step_diff = self.diff - self.diff_old
            x = self.output_values - self.diff * (self.output_values - self.output_values_old) / np.where(
                step_diff == 0, 1e-6, step_diff)  #converging
            if any((abs(x) - abs(2 * self.output_values)) > 100): #catching overshoots for calculation, another check before writing into the net
                x[np.nonzero((abs(x) > abs(100 - abs(self.output_values))))] = np.sign(x[np.nonzero((abs(x) -
                                                                           abs(2 * self.output_values)) > 100)]) * 100
            ###calculate the distribution of the output values
            if self.distribution_method == ControlModusEnum.imported: #when importing net from PF for backwards compatibility
                distribution = output_distribution_values_in_service

            elif self.distribution_method == ControlModusEnum.rel_P: #proportional to the dispatch active power
                dispatched_active_power = read_from_net(net, self.output_element, self.output_element_index, 'p_mw', 'auto')
                dispatched_active_power = np.atleast_1d(dispatched_active_power)[np.array(self.output_element_in_service)]
                distribution = dispatched_active_power/sum(dispatched_active_power)

            elif self.distribution_method == ControlModusEnum.rel_rated_S: #proportional to the rated apparent power
                if not hasattr(self, 'rel_rated_S_warned'):
                    self.rel_rated_S_warned = True
                    logger.warning(f'The standard type attribute containing the rated apparent power for'
                               f' {self.output_element} is not correctly implemented yet (BSC {self.index}).')
                try:
                    s_rated_mva = np.array(net.sgen.loc[self.output_element_index, 'sn_mva']) #todo correct attribute?
                    distribution = s_rated_mva
                    nan_index = np.isnan(distribution)
                    distribution[nan_index] = 50
                    if any(np.atleast_1d(nan_index)):
                        logger.warning(f'{self.output_element} at index {np.atleast_1d(self.output_element_index)[nan_index]}'
                                       f' in Controller {self.index} has no specified rated apparent power, assuming 50 MVA\n')
                    if not all(isinstance(n, numbers.Number) for n in np.atleast_1d(distribution)):
                        logger.warning(f'{self.output_element} in Controller {self.index} has no'
                                       f' specified rated apparent power, assuming 50 MVA\n')
                        distribution = np.full(np.sum(self.output_element_in_service), 50)

                except KeyError:
                    logger.warning(f'{self.output_element} in Controller {self.index} has no defined standard type '
                                    f'or specified rated apparent power, assuming 50 MVA\n')
                    distribution = np.full(np.sum(self.output_element_in_service), 50)

            elif self.distribution_method == ControlModusEnum.set_Q: #individually set Q distribution
                distribution = output_distribution_values_in_service

            elif self.distribution_method == ControlModusEnum.max_Q:  # Maximise Reactive Reserve
                #only consider active sgens who are within their limits
                generators_not_at_limit = (x <= np.array(self.max_q_mvar)[self.output_element_in_service]) \
                                          & (x >= np.array(self.min_q_mvar)[self.output_element_in_service])
                #get Q for sgens
                total_distributable_q = ((np.sum(np.array(x)[generators_not_at_limit]) -
                                          np.sum(np.array(self.min_q_mvar)[self.output_element_in_service][generators_not_at_limit])) /
                                         (np.sum(np.array(self.max_q_mvar)[self.output_element_in_service][generators_not_at_limit]) -
                                          np.sum(np.array(x)[generators_not_at_limit])))
                if np.isnan(total_distributable_q): #no distributable Q
                    total_distributable_q = 0
                #calculate the qs for generators to be considered from total distributable Q
                q_max_q = ((total_distributable_q * np.array(self.max_q_mvar)[self.output_element_in_service][generators_not_at_limit] +
                    np.array(self.min_q_mvar)[self.output_element_in_service][generators_not_at_limit]) / (1 + total_distributable_q))
                ### output gens not to be considered run at max capacity, all others on calculated Q
                #output values must be equal in length to distribution
                if len(np.atleast_1d(q_max_q)) != len(np.atleast_1d(self.output_element_in_service)):
                    counter_values = 0
                    distribution = np.ones(len(np.atleast_1d(self.output_element_in_service)))  #initializing the distribution for correction
                    for i in range(len(np.atleast_1d(generators_not_at_limit))):
                        if np.atleast_1d(generators_not_at_limit)[i]:#calculated Q
                            distribution[i] = np.atleast_1d(q_max_q)[counter_values]
                            counter_values += 1
                        elif not np.atleast_1d(generators_not_at_limit)[i]:#min or max Q
                            distribution[i] = np.atleast_1d(self.max_q_mvar)[i] if (np.atleast_1d(x)[i]
                                                >= 0) else np.atleast_1d(self.min_q_mvar)[i]
                else:
                    distribution = q_max_q

            elif self.distribution_method == ControlModusEnum.rel_V_pu:  # Voltage set point Adaptation
                if len(np.atleast_1d(self.output_element_in_service)) > 1 or sum(
                        np.atleast_1d(self.output_element_in_service)) > 1:#only for multiple elements
                    ###check for multiple output elements who influence the busbar###
                    if (len(net.sgen.bus) != len(set(net.sgen.bus)) or len(net.gen.bus) != len(set(net.gen.bus)) or
                        set(net.sgen.bus).intersection(set(net.gen.bus))):
                            busbar_gen_sgen = list(set(net.sgen.bus).intersection(set(net.gen.bus))) #gens and sgens
                            busbar_gen_sgen = False if len(busbar_gen_sgen) == 0 else busbar_gen_sgen #False if array empty
                            busbar_sgen_sgen = list(np.where(np.bincount(np.array(net.sgen['bus'])) > 1)[0]) #sgens and sgens
                            busbar_sgen_sgen = False if len(busbar_sgen_sgen) == 0 else busbar_sgen_sgen #False if array empty
                            busbar_gen_gen = list(np.where(np.bincount(np.array(net.gen['bus'])) > 1)[0]) #gens and gens
                            busbar_gen_gen = False if len(busbar_gen_gen) == 0 else busbar_gen_gen #False if array empty
                            busbar_all = [busbar_gen_gen, busbar_sgen_sgen, busbar_gen_sgen] #merge all indices
                            if any(busbar_all):#not all busbar with multiple output elements?
                                busbar_all = np.array([x for x in busbar_all if x != False][0]) #delete bools
                                index_sgen = np.where(np.isin(net.sgen['bus'], busbar_all))[0] #indices of sgens
                                index_sgen = [index for i, index in enumerate(index_sgen) if list(net.sgen['in_service'])[i]]#check for service
                                index_gen = np.where(np.isin(net.gen['bus'], busbar_all))[0] #indices of gens
                                index_gen = [index for i, index in enumerate(index_gen) if list(net.gen['in_service'])[i]] #check for service
                                if len(index_sgen) + len(index_gen) > 1:
                                    items_sgen, items_gen, busbar = "Check Sgen:\n", "Check gen:\n", ""#initiate strings
                                    for x in index_sgen: items_sgen += f"{net.sgen.name[x]} with index {x}\n"#append sgen names
                                    for x in index_gen: items_gen += f"{net.gen.name[x]} with index {x}\n" #append gen names
                                    for x in busbar_all: busbar += f"{net.bus.name[x]} with index {x}; " #append busbar names
                                    raise NotImplementedError(f"Multiple Output Elements are controlling the voltage at Busbar(s) {busbar} \n"
                                                        f"Voltage set point adaptation for Controller {self.index} is not possible.\n"
                                                        f"{items_sgen}{items_gen}")

                    if len(self.bus_idx_dist)==0 and (self.output_element == 'sgen' or self.output_element == 'gen'):
                        if self.output_element == 'sgen': #gens are ignored
                            self.bus_idx_dist = np.atleast_1d(net.sgen.bus[self.output_element_index])[self.output_element_in_service]#distributing output elements
                        else:
                            raise UserWarning(f"Output Element {self.output_element} in Controller {self.index} is not supported")

                    ###calculate the voltage set points
                    v_min_pu = np.atleast_1d(self.v_min_pu)[self.output_element_in_service] #adapt min/max and set point for active elements
                    v_max_pu = np.atleast_1d(self.v_max_pu)[self.output_element_in_service]
                    v_set_point_pu = np.atleast_1d(self.v_set_point_pu)[self.output_element_in_service]
                    vm_pu = read_from_net(net, "res_bus", self.bus_idx_dist, "vm_pu", 'auto') #init
                    sum_vm_pu = np.sum(vm_pu) #total
                    bounds = [(L, U) for L, U in zip(v_min_pu, v_max_pu)] #limits
                    result = minimize(
                        lambda v: np.sum((v - v_set_point_pu) ** 2),  #minimize deviation from set point
                        vm_pu,  # Initial guess
                        method='SLSQP',  # Optimization method trust-constr or SLSQP
                        bounds=bounds,  # Soft limits as bounds
                        constraints=[
                            {'type': 'eq', 'fun': lambda v: np.sum(v) - sum_vm_pu},  # Load constraint
                            {'type': 'ineq', 'fun': lambda v: v - v_min_pu},  # Lower soft limits
                            {'type': 'ineq', 'fun': lambda v: v_max_pu - v}  # Upper soft limits
                        ],
                        options={'maxiter': 1000, 'ftol': 1e-9})  #more iterations, small tolerance 'ftol': 1e-9 only with SLSQP
                    voltage = result.x #getting the results of minimize function
                    ### convert sgens to gens, write voltage to gens, read Q and adapt distribution
                    in_service_indices = np.array(self.output_element_index)[self.output_element_in_service]#actual indices
                    counter = 0
                    for i in in_service_indices:
                        if self.output_element == 'sgen': #get all sgens, convert to gens
                            create_gen(net = net,
                                    bus = net.sgen.at[i, 'bus'],
                                    p_mw = net.sgen.at[i, 'p_mw'],
                                    vm_pu = voltage[counter],  # Voltage array
                                    in_service = net.sgen.at[i, 'in_service'],
                                    sn_mva = net.sgen.at[i, 'sn_mva'] if 'sn_mva' in net.sgen.columns else None,
                                    scaling = net.sgen.at[i, 'scaling'] if 'scaling' in net.sgen.columns else None,
                                    min_p_mw = net.sgen.at[i, 'min_p_mw'] if 'min_p_mw' in net.sgen.columns else 0, #for value other then inf min and max must be given
                                    max_p_mw = net.sgen.at[i, 'max_p_mw'] if 'max_p_mw' in net.sgen.columns else 9999,
                                    min_q_mvar = net.sgen.at[i, 'min_q_mvar'] if 'min_q_mvar' in net.sgen.columns and np.isfinite(net.sgen.at[i, 'min_q_mvar']) else -20,
                                    max_q_mvar = net.sgen.at[i, 'max_q_mvar'] if 'max_q_mvar' in net.sgen.columns and np.isfinite(net.sgen.at[i, 'max_q_mvar']) else 20,
                                    description = net.sgen.at[i, 'description'] if 'description' in net.sgen.columns else None,
                                    equipment = net.sgen.at[i, 'equipment'] if 'equipment' in net.sgen.columns else None,
                                    geo = net.sgen.at[i, 'geo'] if 'geo' in net.sgen.columns else None,
                                    current_source = net.sgen.at[
                                        i, 'current_source'] if 'current_source' in net.sgen.columns else None,
                                    name = f'temp_gen_{counter}')#type 'GEN'
                            net.sgen.at[i, 'in_service'] = False #disable sgens
                            counter += 1
                    index = np.array([])
                    for i in net.gen.index: #get index of created gens
                        if net.gen.loc[i, 'name'].startswith("temp_gen_"):
                            index = np.append(index, i)
                    index = index[0] if self.write_flag == 'single_index' else index
                    write_to_net(net, 'gen', index,'vm_pu', voltage, self.write_flag) #write V to net
                    runpp(net, run_control = False, enforce_q_lims=False) #run net
                    distribution = np.array(net.res_gen.loc[index, 'q_mvar']) #read Q from net
                    net.gen.drop(index=index, inplace=True) #delete created gens
                    net.sgen.loc[np.array(self.output_element_index)[self.output_element_in_service], 'in_service'] = True #reactivate sgens
                else: distribution = np.array([1]) #distribution is one for one active output element

            else: #unrecognizable output values distribution, using set_Q
                if (((isinstance(self.distribution_method, list) or isinstance(self.distribution_method, np.ndarray))
                    and all(isinstance(x, numbers.Number) for x in self.distribution_method)) or
                        isinstance(self.distribution_method, numbers.Number)):#numbers
                    logger.warning(f'Controller {self.index}: Distribution_method must be string from available methods'
                                   f' (rel_P, rel_rated_S, set_Q, max_Q or rel_V_pu). Using provided values with method set_Q\n')
                    self.output_values_distribution = np.array(self.distribution_method)
                    self.distribution_method = ControlModusEnum.set_Q
                    distribution = self.output_values_distribution[np.array(self.output_element_in_service)]
                else:
                    raise NotImplementedError(f"Controller {self.index}: Reactive power distribution method {self.distribution_method}"
                                              f" not implemented available methods are (rel_P, rel_rated_S, set_Q, max_Q, rel_V_pu).")
            if self.output_element != 'gen':
                if self.distribution_method == ControlModusEnum.max_Q: #max_Q and voltage gives the correct Qs for the gens
                    if sum(np.atleast_1d(generators_not_at_limit)) == 0:
                        values = (sum(x) - sum(distribution)) / len(np.atleast_1d(distribution))
                        distribution = np.atleast_1d(distribution) + values #todo if respected Q limits only generators_not_at_limit, might not converge
                    else:
                        values = (sum(x) - sum(distribution)) / len(np.atleast_1d(distribution)[generators_not_at_limit])
                        np.atleast_1d(distribution)[generators_not_at_limit] += values
                    x = distribution
                #Voltage set point adaption gives correct Qs but needs convergence
                elif (self.distribution_method == ControlModusEnum.rel_V_pu and (sum(np.atleast_1d(self.output_element_in_service)) > 1
                                                                                 or sum(np.atleast_1d(self.output_element_in_service)) > 1)): #only when multiple elements
                    x = distribution + (sum(x) - sum(distribution)) / len(distribution)
                else: #percentile calculation
                    distribution = np.array(distribution, dtype=np.float64) / np.sum(abs(distribution))  # normalization
                    if (any(abs(x) > 3 for x in np.atleast_1d(distribution)) or  # catching distributions out of bounds
                            len(np.atleast_1d(distribution)) != sum(
                                np.atleast_1d(self.output_element_in_service))):  # catching wrong distributions
                        equal = 1 / sum(self.output_element_in_service)
                        distribution = np.full(np.sum(np.array(self.output_element_in_service)), equal)
                    x = x * distribution if isinstance(x, numbers.Number) else sum(x) * distribution #add distribution to Q values
            ###enforce hard Q limits###
            if not all(self.output_adjustable) and net._options['enforce_q_lims']:
                positions_adjustable = [i for i, val in enumerate(self.output_adjustable) if
                                        val]  # gives which is/are adjustable
                positions_not_adjustable = [i for i, val in enumerate(self.output_adjustable) if
                                            not val]  # can be one or multiple ## gives which is/are not adjustable anymore

                sum_adjustable = sum(x) - sum(self.output_values[
                                                  positions_not_adjustable])  # stations that are still adjustable, rest of the power must be achieved
                x[positions_adjustable] = sum_adjustable * self.distribution_method[positions_adjustable]

                for i in positions_not_adjustable:
                    if self.output_element_in_service[i]:
                        x[i] = self.output_values[i]  # reset value to q_limit
                    else:
                        x[i] = 0  # reset value to 0 because station is out of service

            else:
                if self.distribution_method != ControlModusEnum.max_Q and self.distribution_method != ControlModusEnum.rel_V_pu:
                    x = sum(np.atleast_1d(x)) * distribution

            if self.output_adjustable is not None and net._options.get('enforce_q_lims', False):  # none if output element is a shunt
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
                        max_q_mvar_limit = self.output_max_q_mvar[np.atleast_1d(positions)]

                        # adapt distribution and x
                        self.output_values_distribution[positions] = 0
                        if np.all(self.output_values_distribution == 0):
                            # all stations reached limit, prevent for division with 0 resulting in nan array
                            pass
                        else:
                            self.output_values_distribution /= sum(self.output_values_distribution)
                        x = (sum_old-sum(np.atleast_1d(max_q_mvar_limit)))*self.output_values_distribution
                        x[positions] = max_q_mvar_limit # reset to limit

                    elif any(reached_min_qmvar):
                        positions = [i for i, val in enumerate(reached_min_qmvar) if val is np.True_]
                        reached_index = [self.output_element_index[i] for i in positions]
                        logging.info('Station(s) controlled by %s reached the minimum reactive power limit: %s'
                              % (self.name, ', '.join(net[self.output_element].loc[reached_index].name.tolist())))
                        self.output_adjustable[positions] = False
                        sum_old = sum(x)
                        min_q_mvar_limit = self.output_min_q_mvar[np.atleast_1d(positions)]

                        # adapt distribution and x
                        self.output_values_distribution[positions] = 0
                        if np.all(self.output_values_distribution == 0):
                            # all stations reached limit, prevent for division with 0 resulting in nan array
                            pass
                        else:
                            self.output_values_distribution /= sum(self.output_values_distribution)

                        x = (sum_old-sum(np.atleast_1d(min_q_mvar_limit)))*self.output_values_distribution
                        x[positions] = min_q_mvar_limit # reset to limit

                    self.output_values_old, self.output_values = self.output_values, x
                else:
                    # check when x is a single value (only one adjustable machine)
                    # check if limit is reached
                    self._update_min_max_q_mvar(net)

                    reached_min_qmvar = x < self.output_min_q_mvar
                    reached_max_qmvar = x > self.output_max_q_mvar

                    self.output_values_old = self.output_values
                    if reached_min_qmvar or reached_max_qmvar:
                        logging.info('Station %s controlled by %s reached a reactive power limit.' % (
                        self.output_element_index, self.name))
                        self.output_adjustable = np.array([False], dtype=np.bool)
                        logging.info(
                            f"Station {self.output_element_index} controlled by {self.name} reached a reactive power "
                            f"limit."
                        )
                        self.output_adjustable = np.array([False], dtype=bool)
                        if reached_min_qmvar:
                            x = self.output_min_q_mvar
                        elif reached_max_qmvar:
                            x = self.output_max_q_mvar
            x = np.sign(x) * (np.where(abs(abs(x) - abs(self.output_values)) > 84, 84,
                                   abs(x)))  # catching distributions out of bounds, 84 seems to be the maximum
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
            if isinstance(self.output_values_distribution, str) or getattr(self, 'output_values_distribution', None) is None:
                distribution = np.ones(len(np.atleast_1d(self.output_element_in_service)))/len(np.atleast_1d(self.output_element_in_service))
            else: distribution = self.output_values_distribution
        else:
            distribution = initial_pf_distribution

        # normalize the values distribution
        # set output_values_distribution to 0, if station is not in service
        self.output_values_distribution = [0 if not in_service else value
               for in_service, value in zip(np.atleast_1d(self.output_element_in_service), np.atleast_1d(distribution))]
        total = np.sum(self.output_values_distribution)
        if total is not None and total > 0:  # To avoid division by zero
            self.output_values_distribution = np.array(self.output_values_distribution, dtype=np.float64) / total
        else:
            self.output_values_distribution = np.zeros_like(self.output_values_distribution, dtype=np.float64)

    def _update_min_max_q_mvar(self, net):
        if 'min_q_mvar' in net[self.output_element].columns:
            if not np.all(np.isnan(pd.array(pd.Series(net[self.output_element].loc[self.output_element_index, 'id_q_capability_characteristic']).values, dtype="Int64"))):
                qmin, _ = get_min_max_q_mvar_from_characteristics_object(net, self.output_element, self.output_element_index)
                self.output_min_q_mvar = np.nan_to_num(qmin, nan=-np.inf)
                net[self.output_element].loc[self.output_element_index, 'min_q_mvar'] = self.output_min_q_mvar
            else:
                self.output_min_q_mvar = np.nan_to_num(pd.Series(
                    net[self.output_element].loc[self.output_element_index, 'min_q_mvar']).values, nan=-np.inf)
                net[self.output_element].loc[self.output_element_index, 'min_q_mvar'] = self.output_min_q_mvar
        else:
            self.output_min_q_mvar = list(np.array([-np.inf]*len(self.output_element_index), dtype=np.float64))

        if 'max_q_mvar' in net[self.output_element].columns:
            if not np.all(np.isnan(pd.array(pd.Series(net[self.output_element].loc[self.output_element_index, 'id_q_capability_characteristic']).values, dtype="Int64"))):
                _, qmax = get_min_max_q_mvar_from_characteristics_object(net, self.output_element, self.output_element_index)
                self.output_max_q_mvar = np.nan_to_num(qmax, nan=np.inf)
                net[self.output_element].loc[self.output_element_index, 'max_q_mvar'] = self.output_max_q_mvar
            else:
                self.output_max_q_mvar = np.nan_to_num(pd.Series(
                    net[self.output_element].loc[self.output_element_index, 'max_q_mvar']).values, nan=np.inf)
                net[self.output_element].loc[self.output_element_index, 'max_q_mvar'] = self.output_max_q_mvar
        else:
            self.output_max_q_mvar = list(np.array([np.inf]*len(self.output_element_index), dtype=np.float64))

class DroopControl(Controller):
    """
    The droop controller is used in case of a droop based control. It can operate either as a Q(U) controller or
    as a U(Q) controller and is used in addition to a binary search controller (bsc). The linked binary search
    controller is specified using the controller_index, which refers to the linked bsc. The droop controller
    behaves in a similar way to the station controllers presented in the Power Factory Tech Ref, although not
    all possible settings from Power Factory are yet available.

    Parameters
    ----------
    self : DroopControl
    net : pandapowerNet
        A pandapower grid.
    controller_idx : int
        Index of linked Binary search control (bsc.index).
    in_service : bool
        Whether the droop controller is in service or not. Default is True
    control_modus : str -> ControlModusEnum:
        takes string: Q_ctrl_V_droop, V_ctrl_Q_droop. Formerly called voltage_ctrl.
    q_droop_var : float
        Droop Value in Mvar/p.u.
    vm_set_pu_bsc : float
        Initial voltage set point in case of voltage control. If None will take linked bsc setpoint
    bus_idx : int
        Bus index which is used for Q control.
    q_set_mvar_bsc : float
        Initial voltage set point in case of no voltage control.
    vm_set_lb : float
        Lower band border of dead band
    vm_set_ub : float
        Upper band border of dead band
    tol : float, optional
        Tolerance for controller convergence. Default is 0.001.
    ctrl_in_service : bool, optional
        Whether the controller is in service. Default is True.
    order : int, optional
        Execution order of the controller.
    level : int, optional
        Execution level of the controller.
    drop_same_existing_ctrl : bool, optional
        Whether to drop existing controllers with the same parameters. Default is False
    matching_params : dict, optional
        Parameters used to match controllers. Default is None
    name : str, optional
        Name of the controller.
    kwargs : dict, optional
        Additional keyword arguments.
    """
    def __init__(self, net, q_droop_mvar, controller_idx, control_modus = None, bus_idx=None, tol=1e-6,
                 q_set_mvar_bsc=None, in_service=True, order=-1, level=0, name="", drop_same_existing_ctrl=False,
                 matching_params=None, vm_set_pu_bsc=None, vm_set_lb=None, vm_set_ub=None, **kwargs):
        super().__init__(net, in_service=in_service, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params)
        #TODO: implement maximum and minimum of droop control
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
        if control_modus and value is None:
            self.vm_set_pu_bsc = net.controller.at[controller_idx, "object"].set_point
        else:
            self.vm_set_pu_bsc = value
        self.vm_set_pu = self.vm_set_pu_bsc
        self.vm_set_pu_new = None
        self.lb_voltage = vm_set_lb
        self.ub_voltage = vm_set_ub
        self.controller_idx = controller_idx
        self.control_modus = control_modus
        self.tol = tol
        self.applied = False
        self.read_flag, self.input_variable = _detect_read_write_flag(net, "res_bus", bus_idx, "vm_pu")
        self.q_set_mvar_bsc = q_set_mvar_bsc
        self.q_set_mvar = None
        self.q_set_old_mvar = None
        self.diff = None
        self.converged = False
        self._deprecation_warned = False
        self.check_control_modus_and_values(net)


    def check_control_modus_and_values(self, net):
        if getattr(self, 'control_modus', None) is None:#catching old attribute voltage_ctrl
            if hasattr(self, 'voltage_ctrl'):
                self.control_modus = self.voltage_ctrl
                if getattr(self, '_deprecation_warned', False) is False:#only one message that voltage ctrl is deprecated
                    logger.warning(
                        f"'voltage_ctrl' in Controller {self.index} is deprecated. "
                        "Use 'control_modus' ('Q_ctrl', 'V_ctrl', etc.) instead.")
                    self._deprecation_warned = True
        ###catching old implementation
        if isinstance(self.control_modus, bool) and self.control_modus == True:
            self.control_modus = ControlModusEnum.v_ctrl_q_droop
            logger.warning(f"Deprecated Control Modus in Controller {self.index}, using V_ctrl with Q droop from available types"
                         f" 'Q_ctrl' or 'V_ctrl'\n")
        elif isinstance(self.control_modus, bool) and self.control_modus == False:
            self.control_modus = ControlModusEnum.q_ctrl_v_droop
            logger.warning(f"Deprecated Control Modus in Controller {self.index}, using Q_ctrl with V droop from available types"
                         f" 'Q_ctrl' or 'V_ctrl'\n")
        else:
            try:
                self.control_modus = ControlModusEnum(self.control_modus)
            except ValueError:
                logger.warning(f"Control_modus {self.control_modus} not recognized, using 'Q_ctrl_V_droop' from available"
                               f" types 'Q_ctrl' and 'V_ctrl'\n")
                self.control_modus = ControlModusEnum.q_ctrl_v_droop
        if self.control_modus in ControlModusEnum.pf_modes():#legacy ambiguous
                raise UserWarning(f"Power Factor Droop Control not implemented (in Controller {self.index}).'\n")
        elif self.control_modus in ControlModusEnum.v_modes() and self.control_modus not in ControlModusEnum.droop_modes():
            logger.warning(f"Power Factor Droop Control in Controller {self.index}: Control modus is ambivalent, using"
                           f" 'V_ctrl with Q droop' from available modi.\n")
            self.control_modus = ControlModusEnum.v_ctrl_q_droop
        elif self.control_modus in ControlModusEnum.q_modes() and self.control_modus not in ControlModusEnum.droop_modes():
            logger.warning(f"Power Factor Droop Control in Controller {self.index}: Control modus is ambivalent, using"
                           f" 'Q_ctrl with V droop' from available modi.\n")
            self.control_modus = ControlModusEnum.q_ctrl_v_droop
        if (self.control_modus in ControlModusEnum.v_modes() and not
                    isinstance(getattr(self, 'vm_set_pu', None), numbers.Number)):#catching missing voltage set point
            logger.warning(f"vm_set_pu must be a number, not "
                   f"{isinstance(getattr(self, 'vm_set_pu', None), numbers.Number)} in Controller {self.index}, "
                   f"using 1 as new setpoint")
            self.vm_set_pu = getattr(net.controller.object[self.controller_idx], "set_point", 1)
        #checking if Droop and BS Controller have the same control_modus
        if self.control_modus != net.controller.at[self.controller_idx, 'object'].control_modus:
            if self.control_modus in ControlModusEnum.droop_modes():
                logger.warning(
                    f"Discrepancy between BinarySearchController Modus and Droop Controller Modus in {self.index}."
                    f"Using Droop Controller Modus {self.control_modus} from droop-controller")
                net.controller.at[self.controller_idx, 'object'].control_modus = self.control_modus
            else:
                logger.warning(
                    f"Discrepancy between BinarySearchController Modus and Droop Controller Modus in {self.index}."
                    f"Using Q_ctrl_P_droop from available types 'Q_ctrl', 'V_ctrl' or 'PF_ctrl'\n")
                self.control_modus = net.controller.at[self.controller_idx, 'object'].control_modus

    def is_converged(self, net):
        if (not net.controller.at[self.controller_idx, "object"].in_service or
                net.controller.at[self.controller_idx, "object"].converged):
            self.converged = True
            return self.converged
        ###check control_modus###
        self.check_control_modus_and_values(net)
        if self.control_modus in ControlModusEnum.v_modes():
            self.diff = (net.controller.at[self.controller_idx, "object"].set_point -
                         read_from_net(net, "res_bus", int(self.bus_idx), "vm_pu", self.read_flag))
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
        self.vm_pu = read_from_net(net, "res_bus", self.bus_idx, "vm_pu", flag=self.read_flag)
        if self.control_modus not in ControlModusEnum.v_modes():
            if self.q_set_mvar_bsc is None:
                self.q_set_mvar_bsc = net.controller.at[self.controller_idx, "object"].set_point
            if self.lb_voltage is not None and self.ub_voltage is not None:
                if self.vm_pu > self.ub_voltage:
                    self.q_set_old_mvar, self.q_set_mvar = (
                        self.q_set_mvar, self.q_set_mvar_bsc - (self.ub_voltage - self.vm_pu) * self.q_droop_mvar)
                elif self.vm_pu < self.lb_voltage:
                    self.q_set_old_mvar, self.q_set_mvar = (
                        self.q_set_mvar, self.q_set_mvar_bsc + (self.lb_voltage - self.vm_pu) * self.q_droop_mvar)
                else:
                    self.q_set_old_mvar, self.q_set_mvar = (self.q_set_mvar, self.q_set_mvar_bsc)
            else:
                self.q_set_old_mvar, self.q_set_mvar = (
                    self.q_set_mvar, self.q_set_mvar + (
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
            self.vm_set_pu = getattr(self, 'vm_set_pu', net.controller.object[self.controller_idx].set_point)
            self.vm_set_pu_new = self.vm_set_pu + sum(
                input_values) / self.q_droop_mvar
            net.controller.at[self.controller_idx, "object"].set_point = self.vm_set_pu_new


class VDroopControl_local(Controller):
    """
    The VDroopControl_local is used in case of a local droop based voltage control. It is used in addition to
    a binary search controller (bsc). The linked binary search controller is specified using the controller index,
    which refers to the linked bsc.

    Parameters:
        net: A pandapower grid.
        q_droop_var: Droop Value in Mvar/p.u.
        vm_set_pu_bsc: Initial voltage set point.
        controller_idx: Index of linked Binary< search control (if present).
        tol: Tolerance criteria of controller convergence.
        vm_set_lb: Lower band border of dead band
        vm_set_ub: Upper band border of dead band
    """

    def __init__(self, net, q_droop_mvar, controller_idx, bus_idx, control_modus = None, tol=1e-6, in_service=True, order=-1, level=0,
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
        self.vm_set_pu_bsc =  vm_set_pu_bsc if vm_set_pu_bsc is not None else kwargs.get('vm_set_pu')
        self.vm_set_pu_new = None
        self.q_set_mvar = q_set_mvar
        self.lb_voltage = vm_set_lb
        self.ub_voltage = vm_set_ub
        self.controller_idx = controller_idx
        self.bus_idx = bus_idx
        try:
            self.control_modus = ControlModusEnum(control_modus)
        except ValueError:
            logger.warning(f"Control_modus {control_modus} not recognized, using 'V_ctrl_Q_droop_local' \n")
            self.control_modus = ControlModusEnum.v_ctrl_q_droop_local
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

class ControlModusEnum(Enum):
    v_ctrl = "V_ctrl"
    v_ctrl_q_droop = "V_ctrl_Q_droop"
    v_ctrl_q_droop_local = "V_ctrl_Q_droop_local"
    q_ctrl = "Q_ctrl"
    q_ctrl_v_droop = "Q_ctrl_V_droop"
    PF_ctrl = "PF_ctrl"
    PF_ctrl_ind = "PF_ctrl_ind"
    PF_ctrl_cap = "PF_ctrl_cap"
    tan_phi_ctrl = "tan_phi_ctrl"
    rel_P = "rel_P"
    rel_rated_S = "rel_rated_S"
    max_Q = "max_Q"
    rel_V_pu = "rel_V_pu"
    set_Q = "set_Q"
    imported = 'imported'

    @classmethod
    def pf_modes(cls):
        return {
            cls.PF_ctrl,
            cls.PF_ctrl_cap,
            cls.PF_ctrl_ind,
        }

    @classmethod
    def v_modes(cls):
        return {
            cls.v_ctrl,
            cls.v_ctrl_q_droop,
            cls.v_ctrl_q_droop_local,
        }

    @classmethod
    def q_modes(cls):
        return {
            cls.q_ctrl,
            cls.q_ctrl_v_droop,
        }

    @classmethod
    def droop_modes(cls):
        return {
            cls.v_ctrl_q_droop,
            cls.v_ctrl_q_droop_local,
            cls.q_ctrl_v_droop,
        }