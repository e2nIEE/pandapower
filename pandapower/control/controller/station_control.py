import numpy as np
import numbers
from collections.abc import Sequence
import logging

from pandapower.control.basic_controller import Controller
from pandapower.auxiliary import _detect_read_write_flag, read_from_net, write_to_net
from pandapower.control.util.auxiliary import get_min_max_q_mvar_from_characteristics_object
from enum import Enum
logger = logging.getLogger(__name__)


class BinarySearchControl(Controller):
    """
    The Binary search control is a controller that adjusts output values in order to reach a given set point.
    It can be used for reactive power control, voltage control, cosines(phi) or tangens(phi) control. The control modus
    can be set via the control_modus parameter. Input and output elements and indexes can be lists. Input elements can
    be transformers, switches, lines or buses (only in voltage control). the controlled bus must be
    given to input_element_index. Output elements are sgens, where active and reactive power can be set. The
    "output_values_distribution" describes the distribution of reactive power provision between multiple
    "output_elements" and will be normalized to 100 % (1).

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
    output_values_distribution : int, float or list of float
        Distribution of reactive power provision among output elements (must sum to 1).
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
        Element of input element in net. Controlled bus in case of Voltage control. Can be
        given the string ``"auto"`` in control_modus ``"V_ctrl"`` to automatically select a bus whose nominal voltage is >= X kV.
        The X must be given to set_point. Will take target voltage of the encountered bus. If no bus is found,
        uses the bus next to the controlled generator group. Not completely implemented, generators on multiple buses
        are not correctly handled.
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
    def __init__(self, net, ctrl_in_service, output_element, output_variable, output_element_index,
                 output_element_in_service, output_values_distribution, input_element, input_variable,
                 input_element_index, set_point, control_modus:str=None, name="", input_inverted=None,
                 tol=0.001, in_service=True, order=0, level=0, drop_same_existing_ctrl=False,
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
        if isinstance(output_element_index, list) or isinstance(output_element_index, np.ndarray):
            self.output_element_index = [int(item) for item in output_element_index]
        else:
            self.output_element_index = []
            self.output_element_index.append(output_element_index)
            self.output_element_index = output_element_index
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
            inv_list = list(np.atleast_1d(input_inverted))[:n]
            if len(inv_list) < n:
                inv_list += [False] * (n - len(inv_list))
            self.input_sign = [-1 if inv else 1 for inv in inv_list]
        self.output_element = output_element
        self.output_element_index = output_element_index
        self.output_element_in_service = output_element_in_service

        # normalize the values distribution:
        self._normalize_distribution_in_service(initial_pf_distribution=output_values_distribution)

        self.output_adjustable = np.array([False if not distribution else service
                                            for distribution, service in zip(np.atleast_1d(self.output_values_distribution),
                                                np.atleast_1d(self.output_element_in_service))], dtype=bool)
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
                         f" types 'Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan(phi)_ctrl'\n")
        elif isinstance(control_modus, bool) and control_modus == False: #Only functions written out!?!
            self.control_modus = ControlModusEnum.q_ctrl
            logger.warning(f"Deprecated Controller control_modus for Controller {self.index}, using Q_ctrl from available"
                         f" types 'Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan(phi)_ctrl'\n")
        else:
            try:
                self.control_modus = ControlModusEnum(control_modus)
            except ValueError:
                logger.warning(f"Control_modus {control_modus} not recognized, using 'Q_ctrl' from available"
                               f" types 'Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan_phi_ctrl'\n")
                self.control_modus = ControlModusEnum.q_ctrl
        if self.control_modus == ControlModusEnum.PF_ctrl_cap: #-1 for capacitive, 1 for inductive systems
            self.reactance= -1
        else:
            if control_modus == ControlModusEnum.PF_ctrl:
                logger.warning(
                    f"Ambivalent reactive power flow direction for Controller {self.index}, using inductive direction.\n")
                self.control_modus = ControlModusEnum.PF_ctrl_ind
            self.reactance = 1

        if self.control_modus in ControlModusEnum.pf_modes(): #checking cos(phi) limits
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
                    _, input_variable_temp_p = _detect_read_write_flag(net, self.input_element,input_index,
                                                                       input_variable_p)
                self.input_variable_p.append(input_variable_temp_p)  #read flag p not necessary, flag same as Q variables
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
                self._deprecation_warned = True  #only one message that voltage ctrl is deprecated
            return self.voltage_ctrl
        if name == 'bus_idx':
            if not hasattr(self, '_deprecation_warned_bus_idx'):
                logger.warning(
                    f"Variable 'bus_idx' in Binary Search Control {self.index} for control_modus V_ctrl is deprecated. "
                    f"Give index of controlled bus to input_element_index. Input_variable must be 'vm_pu' and"
                    f" input_element 'res_bus'"
                )
                self._deprecation_warned_bus_idx = True  #only one warning about bus_idx deprecation
            return self.input_element_index
        raise AttributeError(f"{self.__class__.__name__!r} has no attribute {name!r}")


    # derived per-run state (_vstate, cached droop links) must never end up in saved nets
    json_excludes = Controller.json_excludes + ["_vstate", "_linked_droop_objs"]

    # maps the result table an input measurement is taken from to the element table that
    # carries the in_service information
    _RES_TO_ELEMENT = {"res_line": "line", "res_trafo": "trafo", "res_trafo3w": "trafo3w",
                       "res_switch": "switch", "res_impedance": "impedance", "res_bus": "bus",
                       "res_gen": "gen"}

    def initialize_control(self, net):
        if getattr(self, 'stations', None):
            self._initialize_stations(net)
            return
        output_element_index = np.atleast_1d(self.output_element_index)[0] if self.write_flag == 'single_index' else \
                self.output_element_index #ruggedize for single index
        self.output_values = read_from_net(net, self.output_element, output_element_index, self.output_variable,
                                           self.write_flag)
        self.output_values_old = None
        self.output_adjustable = np.array([False if not distribution else service
                                            for distribution, service in zip(np.atleast_1d(self.output_values_distribution),
                                                                            np.atleast_1d(self.output_element_in_service))],
                                            dtype=bool)
        self._build_vstate(net)

    def _build_vstate(self, net):
        """Precompute positional indices and cached lookups for the per-iteration hot path.

        Rebuilt at the beginning of every run_control (initialize_control) and lazily for
        controllers restored from JSON (from_dict does not call __init__). Never serialized
        (see json_excludes). Falls back to the legacy label-based access paths whenever the
        preconditions for positional access are not met.
        """
        vs = {}
        # inputs: positional indices into the element table (for in_service masks)
        element_table = self._RES_TO_ELEMENT.get(self.input_element)
        input_idx = ([] if self.input_element_index is None
                     else list(np.atleast_1d(self.input_element_index)))
        read_flags = list(np.atleast_1d(getattr(self, 'read_flag', [])))
        fast_input = (element_table is not None and element_table in net
                      and len(read_flags) == len(input_idx)
                      and all(flag == 'single_index' for flag in read_flags))
        if fast_input:
            pos = net[element_table].index.get_indexer(input_idx)
            fast_input = not np.any(pos == -1)
            vs['input_pos'] = pos
        vs['fast_input'] = fast_input
        vs['input_idx'] = input_idx
        vs['input_element_table'] = element_table
        vs['input_in_service_col'] = 'closed' if element_table == 'switch' else 'in_service'
        # positional indices into the result table (for value reads) are resolved lazily on
        # the first read because result tables are only guaranteed to exist after a powerflow
        vs['res_pos'] = None
        # outputs: positional indices into the output element table
        output_idx = list(np.atleast_1d(self.output_element_index))
        fast_output = (self.output_element in ('gen', 'sgen', 'shunt')
                       and self.output_element in net)
        if fast_output:
            pos = net[self.output_element].index.get_indexer(output_idx)
            fast_output = not np.any(pos == -1)
            vs['output_pos'] = pos
        vs['fast_output'] = fast_output
        self._vstate = vs
        # cache controllers linked to this one (droop controllers reference their binary
        # search controller via controller_idx); avoids an O(n_controllers) scan of
        # net.controller in every is_converged call. controller_idx is looked up via
        # __dict__ because getattr would fall through to the (slow) __getattr__ shim on
        # every controller that has no controller_idx
        self._linked_droop_objs = [
            obj for obj in net.controller['object'].values
            if getattr(obj, '__dict__', {}).get('controller_idx') == self.index
            and obj is not self]
        return vs

    def _refresh_in_service(self, net, vs):
        """Update input_element_in_service / output_element_in_service from the net."""
        if vs['fast_input']:
            column = vs['input_in_service_col']
            self.input_element_in_service = list(
                net[vs['input_element_table']][column].values[vs['input_pos']])
        else:
            self.input_element_in_service = []
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
        if vs['fast_output']:
            self.output_element_in_service = list(
                net[self.output_element]['in_service'].values[vs['output_pos']])
        else:
            self.output_element_in_service = []
            for output_index in np.atleast_1d(self.output_element_index):
                if self.output_element == "gen":
                    self.output_element_in_service.append(net.gen.in_service[output_index])
                elif self.output_element == "sgen":
                    self.output_element_in_service.append(net.sgen.in_service[output_index])
                elif self.output_element == "shunt":
                    self.output_element_in_service.append(net.shunt.in_service[output_index])

    def _read_input_values(self, net, vs, need_p):
        """Read the measurement values of all in-service input elements.

        Returns plain lists in the same order as the legacy per-element read loop, so all
        downstream arithmetic (sign multiplication, summation) is unchanged.
        """
        input_values, p_input_values = [], []
        fast_read = vs['fast_input']
        if fast_read:
            res_pos = vs['res_pos']
            if res_pos is None:
                res_pos = net[self.input_element].index.get_indexer(vs['input_idx'])
                if np.any(res_pos == -1):
                    fast_read = False
                    vs['fast_input'] = False
                else:
                    vs['res_pos'] = res_pos
        if fast_read:
            res_table = net[self.input_element]
            columns = {}
            for counter, pos in enumerate(vs['res_pos']):
                if not self.input_element_in_service[counter]:
                    continue
                column = self.input_variable[counter]
                values = columns.get(column)
                if values is None:
                    values = columns[column] = res_table[column].values
                input_values.append(values[pos])
                if need_p:
                    p_column = self.input_variable_p[counter]
                    p_values = columns.get(p_column)
                    if p_values is None:
                        p_values = columns[p_column] = res_table[p_column].values
                    p_input_values.append(p_values[pos])
        else:
            counter = 0
            for input_index in self.input_element_index:
                if self.input_element_in_service[counter]:
                    input_values.append(read_from_net(net, self.input_element, input_index,
                                                      self.input_variable[counter], self.read_flag[counter]))
                    if need_p:
                        p_input_values.append(read_from_net(net, self.input_element, input_index,
                                                            self.input_variable_p[counter], self.read_flag[counter]))
                counter += 1
        return input_values, p_input_values

    def _limits_reached_else_refresh(self, log_prefix):
        """Handle the shared "are any outputs still adjustable" block of all control modi.

        Returns True (and sets converged) if every output element has reached its reactive
        power limit; otherwise drops out-of-service outputs from output_adjustable and
        renormalizes the distribution.
        """
        if not any(self.output_adjustable):
            logging.info(log_prefix + 'All stations controlled by %s reached reactive power limits.' % self.name)
            self.converged = True
            return True
        self.output_adjustable = np.array([in_service and adjustable for in_service, adjustable in zip(
            self.output_element_in_service, self.output_adjustable)], dtype=bool)
        self._normalize_distribution_in_service()
        return False

    def is_converged(self, net):
        """
        Actual implementation of the convergence criteria: If controller is applied, it can stop
        """
        # if controller not in_service, return True
        self.in_service = net.controller.in_service[self.index]
        if not self.in_service:
            self.converged = True
            return self.converged
        if getattr(self, 'stations', None):
            return self._is_converged_stations(net)
        # derived state is built by initialize_control; build lazily for controllers
        # restored from JSON or used outside run_control
        vs = getattr(self, '_vstate', None)
        if vs is None:
            vs = self._build_vstate(net)
        ###updating input & output elements in service lists
        self._refresh_in_service(net, vs)

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
                    f' at index {np.array(self.output_element_index)}'
                    f' will provide 100% of the reactive power in Controller {self.index}.\n')
            else:
                in_service_mask = np.asarray(self.output_element_in_service, dtype=bool)
                logger.warning(
                    f'Reactive Power Distribution for one output element cannot be modified. The active '
                    f'{self.output_element} at index '
                    f'{np.asarray(self.output_element_index)[in_service_mask]} will provide 100% of the'
                    f' reactive power in Controller {self.index}.\n')

        # read input values
        input_values = [] #reactive power q
        p_input_values = [] #active power p for power factor controllers
        if self.input_element != 'res_bus':
            need_p = (self.control_modus in ControlModusEnum.pf_modes()
                      or self.control_modus == ControlModusEnum.tan_phi_ctrl)
            input_values, p_input_values = self._read_input_values(net, vs, need_p)
            input_values = (self.input_sign * np.asarray(input_values)).tolist()
            if need_p:
                p_input_values = (self.input_sign * np.asarray(p_input_values)).tolist()
        # compare old and new set values
        if self.control_modus in ControlModusEnum.q_modes() or (self.control_modus in ControlModusEnum.v_modes()
                                                                and self.input_element_index is None):
            if self.control_modus in ControlModusEnum.v_modes():
                logger.warning('Missing attribute self.input_element_index, defaulting to Q_ctrl\n')
                self.control_modus = ControlModusEnum.q_ctrl
            self.diff_old = self.diff
            if self._limits_reached_else_refresh(''):
                return self.converged

            self.diff = self.set_point - sum(input_values)
            self.converged = np.all(np.abs(self.diff) < self.tol)

        elif self.control_modus in ControlModusEnum.pf_modes():#capacitive => reactance = -1, inductive => reactance = 1
            if self.control_modus == ControlModusEnum.PF_ctrl_ind:
                self.reactance = 1
            else:
                self.control_modus = ControlModusEnum.PF_ctrl_cap
                self.reactance = -1

            self.diff_old = self.diff
            if self._limits_reached_else_refresh('PF_ctrl: '):
                return self.converged
            set_point = self.set_point
            if -0.012 < set_point < 0.012: #clip set_point to handle pf=0, without mutating self.set_point
                set_point = 0.012 if set_point >= 0 else -0.012
                if not vs.get('pf_clip_warned', False):
                    vs['pf_clip_warned'] = True
                    logger.warning(f"Power factor calculation with set_point 0 not possible with BSC {self.index}.\n"
                                   f"Maximizing Q output by clipping set_point to {set_point}\n")
            q_set = self.reactance * sum(p_input_values)/len(p_input_values) * (np.tan(np.arccos(set_point)))
            self.diff = q_set - sum(input_values)/len(input_values)
            self.converged = np.all(np.abs(self.diff)<self.tol)

        elif self.control_modus == ControlModusEnum.tan_phi_ctrl:
            self.diff_old = self.diff
            if self._limits_reached_else_refresh('tan(phi)_ctrl: '):
                return self.converged

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
                if self.input_element != 'res_bus':  # and not any(getattr(net.controller.at[x, 'object'], 'controller_idx', False) ==
                    if hasattr(self, 'bus_idx') and getattr(self, 'bus_idx') is not None:  # legacy
                        self.diff_old = self.diff
                        if self._limits_reached_else_refresh('Q_ctrl: '):
                            return self.converged

                        self.diff = self.set_point - net.res_bus.vm_pu.at[self.bus_idx]
                        self.converged = np.all(np.abs(self.diff) < self.tol)
                    else:
                        logger.warning(f"'input_element' must be 'res_bus' for V_ctrl not {self.input_element}, correcting.")
                        self.input_element = 'res_bus'
                        if np.atleast_1d(self.input_variable)[0] != 'vm_pu':
                            logger.warning(f"'input_variable' must be 'vm_pu' for V_ctrl not {self.input_variable}, correcting ")
                            self.input_variable = 'vm_pu'
                        self.diff_old = self.diff  # V_ctrl
                        if self._limits_reached_else_refresh('V_ctrl: '):
                            return self.converged

                        self.diff = self.set_point - net.res_bus.vm_pu.at[np.atleast_1d(self.input_element_index)[0]]
                        self.converged = np.all(np.abs(self.diff) < self.tol)
                else:
                    self.diff_old = self.diff  # V_ctrl
                    if self._limits_reached_else_refresh('V_ctrl: '):
                        return self.converged

                    self.diff = self.set_point - net.res_bus.vm_pu.at[np.atleast_1d(self.input_element_index)[0]]
                    self.converged = np.all(np.abs(self.diff) < self.tol)
            else:
                if self.control_modus not in ControlModusEnum.q_modes():
                    logger.warning(f"No Controller Modus specified for Controller {self.index}, using Q_ctrl.\n"
                                   "Please specify 'control_modus' ('Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan(phi)_ctrl')\n")
                    self.control_modus = ControlModusEnum.q_ctrl
                self.diff_old = self.diff  # Q_ctrl
                if self._limits_reached_else_refresh('Q_ctrl: '):
                    return self.converged

                self.diff = self.set_point - sum(input_values)
                self.converged = np.all(np.abs(self.diff) < self.tol)
        ###check convergence of linked droop controllers (if any); the list is cached in
        ###_build_vstate to avoid scanning all controllers in every iteration
        if self.converged and any(not getattr(obj, 'converged', True)
                                  for obj in self._linked_droop_objs):
            self.converged = False
        return self.converged

    def control_step(self, net):
        if getattr(self, 'stations', None):
            self._control_step_stations(net)
            return
        self._binary_search_control_step(net)

    def _binary_search_control_step(self, net):
        if not self.in_service:
            return
        vs = getattr(self, '_vstate', None)
        if vs is None:
            vs = self._build_vstate(net)
        damping = float(getattr(self, 'damping_factor', 1.0) or 1.0)
        if self.output_values_old is None:  # first step
            # is ok that values are set for all stations even though they are out of service or not adjustable --> following step will correct this
            # output_values keeps one entry per output element (also out-of-service ones);
            # out-of-service entries are excluded when writing to the net
            values = np.atleast_1d(self.output_values).astype(np.float64)
            probe_total = self._initial_probe_total(values, damping)
            if probe_total is None:
                # V modi: the voltage response in MVAr/pu is grid specific, keep the small
                # legacy probe (update_method="jacobian" will compute the true sensitivity)
                self.output_values_old, self.output_values = (values, values + 1e-3)
            else:
                distribution = np.atleast_1d(self.output_values_distribution).astype(np.float64)
                self.output_values_old, self.output_values = (values, values + probe_total * distribution)
            positions_not_adjustable = [i for i, val in enumerate(self.output_adjustable) if not val]
            for i in positions_not_adjustable:
                if self.output_values_distribution[i]==0 or not self.output_element_in_service[i] :
                    self.output_values[i] = 0
                else:
                    continue
        else:  #second step
            # another controller or enforce_q_lims may have modified the written values in
            # the net since the last step -- the powerflow saw the net values, so they are
            # the true evaluation point of the secant
            self._resync_output_values(net, vs)
            x_total = self._safeguarded_secant_total(vs, damping)
            distribution = np.atleast_1d(self.output_values_distribution).astype(np.float64)
            x = x_total * distribution

            if not all(self.output_adjustable) and net._options.get('enforce_q_lims', False):
                positions_adjustable = [i for i, val in enumerate(self.output_adjustable) if val]  # gives which is/are adjustable
                positions_not_adjustable = [i for i, val in enumerate(self.output_adjustable) if not val]  # can be one or multiple ## gives which is/are not adjustable anymore

                sum_adjustable = sum(x) - sum(self.output_values[positions_not_adjustable])  # stations that are still adjustable, rest of the power must be achieved
                x[positions_adjustable] = sum_adjustable * self.output_values_distribution[positions_adjustable]

                for i in positions_not_adjustable:
                    if self.output_element_in_service[i]:
                        x[i] = self.output_values[i]  # reset value to q_limit
                    else:
                        x[i] = 0  # reset value to 0 because station is out of service

            else:
                x = sum(x) * self.output_values_distribution

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
                        logging.info(
                            f"Station {self.output_element_index} controlled by {self.name} reached a reactive power "
                            f"limit."
                        )
                        self.output_adjustable = np.array([False], dtype=bool)
                        if reached_min_qmvar:
                            self.output_values = self.output_min_q_mvar
                        elif reached_max_qmvar:
                            self.output_values = self.output_max_q_mvar
                    else:
                        self.output_values = x
            else:
                self.output_values_old, self.output_values = self.output_values, x
        ### write new set of Q values to output elements (out-of-service outputs excluded)###
        in_service_mask = np.asarray(self.output_element_in_service, dtype=bool)
        values = np.atleast_1d(self.output_values)
        if len(values) == len(in_service_mask):
            values = values[in_service_mask]
        if self.write_flag == 'single_index':
            output_element_index = list(np.atleast_1d(self.output_element_index)[in_service_mask])[0]
            output_values = list(values)[0]
        else:
            output_element_index = list(np.array(self.output_element_index)[in_service_mask])
            output_values = list(values)
        write_to_net(net, self.output_element, output_element_index, self.output_variable, output_values, self.write_flag)

    def _resync_output_values(self, net, vs):
        """Align the internal output state with the values currently in the net tables."""
        if not vs.get('fast_output', False) or not isinstance(self.output_variable, str):
            return
        current = net[self.output_element][self.output_variable].values[vs['output_pos']]
        values = np.atleast_1d(self.output_values).astype(np.float64)
        mask = np.asarray(self.output_element_in_service, dtype=bool)
        if len(current) != len(values) or len(mask) != len(values):
            return
        values[mask] = current[mask]
        self.output_values = values

    def _initial_probe_total(self, values, damping):
        """Total first-step perturbation for Q-type control modi, or None for the legacy probe.

        For Q/PF/tan(phi) control the measured quantity follows the summed station output
        nearly 1:1, so a residual-sized first step is already close to the Newton step; the
        secant update afterwards corrects the remaining slope error. For V modi the voltage
        response in MVAr/pu is grid specific, so None is returned and the caller keeps the
        small legacy probe.
        """
        if self.control_modus in ControlModusEnum.v_modes():
            return None
        if self.diff is None or np.ndim(self.diff) != 0 or not np.isfinite(self.diff):
            return None
        cap = 2.0 * float(np.abs(values).sum()) + 50.0
        probe_total = float(np.clip(damping * float(self.diff), -cap, cap))
        if abs(probe_total) < 1e-3:
            probe_total = 1e-3  # keep the perturbation measurable for the secant slope
        return probe_total

    def _safeguarded_secant_total(self, vs, damping):
        """Next total station output from a bracketing-safeguarded secant update.

        The update works on the summed station output. As soon as two iterates with opposite
        residual sign are known, the solution is bracketed and an Illinois-damped regula
        falsi keeps all further iterates inside the bracket. Without a bracket, a (damped)
        secant step with a step-size cap is taken; a flat measurement response takes a
        bounded unit-slope step instead of dividing by a near-zero slope.
        """
        solver = vs.setdefault('solver', self._new_solver_state(self.set_point))
        values = np.atleast_1d(self.output_values).astype(np.float64)
        values_old = np.atleast_1d(self.output_values_old).astype(np.float64)
        total = float(values.sum())
        total_old = float(values_old.sum())
        try:
            f = float(self.diff)
            f_old = f if self.diff_old is None else float(self.diff_old)
        except (TypeError, ValueError):
            # non-scalar residual: legacy per-element secant as fallback
            step_diff = self.diff - self.diff_old
            x = values - self.diff * (values - values_old) / np.where(step_diff == 0, 1e-6, step_diff)
            cap = 2 * (np.abs(values) + 1e-6) + 50
            return float(np.sum(values + np.clip(x - values, -cap, cap)))
        cap = 2.0 * float(np.abs(values).sum()) + 50.0
        return self._secant_core(solver, f, f_old, total, total_old, damping,
                                 self.set_point, cap, "%s (index %s)" % (self.name, self.index))

    @staticmethod
    def _new_solver_state(set_point):
        return {'lo': None, 'hi': None, 'side': 0, 'best': None, 'stall': 0,
                'stall_warned': False, 'slope': None, 'set_point': set_point}

    @staticmethod
    def _secant_core(solver, f, f_old, total, total_old, damping, set_point, cap, label):
        """Bracketing-safeguarded secant update on a scalar residual, see
        _safeguarded_secant_total. ``solver`` carries the state between calls."""
        # a changed set point (e.g. written by a chained droop controller) changes the
        # residual function, previously collected bracket points are no longer valid
        if solver['set_point'] != set_point:
            solver['lo'] = solver['hi'] = None
            solver['side'] = 0
            solver['set_point'] = set_point

        # remember the most recent meaningful secant slope (df/dQ_total); used when the last
        # two iterates collapse onto each other and no local slope can be computed
        if abs(total - total_old) > 1e-12 and abs(f - f_old) > 1e-12 * max(1.0, abs(f)):
            solver['slope'] = (f - f_old) / (total - total_old)

        # stagnation diagnostics; a frozen residual with an active bracket means the bracket
        # was collected while other controllers still moved the operating point (stale) --
        # discard it and continue with plain secant steps on fresh information
        if solver['best'] is None or abs(f) < 0.9 * solver['best']:
            solver['best'] = abs(f) if solver['best'] is None else min(abs(f), solver['best'])
            solver['stall'] = 0
        else:
            solver['stall'] += 1
            if solver['stall'] >= 3 and solver['lo'] is not None:
                logger.debug("BinarySearchControl %s: discarding stale bracket" % label)
                solver['lo'] = solver['hi'] = None
                solver['side'] = 0
                solver['best'] = abs(f)
                solver['stall'] = 0
            elif solver['stall'] >= 8 and not solver['stall_warned']:
                solver['stall_warned'] = True
                logger.warning(
                    "BinarySearchControl %s: residual %.3g is not decreasing "
                    "after %d control steps" % (label, abs(f), solver['stall']))

        # maintain the bracket around the zero crossing
        if solver['lo'] is None:
            if f_old * f < 0:
                first, second = (total_old, f_old), (total, f)
                solver['lo'], solver['hi'] = ((first, second) if first[0] <= second[0]
                                              else (second, first))
                solver['side'] = 0
        else:
            lo_x, lo_f = solver['lo']
            hi_x, hi_f = solver['hi']
            if f == 0.0:
                return total
            if f * lo_f > 0:
                solver['lo'] = (total, f)
                if solver['side'] == -1:
                    solver['hi'] = (hi_x, hi_f * 0.5)  # Illinois damping
                solver['side'] = -1
            elif f * hi_f > 0:
                solver['hi'] = (total, f)
                if solver['side'] == 1:
                    solver['lo'] = (lo_x, lo_f * 0.5)  # Illinois damping
                solver['side'] = 1

        if solver['lo'] is not None:
            lo_x, lo_f = solver['lo']
            hi_x, hi_f = solver['hi']
            x_new = (lo_x * hi_f - hi_x * lo_f) / (hi_f - lo_f)
            if not (min(lo_x, hi_x) < x_new < max(lo_x, hi_x)):
                x_new = 0.5 * (lo_x + hi_x)  # numerical safety: bisect
            return x_new

        # no bracket yet: plain secant with a step-size cap. damping_factor is deliberately
        # not applied to regular secant steps (it would slow every well-behaved controller);
        # it only softens the fallback steps below and the first probe
        step_diff = f - f_old
        if abs(step_diff) <= 1e-12 * max(1.0, abs(f)) or abs(total - total_old) <= 1e-12:
            if solver['stall'] >= 5:
                # the measurement does not respond to the output at all: hold the position
                # instead of pushing ever more reactive power into the grid (cumulative
                # fallback steps would eventually make the powerflow itself collapse);
                # run_control reports ControllerNotConverged via max_iter
                return total
            if solver['slope']:
                # local slope unavailable (iterates collapsed): Newton with remembered slope
                x_new = total - damping * f / solver['slope']
            else:
                x_new = total + damping * f  # flat response: bounded unit-slope step
        else:
            x_new = total - f * (total - total_old) / step_diff
        return total + float(np.clip(x_new - total, -cap, cap))

    # ------------------------------------------------------------------------------------
    # multi-station mode: one controller instance manages many stations, each with its own
    # control modus, set point, measurement, outputs and (optionally) droop characteristic.
    # Droop is part of the station residual (single fixed-point loop), not a chained
    # controller. Created via BinarySearchControl.for_stations; single-station controllers
    # created through __init__ keep the legacy code path above.
    # ------------------------------------------------------------------------------------

    @classmethod
    def for_stations(cls, net, stations, output_element="sgen", output_variable="q_mvar",
                     tol=1e-3, in_service=True, order=0, level=0, name="",
                     update_method="secant", drop_same_existing_ctrl=False,
                     matching_params=None, **kwargs):
        """Create one BinarySearchControl instance controlling multiple stations.

        Parameters
        ----------
        net : pandapowerNet
        stations : list of dict
            One dict per station with the keys:

            - ``control_modus`` (str): ``"Q_ctrl"``, ``"V_ctrl"``, ``"PF_ctrl_ind"``,
              ``"PF_ctrl_cap"``, ``"tan_phi_ctrl"``, ``"Q_ctrl_V_droop"``,
              ``"V_ctrl_Q_droop"`` or ``"V_ctrl_Q_droop_local"``
            - ``set_point`` (float): reactive power / voltage / power factor / tan(phi)
              set point (base set point for droop modi)
            - ``input_element`` (str): result table of the measurement, e.g. ``"res_line"``,
              ``"res_trafo"``; ``"res_bus"`` for plain V_ctrl
            - ``input_variable`` (str or list of str): measured column(s), e.g.
              ``"q_to_mvar"``; ``"vm_pu"`` for plain V_ctrl
            - ``input_element_index`` (int or list of int)
            - ``input_inverted`` (bool or list of bool, optional)
            - ``output_element_index`` (list of int): controlled elements in the (shared)
              output table
            - ``output_values_distribution`` (list of float): Q distribution among outputs
            - ``tol`` (float, optional): per-station tolerance override
            - ``name`` (str, optional)
            - ``droop`` (dict, required for droop modi):
              ``q_droop_mvar`` (Mvar/pu), ``bus_idx`` (measured bus),
              ``vm_set_lb``/``vm_set_ub`` (deadband, Q_ctrl_V_droop),
              ``vm_set_pu`` (no-deadband voltage reference, Q_ctrl_V_droop),
              ``q_set_mvar`` (local Q reference, V_ctrl_Q_droop_local)
        output_element : str
            Output table shared by all stations of this instance (``"sgen"``, ``"gen"`` or
            ``"shunt"``). Stations with different output tables need separate instances.
        output_variable : str
            Written column, e.g. ``"q_mvar"`` or ``"step"``.
        update_method : str
            ``"secant"`` (default) or ``"jacobian"``. With ``"jacobian"``, plain ``V_ctrl``
            stations take coupled Newton steps based on the dVm/dQ sensitivities from the
            powerflow Jacobian (captures the interaction of electrically close stations and
            reduces the number of powerflows); all other modi and any failure case
            automatically fall back to the safeguarded secant update.
        """
        self = cls.__new__(cls)
        Controller.__init__(self, net, in_service=in_service, order=order, level=level,
                            drop_same_existing_ctrl=drop_same_existing_ctrl,
                            matching_params=matching_params)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.name = name
        self.output_element = output_element
        self.output_variable = output_variable
        self.write_flag = 'loc'
        self.tol = tol
        self.in_service = in_service
        self.update_method = update_method
        self.converged = False
        # harmless flat attributes for __str__ / external inspection
        self.input_element = 'stations'
        self.input_variable = []
        self.output_element_index = []
        self.set_point = None
        self.diff = None
        self.diff_old = None
        self.stations = [cls._normalize_station(s, k, tol) for k, s in enumerate(stations)]
        return self

    @staticmethod
    def _normalize_station(station, position, default_tol):
        """Validate a station dict and normalize it to canonical (JSON-safe) form."""
        s = dict(station)
        try:
            modus = ControlModusEnum(s.get('control_modus'))
        except ValueError:
            raise UserWarning(f"station {position}: unknown control_modus "
                              f"{s.get('control_modus')!r}")
        s['control_modus'] = modus.value
        if 'set_point' not in s:
            raise UserWarning(f"station {position}: set_point is required")
        s['set_point'] = float(s['set_point'])
        if modus in ControlModusEnum.pf_modes() and abs(s['set_point']) > 1:
            raise UserWarning(f"station {position}: power factor set point out of range [-1, 1]")
        s['tol'] = float(s.get('tol', default_tol))
        s['input_element_index'] = [int(i) for i in np.atleast_1d(s['input_element_index'])]
        n_inputs = len(s['input_element_index'])
        variables = s['input_variable']
        s['input_variable'] = ([variables] * n_inputs if isinstance(variables, str)
                               else list(variables))
        if len(s['input_variable']) != n_inputs:
            raise UserWarning(f"station {position}: input_variable and input_element_index "
                              f"lengths differ")
        if (modus in ControlModusEnum.pf_modes() or modus == ControlModusEnum.tan_phi_ctrl) \
                and s['input_element'] == 'res_bus':
            raise UserWarning(f"station {position}: {modus.value} needs a branch measurement, "
                              f"not res_bus")
        inverted = np.atleast_1d(s.get('input_inverted', False))
        if len(inverted) == 1:
            inverted = np.repeat(inverted, n_inputs)
        if len(inverted) != n_inputs:
            raise UserWarning(f"station {position}: input_inverted and input_element_index "
                              f"lengths differ")
        s['input_sign'] = [-1.0 if inv else 1.0 for inv in inverted]
        s.pop('input_inverted', None)
        s['output_element_index'] = [int(i) for i in np.atleast_1d(s['output_element_index'])]
        distribution = np.asarray(
            np.atleast_1d(s.get('output_values_distribution',
                                [1.0] * len(s['output_element_index']))), dtype=np.float64)
        if len(distribution) != len(s['output_element_index']):
            raise UserWarning(f"station {position}: output_values_distribution and "
                              f"output_element_index lengths differ")
        s['output_values_distribution'] = [float(v) for v in distribution / distribution.sum()]
        droop = s.get('droop')
        if modus in ControlModusEnum.droop_modes():
            if not droop or 'q_droop_mvar' not in droop or 'bus_idx' not in droop:
                raise UserWarning(f"station {position}: droop modus {modus.value} requires a "
                                  f"droop dict with q_droop_mvar and bus_idx")
            if (modus == ControlModusEnum.q_ctrl_v_droop
                    and ('vm_set_lb' in droop) != ('vm_set_ub' in droop)):
                raise UserWarning(f"station {position}: Q_ctrl_V_droop needs both or none of "
                                  f"vm_set_lb/vm_set_ub")
            if (modus == ControlModusEnum.q_ctrl_v_droop and 'vm_set_lb' not in droop
                    and 'vm_set_pu' not in droop):
                raise UserWarning(f"station {position}: Q_ctrl_V_droop without deadband needs "
                                  f"a vm_set_pu voltage reference")
        elif modus in ControlModusEnum.v_modes():
            if s['input_element'] != 'res_bus' and (not droop or 'bus_idx' not in droop):
                raise UserWarning(f"station {position}: V_ctrl needs input_element 'res_bus' "
                                  f"or a droop dict with bus_idx")
        return s

    def _initialize_stations(self, net):
        """Build the runtime state (positional indices, solver state) for all stations."""
        vs = {'stations': [], 'res_resolved': False}
        output_table = net[self.output_element]
        output_values = output_table[self.output_variable].values
        for k, s in enumerate(self.stations):
            rt = {'cfg': s, 'label': s.get('name') or f"{self.name}[{k}]"}
            rt['modus'] = ControlModusEnum(s['control_modus'])
            rt['set_point'] = s['set_point']
            rt['tol'] = s['tol']
            rt['droop'] = s.get('droop')
            rt['out_idx'] = np.asarray(s['output_element_index'])
            rt['out_pos'] = output_table.index.get_indexer(rt['out_idx'])
            if np.any(rt['out_pos'] == -1):
                raise UserWarning(f"station {rt['label']}: output element(s) "
                                  f"{s['output_element_index']} not found in "
                                  f"{self.output_element}")
            rt['dist_base'] = np.asarray(s['output_values_distribution'], dtype=np.float64)
            rt['at_limit'] = np.zeros(len(rt['out_idx']), dtype=bool)
            rt['values'] = output_values[rt['out_pos']].astype(np.float64)
            rt['values_old'] = None
            rt['solver'] = self._new_solver_state(rt['set_point'])
            rt['f'] = None
            rt['f_old'] = None
            rt['converged'] = False
            rt['disabled'] = False
            rt['warned'] = False
            element_table = self._RES_TO_ELEMENT.get(s['input_element'])
            if element_table is None or element_table not in net:
                raise UserWarning(f"station {rt['label']}: unsupported input_element "
                                  f"{s['input_element']!r}")
            rt['in_res'] = s['input_element']
            rt['in_element_table'] = element_table
            rt['in_service_col'] = 'closed' if element_table == 'switch' else 'in_service'
            rt['in_idx'] = list(s['input_element_index'])
            rt['in_pos'] = net[element_table].index.get_indexer(rt['in_idx'])
            if np.any(rt['in_pos'] == -1):
                raise UserWarning(f"station {rt['label']}: input element(s) {rt['in_idx']} "
                                  f"not found in {element_table}")
            rt['in_cols'] = list(s['input_variable'])
            rt['in_sign'] = np.asarray(s['input_sign'], dtype=np.float64)
            rt['res_pos'] = None  # resolved lazily against the result table
            if rt['modus'] in ControlModusEnum.pf_modes() or rt['modus'] == ControlModusEnum.tan_phi_ctrl:
                rt['p_cols'] = [c.replace('q', 'p').replace('var', 'w') for c in rt['in_cols']]
            rt['reactance'] = -1.0 if rt['modus'] == ControlModusEnum.PF_ctrl_cap else 1.0
            # controlled bus (V modi and droop modi)
            bus = None
            if rt['droop'] and 'bus_idx' in rt['droop']:
                bus = rt['droop']['bus_idx']
            elif rt['modus'] in ControlModusEnum.v_modes():
                bus = rt['in_idx'][0]
            rt['bus'] = bus
            rt['bus_pos'] = None if bus is None else net.bus.index.get_loc(bus)
            vs['stations'].append(rt)
        self._vstate = vs
        self._linked_droop_objs = []

    @staticmethod
    def _station_effective_residual(rt, q_meas, vm):
        """Residual of one station; droop characteristics are folded into the residual."""
        modus = rt['modus']
        set_point = rt['set_point']
        droop = rt['droop']
        if modus == ControlModusEnum.q_ctrl:
            return set_point - q_meas
        if modus == ControlModusEnum.q_ctrl_v_droop:
            k = droop['q_droop_mvar']
            if droop.get('vm_set_lb') is not None and droop.get('vm_set_ub') is not None:
                if vm > droop['vm_set_ub']:
                    q_set = set_point - (droop['vm_set_ub'] - vm) * k
                elif vm < droop['vm_set_lb']:
                    q_set = set_point + (droop['vm_set_lb'] - vm) * k
                else:
                    q_set = set_point
            else:
                q_set = set_point + (droop['vm_set_pu'] - vm) * k
            return q_set - q_meas
        if modus == ControlModusEnum.v_ctrl:
            return set_point - vm
        if modus == ControlModusEnum.v_ctrl_q_droop:
            return set_point + q_meas / droop['q_droop_mvar'] - vm
        if modus == ControlModusEnum.v_ctrl_q_droop_local:
            q_set = droop.get('q_set_mvar', 0.0) or 0.0
            return set_point - (q_meas - q_set) / droop['q_droop_mvar'] - vm
        raise UserWarning(f"unsupported control modus {modus} in multi-station mode")

    def _is_converged_stations(self, net):
        vs = getattr(self, '_vstate', None)
        if vs is None or 'stations' not in vs:
            self._initialize_stations(net)
            vs = self._vstate
        cache = {}

        def col(table, column):
            key = (table, column)
            if key not in cache:
                cache[key] = net[table][column].values
            return cache[key]

        all_converged = True
        for rt in vs['stations']:
            if rt['disabled']:
                continue
            out_in_service = col(self.output_element, 'in_service')[rt['out_pos']].astype(bool)
            in_mask = col(rt['in_element_table'],
                          rt['in_service_col'])[rt['in_pos']].astype(bool)
            if not out_in_service.any() or not in_mask.any():
                if not rt['warned']:
                    logger.warning("station %s: all input or output elements out of service, "
                                   "skipping station" % rt['label'])
                    rt['warned'] = True
                rt['disabled'] = True
                continue
            rt['out_in_service'] = out_in_service
            rt['in_mask'] = in_mask
            # distribution over outputs that are in service and not at a Q limit
            dist = rt['dist_base'] * out_in_service * ~rt['at_limit']
            total_dist = dist.sum()
            adjustable = dist != 0
            if total_dist > 0:
                dist = dist / total_dist
            rt['dist'] = dist
            rt['adjustable'] = adjustable
            if not adjustable.any():
                if not rt['warned']:
                    logging.info('All outputs of station %s reached their reactive power '
                                 'limits.' % rt['label'])
                    rt['warned'] = True
                rt['converged'] = True
                continue

            if rt['in_res'] == 'res_bus':
                q_meas = 0.0
            else:
                res_pos = rt['res_pos']
                if res_pos is None:
                    res_pos = net[rt['in_res']].index.get_indexer(rt['in_idx'])
                    rt['res_pos'] = res_pos
                q_meas = 0.0
                p_meas = 0.0
                n_active = 0
                for i, pos in enumerate(res_pos):
                    if not in_mask[i]:
                        continue
                    q_meas += rt['in_sign'][i] * col(rt['in_res'], rt['in_cols'][i])[pos]
                    if 'p_cols' in rt:
                        p_meas += rt['in_sign'][i] * col(rt['in_res'], rt['p_cols'][i])[pos]
                    n_active += 1
            vm = None if rt['bus_pos'] is None else col('res_bus', 'vm_pu')[rt['bus_pos']]
            modus = rt['modus']
            if modus in ControlModusEnum.pf_modes():
                set_point = rt['set_point']
                if -0.012 < set_point < 0.012:
                    set_point = 0.012 if set_point >= 0 else -0.012
                q_set = rt['reactance'] * p_meas / n_active * np.tan(np.arccos(set_point))
                f = q_set - q_meas / n_active
            elif modus == ControlModusEnum.tan_phi_ctrl:
                f = p_meas / n_active * rt['set_point'] - q_meas / n_active
            else:
                f = self._station_effective_residual(rt, q_meas, vm)
            rt['f_old'], rt['f'] = rt['f'], float(f)
            rt['converged'] = abs(rt['f']) < rt['tol']
            if not rt['converged']:
                all_converged = False
        self.converged = all_converged
        return self.converged

    def _jacobian_deltas(self, net, vs, damping):
        """Coupled Newton steps {station position: dQ_total} for plain V_ctrl stations.

        Builds the cross-station sensitivity matrix M[s, t] = dVm(bus_s)/dQ_total(t) from the
        Newton-Raphson Jacobian of the last powerflow and solves M * dQ = r for all eligible
        stations simultaneously -- this captures the interaction between electrically close
        stations that makes independent per-station updates oscillate. Any failure returns {}
        and the caller falls back to the safeguarded secant for this iteration.
        """
        from pandapower.control.util.sensitivity import calc_dvm_dq
        stations = vs['stations']
        candidates = [k for k, rt in enumerate(stations)
                      if not rt['disabled'] and not rt['converged'] and rt['f'] is not None
                      and rt['modus'] == ControlModusEnum.v_ctrl and rt['bus'] is not None
                      and rt['adjustable'].any()]
        if not candidates:
            return {}
        output_buses = net[self.output_element]['bus'].values
        vm_buses = [stations[k]['bus'] for k in candidates]
        q_buses, q_slices = [], []
        for k in candidates:
            buses = output_buses[stations[k]['out_pos']]
            q_slices.append((len(q_buses), len(q_buses) + len(buses)))
            q_buses.extend(buses)
        sensitivity = calc_dvm_dq(net, q_buses, vm_buses)
        if sensitivity is None:
            logger.debug("%s: no Jacobian available, secant fallback" % self.name)
            return {}
        # station-total sensitivities: outputs weighted with the current distribution.
        # deliberately no dense BLAS calls (@ / dot) anywhere in this method: powerflow
        # backends shipping their own BLAS (e.g. lightsim2grid with MKL numpy on Windows)
        # crash inside dense LAPACK/BLAS kernels
        matrix = np.empty((len(candidates), len(candidates)))
        for j, k in enumerate(candidates):
            start, stop = q_slices[j]
            matrix[:, j] = np.sum(sensitivity[:, start:stop] * stations[k]['dist'], axis=1)
        # drop stations touching non-PQ buses (NaN sensitivities)
        valid = ~(np.isnan(matrix).any(axis=1) | np.isnan(matrix).any(axis=0))
        if not valid.all():
            logger.debug("%s: stations at non-PQ buses use the secant fallback" % self.name)
            candidates = [k for k, ok in zip(candidates, valid) if ok]
            if not candidates:
                return {}
            matrix = matrix[np.ix_(valid, valid)]
        residual = np.array([stations[k]['f'] for k in candidates])
        # step rejection: if the previous jacobian step increased the residual norm, take a
        # safeguarded secant step on fresh information instead
        jac_state = vs.setdefault('jac', {'prev_rnorm': None})
        rnorm = float(np.max(np.abs(residual)))
        if jac_state['prev_rnorm'] is not None and rnorm > jac_state['prev_rnorm']:
            jac_state['prev_rnorm'] = None
            logger.debug("%s: jacobian step increased the residual, secant fallback" % self.name)
            return {}
        diagonal = np.diag(matrix)
        if np.any(diagonal == 0):
            return {}
        # the solve goes through SuperLU (like the sensitivity computation) instead of dense
        # LAPACK: environments where a powerflow backend ships its own BLAS (e.g.
        # lightsim2grid + MKL numpy on Windows) crash inside dense LAPACK calls
        from scipy.sparse import csc_matrix
        from scipy.sparse.linalg import spsolve
        sparse_matrix = csc_matrix(matrix)
        try:
            dq = np.atleast_1d(spsolve(sparse_matrix, residual))
            # validate instead of a cond() estimate: fall back to the decoupled diagonal
            # update when the solution is unusable
            if (not np.all(np.isfinite(dq))
                    or np.max(np.abs(sparse_matrix.dot(dq) - residual)) > 1e-8 * max(1.0, rnorm)):
                dq = residual / diagonal
        except RuntimeError:
            dq = residual / diagonal
        if not np.all(np.isfinite(dq)):
            return {}
        jac_state['prev_rnorm'] = rnorm
        return {k: damping * delta for k, delta in zip(candidates, dq)}

    def _control_step_stations(self, net):
        vs = self._vstate
        damping = float(getattr(self, 'damping_factor', 1.0) or 1.0)
        enforce_q_lims = net._options.get('enforce_q_lims', False)
        output_table = net[self.output_element]
        current_values = output_table[self.output_variable].values
        min_q = max_q = None
        if enforce_q_lims and 'min_q_mvar' in output_table.columns:
            min_q = np.nan_to_num(output_table['min_q_mvar'].values.astype(np.float64),
                                  nan=-np.inf)
        if enforce_q_lims and 'max_q_mvar' in output_table.columns:
            max_q = np.nan_to_num(output_table['max_q_mvar'].values.astype(np.float64),
                                  nan=np.inf)
        jacobian_deltas = {}
        if getattr(self, 'update_method', 'secant') == 'jacobian':
            jacobian_deltas = self._jacobian_deltas(net, vs, damping)
        write_index, write_values = [], []
        for position, rt in enumerate(vs['stations']):
            if rt['disabled'] or rt['converged'] or rt['f'] is None:
                continue
            out_in_service = rt['out_in_service']
            # the powerflow saw the values currently in the net -> true evaluation point
            values = current_values[rt['out_pos']].astype(np.float64) * out_in_service
            f = rt['f']
            cap = 2.0 * float(np.abs(values).sum()) + 50.0
            frozen = rt['at_limit'] & out_in_service
            if position in jacobian_deltas:
                total = float(values.sum())
                x_total = total + float(np.clip(jacobian_deltas[position], -cap, cap))
                x = (x_total - float(values[frozen].sum())) * rt['dist']
                x[frozen] = values[frozen]
            elif rt['values_old'] is None:  # first step: probe
                if rt['modus'] in ControlModusEnum.v_modes():
                    x = values + 1e-3 * (rt['dist'] > 0)
                else:
                    probe_total = float(np.clip(damping * f, -cap, cap))
                    if abs(probe_total) < 1e-3:
                        probe_total = 1e-3
                    x = values + probe_total * rt['dist']
            else:
                total = float(values.sum())
                total_old = float((rt['values_old'] * out_in_service).sum())
                f_old = f if rt['f_old'] is None else rt['f_old']
                x_total = self._secant_core(rt['solver'], f, f_old, total, total_old,
                                            damping, rt['set_point'], cap, rt['label'])
                # outputs at a limit keep their clamped value, the rest shares the remainder
                x = (x_total - float(values[frozen].sum())) * rt['dist']
                x[frozen] = values[frozen]
            if enforce_q_lims and (min_q is not None or max_q is not None):
                station_min = (min_q[rt['out_pos']] if min_q is not None
                               else np.full(len(x), -np.inf))
                station_max = (max_q[rt['out_pos']] if max_q is not None
                               else np.full(len(x), np.inf))
                over = (x > station_max) & rt['adjustable']
                under = (x < station_min) & rt['adjustable']
                if over.any() or under.any():
                    reached = over | under
                    x = np.where(over, station_max, x)
                    x = np.where(under, station_min, x)
                    rt['at_limit'] = rt['at_limit'] | reached
                    logging.info('Station %s: output element(s) %s reached a reactive power '
                                 'limit.' % (rt['label'],
                                             list(rt['out_idx'][reached])))
            rt['values_old'], rt['values'] = values, x
            write_index.extend(rt['out_idx'][out_in_service])
            write_values.extend(x[out_in_service])
        if write_index:
            write_to_net(net, self.output_element, write_index, self.output_variable,
                         write_values, 'loc')

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
            if not np.all(np.isnan(net[self.output_element].loc[self.output_element_index, 'id_q_capability_characteristic'].values)):
                qmin, _ = get_min_max_q_mvar_from_characteristics_object(net, self.output_element, self.output_element_index)
                self.output_min_q_mvar = np.nan_to_num(qmin, nan=-np.inf)
                net[self.output_element].loc[self.output_element_index, 'min_q_mvar'] = self.output_min_q_mvar
            else:
                self.output_min_q_mvar = np.nan_to_num(net[self.output_element].loc[self.output_element_index, 'min_q_mvar'].values, nan=-np.inf)
                net[self.output_element].loc[self.output_element_index, 'min_q_mvar'] = self.output_min_q_mvar
        else:
            self.output_min_q_mvar = list(np.array([-np.inf]*len(self.output_element_index), dtype=np.float64))

        if 'max_q_mvar' in net[self.output_element].columns:
            if not np.all(np.isnan(net[self.output_element].loc[self.output_element_index, 'id_q_capability_characteristic'].values)):
                _, qmax = get_min_max_q_mvar_from_characteristics_object(net, self.output_element, self.output_element_index)
                self.output_max_q_mvar = np.nan_to_num(qmax, nan=np.inf)
                net[self.output_element].loc[self.output_element_index, 'max_q_mvar'] = self.output_max_q_mvar
            else:
                self.output_max_q_mvar = np.nan_to_num(net[self.output_element].loc[self.output_element_index, 'max_q_mvar'].values, nan=np.inf)
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
        if isinstance(self.control_modus, bool) and self.control_modus:
            self.control_modus = ControlModusEnum.v_ctrl_q_droop
            logger.warning(f"Deprecated Control Modus in Controller {self.index}, using V_ctrl with Q droop from available types"
                         f" 'Q_ctrl' or 'V_ctrl'\n")
        elif isinstance(self.control_modus, bool) and not self.control_modus:
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
        if self.control_modus in ControlModusEnum.pf_modes():  #legacy ambiguous
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
                    isinstance(getattr(self, 'vm_set_pu', None), numbers.Number)):  #catching missing voltage set point
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
        bsc = net.controller.at[self.controller_idx, "object"]
        if not bsc.in_service or bsc.converged:
            self.converged = True
            return self.converged
        ###check control_modus###
        self.check_control_modus_and_values(net)
        if self.control_modus in ControlModusEnum.v_modes():
            self.diff = (bsc.set_point -
                         read_from_net(net, "res_bus", int(self.bus_idx), "vm_pu", self.read_flag))
        else:
            counter = 0
            input_values = []
            for input_index in bsc.input_element_index:
                input_values.append(
                    read_from_net(net, bsc.input_element, input_index,
                                  bsc.input_variable[counter],
                                  bsc.read_flag[counter]))
                counter += 1
            input_sign = np.asarray(bsc.input_sign)
            input_values = (input_sign * np.asarray(input_values)).tolist()
            self.diff = (bsc.set_point - sum(input_values))
        self.converged = np.all(np.abs(self.diff) < self.tol)
        return self.converged

    def control_step(self, net):
        self._droop_control_step(net)

    def _droop_control_step(self, net):
        bsc = net.controller.at[self.controller_idx, "object"]
        self.vm_pu_old = self.vm_pu
        self.vm_pu = read_from_net(net, "res_bus", self.bus_idx, "vm_pu", flag=self.read_flag)
        if self.control_modus not in ControlModusEnum.v_modes():
            if self.q_set_mvar_bsc is None:
                self.q_set_mvar_bsc = bsc.set_point
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
                bsc.set_point = self.q_set_mvar

        else:
            input_element = bsc.input_element
            input_element_index = bsc.input_element_index
            input_variable = bsc.input_variable
            read_flag = bsc.read_flag
            input_values = []
            counter = 0
            for input_index in input_element_index:
                input_values.append(read_from_net(net, input_element, input_index,
                                                  input_variable[counter], read_flag[counter]))
            input_values = (bsc.input_sign * np.asarray(input_values)).tolist()
            self.vm_set_pu = getattr(self, 'vm_set_pu', bsc.set_point)
            self.vm_set_pu_new = self.vm_set_pu + sum(
                input_values) / self.q_droop_mvar
            bsc.set_point = self.vm_set_pu_new


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
            cls.q_ctrl_v_droop
        }
    @classmethod
    def droop_modes(cls):
        return {
            cls.v_ctrl_q_droop,
            cls.v_ctrl_q_droop_local,
            cls.q_ctrl_v_droop,
        }