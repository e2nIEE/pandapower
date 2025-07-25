import numbers
from cmath import isnan
import numpy as np
from numba.core.ir import Raise
from numpy.ma.extras import atleast_1d
from scipy.optimize import minimize
from pandapower import create_gen
from pandapower import create_sgen
from pandas import concat

from pandapower.control.basic_controller import Controller
from pandapower.auxiliary import _detect_read_write_flag, read_from_net, write_to_net
import pandapower.topology as top
import networkx as nx
try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

class BinarySearchControl(Controller):
    """
        The Binary search control is a controller which is used to reach a given set point. It can be used for
        reactive power control, voltage control, cosines(phi) or tangens(phi) control. The control modus can be set via
        the modus parameter. Input and output elements and indexes can be lists. Input elements can be transformers,
        switches, lines or buses (only in case of voltage control). in case of voltage control, the controlled bus must be
        given to input_element_index. Output elements are sgens, where active and reactive power can
        be set. The output value distribution takes a string and selects the type of reactive power distribution.
        The output distribution value describes the distribution of reactive power provision between multiple
        output_elements and will be normalized to 100 %.

        INPUT:
            **self**

            **net** - A pandapower grid

            **ctrl_in_service** - Whether the controller is in service or not.

            **output_element** - Output element of the controller. Takes a string value "gen" or "sgen", with
            reactive power control, currently only "sgen" is possible.

            **output_variable** - Output variable of that element, normally "q_mvar".

            **output_element_index** - Index of output element in e.g. "net.sgen".

            **output_element_in_service** - Whether output elements are in service or not.

            **input_element** - Measurement location, can be a transformer, switches or lines. Must be a bus for
            V_ctrl. Indicated by string value "res_trafo", "res_switch", "res_line" or "res_bus". In case of
            "res_switch": an additional small impedance is introduced in the switch.

            **input_variable** - Variable which is used to take the measurement from. Indicated by string value. Must
            be 'vm_pu' for 'V_ctrl'.

            **input_element_index** - Element of input element in net. Controlled bus in case of Voltage control. Can be
            given the string 'auto' in modus 'V_ctrl' to automatically select a bus whose nominal voltage is >= X kV.
            The X must be given to 'set_point'. Will take target voltage of the encountered bus. If no bus is found,
            uses the bus next to the controlled generator group. Not completely implemented, generators on multiple buses
            are not correctly handled.

            **set_point** - Set point of the controller, can be a reactive power provision or a voltage set point. In
            case of voltage set point, modus must be V_ctrl, input_element_index must be a bus (input_variable must be
            'vm_pu' input_element must be 'res_bus'). Can be overwritten by a droop controller chained with the binary
            search control. If 'V_ctrl' and automated bus selection (input_element_index == 'auto'), set_point will be
            the search criteria in kV for the controlled bus (V_bus >= V_set_point).

            **output_values_distribution** - Takes string to select one of the different available reactive power distribution
            methods: 'rel_P' -Q is relative to used Power, 'rel_rated_S' -Q is relative to the rated power S, currently
            using the sgen attribute 'sn_mva', 'set_Q' -set individual reactive power for each output element,
            'max_Q' -maximized reactive power reserve for the output elements, 'rel_V_pu' -Q is relative to the voltage
            limits of the output element.

            **output_distribution_values=None** -The values of the Q distribution, only applicable if q_distribution = 'set_Q'
            or rel_V_pu. For 'set_Q': Must be list containing the Q Value of each controlled element in percent in the same
            order as the controlled elements. For 'rel_V_pu': must be a list containing [Target Voltage, minimal allowed
            Voltage, maximal allowed Voltage] for each output element.

            **modus=None** - Enables the selection of the available control modi by taking one of the strings: Q_ctrl, V_ctrl,
            PF_ctrl (PF_ctrl_ind or PF_ctrl_cap for reactance of PF_ctrl) or tan(phi)_ctrl. Formerly called Voltage_ctrl

            **tol=0.001** - Tolerance criteria of controller convergence.
       """
    def __init__(self, net, ctrl_in_service:bool, output_element, output_variable, output_element_index,
                 output_element_in_service, input_element, input_variable,
                 input_element_index, set_point:float, output_values_distribution:str, output_distribution_values = None,
                 modus:str = None, tol=0.001, order=0, level=0,
                 drop_same_existing_ctrl=False, matching_params=None, **kwargs):
        super().__init__(net, in_service=ctrl_in_service, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params)
        self.redistribute_values = None  # Values to save for redistributed gens
        self.counter_warning = False #only one message that only one active output element
        self.in_service = ctrl_in_service
        self.input_element = input_element #point to be controlled
        self.output_element = output_element #typically sgens, output of Q
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
        self.output_values_distribution = (output_values_distribution[0] if ((isinstance(output_values_distribution, list)
            or isinstance(output_values_distribution, np.ndarray)) and isinstance(output_values_distribution[0], str)
            ) else output_values_distribution)#ruggedized code for miss input
        self.output_distribution_values = output_distribution_values
        self.set_point = set_point
        self.input_element_index = []  # for boundaries
        if input_element_index == 'auto':
            self.automatic_selection(net)
        elif isinstance(input_element_index, list) or isinstance(input_element_index, np.ndarray):
            for element in input_element_index:
                self.input_element_index.append(element)
        else:
            self.input_element_index.append(input_element_index)
        self.tol = tol #tolerance
        if self.tol is None: #old order
            self.tol = 0.001
        self.output_values = None
        self.output_values_old = None
        self.diff = None
        self.diff_old = None
        self.converged = False #criteria for success of controller
        self.overwrite_convergence = False #for droop
        self.write_flag, self.output_variable = _detect_read_write_flag(net, output_element, output_element_index,
                                                                        output_variable)
        # write kwargs in self
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.read_flag = [] #type of read value
        self.input_variable = [] #unit of controlled element Q
        self.input_variable_p = [] #unit of controlled element P
        self.input_element_in_service = []
        self.max_q_mvar = [] #limits of output element Q
        self.min_q_mvar = []

        if self.output_values_distribution == 'rel_V_pu':
            self.bus_idx_dist = [] #initializing bus idx
            output_distribution_values = np.array(self.output_distribution_values) #forming limit arrays
            if output_distribution_values.ndim == 1: #one controlled sgen
                try:
                    self.v_set_point_pu = np.array(output_distribution_values)[0]
                    self.v_min_pu = np.minimum(np.array(output_distribution_values)[1], np.array(output_distribution_values)[2])
                    self.v_max_pu = np.maximum(np.array(output_distribution_values)[1], np.array(output_distribution_values)[2])
                except IndexError: #insufficient values in array
                    logger.warning(f"Insufficient values in distribution rel_V_pu {self.output_distribution_values} In "
                                   f"Controller {self.index}. Using set point 1 pu and min/max 0.9/1.1 pu\n")
                    equal_array = [1, 0.9, 1.1]
                    self.output_distribution_values = np.tile(equal_array, (len(np.array(self.output_element_in_service)), 1))[0]
                    output_distribution_values = np.array(self.output_distribution_values)  # forming limit arrays
                    self.v_set_point_pu = output_distribution_values[0]
                    self.v_min_pu = output_distribution_values[1]
                    self.v_max_pu = output_distribution_values[2]

            elif output_distribution_values.ndim >= 2: #more than one controlled sgen
                try:#insufficient values in arrays
                    self.v_set_point_pu = np.array(output_distribution_values)[:, 0]
                    self.v_min_pu = np.minimum(np.array(output_distribution_values)[:, 1],np.array(output_distribution_values)[:, 2])
                    self.v_max_pu = np.maximum(np.array(output_distribution_values)[:, 1],np.array(output_distribution_values)[:, 2])
                except IndexError:
                    logger.warning(f"Insufficient values in distribution rel_V_pu {self.output_distribution_values} In "
                                   f"Controller {self.index}. Using set point 1 pu and min/max 0.9/1.1 pu\n")
                    equal_array = [1, 0.9, 1.1]
                    self.output_distribution_values = np.full(len(np.array(self.output_element_in_service)),equal_array)
                    output_distribution_values = np.array(self.output_distribution_values)  # forming limit arrays
                    self.v_set_point_pu = output_distribution_values[:, 0]
                    self.v_min_pu = output_distribution_values[:, 1]
                    self.v_max_pu = output_distribution_values[:, 2]
            else:
                self.output_distribution_values = None
        ###finding correct modus, catching deprecated voltage_ctrl argument###
        if modus is None: #catching old attribute voltage_ctrl
            if hasattr(self, 'voltage_ctrl'):
                modus = self.voltage_ctrl
                if not hasattr(self, '_deprecation_warned'):#only one message that voltage ctrl is deprecated
                    logger.warning(
                        f"'voltage_ctrl' in Controller {self.index} is deprecated. "
                        "Use 'modus' ('Q_ctrl', 'V_ctrl', etc.) instead.")
                    self._deprecation_warned = True

        if type(modus) == bool and modus == True: #Only functions written out!?!
            self.modus = "V_ctrl"
            logger.warning(f"Deprecated Controller modus for Controller {self.index}, using 'V_ctrl' from available"
                         f" types 'Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan(phi)_ctrl'\n")
        elif type(modus) == bool and modus == False: #Only functions written out!?!
            self.modus = "Q_ctrl"
            logger.warning(f"Deprecated Controller modus for Controller {self.index}, using Q_ctrl from available"
                         f" types 'Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan(phi)_ctrl'\n")
        elif modus == "PF_ctrl_cap": # -1 for capacitive, 1 for inductive systems
            self.modus = "PF_ctrl"
            self.reactance= -1
        elif modus == "PF_ctrl_ind":
            self.modus = "PF_ctrl"
            self.reactance = 1
        elif modus == "PF_ctrl":
            logger.warning(f"Ambivalent reactive power flow direction for Controller {self.index}, using capacitive direction.\n")
            self.modus = modus
            self.reactance = -1
        else:
            if modus == "tan(phi)_ctrl" or modus == "V_ctrl":
                self.modus = modus
            else:
                if modus != 'Q_ctrl':
                    logger.warning(f"Modus {modus} not recognized, using 'Q_ctrl' from available"
                             f" types 'Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan(phi)_ctrl'\n")
                self.modus = 'Q_ctrl'

        if self.modus == 'PF_ctrl': #checking cos(phi) limits
            if abs(self.set_point) >1:
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
            elif self.input_element == "res_bus":
                self.input_element_in_service.append(net.bus.in_service[input_index])

            if isinstance(input_variable, list): #get Q variable and read flags for input elements
                read_flag_temp, input_variable_temp = _detect_read_write_flag(net, self.input_element, input_index,
                                                                              input_variable[counter])
            else:
                read_flag_temp, input_variable_temp = _detect_read_write_flag(net, self.input_element,input_index,
                                                                              input_variable)
            ###get p variables for input elements for Phi controller
            if self.modus == "PF_ctrl" or self.modus=='tan(phi)_ctrl':
                if isinstance(input_variable, list):
                    input_variable_p = input_variable[counter].replace('q', 'p').replace('var','w')
                    read_flag_temp_p, input_variable_temp_p = _detect_read_write_flag(net, self.input_element,input_index,
                                                                                  input_variable_p)
                else:
                    input_variable_p = input_variable.replace('q', 'p').replace('var','w')
                    read_flag_temp_p, input_variable_temp_p = _detect_read_write_flag(net, self.input_element,input_index,
                                                                                  input_variable_p)
                self.input_variable_p.append(input_variable_temp_p) #read flag p not necessary, flag same as Q variables
            self.read_flag.append(read_flag_temp)
            self.input_variable.append(input_variable_temp)
            counter += 1

        ###reading Q limits###
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

    def __str__(self):
        return super().__str__() + " [%s.%s.%s.%s]" % (
            self.input_element, self.input_variable, self.output_element, self.output_variable)

    def __getattr__(self, name):
        if name == "modus":
            if not hasattr(self, '_deprecation_warned'):
                logger.warning(
                    f"'voltage_ctrl' in Controller {self.index} is deprecated. "
                    "Use 'modus' ('Q_ctrl', 'V_ctrl', etc.) instead."
                )
                self._deprecation_warned = True#only one message that voltage ctrl is deprecated
            return self.voltage_ctrl
        if name == 'bus_idx':
            if not hasattr(self, '_deprecation_warned_bus_idx'):
                logger.warning(
                    f"Variable 'bus_idx' in Binary Search Control {self.index} for modus V_ctrl is deprecated. "
                    f"Give index of controlled bus to input_element_index. Input_variable must be 'vm_pu' and"
                    f" input_element 'res_bus'"
                )
                self._deprecation_warned_bus_idx = True#only one warning about bus_idx deprecation
            return self.input_element_index
        raise AttributeError(f"{self.__class__.__name__!r} has no attribute {name!r}")


    def initialize_control(self, net, converged = False):
        ###For V_ctrl, concatenate all gens to a single gen. Redistribution in finalize_control()###
        active_gens = (self.output_element_in_service if isinstance(self.output_element_in_service[0], bool) else
                        np.atleast_1d(self.output_element_in_service)[:, 0].tolist()) #ugly
        if (self.modus == 'V_ctrl' and self.output_element == 'gen' and
                        len(np.atleast_1d(self.output_element_index)[active_gens]) >= 2):
            fused_bus_by_switch = False
            fused_bus_index = []
            for i in net.switch.index: #check if ideal switch between buses with controlled gens
                if (net.switch.at[i, 'et'] == 'b' and net.switch.at[i, 'closed'] and net.switch.at[i, 'z_ohm'] == 0 and
                        net.switch.at[i, 'bus'] in np.atleast_1d(
                            net.gen.loc[np.atleast_1d(self.output_element_index)[active_gens], 'bus'])):  # fused buses by ideal switch
                    if (net.switch.at[i, 'element'] in
                        np.atleast_1d(net.gen.loc[np.atleast_1d(self.output_element_index)[active_gens], 'bus'])): #controlled gens at buses
                        fused_bus_by_switch = True
                        fused_bus_index.append([[net.switch.at[i, 'bus'], net.switch.at[i, 'element']],
                            [np.atleast_1d(self.output_element_index)[active_gens][np.where(
                                np.atleast_1d(net.gen.loc[np.atleast_1d(self.output_element_index)[active_gens], 'bus']) ==
                                net.switch.at[i, 'bus'])[0][0]],
                             np.atleast_1d(self.output_element_index)[active_gens][np.where(
                                 np.atleast_1d(net.gen.loc[np.atleast_1d(self.output_element_index)[active_gens], 'bus']) ==
                                 net.switch.at[i, 'element'])[0][0]]]])#append buses in groups with indexes and gen index
            if not net.gen.loc[np.atleast_1d(self.output_element_index)[active_gens], "bus"].is_unique or fused_bus_by_switch:
                logger.info(f'concatenated multiple gens to single gen in Voltage Controller {self.index}')
                bus_series = net.gen.loc[np.atleast_1d(self.output_element_index)[active_gens], "bus"]
                #non_unique_indices = bus_series[bus_series.duplicated(keep=False)].index.tolist()
                duplicated_buses = bus_series[bus_series.duplicated(keep=False)]#check if buses of gens are unique
                # Groups of indices of buses of gens at same bus
                duplicated_gen_groups = duplicated_buses.groupby(duplicated_buses).apply(lambda x: x.index.tolist())
                ###concatenate gens at same bus or at ideally connected bus###
                index_replaced_fused = []
                index_replaced = []
                for i in fused_bus_index: #first fused buses
                    bus_number, indices = i[0], i[1]
                    if sum(net.gen.loc[indices, 'in_service']) >= 2:
                        net.gen.loc[indices, 'in_service'] = False
                        temp_gen_fused = create_gen(net, bus_number[0], net.gen.loc[indices, 'p_mw'].sum(),
                                                       net.gen.loc[indices[0], 'vm_pu'])
                        index_replaced_fused.append([temp_gen_fused])
                for bus_number, indices in duplicated_gen_groups.items(): #second normal buses
                    if all(net.gen.loc[indices, 'in_service']):
                        net.gen.loc[indices, "in_service"] = False
                        temp_gen = create_gen(net, bus_number, net.gen.loc[indices, "p_mw"].sum(),
                                                 net.gen.loc[indices[0], "vm_pu"])
                        index_replaced.append([temp_gen])
                #save values for redistribution
                self.redistribute_values = [self.output_element_index, self.output_element_in_service, self.min_q_mvar,
                                            self.max_q_mvar, duplicated_gen_groups, fused_bus_index]
                #replace values in controller
                self.output_element_index = np.atleast_1d(np.array(index_replaced_fused + index_replaced)).flatten()
                self.output_element_in_service = np.full(len(self.output_element_index), True, dtype=bool)
                self.min_q_mvar = np.full(len(np.atleast_1d(self.output_element_index)),
                                      (sum(np.atleast_1d(self.min_q_mvar)) / len(np.atleast_1d(self.min_q_mvar))), float)
                self.max_q_mvar = np.full(len(np.atleast_1d(self.output_element_index)),
                                      (sum(np.atleast_1d(self.max_q_mvar)) / len(np.atleast_1d(self.max_q_mvar))), float)
        #reread output elements
        #net.controller.at[self.index, 'object'].converged = converged
        output_element_index = self.output_element_index[0] if self.write_flag == 'single_index' else\
                                            self.output_element_index #ruggedize for single index
        self.output_values = read_from_net(net, self.output_element, output_element_index, self.output_variable,
                                            self.write_flag)
        self.output_values_old = None

    def is_converged(self, net):
        """
        Actual implementation of the convergence criteria: If controller is applied, it can stop
        """
        # if controller not in_service, return True
        self.in_service = net.controller.in_service[self.index]
        if not self.in_service:
            return True
        ###updating input & output elements in service lists
        self.input_element_in_service = list(self.input_element_in_service)
        self.output_element_in_service = list(self.output_element_in_service)
        self.input_element_in_service.clear()
        self.output_element_in_service.clear()
        for input_index in np.atleast_1d(self.input_element_index):
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
            logger.warning(f"Controller {self.index} has no active output elements: {self.output_element}:"
                           f" {self.output_element_index} are disabled.\n Control aborted\n")
            self.converged = True
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
                    if self.modus == "PF_ctrl" or self.modus == 'tan(phi)_ctrl':
                        p_input_values.append(read_from_net(net,self.input_element, input_index,
                                                        self.input_variable_p[counter], self.read_flag[counter]))
                counter += 1
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
        if self.modus == "Q_ctrl" or (self.modus=='V_ctrl' and self.input_element_index is None):
            if self.modus == 'V_ctrl':
                logger.warning('Missing attribute self.input_element_index, defaulting to Q_ctrl\n')
                self.modus = 'Q_ctrl'
            self.diff_old = self.diff
            self.diff = self.set_point - sum(input_values)
            self.converged = np.all(np.abs(self.diff) < self.tol)

        elif str(self.modus).startswith("PF_ctrl"):#capacitive => reactance = -1, inductive => reactance = 1
            if self.modus == 'PF_ctrl_ind':
                self.modus = 'PF_ctrl'
                self.reactance = 1
            elif self.modus == 'PF_ctrl_cap':
                self.modus = 'PF_ctrl'
                self.reactance = -1

            self.diff_old = self.diff
            q_set = self.reactance * sum(p_input_values)/len(p_input_values) * (np.tan(np.arccos(self.set_point)))
            self.diff = q_set - sum(input_values)/len(input_values)
            self.converged = np.all(np.abs(self.diff)<self.tol)

        elif self.modus == "tan(phi)_ctrl":
            self.diff_old = self.diff
            q_set = sum(p_input_values)/len(p_input_values) * self.set_point
            self.diff = q_set - sum(input_values)/len(input_values)
            self.converged = np.all(np.abs(self.diff) < self.tol)
        else:
            ###catching deprecated modi from old imports
            if type(self.modus) == bool and self.modus == True and self.input_element_index is not None:
                self.modus = "V_ctrl"  # catching old implementation
                logger.warning(
                    f"Deprecated Control Modus in Controller {self.index}, using V_ctrl from available types\n")
            elif (type(self.modus) == bool and self.modus == False) or (type(self.modus) == bool and self.modus == True
                and self.input_element_index is None):
                if self.modus is True:
                    logger.warning(f'Deprecated Control Modus in Controller {self.index}, attempted to use "V_ctrl" but '
                                   f'missing attribute input_element_index, defaulting to Q_ctrl\n')
                else:
                    logger.warning(
                        f"Deprecated Control Modus in Controller {self.index}, using Q_ctrl from available types\n")
                self.modus = "Q_ctrl"

            if self.modus == "V_ctrl":
                if self.input_element != 'res_bus' and not any(getattr(net.controller.at[x, 'object'], 'controller_idx', False) ==
                                                                        self.index for x in net.controller.index):
                    logger.warning(f"'input_element' must be 'res_bus' for V_ctrl not {self.input_element}, correcting.")
                    self.input_element = 'res_bus'
                    if np.atleast_1d(self.input_variable)[0] != 'vm_pu':
                        logger.warning(f"'input_variable' must be 'vm_pu' for V_ctrl not {self.input_variable}, correcting ")
                        self.input_variable = 'vm_pu'

                self.diff_old = self.diff #V_ctrl
                self.diff = self.set_point - net.res_bus.vm_pu.at[np.atleast_1d(self.input_element_index)[0]]
                self.converged = np.all(np.abs(self.diff) < self.tol)
            else:
                if self.modus != 'Q_ctrl':
                    logger.warning(f"No Controller Modus specified for Controller {self.index}, using Q_ctrl.\n"
                      "Please specify 'modus' ('Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan(phi)_ctrl')\n")
                    self.modus = 'Q_ctrl'
                self.diff_old = self.diff #Q_ctrl
                self.diff = self.set_point - sum(input_values)
                self.converged = np.all(np.abs(self.diff) < self.tol)

        ### check limits after convergence###
        if self.converged and not getattr(self, 'overwrite_convergence', False):
            if self.output_values_distribution == 'rel_V_pu':
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
                exceed_limit_min = np.where(self.output_values < np.array(self.min_q_mvar)[np.array(self.output_element_in_service)])[0]
                exceed_limit_max = np.where(self.output_values > np.array(self.max_q_mvar)[np.array(self.output_element_in_service)])[0]
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

        if getattr(self,'overwrite_convergence', False): ###overwrite convergence in case of droop controller
            self.overwrite_convergence = False
            return False
        else:
            return self.converged

    def control_step(self, net):
        self._binary_search_control_step(net)

    def _binary_search_control_step(self, net):
        from pandapower import runpp #to avoid circular imports, import here
        generators_not_at_limit = None
        if not self.in_service: #redundant
            return
        ### Distribution warnings###
        if getattr(self, 'output_distribution_values', None) is not None: #catch warnings
            if (self.output_values_distribution == 'rel_P' or self.output_values_distribution == 'rel_rated_S' or
                    self.output_values_distribution == "max_Q"):
                logger.warning(f'The inserted values for output distribution values {self.output_distribution_values} '
                               f'will have no effect on the reactive power distribution\n')
                self.output_distribution_values, output_distribution_values_in_service = None, None
            elif self.output_values_distribution == 'imported' or self.output_values_distribution == "set_Q":
                if len(self.output_distribution_values) < len(np.array(self.output_element_in_service)):#check if enough values
                    equal_val = 1 / (len(np.array(self.output_element_in_service)-len(self.output_distribution_values)))
                    logger.warning(
                        f'Mismatched lengths of output elements {self.output_element} and output_distribution_values'
                        f'{len(np.array(self.output_element_in_service))} > {len(self.output_distribution_values)}'
                        f' in Controller {self.index}.\n' f'Appending values {equal_val} \n')
                    self.output_distribution_values = (np.append(self.output_distribution_values, [equal_val] *
                                         (len(self.output_element_in_service) - len(self.output_distribution_values))))
                output_element_in_service = np.array(self.output_element_in_service)#ruggedizing code for wrong inputs
                output_element_in_service.resize((len(np.array(self.output_distribution_values)),), refcheck=False)
                output_distribution_values_in_service = (np.array(self.output_distribution_values)
                [np.array(output_element_in_service)]) ###only distributing between active output elements
            elif self.output_values_distribution == 'rel_V_pu':
                if ((np.array(self.output_distribution_values).ndim > 1 and any(len(element) != 3 for element in self.output_distribution_values))
                        or (np.array(self.output_distribution_values).ndim == 1 and len(self.output_distribution_values) != 3)):
                    logger.warning(f"Insufficient values in distribution rel_V_pu {self.output_distribution_values} In "
                               f"Controller {self.index}. Using set point 1 pu and min/max 0.9/1.1 pu\n")
                    equal_array = [1, 0.9, 1.1]
                    self.output_distribution_values = self.output_distribution_values = np.tile(equal_array,
                                                            (len(np.array(self.output_element_in_service)), 1))[0]
                    output_distribution_values = np.array(self.output_distribution_values)  # forming limit arrays
                    self.v_set_point_pu = output_distribution_values[:, 0]
                    self.v_min_pu = output_distribution_values[:, 1]
                    self.v_max_pu = output_distribution_values[:, 2]
                output_distribution_values_in_service = None
            else: output_distribution_values_in_service, self.output_distribution_values = None, None
        elif getattr(self, 'output_distribution_values', None) is None:
            if self.output_values_distribution == 'set_Q' or self.output_values_distribution == 'imported':
                if self.output_values_distribution == 'set_Q':
                    logger.warning(f'Reactive Power Distribution method "set_Q" needs values given to output_distribution_values '
                         f'in Controller {self.index}. Distributing the reactive power equally between all available output elements.\n')
                else:#self.output_values_distribution == 'imported'
                    logger.warning(f"Something went wrong while importing output distribution values in Controller"
                         f" {self.index}. Distributing the reactive power equally between all available output elements.\n")
                equal = 1 / sum(np.array(self.output_element_in_service))
                self.output_distribution_values = np.full(len(np.array(self.output_element_in_service)), equal)
                output_distribution_values_in_service = self.output_distribution_values[np.array(self.output_element_in_service)]  #only distributing between active output elements
            elif self.output_values_distribution == 'rel_V_pu':
                logger.warning(f"Missing values for output distribution values 'rel_V_pu in Controller {self.index}. "
                             f"Using set point 1 pu and min/max 0.9/1.1 pu\n ")
                equal_array = [1, 0.9, 1.1]
                self.output_distribution_values = self.output_distribution_values = np.tile(equal_array,
                                                        (len(np.array(self.output_element_in_service)), 1))[0]#new vals
                output_distribution_values = np.atleast_2d(self.output_distribution_values)  # forming limit arrays
                self.v_set_point_pu = output_distribution_values[:, 0]
                self.v_min_pu = output_distribution_values[:, 1]
                self.v_max_pu = output_distribution_values[:, 2]
                output_distribution_values_in_service = None
            else: self.output_distribution_values, output_distribution_values_in_service = None, None#rel_rated_S and rel_P, max_Q
        else: raise UserWarning(f"Output_distribution_values in Controller {self.index} is {self.output_distribution_values}")

        ###calculate output values###
        if self.output_values_old is None:  # first step
            self.output_values_old, self.output_values = (
            np.atleast_1d(self.output_values)[self.output_element_in_service],
            np.atleast_1d(self.output_values)[self.output_element_in_service] + 1e-3)
        else:#second step
            step_diff = self.diff - self.diff_old
            x = self.output_values - self.diff * (self.output_values - self.output_values_old) / np.where(
                step_diff == 0, 1e-6, step_diff)  #converging
            if any((abs(x) - abs(2 * self.output_values)) > 100): #catching overshoots for calculation, another check before writing into the net
                x[np.where((abs(x) > abs(100 - abs(self.output_values))))[0]] = np.sign(x[np.where((abs(x) -
                                                                           abs(2 * self.output_values)) > 100)[0]]) * 100
            ###calculate the distribution of the output values
            if self.output_values_distribution == 'imported': #when importing net from PF for backwards compatibility
                distribution = output_distribution_values_in_service

            elif self.output_values_distribution == 'rel_P': #proportional to the dispatch active power
                dispatched_active_power = read_from_net(net, self.output_element, self.output_element_index, 'p_mw', 'auto')
                dispatched_active_power = dispatched_active_power[np.array(self.output_element_in_service)]
                distribution = dispatched_active_power/sum(dispatched_active_power)

            elif self.output_values_distribution == 'rel_rated_S': #proportional to the rated apparent power
                if not hasattr(self, 'rel_rated_S_warned'):
                    self.rel_rated_S_warned = True
                    logger.warning(f'The standard type attribute containing the rated apparent power for'
                               f' {self.output_element} is not correctly implemented yet (BSC {self.index}).')
                try:
                    s_rated_mva = np.array(net.sgen.loc[self.output_element_index, 'sn_mva']) #todo correct attribute?
                    distribution = s_rated_mva
                    nan_index = np.isnan(distribution)
                    distribution[nan_index] = 50
                    if any(nan_index):
                        logger.warning(f'{self.output_element} at index {np.atleast_1d(self.output_element_index)[nan_index]}'
                                       f' in Controller {self.index} has no specified rated apparent power, assuming 50 MVA\n')
                    if not all(isinstance(n, numbers.Number) for n in distribution):
                        logger.warning(f'{self.output_element} in Controller {self.index} has no'
                                       f' specified rated apparent power, assuming 50 MVA\n')
                        distribution = np.full(np.sum(self.output_element_in_service), 50)

                except KeyError:
                    logger.warning(f'{self.output_element} in Controller {self.index} has no defined standard type '
                                    f'or specified rated apparent power, assuming 50 MVA\n')
                    distribution = np.full(np.sum(self.output_element_in_service), 50)

            elif self.output_values_distribution == 'set_Q': #individually set Q distribution
                distribution = output_distribution_values_in_service

            elif self.output_values_distribution == 'max_Q':  # Maximise Reactive Reserve
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
                if len(np.atleast_1d(q_max_q)) != len(atleast_1d(self.output_element_in_service)):
                    counter_values = 0
                    distribution = np.ones(len(np.atleast_1d(self.output_element_in_service)))  #initializing the distribution for correction
                    for i in range(len(atleast_1d(generators_not_at_limit))):
                        if np.atleast_1d(generators_not_at_limit)[i]:#calculated Q
                            distribution[i] = np.atleast_1d(q_max_q)[counter_values]
                            counter_values += 1
                        elif not np.atleast_1d(generators_not_at_limit)[i]:#min or max Q
                            distribution[i] = np.atleast_1d(self.max_q_mvar)[i] if (np.atleast_1d(x)[i]
                                                >= 0) else np.atleast_1d(self.min_q_mvar)[i]
                else:
                    distribution = q_max_q

            elif self.output_values_distribution == 'rel_V_pu':  # Voltage set point Adaptation
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
                            if not not any(busbar_all):#not all busbar with multiple output elements?
                                busbar_all = np.array([x for x in busbar_all if x != False][0]) #delete bools
                                index_sgen = np.where(np.isin(net.sgen['bus'], busbar_all))[0] #indices of sgens
                                index_sgen = [index for i, index in enumerate(index_sgen) if list(net.sgen['in_service'])[i]]#check for service
                                index_gen = np.where(np.isin(net.gen['bus'], busbar_all))[0] #indices of gens
                                index_gen = [index for i, index in enumerate(index_gen) if list(net.gen['in_service'])[i]] #check for service
                                if len(index_sgen) + len(index_gen) > 1:
                                    items_sgen, items_gen, busbar = f"Check Sgen:\n", f"Check gen:\n", f""#initiate strings
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
                                    min_p_mw = net.sgen.at[i, 'min_p_mw'] if 'min_p_mw' in net.sgen.columns else None,
                                    max_p_mw = net.sgen.at[i, 'max_p_mw'] if 'max_p_mw' in net.sgen.columns else None,
                                    min_q_mvar = net.sgen.at[i, 'min_q_mvar'] if 'min_q_mvar' in net.sgen.columns else None,
                                    max_q_mvar = net.sgen.at[i, 'max_q_mvar'] if 'max_q_mvar' in net.sgen.columns else None,
                                    description = net.sgen.at[i, 'description'] if 'description' in net.sgen.columns else None,
                                    equipment = net.sgen.at[i, 'equipment'] if 'equipment' in net.sgen.columns else None,
                                    geo = net.sgen.at[i, 'geo'] if 'geo' in net.sgen.columns else None,
                                    current_source = net.sgen.at[
                                        i, 'current_source'] if 'current_source' in net.sgen.columns else None,
                                    name = f'temp_gen_{counter}')#type='GEN'
                            net.sgen.at[i, 'in_service'] = False #disable sgens
                            counter += 1
                    index = np.array([])
                    for i in net.gen.index: #get index of created gens
                        if net.gen.loc[i, 'name'].startswith("temp_gen_"):
                            index = np.append(index, i)
                    index = index[0] if self.write_flag == 'single_index' else index
                    write_to_net(net, 'gen', index,'vm_pu', voltage, self.write_flag) #write V to net
                    runpp(net, run_control = False) #run net
                    distribution = np.array(net.res_gen.loc[index, 'q_mvar']) #read Q from net
                    net.gen.drop(index=index, inplace=True) #delete created gens
                    net.sgen.loc[np.array(self.output_element_index)[self.output_element_in_service], 'in_service'] = True #reactivate sgens
                else: distribution = np.array([1]) #distribution is one for one active output element

            else: #unrecognizable output values distribution, using set_Q
                if (((isinstance(self.output_values_distribution, list) or isinstance(self.output_values_distribution, np.ndarray))
                    and all(isinstance(x, numbers.Number) for x in self.output_values_distribution)) or
                        isinstance(self.output_values_distribution, numbers.Number)):#numbers
                    logger.warning(f'Controller {self.index}: Output_values_distribution must be string from available methods'
                                   f' (rel_P, rel_rated_S, set_Q, max_Q or rel_V_pu). Using provided values with method set_Q\n')
                    self.output_distribution_values = np.array(self.output_values_distribution)
                    self.output_values_distribution = 'set_Q'
                    distribution = self.output_distribution_values[np.array(self.output_element_in_service)]
                else:
                    raise NotImplementedError(f"Controller {self.index}: Reactive power distribution method {self.output_values_distribution}"
                                              f" not implemented available methods are (rel_P, rel_rated_S, set_Q, max_Q, rel_V_pu).")
            if self.output_element != 'gen':
                if self.output_values_distribution == 'max_Q': #max_Q and voltage gives the correct Qs for the gens
                    if sum(np.atleast_1d(generators_not_at_limit)) == 0:
                        values = (sum(x) - sum(distribution)) / len(np.atleast_1d(distribution))
                        distribution = np.atleast_1d(distribution) + values #todo if respected Q limits only generators_not_at_limit, might not converge
                    else:
                        values = (sum(x) - sum(distribution)) / len(np.atleast_1d(distribution)[generators_not_at_limit])
                        np.atleast_1d(distribution)[generators_not_at_limit] += values
                    x = distribution
                #Voltage set point adaption gives correct Qs but needs convergence
                elif (self.output_values_distribution == 'rel_V_pu' and (sum(np.atleast_1d(self.output_element_in_service)) > 1
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
            x = np.sign(x) * (np.where(abs(abs(x) - abs(self.output_values)) > 84, 84, abs(x)))  # catching distributions out of bounds, 84 seems to be the maximum
            self.output_values_old, self.output_values = self.output_values, x

            ### write new set of Q values to output elements###
        output_element_index = (list(np.atleast_1d(self.output_element_index)[self.output_element_in_service])[0] if self.write_flag
            == 'single_index' else list(np.array(self.output_element_index)[self.output_element_in_service])) #ruggedizing code
        output_values = (list(self.output_values)[0] if self.write_flag
            == 'single_index' else list(self.output_values))  # ruggedizing code
        write_to_net(net, self.output_element, output_element_index, self.output_variable, output_values, self.write_flag)

    def automatic_selection(self, net):  # automatic selection of controlled Busbar in V_ctrl. Only new creation of nets
        target_buses = net.bus[net.bus.vn_kv >= self.set_point].index.tolist()  # All buses fulfilling the criteria
        ref_buses = net.sgen.loc[self.output_element_index, 'bus'].tolist() if self.output_element == 'sgen' \
            else net.gen.loc[self.output_element_index, 'bus'].tolist() # the start buses
        ref_buses = np.unique(ref_buses)  #if machines at one bus
        distances_list = []
        if len(ref_buses) != 1:  # control group for multiple generators
            for i in ref_buses:  # get distances of bus to possible buses
                if len(net.ext_grid.bus) > 1: # multiple external nets dont work
                    raise UserWarning(
                        f'Multiple External Grids for control group in controller {self.index}, auto-selection'
                        f'of controlled busbar not possible, aborting\n')
                g = top.create_nxgraph(net, respect_switches=True) #create graph for connecting buses
                distances_list.append(nx.shortest_path(g, source=i, target=int(net.ext_grid.bus.values)))#all buses between generators and external net
            common = set(distances_list[0]).intersection(*distances_list[1:]) #all buses in same lists
            control_group_bus = next((x for x in distances_list[0] if x in common), None) #closest bus to gens
        else:
            control_group_bus = ref_buses[0]
        distances = top.calc_distance_to_bus(net, control_group_bus, weight=None)  # criteria is the distance in the network, not in km
        distances_list = [distances.loc[target_buses]]
        distances_filtered = concat(distances_list, axis=0, ignore_index=False)
        if distances_filtered.empty:  # no possible busbar -> Bus next to gen, in PF using gen target vm_pu
            self.input_element_index = net.sgen.at[np.atleast_1d(self.output_element_index)[0], 'bus'] if self.output_element == 'sgen'\
                else net.gen.at[np.atleast_1d(self.output_element_index)[0], 'bus'] # here bus target v cause no gen target vm_pu
        else:
            self.input_element_index = distances_filtered.idxmin()  # minimal distances index
            min_distance = np.atleast_1d(distances_filtered).min()  # minimal distance
            distances_filtered = distances_filtered.drop(self.input_element_index)  # check if multiple buses within minimal distance
            if min_distance in distances_filtered.values:
                raise UserWarning  # what todo when multiple buses within minimal distance
        try:
            self.set_point = net.bus.at[self.input_element_index, 'set_pu'] #todo correct attribute?
        except KeyError:
            logger.error(f"The automatically selected bus {self.input_element_index} in Controller {self.index} "
                         f"has no target voltage (attribute 'set_pu', trying target voltage 1 pu\n") #todo attribute
            self.set_point = 1

    def finalize_control(self, net):
        from pandapower import runpp  # to avoid circular imports, import here
        ###redistribute the gens to multiple gens if they were concatenated in initialize_control()
        if getattr(self, "redistribute_values", None) is not None:
            logger.info(f'Redistributed gens to sgens in Voltage controller {self.index}\n')
            counter = 0
            temp_gen_index = self.output_element_index
            gen_bus_index = []
            for i in self.redistribute_values[5]:
                bus_number, indices = i[0], i[1]
                net.res_gen.loc[indices, 'q_mvar'] = net.res_gen.at[
                                                     np.atleast_1d(temp_gen_index)[counter], 'q_mvar'] / len(indices)
                gen_bus_index.extend(np.atleast_1d(indices).tolist())#get bus index for fused bus
                counter += 1
            for bus_number, indices in self.redistribute_values[4].items():#get bus index for normal bus
                net.res_gen.loc[indices, 'q_mvar'] = net.res_gen.at[
                                                     np.atleast_1d(temp_gen_index)[counter], 'q_mvar'] / len(indices)
                gen_bus_index.extend(np.atleast_1d(indices).tolist())
                counter += 1
            net.gen.drop(self.output_element_index, inplace=True)#delete the replacement gen
            ###change the gens to sgens for Q_distribution
            counter = 0
            index = []
            active_gens = np.atleast_1d(self.redistribute_values[1])[:, 0].tolist()
            for i in self.redistribute_values[0]:  #self.output_element_index: #only gens of controller
                if not net.gen.at[i, 'in_service']:  # get all gens of controller who where disabled, convert to sgens. also offline from start
                    sgen = create_sgen(net=net,#create sgens
                                          bus=net.gen.at[i, 'bus'],
                                          p_mw=net.gen.at[i, 'p_mw'],
                                          q_mvar=net.res_gen.at[i, 'q_mvar'],
                                          in_service= bool(np.atleast_1d(active_gens)[counter]),
                                          sn_mva=net.gen.at[i, 'sn_mva'] if 'sn_mva' in net.gen.columns and not isnan(
                                              net.gen.at[i, 'sn_mva']) else None,
                                          scaling=net.gen.at[
                                              i, 'scaling'] if 'scaling' in net.gen.columns and not isnan(
                                              net.gen.at[i, 'scaling']) else None,
                                          min_p_mw=net.gen.at[
                                              i, 'min_p_mw'] if 'min_p_mw' in net.gen.columns and not isnan(
                                              net.gen.at[i, 'min_p_mw']) else None,
                                          max_p_mw=net.gen.at[
                                              i, 'max_p_mw'] if 'max_p_mw' in net.gen.columns and not isnan(
                                              net.gen.at[i, 'max_p_mw']) else None,
                                          min_q_mvar=net.gen.at[
                                              i, 'min_q_mvar'] if 'min_q_mvar' in net.gen.columns and not isnan(
                                              net.gen.at[i, 'min_q_mvar']) else None,
                                          max_q_mvar=net.gen.at[
                                              i, 'max_q_mvar'] if 'max_q_mvar' in net.gen.columns and not isnan(
                                              net.gen.at[i, 'max_q_mvar']) else None,
                                          description=net.gen.at[
                                              i, 'description'] if 'description' in net.gen.columns and type(
                                              net.gen.at[i, 'description']) == str else None,
                                          equipment=net.gen.at[
                                              i, 'equipment'] if 'equipment' in net.gen.columns and type(
                                              net.gen.at[i, 'equipment']) == str else None,
                                          geo=net.gen.at[i, 'geo'] if 'geo' in net.gen.columns and type(
                                              net.gen.at[i, 'geo']) == str else None,
                                          # current_source=(net.gen.at[i, 'current_source'] if 'current_source' and
                                          #not isnan(net.gen.at[i, 'current_source']) in net.gen.columns else None),
                                          controllable=net.gen.at[i, 'controllable'],
                                          name=net.gen.at[i, 'name'])  # type='SGEN'
                    index.append(sgen)
                    counter += 1
            net.gen.drop(self.redistribute_values[0], inplace=True) #delete now obsolete gens
            #change controller values
            self.output_element_index = index
            self.output_element = 'sgen'
            self.output_variable = 'q_mvar'
            self.output_values = np.atleast_1d(net.sgen.loc[index, 'q_mvar'])[active_gens]
            self.output_values_old = self.output_values + 10 #to have the controller not start as converged
            self.output_element_in_service = active_gens
            self.min_q_mvar = self.redistribute_values[2]
            self.max_q_mvar = self.redistribute_values[3]
            self.redistribute_values = None #to avoid being stuck in loop
            self._binary_search_control_step(net)
            runpp(net)

class DroopControl(Controller):
    """
            The droop controller is used in case of a droop based control. It can operate either as a Q(U) controller,
            as a U(Q) controller, as a cosphi(P) controller or as a cosphi(U) controller and is used in tandem with
            a binary search controller (bsc). The linked binary search controller is specified using the
            controller index, which refers to the linked bsc (bsc.index). The droop controller behaves in a similar way
            to the station controllers presented in the Power Factory Tech Ref.

            INPUT:
                **self**

                **net** - A pandapower grid.

                **controller_idx** - Index of linked Binary search control (bsc.index).

                **in_service = True** - Whether the droop controller is in service or not.

                **modus** - takes string: Q_ctrl, V_ctrl or PF_ctrl. Select droop variety of PF_ctrl by
                choosing 'PF_ctrl_P' for P-Characteristic or 'PF_ctrl_V' for V-Characteristic. PF_ctrl_P takes the active
                power at the input_element as reference, for PF_ctrl_V the reference voltage must be defined via the
                bus_idx. Formerly called voltage_ctrl.

                **q_droop_var = None** - Droop Value in Mvar/p.u. in case of Q or V control.

                **bus_idx = None** - Bus index which is used for PF(V) control and Q control.

                **vm_set_lb = None** - Lower band border of dead band; The Power [MW] or Voltage[pu] at which Phi is static
                and underexcited (inductive) in case of PF_ctrl

                **vm_set_ub = None** - Upper band border of dead band; The Power [MW] or Voltage[pu] at which Phi is static
                and overexcited (capacitive) in case of PF_ctrl

                **pf_overexcited = None** - Static overexcited limit for Phi in case of PF_ctrl.

                **pf_underexcited=None** - Static underexcited limit for Phi in case of PF_ctrl.

                **input_type_q_meas=None** - Type of element(s) Q measurement for voltage control with droop is taken from
                according to v_set_point_new = v_set_point + Q_meas / q_droop_mvar. Takes string.

                **input_variable_q_meas=None** - Variable of element(s) Q measurement for voltage control is taken from
                according to v_set_point_new = v_set_point + Q_meas / q_droop_mvar. Takes string or list of strings.

                **input_element_index_q_meas=None** - Index of element(s) Q measurement for voltage control is taken from
                according to v_set_point_new = v_set_point + Q_meas / q_droop_mvar. Takes integer or list of integers.
                If left to None, Q_meas will be set to 0.

                **tol = 1e-6** - Tolerance criteria of controller convergence.
           """
    def __init__(self, net, controller_idx:int, in_service:bool=True, modus:str = None, q_droop_mvar = None,
                 bus_idx=None, vm_set_lb=None, vm_set_ub=None, pf_overexcited=None, pf_underexcited=None,
                 input_element_q_meas:str = None, input_variable_q_meas = None, input_element_index_q_meas = None, tol=1e-6,
                 order=-1, level=0, drop_same_existing_ctrl=False, matching_params=None, **kwargs):
        super().__init__(net, in_service=in_service, order=order, level=level, drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params)
        # TODO: implement maximum and minimum of droop control
        # write kwargs in self
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.q_droop_mvar = q_droop_mvar #droop in Q_ctrl
        self.input_element_q_meas = input_element_q_meas
        self.input_variable_q_meas = input_variable_q_meas
        self.input_element_index_q_meas = input_element_index_q_meas
        self.bus_idx = bus_idx
        self.vm_pu = None
        self.vm_pu_old = self.vm_pu
        self.controller_idx = controller_idx
        self.vm_set_pu = net.controller.at[self.controller_idx, "object"].set_point
        self.vm_set_pu_new = None
        self.lb_voltage = vm_set_lb
        self.ub_voltage = vm_set_ub
        self.tol = tol
        self.read_flag, self.input_variable = _detect_read_write_flag(net, "res_bus", bus_idx, "vm_pu")
        self.q_set_mvar_bsc = None
        self.q_set_mvar = None
        self.q_set_old_mvar = None
        self.diff = None
        self.converged = False
        self.pf_over = pf_overexcited
        self.pf_under = pf_underexcited
        self.p_cosphi = None #selection of droop modus for pf_ctrl
        ###catch modus and deprecated attribute voltage_ctrl
        if modus is None:#catching old attribute voltage_ctrl
            if hasattr(self, 'voltage_ctrl'):
                modus = self.voltage_ctrl
                if not hasattr(self, '_deprecation_warned'):#only one message that voltage ctrl is deprecated
                    logger.warning(
                        f"'voltage_ctrl' in Controller {self.index} is deprecated. "
                        "Use 'modus' ('Q_ctrl', 'V_ctrl', etc.) instead.")
                    self._deprecation_warned = True
        ###atching old implementation
        if type(modus) == bool and modus == True:
            modus = "V_ctrl"
            logger.warning(f"Deprecated Control Modus in Controller {self.index}, using V_ctrl from available types"
                         f" 'Q_ctrl', 'V_ctrl' or 'PF_ctrl'\n")
        elif type(modus) == bool and modus == False:
            modus = "Q_ctrl"
            logger.warning(f"Deprecated Control Modus in Controller {self.index}, using Q_ctrl from available types"
                         f" 'Q_ctrl', 'V_ctrl' or 'PF_ctrl'\n")

        if modus == "PF_ctrl_cap" or modus == "PF_ctrl_ind" or modus == 'PF_ctrl' or modus == 'PF_ctrl_P':#PF(P) control
            if modus != 'PF_ctrl_P':
                logger.warning(f"Power Factor Droop Control in Controller {self.index}: Modus is ambivalent, using"
                               f" 'PF_ctrl_P' from available modi: 'PF_ctrl_P' and 'PF_ctrl_V'\n")
            self.modus = 'PF_ctrl'
            self.p_cosphi = True
        elif modus == 'PF_ctrl_V':#PF(V) control
            self.modus = 'PF_ctrl'
            self.p_cosphi = False
        else:
            if modus == 'Q_ctrl' or modus == "V_ctrl":
                if self.vm_set_pu is None and modus == 'V_ctrl': #catching missing voltage set point
                    raise UserWarning(f'vm_set_pu must be a number, not {type(self.vm_set_pu)} in Controller {self.index}')
                self.modus = modus
            else:
                raise UserWarning(f'Droop Control Modus {modus} not decipherable in Controller {self.index}')
        #checking if Droop and BS Controller have the same modus
        if self.modus != net.controller.at[self.controller_idx, 'object'].modus:
            if (self.modus != 'PF_ctrl_P' and self.modus != 'PF_ctrl_V' and #droop included in modus string
                net.controller.at[self.controller_idx, 'object'].modus != True and self.modus != True):#conversion in progress
                logger.warning(f"Discrepancy between BinarySearchController Modus and Droop Controller Modus in {self.index}."
                               f"Using Droop Modus {net.controller.at[self.controller_idx, 'object'].modus}")
                self.modus = net.controller.at[self.controller_idx, 'object'].modus
        ###checking for values
        if self.modus == 'PF_ctrl': #catching missing values
            if self.lb_voltage is None or self.ub_voltage is None:
                raise UserWarning(f'Input error, vm_set_lb and vm_set_ub must be a number in Controller {self.index}')
            if self.lb_voltage < 0 or self.ub_voltage < 0:
                if self.p_cosphi:
                    raise UserWarning(f'P_Maximum (vm_set_ub) and P_Minimum (vm_set_lb) must be >= 0 W in Controller {self.index}')
                elif not self.p_cosphi:
                    raise UserWarning(f'V_Maximum (vm_set_ub) and V_Minimum (vm_set_lb) must be >= 0 pu in Controller {self.index}')
                else:
                    raise UserWarning(f'Something wrong with the entered values {self.lb_voltage, self.ub_voltage} in Controller {self.index}')
            if self.pf_over is None or self.pf_under is None:
                logger.warning(f'pf_overexcited and pf_underexcited must be number, not {self.pf_over}, {self.pf_under}. Using'
                               f'0.8 and 0.2.')
                self.pf_over = 0.8
                self.pf_under = 0.2
            if  1 < self.pf_over < 0 or 1 < self.pf_under < 0:
                raise UserWarning(f'Power Factor limits pf_overexcited and pf_underexcited must be between 0 and 1 in Controller {self.index}')
            if self.lb_voltage == self.ub_voltage:
                if self.p_cosphi:
                    raise UserWarning(f'P_Maximum and P_Minimum may not be the same value in Controller {self.index}')
                elif not self.p_cosphi:
                    raise UserWarning(f'V_Maximum and V_Minimum must not be the same value in Controller {self.index}')
                else:
                    raise UserWarning(f'Something wrong with the entered values {self.lb_voltage, self.ub_voltage} in Controller {self.index}')


    def __getattr__(self, name):
        if name == "modus":
            if not hasattr(self, '_deprecation_warned'):
                logger.warning(
                    f"'voltage_ctrl' in Controller {self.index} is deprecated. "
                    "Use 'modus' ('Q_ctrl', 'V_ctrl', etc.) instead."
                )
                self._deprecation_warned = True  # only one message that voltage ctrl is deprecated
            return self.voltage_ctrl
        raise AttributeError(f"{self.__class__.__name__!r} has no attribute {name!r}")# Raises AttributeError if missing

    def is_converged(self, net):
        ###check convergence
        self.in_service = net.controller.in_service[self.index]
        if not self.in_service:
            return True
        if self.modus != net.controller.at[self.controller_idx, 'object'].modus:#checking if droop and bsc have the same modus
            if (self.modus != 'PF_ctrl_P' and self.modus != 'PF_ctrl_V' and #here the droop is included in the string
                (net.controller.at[self.controller_idx, 'object'].modus != True and self.modus != True)):#converting in process
                logger.warning(f"Discrepancy between BinarySearchController Modus and Droop Controller Modus in {self.index}."
                               f"Using Droop Modus {net.controller.at[self.controller_idx, 'object'].modus}")
                self.modus = net.controller.at[self.controller_idx, 'object'].modus
        if type(self.modus) == bool and self.modus == True:#catching deprecated modi in old imports
            self.modus = "V_ctrl"  # catching old implementation
            logger.warning(f"Deprecated Control Modus in Controller {self.index}, using V_ctrl from available types\n")
        elif type(self.modus) == bool and self.modus == False:
            self.modus = "Q_ctrl"
            logger.warning(f"Deprecated Control Modus in Controller {self.index}, using Q_ctrl from available types\n")

        if self.modus == 'V_ctrl': #voltage droop
            ###backwards compatibility
            if (hasattr(self, 'bus_idx') and net.controller.at[
                self.controller_idx, 'object'].input_element != "res_bus" and
                    self.bus_idx is not None):
                logger.warning(f"Attribute 'bus_idx' in Droop controller {self.index} is deprecated for modus V_ctrl,"
                               f"please select the bus via the 'input_element_index' attribute of the linked binary search"
                               f"controller {self.controller_idx}. Attempting to use bus index {self.bus_idx}.")
                self.input_element_q_meas = net.controller.at[self.controller_idx, 'object'].input_element
                self.input_variable_q_meas = net.controller.at[self.controller_idx, 'object'].input_variable
                self.input_element_index_q_meas = net.controller.at[self.controller_idx, 'object'].input_element_index
                net.controller.at[self.controller_idx, 'object'].input_element = "res_bus"
                net.controller.at[self.controller_idx, 'object'].input_element_index = self.bus_idx
                net.controller.at[self.controller_idx, 'object'].input_variable = "vm_pu"
                self.bus_idx = None
            if hasattr(self, 'bus_idx') and self.bus_idx is not None: #Q_ctrl
                logger.warning(f"Specified 'bus_idx' in Controller {self.index} for modus 'V_ctrl', defaulting to "
                               f"Q_ctrl\n")
                self.modus = 'Q_ctrl'
                counter = 0
                input_values = []  # getting Q values
                for input_index in net.controller.at[self.controller_idx, "object"].input_element_index:
                    input_values.append(
                        read_from_net(net, net.controller.at[self.controller_idx, "object"].input_element, input_index,
                                      net.controller.at[self.controller_idx, "object"].input_variable[counter],
                                      net.controller.at[self.controller_idx, "object"].read_flag[counter]))
                    counter += 1
                self.diff = net.controller.at[self.controller_idx, "object"].set_point - sum(input_values)
            else: #true V_ctrl
                self.diff = (net.controller.at[self.controller_idx, "object"].set_point -
                    read_from_net(net, "res_bus", np.atleast_1d(
                    net.controller.at[self.controller_idx,'object'].input_element_index)[0], "vm_pu", 'auto'))
        elif str(self.modus).startswith('PF_ctrl'):
            if self.q_set_old_mvar is not None and self.q_set_mvar:
                self.diff = self.q_set_mvar - self.q_set_old_mvar
            else:
                counter = 0
                input_values = []
                p_input_values = []
                for input_index in net.controller.at[self.controller_idx, "object"].input_element_index:
                    input_values.append( #getting Q values
                        read_from_net(net, net.controller.at[self.controller_idx, "object"].input_element, input_index,
                                      net.controller.at[self.controller_idx, "object"].input_variable[counter],
                                      net.controller.at[self.controller_idx, "object"].read_flag[counter]))
                    p_input_values.append( #getting P values
                        read_from_net(net, net.controller.at[self.controller_idx, "object"].input_element, input_index,
                                      net.controller.at[self.controller_idx, "object"].input_variable_p[counter],
                                      net.controller.at[self.controller_idx, "object"].read_flag[counter]))
                    counter += 1
                q_set = net.controller.at[self.controller_idx, "object"].reactance * sum(p_input_values) * (
                    np.tan(np.arccos(net.controller.at[self.controller_idx, "object"].set_point)))#calculating set point from linked controller
                self.diff = q_set - sum(input_values)/len(input_values)

        elif self.modus == 'tan(phi)_ctrl':
            raise UserWarning(f'No droop option for tan(phi) controller {self.index}')
        else:
            if self.modus != 'Q_ctrl':
                logger.warning(f'No specified modus in droop controller {self.index}, using Q_ctrl\n')
            counter = 0
            input_values = [] #getting Q values
            for input_index in net.controller.at[self.controller_idx, "object"].input_element_index:
                input_values.append(
                    read_from_net(net, net.controller.at[self.controller_idx, "object"].input_element, input_index,
                                  net.controller.at[self.controller_idx, "object"].input_variable[counter],
                                  net.controller.at[self.controller_idx, "object"].read_flag[counter]))
                counter += 1
            self.diff = net.controller.at[self.controller_idx, "object"].set_point - sum(input_values)
        # bigger differences with switches as input elements, increase tolerance
        #if net.controller.at[self.controller_idx, "object"].input_element == "res_switch":
        #    self.tol = 0.2

        if self.modus != 'V_ctrl' and self.modus != 'PF_ctrl': #Convergence
            self.converged = np.all(np.abs(self.diff) < self.tol)
        else: #Convergence for voltage control and PF_ctrl
            if np.all(np.abs(self.diff) < self.tol):
                self.converged = net.controller.at[self.controller_idx, "object"].converged
            elif net.controller.at[self.controller_idx, "object"].diff_old is not None:
                net.controller.at[self.controller_idx, "object"].overwrite_convergence = True
        return self.converged

    def control_step(self, net):
        self._droop_control_step(net)

    def _droop_control_step(self, net):
        ###calculating new set point###
        if type(self.modus) == bool and self.modus == True:
            self.modus = "V_ctrl" #catching old implementation when importing from json
        elif type(self.modus) == bool and self.modus == False:
            self.modus = "Q_ctrl"
        if self.modus != 'V_ctrl' and not getattr(self, 'p_cosphi', False): #getting voltage
            self.vm_pu = read_from_net(net, "res_bus", self.bus_idx, "vm_pu", self.read_flag)
        elif self.modus == 'V_ctrl':
            self.vm_pu = net.controller.at[self.controller_idx,'object'].set_point
        self.vm_pu_old = self.vm_pu

        if self.modus=='Q_ctrl':
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

        elif self.modus == 'PF_ctrl':
            counter = 0
            input_values = []
            p_input_values = [] #P_values if p_cosphi, V_values if not
            for input_index in net.controller.at[self.controller_idx, "object"].input_element_index:
                input_values.append( #getting Q values
                    read_from_net(net, net.controller.at[self.controller_idx, "object"].input_element, input_index,
                                  net.controller.at[self.controller_idx, "object"].input_variable[counter],
                                  net.controller.at[self.controller_idx, "object"].read_flag[counter]))
                if self.p_cosphi:
                    p_input_values.append( #getting P values if PF(P) control
                        read_from_net(net, net.controller.at[self.controller_idx, "object"].input_element, input_index,
                                      net.controller.at[self.controller_idx, "object"].input_variable_p[counter],
                                      net.controller.at[self.controller_idx, "object"].read_flag[counter]))
                else: #PF_U control
                    if counter == 0: p_input_values = np.array([self.vm_pu]) #only get values once
                counter += 1
            ###getting reactance and checking if limit is reached
            if self.lb_voltage > self.ub_voltage: #phi overexcited > phi underexcited
                if self.lb_voltage <= sum(p_input_values)/len(p_input_values) or sum(p_input_values)/len(p_input_values) <= -self.lb_voltage:#underexcited limit(-1)
                    pf_cosphi = self.pf_under
                    net.controller.at[self.controller_idx, "object"].reactance = -1
                elif -self.ub_voltage <= sum(p_input_values)/len(p_input_values) <= self.ub_voltage:# overexcited limit (1)
                    pf_cosphi = self.pf_over
                    net.controller.at[self.controller_idx, "object"].reactance = 1
                else: #droop
                    m = ((1-self.pf_under) + (1-self.pf_over)) / (self.ub_voltage - self.lb_voltage) #getting function
                    b = (1-self.pf_over) - m * self.ub_voltage
                    ##droop set point##
                    if sum(p_input_values)/len(p_input_values) >= 0:
                        droop_set_point = m * sum(p_input_values)/len(p_input_values) + b
                    else:#f(x)=f(-x) for p<0
                        droop_set_point = m * -sum(p_input_values)/len(p_input_values) + b
                    if droop_set_point < 0: #reactance from droop_set_point
                        net.controller.at[self.controller_idx, "object"].reactance = -1
                    else:
                        net.controller.at[self.controller_idx, "object"].reactance = 1
                    pf_cosphi = (1-abs(droop_set_point)) #pass on new set point

            elif self.lb_voltage < self.ub_voltage: #phi overexcited < phi underexcited
                if -self.ub_voltage >= sum(p_input_values)/len(p_input_values) or sum(p_input_values)/len(p_input_values) >= self.ub_voltage:#overexcited limit (1)
                    pf_cosphi = self.pf_over
                    net.controller.at[self.controller_idx, "object"].reactance = 1
                elif -self.lb_voltage <= sum(p_input_values)/len(p_input_values) <= self.lb_voltage:# underexcited limit(-1)
                    pf_cosphi = self.pf_under
                    net.controller.at[self.controller_idx, "object"].reactance = -1
                else:#droop
                    m = ((1-self.pf_under)+ (1-self.pf_over)) / (self.ub_voltage - self.lb_voltage) #getting function
                    b = -(1-self.pf_under) - m * self.lb_voltage
                    ##droop set point##
                    if sum(p_input_values)/len(p_input_values) >= 0:
                        droop_set_point = (m * sum(p_input_values)/len(p_input_values) + b)
                    else: #f(x) = f(-x) for p<0
                        droop_set_point = (m * -sum(p_input_values)/len(p_input_values) + b)
                    if droop_set_point >= 0: #reactance from droop_set_point
                        net.controller.at[self.controller_idx, "object"].reactance = 1
                    else:
                        net.controller.at[self.controller_idx, "object"].reactance = -1
                    pf_cosphi = (1-abs(droop_set_point)) #pass on new set point
            else:
                raise UserWarning(f'error with limits {self.lb_voltage, self.ub_voltage} in Controller {self.index}')
            self.q_set_old_mvar, self.q_set_mvar = self.q_set_mvar, pf_cosphi

        else: #V_ctrl and wrong strings
            if self.modus != "V_ctrl":
                logger.error(f"No Droop Controller Modus specified for Controller {self.index}, using V_ctrl.\n"
                             "Please specify 'modus' ('Q_ctrl', 'V_ctrl', 'PF_ctrl' or 'tan(phi)_ctrl')\n")
            if self.q_set_mvar is not None:
                self.q_set_old_mvar, self.q_set_mvar = (
                self.q_set_mvar, self.q_set_mvar - (self.vm_set_pu - self.vm_pu) * self.q_droop_mvar)

        ###applying new set point###
        if self.q_set_old_mvar is not None: #second step
            self.diff = self.q_set_mvar - self.q_set_old_mvar
        if self.q_set_mvar is not None: #q_set_mvar was calculated beforehand
            net.controller.at[self.controller_idx, "object"].set_point = self.q_set_mvar
        else:
            if not hasattr(self, 'input_element_index_q_meas'):
                logger.error(f"No measurement point for Q value specified in Droop controller {self.index}, attempting to"
                             f"use point specified in controller {self.controller_idx}.")
                self.input_element_index_q_meas = net.controller.at[self.controller_idx, 'object'].input_element_index
                self.input_element_q_meas = net.controller.at[self.controller_idx, 'object'].input_element
                self.input_variable_q_meas = net.controller.at[self.controller_idx, 'object'].input_variable
            if self.input_element_index_q_meas is None or self.input_element_q_meas == 'res_bus':
                input_values = np.atleast_1d(0)#Q_meas = 0
            else:
                input_element = self.input_element_q_meas #net.controller.at[self.controller_idx, "object"].input_element
                input_element_index = self.input_element_index_q_meas #net.controller.at[self.controller_idx, "object"].input_element_index
                input_variable = np.atleast_1d(self.input_variable_q_meas)#net.controller.at[self.controller_idx, "object"].input_variable
                read_flag = net.controller.at[self.controller_idx, "object"].read_flag
                input_values = []
                counter = 0
                for input_index in np.atleast_1d(input_element_index):
                    input_values.append(read_from_net(net, input_element, input_index, str(input_variable[counter]), read_flag[counter]))
            self.vm_set_pu_new = self.vm_set_pu + sum(input_values) / self.q_droop_mvar #only sum, not divided by elements
            net.controller.at[self.controller_idx, "object"].set_point = self.vm_set_pu_new