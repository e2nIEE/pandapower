import numpy as np
from collections.abc import Sequence
import logging

from pandapower.control.basic_controller import Controller
from pandapower.auxiliary import _detect_read_write_flag, read_from_net, write_to_net

import logging

logger = logging.getLogger(__name__)


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

        **output_element_index** - Index or list of indices of the output element(s) in net.

        **output_element_in_service** - List indicating whether each output element is in service.

        **output_values_distribution** - Distribution of reactive power provision among output elements.


        **input_element** - Measurement location, can be a transformers, switches, lines or busses (only with
        voltage_ctrl), indicated by string value "res_trafo", "res_switch", "res_line" or "res_bus". In case of
        "res_switch", an additional small impedance is introduced in the switch.

        **input_variable** - Variable which is used to take the measurement from. Indicated by string value.

        **input_element** - Measurement location, can be "res_trafo", "res_switch", "res_line", or "res_bus".

        **input_inverted** - Boolean that indicates if the measurement of the input element must be inverted. Required
        when importing from PowerFactory.

        **input_element_index** - Element of input element in net.


        **input_element_index** - Index or list of indices of the input element(s) in net.
            
        **set_point** - Set point of the controller, can be a reactive power provision or a voltage set point. In
        case of voltage set point, voltage control must be set to true, bus_idx must be set to measurement bus and
        input_element must be "res_bus". Can be overwritten by a droop controller chained with the binary search
        control.

        **gen_Q_response** - List of +/- 1 that indicates the Q gen response of the measurement location. Used in
        order to invert the droop value of the controller.

        **voltage_ctrl** - Whether the controller is used for voltage control.

        **output_min_q_mvar** - Minimum reactive power limits for each output element.

        **output_max_q_mvar** - Maximum reactive power limits for each output element.

        **bus_idx=None** - Bus index used for voltage control.

        **tol=0.001** - Tolerance for controller convergence.

        **in_service=True** - Whether the controller itself is in service.

        **order=0** - Execution order of the controller.

        **level=0** - Execution level of the controller.

        **drop_same_existing_ctrl=False** - Whether to drop existing controllers with the same parameters.

        **matching_params=None** - Parameters for matching controllers.

        **name=""** - Name of the controller.

        **kwargs** - Additional keyword arguments.
    """
    def __init__(self, net, ctrl_in_service, output_element, output_variable, output_element_index,
                 output_element_in_service, output_values_distribution, input_element, input_variable,
                 input_element_index, set_point, voltage_ctrl, name="", input_inverted=[], gen_Q_response=[],
                 bus_idx=None, tol=0.001, in_service=True, order=0, level=0, drop_same_existing_ctrl=False,
                 matching_params=None, **kwargs):
        super().__init__(net, in_service=in_service, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params)
        self.name = name
        self.in_service = ctrl_in_service
        self.input_element = input_element
        self.input_element_index = []
        if isinstance(input_element_index, list):
            for element in input_element_index:
                self.input_element_index.append(element)
        else:
            self.input_element_index.append(input_element_index)
        self.input_sign = []
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
                                            for distribution, service in zip(self.output_values_distribution,
                                                                            self.output_element_in_service)],
                                            dtype=np.bool)

        n = len(self.output_element_index)
        if gen_Q_response is None or (isinstance(gen_Q_response, Sequence) and len(gen_Q_response) == 0):
            # empty, then set all entries to 1
            self.gen_Q_response = [1] * n
        else:
            if len(gen_Q_response) < n:
                gen_Q_response += [1] * (n - len(gen_Q_response))  # missing entries with +1
            self.gen_Q_response = gen_Q_response
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
        self.write_flag, self.output_variable = _detect_read_write_flag(net, output_element,
                                                                        output_element_index,
                                                                        output_variable)
        # write kwargs in self
        for key, value in kwargs.items():
            setattr(self, key, value)

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
            elif self.input_element == "res_impedance":
                self.input_element_in_service.append(net.impedance.in_service[input_index])
            elif self.input_element == "res_bus":
                self.input_element_in_service.append(net.bus.in_service[input_index])
            elif self.input_element == "res_gen":
                self.input_element_in_service.append(ctrl_in_service)


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
        self.output_values = read_from_net(net, self.output_element, self.output_element_index,
                                           self.output_variable, self.write_flag)
        self.output_adjustable = np.array([False if not distribution else service
                                            for distribution, service in zip(self.output_values_distribution,
                                                                            self.output_element_in_service)],
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
        self.input_element_in_service.clear()
        self.output_element_in_service.clear()
        for input_index in self.input_element_index:
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
        for output_index in self.output_element_index:
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

        # read input values
        input_values = []
        counter = 0
        for input_index in self.input_element_index:
            input_values.append(read_from_net(net, self.input_element, input_index,
                                              self.input_variable[counter], self.read_flag[counter]))
            counter += 1
        input_values = (self.input_sign * np.asarray(input_values)).tolist()
        # read previous set values
        # compare old and new set values
        if not self.voltage_ctrl or self.bus_idx is None:
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

        else:
            self.diff_old = self.diff

            if not any(self.output_adjustable):
                logging.info('V_Ctrl: All stations controlled by %s reached reactive power limits.' %self.name)
                self.converged = True
                return self.converged
            else:
                # adapt output adjustable depending on in_service
                self.output_adjustable = np.array([in_service and adjustable for in_service, adjustable
                                                   in zip(self.output_element_in_service, self.output_adjustable)], dtype=np.bool)

                # normalize the values distribution
                self._normalize_distribution_in_service()

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
            # initial or first step
            # is ok that values are set for all stations even though they are out of service or not adjustable --> following step will correct this
            self.output_values_old, self.output_values = self.output_values, self.output_values + 1e-3

            positions_not_adjustable = [i for i, val in enumerate(self.output_adjustable) if not val]
            for i in positions_not_adjustable:
                if self.output_values_distribution[i]==0 or not self.output_element_in_service[i] :
                    self.output_values[i] = 0
                else:
                    continue
        else:
            step_diff = self.diff - self.diff_old
            x = self.output_values - self.diff * (self.output_values - self.output_values_old) / np.where(
                step_diff == 0, 1e-6, step_diff)

            rel_cap = 2
            cap = rel_cap * (np.abs(self.output_values) + 1e-6) + 50  # add epsilon to avoid zero; absolute cap +50 MVAr

            delta = x - self.output_values
            delta = np.clip(delta, -cap, +cap)

            x = self.output_values + delta

            if not all(self.output_adjustable) and net._options['enforce_q_lims']:
                positions_adjustable = [i for i, val in enumerate(self.output_adjustable) if val] # gives which is/are adjustable
                positions_not_adjustable = [i for i, val in enumerate(self.output_adjustable) if not val] # can be one or multiple ## gives which is/are not adjustable anymore

                sum_adjustable = sum(x) - sum(self.output_values[positions_not_adjustable]) # anlagen, die noch adjustable sind, rest der Leistung muss noch erreicht werden
                x[positions_adjustable] = sum_adjustable * self.output_values_distribution[positions_adjustable]

                for i in positions_not_adjustable:
                    if self.output_element_in_service[i]:
                        x[i] = self.output_values[i] # reset value to q_limit
                    else:
                        x[i] = 0 # reset value to 0 because station is oout of service

            else:
                x = sum(x) * self.output_values_distribution

            if self.output_adjustable is not None and net._options['enforce_q_lims']: # none if output element is a shunt
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
                        positions = [i for i, val in enumerate(reached_max_qmvar) if val is np.True_] # can be one or multiple
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

        # write new set values
        write_to_net(net, self.output_element, self.output_element_index, self.output_variable, self.output_values,
                     self.write_flag)

    def _normalize_distribution_in_service(self, initial_pf_distribution=None):
        # normalize distribution depending on in service of stations
        if initial_pf_distribution is None:
            distribution = self.output_values_distribution
        else:
            distribution = initial_pf_distribution

        # normalize the values distribution
        # set output_values_distribution to 0, if station is not in service
        self.output_values_distribution = [0 if not in_service else value
                                           for in_service, value in zip(self.output_element_in_service, distribution)]
        total = np.sum(self.output_values_distribution)
        if total > 0:  # To avoid division by zero
            self.output_values_distribution = np.array(self.output_values_distribution, dtype=np.float64) / total
        else:
            self.output_values_distribution = np.zeros_like(self.output_values_distribution, dtype=np.float64)

    def _update_min_max_q_mvar(self, net):
        if 'min_q_mvar' in net[self.output_element].columns:
            self.output_min_q_mvar = np.nan_to_num(net[self.output_element].loc[self.output_element_index, 'min_q_mvar'].values, nan=-np.inf)
        else:
            self.output_min_q_mvar = np.array([-np.inf]*len(self.output_element_index), dtype=np.float64)
        if 'max_q_mvar' in net[self.output_element].columns:
            self.output_max_q_mvar = np.nan_to_num(net[self.output_element].loc[self.output_element_index, 'max_q_mvar'].values, nan=np.inf)
        else:
            self.output_max_q_mvar = np.array([np.inf]*len(self.output_element_index), dtype=np.float64)


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

        **vm_set_pu_bsc** - Inital voltage set point in case of voltage control.

        **controller_idx** - Index of linked Binary< search control (if present).

        **voltage_ctrl** - Whether the controller is used for voltage control or not.

        **bus_idx=None** - Bus index which is used for voltage control.

        **q_set_mvar_bsc** - Inital voltage set point in case of no voltage control.

        **tol=1e-6** - Tolerance criteria of controller convergence.

        **vm_set_lb=None** - Lower band border of dead band

        **vm_set_ub=None** - Upper band border of dead band
       """

    def __init__(self, net, q_droop_mvar, bus_idx, controller_idx, voltage_ctrl, tol=1e-6,
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
        if self.bus_idx is None:
            self.converged = np.all(np.abs(self.diff) < self.tol)
        else:
            self.converged = net.controller.at[self.controller_idx, "object"].converged

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
                    self.q_set_old_mvar, self.q_set_mvar = (self.q_set_mvar, self.q_set_mvar_bsc +
                                                            net.controller.object[self.controller_idx].gen_Q_response[0
                                                            ] * (self.ub_voltage - self.vm_pu) * self.q_droop_mvar)
                elif self.vm_pu < self.lb_voltage:
                    self.q_set_old_mvar, self.q_set_mvar = (self.q_set_mvar, self.q_set_mvar_bsc +
                                                            net.controller.object[self.controller_idx].gen_Q_response[0]
                                                             * (self.lb_voltage - self.vm_pu) * self.q_droop_mvar)
                else:
                    self.q_set_old_mvar, self.q_set_mvar = (self.q_set_mvar, self.q_set_mvar_bsc)
            else:
                self.q_set_old_mvar, self.q_set_mvar = (
                    self.q_set_mvar, self.q_set_mvar + net.controller.object[self.controller_idx].gen_Q_response[0] * (
                                self.vm_set_pu - self.vm_pu) * self.q_droop_mvar)

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
