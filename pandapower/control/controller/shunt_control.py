import numpy as np
from pandapower.control.basic_controller import Controller

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

class ShuntController(Controller):
    """
    Base Shunt Controller for controlling the steps of a shunt in a power network.

    The `ShuntController` provides a basic framework for controlling a shunt in a pandapower network. It adjusts
    the reactive power compensation provided by a shunt in discrete steps to influence the voltage level at a
    particular bus. This class is intended to be extended by more specific controller implementations, such as
    the `DiscreteShuntController`.

    INPUT:
        **net** (attrdict) - pandapower network object

        **shunt_index** (int) - The index of the shunt in the pandapower network that is being controlled.

    OPTIONAL:
        **bus_index** (int, None) - The index of the bus that the controller monitors and regulates the voltage at.
                                    If None, the bus connected to the shunt is used by default.

        **tol** (float, 0.001) - Voltage tolerance in per-unit (pu) for controlling the bus voltage. This defines
                                 the acceptable range around the setpoint where no control action is taken.

        **in_service** (bool, True) - Boolean flag indicating whether the controller is currently active or not.

        **check_step_bounds** (bool, True) - Flag to check if the shunt steps should be constrained within predefined
                                             minimum and maximum step limits.

        **order** (int, 0) - The execution order of the controller, which determines when this controller is executed
                             relative to others in the simulation.

        **level** (int, 0) - The hierarchy level of the controller in multi-level control schemes, allowing control
                             actions to be prioritized.

        **kwargs** - Additional keyword arguments passed to the base Controller class.
    """
    def __init__(self, net, shunt_index, bus_index=None, tol=1e-3, in_service=True,
                 check_step_bounds=True, order=0, level=0, **kwargs):
        super().__init__(net, in_service=in_service, order=order, level=level,
                         **kwargs)
        self.tol = tol
        self.shunt_index = shunt_index
        self.element_in_service = net.shunt.loc[self.shunt_index, 'in_service']
        if bus_index is None:
            self.controlled_bus = net.shunt.at[self.shunt_index, 'bus']
        else:
            self.controlled_bus = bus_index

        self.step = net.shunt.at[shunt_index, 'step']

        self.check_step_bounds = check_step_bounds
        if check_step_bounds:
            self.step_min = 0
            self.step_max = net.shunt.at[self.shunt_index, 'max_step']

        ext_grid_buses = net.ext_grid.loc[net.ext_grid.in_service, 'bus'].values
        if self.controlled_bus in ext_grid_buses:
            logging.warning("Controlled Bus is Slack Bus - deactivating controller")
            self.set_active(net, False)

class DiscreteShuntController(ShuntController):
    """
    Discrete Shunt Controller that controls the shunt steps in a discrete manner to regulate the voltage
    at a specific bus. This controller adjusts the active and reactive power of a shunt in steps based on the
    voltage deviation from a setpoint.

    INPUT:
        **net** (attrdict) - pandapower network object

        **shunt_index** (int) - The index of the shunt in the pandapower network to be controlled.

        **vm_set_pu** (float) - The voltage setpoint in per-unit (pu) for controlling the bus voltage.

    OPTIONAL:
        **bus_index** (int, None) - The index of the bus where voltage control is applied. If None, the bus
                                    connected to the shunt is used by default.

        **tol** (float, 0.001) - Voltage tolerance band in per-unit (pu) for control action (default is 1% or 0.01 pu).

        **increment** (int, 1) - Step increment size for controlling the shunt. The controller adjusts the
                                 shunt steps by this increment based on the voltage deviation.

        **reset_at_init** (bool, False) - If True, the shunt steps will be reset to 0 during the initialization
                                          of the controller.

        **in_service** (bool, True) - Boolean flag to indicate whether the controller is active or not.

        **check_step_bounds** (bool, True) - If True, the controller will check and enforce the step boundaries
                                             (minimum and maximum) for the shunt.

        **order** (int, 0) - Execution order of the controller. Controllers with lower order are executed first.

        **level** (int, 0) - Controller level.

        **matching_params** (dict, None) - Dictionary of parameters used to match this controller with the appropriate
                                           elements in the network. Defaults to shunt_index and bus_index.

    """
    def __init__(self, net, shunt_index, vm_set_pu, bus_index=None, tol=1e-3, increment=1, reset_at_init=False,
                 in_service=True, check_step_bounds=True, order=0, level=0, matching_params=None, **kwargs):
        if matching_params is None:
            matching_params = {"shunt_index": shunt_index, "bus_index": bus_index}
        super().__init__(net, shunt_index=shunt_index, bus_index=bus_index, tol=tol,
                         in_service=in_service,
                         check_step_bounds=check_step_bounds, order=order, level=level,
                         matching_params=matching_params, **kwargs)
        self.reset_at_init = reset_at_init

        self.vm_set_pu = vm_set_pu
        self.step = net.shunt.at[self.shunt_index, 'step']
        self.increment = increment
        if not check_step_bounds:
            net.shunt.step = net.shunt.step.astype(np.int64)

    def initialize_control(self, net):
        if self.reset_at_init:
            self.step = 0
            net.shunt.at[self.shunt_index, 'step'] = 0

    def control_step(self, net):
        vm_pu = net.res_bus.at[self.controlled_bus, 'vm_pu']
        self.step = net.shunt.at[self.shunt_index, "step"]

        sign = np.sign(net.shunt.at[self.shunt_index, 'q_mvar'])
        if vm_pu > self.vm_set_pu + self.tol:
            self.step += self.increment * sign
        elif vm_pu <= self.vm_set_pu - self.tol:
            self.step -= self.increment * sign

        if self.check_step_bounds:
            self.step = np.clip(self.step, self.step_min, self.step_max)

        # Write to net
        net.shunt.at[self.shunt_index, 'step'] = self.step

    def is_converged(self, net):
        if not net.shunt.at[self.shunt_index, 'in_service']:
            return True

        vm_pu = net.res_bus.at[self.controlled_bus, "vm_pu"]
        if abs(vm_pu - self.vm_set_pu) < self.tol:
            return True

        if self.check_step_bounds:
            if net.shunt.at[self.shunt_index, 'q_mvar'] >= 0:
                if vm_pu < self.vm_set_pu and self.step == self.step_min:
                    return True
                elif vm_pu > self.vm_set_pu and self.step == self.step_max:
                    return True
            else:
                if vm_pu < self.vm_set_pu and self.step == self.step_max:
                    return True
                elif vm_pu > self.vm_set_pu and self.step == self.step_min:
                    return True
        return False