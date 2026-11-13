import logging

import numpy as np

from pandapower.control.basic_controller import Controller
from pandapower.network import pandapowerNet
from pandapower.pp_types import Int

logger = logging.getLogger(__name__)


class ShuntController(Controller):
    """
    Base Shunt Controller for controlling the steps of a shunt in a power network.

    The `ShuntController` provides a basic framework for controlling a shunt in a pandapower network. It adjusts the
    reactive power compensation provided by a shunt in discrete steps to influence the voltage level at a particular
    bus. This class is intended to be extended by more specific controller implementations, such as the
    `DiscreteShuntController`.

    Parameters:
        net: pandapower network object
        shunt_index: The index of the shunt in the pandapower network that is being controlled.
        bus_index: The index of the bus that the controller monitors and regulates the voltage at. If None,
            the bus connected to the shunt is used by default.
        tol: Voltage tolerance in per-unit (pu) for controlling the bus voltage. This defines the
            acceptable range around the setpoint where no control action is taken.
        in_service: Boolean flag indicating whether the controller is currently active or not.
        check_step_bounds: Flag to check if the shunt steps should be constrained within predefined minimum
            and maximum step limits.
        order: The execution order of the controller, which determines when this controller is executed
            relative to others in the simulation.
        level: The hierarchy level of the controller in multi-level control schemes, allowing control actions
            to be prioritized.

    Keyword Args:
        Additional keyword arguments passed to the base Controller class.
    """

    def __init__(
        self,
        net: pandapowerNet,
        shunt_index: Int,
        bus_index: Int | None = None,
        tol: float = 1e-3,
        in_service: bool = True,
        check_step_bounds: bool = True,
        order: int = 0,
        level: int = 0,
        **kwargs,
    ):
        super().__init__(net, in_service=in_service, order=order, level=level, **kwargs)
        self.tol = tol
        self.shunt_index = shunt_index
        self.element_in_service = net.shunt.loc[self.shunt_index, "in_service"]
        if bus_index is None:
            self.controlled_bus = net.shunt.at[self.shunt_index, "bus"]
        else:
            self.controlled_bus = bus_index

        self.step = net.shunt.at[shunt_index, "step"]

        self.check_step_bounds = check_step_bounds
        if check_step_bounds:
            self.step_min = 0
            self.step_max = net.shunt.at[self.shunt_index, "max_step"]

        ext_grid_buses = net.ext_grid.loc[net.ext_grid.in_service, "bus"].values
        if self.controlled_bus in ext_grid_buses:
            logger.warning("Controlled Bus is Slack Bus - deactivating controller")
            self.set_active(net, False)


class DiscreteShuntController(ShuntController):
    """
    Discrete Shunt Controller that controls the shunt steps in a discrete manner to regulate the voltage
    at a specific bus. This controller adjusts the active and reactive power of a shunt in steps based on the
    voltage deviation from a setpoint.

    Parameters:
        net: pandapower network object
        shunt_index: The index of the shunt in the pandapower network to be controlled.
        vm_set_pu: The voltage setpoint in per-unit (pu) for controlling the bus voltage.
        bus_index: The index of the bus where voltage control is applied. If None, the bus connected to the
            shunt is used by default.
        tol: Voltage tolerance band in per-unit (pu) for control action (default is 1% or 0.01 pu).
        increment: Step increment size for controlling the shunt. The controller adjusts the shunt steps by
            this increment based on the voltage deviation.
        reset_at_init: If True, the shunt steps will be reset to 0 during the initialization of the
            controller.
        in_service: Boolean flag to indicate whether the controller is active or not.
        check_step_bounds: If True, the controller will check and enforce the step boundaries
            (minimum and maximum) for the shunt.
        order: Execution order of the controller. Controllers with lower order are executed first.
        level: Controller level.
        matching_params: Dictionary of parameters used to match this controller with the appropriate
            elements in the network. Defaults to shunt_index and bus_index.
    """

    def __init__(
        self,
        net: pandapowerNet,
        shunt_index: Int,
        vm_set_pu: float,
        bus_index: Int | None = None,
        tol: float = 1e-3,
        increment: int = 1,
        reset_at_init: bool = False,
        in_service: bool = True,
        check_step_bounds: bool = True,
        order: int = 0,
        level: int = 0,
        matching_params: dict | None = None,
        **kwargs,
    ):
        if matching_params is None:
            matching_params = {"shunt_index": shunt_index, "bus_index": bus_index}
        super().__init__(
            net,
            shunt_index=shunt_index,
            bus_index=bus_index,
            tol=tol,
            in_service=in_service,
            check_step_bounds=check_step_bounds,
            order=order,
            level=level,
            matching_params=matching_params,
            **kwargs,
        )
        self.reset_at_init = reset_at_init

        self.vm_set_pu = vm_set_pu
        self.step = net.shunt.at[self.shunt_index, "step"]
        self.increment = increment
        if not check_step_bounds:
            net.shunt.step = net.shunt.step.astype(np.int64)

    def initialize_control(self, net):
        if self.reset_at_init:
            self.step = 0
            net.shunt.at[self.shunt_index, "step"] = 0

    def control_step(self, net):
        vm_pu = net.res_bus.at[self.controlled_bus, "vm_pu"]
        self.step = net.shunt.at[self.shunt_index, "step"]

        sign = np.sign(net.shunt.at[self.shunt_index, "q_mvar"])
        if vm_pu > self.vm_set_pu + self.tol:
            self.step += self.increment * sign
        elif vm_pu <= self.vm_set_pu - self.tol:
            self.step -= self.increment * sign

        if self.check_step_bounds:
            self.step = np.clip(self.step, self.step_min, self.step_max)

        # Write to net
        net.shunt.at[self.shunt_index, "step"] = self.step

    def is_converged(self, net):
        if not net.shunt.at[self.shunt_index, "in_service"]:
            return True

        vm_pu = net.res_bus.at[self.controlled_bus, "vm_pu"]
        if abs(vm_pu - self.vm_set_pu) < self.tol:
            return True

        if self.check_step_bounds:
            if net.shunt.at[self.shunt_index, "q_mvar"] >= 0:
                if (
                    vm_pu < self.vm_set_pu
                    and self.step == self.step_min
                    or vm_pu > self.vm_set_pu
                    and self.step == self.step_max
                ):
                    return True
            else:
                if (
                    vm_pu < self.vm_set_pu
                    and self.step == self.step_max
                    or vm_pu > self.vm_set_pu
                    and self.step == self.step_min
                ):
                    return True
        return False
