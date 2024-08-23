import numpy as np
import pandapower as pp
from pandapower.control.basic_controller import Controller

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

class ShuntController(Controller):
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

# TODO: remove net copy, make similar to secant controller
class ContinuousShuntController(ShuntController):
    def __init__(self, net, shunt_index, vm_set_pu, bus_index=None, tol=1e-3, eps=1e-4, in_service=True,
                 check_step_bounds=True, order=0, level=0, **kwargs):
        super().__init__(net, shunt_index=shunt_index, bus_index=bus_index, tol=tol,
                         in_service=in_service, check_step_bounds=check_step_bounds, order=order,
                         level=level, **kwargs)
        self.vm_set_pu = vm_set_pu
        self.eps = eps
        net.shunt.step = net.shunt.step.astype(np.float64)

    def control_step(self, net):
        delta_vm_pu = net.res_bus.at[self.controlled_bus, 'vm_pu'] - self.vm_set_pu

        copy_net = net.deepcopy()
        copy_net.shunt.loc[self.shunt_index, 'step'] = self.step + self.eps
        pp.runpp(copy_net)
        delta_vm_pu_p = copy_net.res_bus.at[self.controlled_bus, 'vm_pu'] - self.vm_set_pu

        tc = self.step - delta_vm_pu / ((delta_vm_pu_p - delta_vm_pu) / self.eps)
        # logger.info(f"delta_vm_pu: {delta_vm_pu}, delta_vm_pu_p: {delta_vm_pu_p}, "
        #             f"step: {self.step}, tc: {tc}")

        self.step = tc
        if self.check_step_bounds:
            self.step = np.clip(self.step, self.step_min, self.step_max)

        # Write to net
        net.shunt.at[self.shunt_index, 'step'] = self.step

    def is_converged(self, net):
        if not net.shunt.at[self.shunt_index, 'in_service']:
            return True

        vm_pu = net.res_bus.at[self.controlled_bus, "vm_pu"]
        difference = 1 - self.vm_set_pu / vm_pu

        if self.check_step_bounds:
            if vm_pu > self.vm_set_pu and self.step == self.step_min:
                return True
            elif vm_pu < self.vm_set_pu and self.step == self.step_max:
                return True

        return abs(difference) < self.tol


class DiscreteShuntController(ShuntController):
    def __init__(self, net, shunt_index, vm_set_pu, bus_index=None, tol=1e-3, reset_at_init=False,
                 in_service=True, check_step_bounds=True, order=0, level=0, matching_params=None,
                 increment=1, **kwargs):
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
            # logger.debug(f"tol: {net.shunt.at[self.shunt_index, 'name']} vm_pu: {vm_pu}, "
            #              f"set_u: {self.vm_set_pu}")
            return True

        if self.check_step_bounds:
            if net.shunt.at[self.shunt_index, 'q_mvar'] >= 0:
                if vm_pu < self.vm_set_pu and self.step == self.step_min:
                    # logger.debug(f"step min: {net.shunt.at[self.shunt_index, 'name']}")
                    return True
                elif vm_pu > self.vm_set_pu and self.step == self.step_max:
                    # logger.debug(f"step max: {net.shunt.at[self.shunt_index, 'name']}")
                    return True
            else:
                if vm_pu < self.vm_set_pu and self.step == self.step_max:
                    # logger.debug(f"step min: {net.shunt.at[self.shunt_index, 'name']}")
                    return True
                elif vm_pu > self.vm_set_pu and self.step == self.step_min:
                    # logger.debug(f"step max: {net.shunt.at[self.shunt_index, 'name']}")
                    return True
        return False