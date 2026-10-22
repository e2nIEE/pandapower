# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import logging

import numpy as np

from pandapower.control.basic_controller import Controller

logger = logging.getLogger(__name__)


class StorageController(Controller):
    """
    Battery storage controller for time series simulation.

    In each time step the requested grid-side power ``p_request_mw`` (positive means charging) is clipped to the
    power limits and the remaining energy headroom, and the result is written to ``net.storage.p_mw``. The state
    of charge is committed in ``finalize_step``, after the output writer and only if the power flow converged. The
    OutputWriter therefore logs the state of charge at the start of each time step.

    Each time series run starts from the state of charge given at construction. Calls of ``runpp`` or
    ``run_control`` outside a time series do not change the state of charge.

    The energy limits are the storage capacity in MWh. The power limits and ``p_request_mw`` refer to the grid-side
    power ``p_mw * scaling`` in MW.

    ``energy_not_charged_mwh`` and ``energy_not_discharged_mwh`` are the parts of the request that the battery could
    not realise in the last committed step (grid-side MWh). The ``_total_mwh`` attributes sum them over the run. This
    energy is not lost.

    Parameters:
        net (pandapowerNet): The net that contains the storage
        element_index (int): Index of the storage in net.storage
        max_charge_mw (float): Maximum grid-side charging power [MW]
        max_discharge_mw (float): Maximum grid-side discharging power [MW]
        eta_charge (float, 1.0): Charging efficiency [-]
        eta_discharge (float, 1.0): Discharging efficiency [-]
        delta_t_h (float, 0.25): Length of one time step [h]
        p_request_mw (float, 0.0): Requested grid-side power [MW]
        in_service (bool, True): Controller in service
        order (int, 0), level (int, 0): Controller order and level
        drop_same_existing_ctrl (bool, False): Drop existing controllers with the same element_index
        **kwargs: Attributes that are set on the controller
    """

    def __init__(
        self,
        net,
        element_index,
        max_charge_mw,
        max_discharge_mw,
        eta_charge=1.0,
        eta_discharge=1.0,
        delta_t_h=0.25,
        p_request_mw=0.0,
        in_service=True,
        order=0,
        level=0,
        drop_same_existing_ctrl=False,
        **kwargs,
    ):
        """Create the controller for the storage ``element_index`` and read its capacity and state of charge."""
        storage = net.storage.loc[element_index]
        super().__init__(
            net,
            in_service=in_service,
            order=order,
            level=level,
            drop_same_existing_ctrl=drop_same_existing_ctrl,
            matching_params={"element_index": element_index},
        )
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.element_index = element_index
        self.max_charge_mw = max_charge_mw
        self.max_discharge_mw = max_discharge_mw
        self.eta_charge = eta_charge
        self.eta_discharge = eta_discharge
        self.delta_t_h = delta_t_h
        self.p_request_mw = p_request_mw
        self.max_e_mwh = storage.max_e_mwh
        self.min_e_mwh = storage.min_e_mwh
        self._initial_e_mwh = storage.soc_percent / 100 * self.max_e_mwh
        self.e_mwh = self._initial_e_mwh
        self._last_time = None
        self.last_committed_time = None
        self.energy_not_charged_mwh = 0.0
        self.energy_not_discharged_mwh = 0.0
        self.energy_not_charged_total_mwh = 0.0
        self.energy_not_discharged_total_mwh = 0.0

    def time_step(self, net, time):
        """Called at the start of each time step. A new run (first step, or time going back) resets the state."""
        if self._last_time is None or time <= self._last_time:
            self.e_mwh = self._initial_e_mwh
            self.last_committed_time = None
            self.energy_not_charged_mwh = self.energy_not_discharged_mwh = 0.0
            self.energy_not_charged_total_mwh = self.energy_not_discharged_total_mwh = 0.0
            net.storage.at[self.element_index, "soc_percent"] = self._initial_e_mwh / self.max_e_mwh * 100
        self._last_time = time

    def _p_mw_feasible(self, net):
        """Return the value for net.storage.p_mw [MW] that is allowed by the power and energy limits.

        Called by is_converged and control_step. Divides the grid-side power by the scaling factor.
        """
        charge_max = min(self.max_charge_mw, (self.max_e_mwh - self.e_mwh) / (self.eta_charge * self.delta_t_h))
        discharge_max = min(self.max_discharge_mw, (self.e_mwh - self.min_e_mwh) * self.eta_discharge / self.delta_t_h)
        p_grid = min(max(self.p_request_mw, -discharge_max), charge_max)
        return p_grid / net.storage.at[self.element_index, "scaling"]

    def is_converged(self, net):
        """Return True if net.storage.p_mw already has the allowed value, or if the storage is out of service.

        Called in the control loop of each time step.
        """
        if not net.storage.at[self.element_index, "in_service"]:
            return True
        return bool(np.isclose(net.storage.at[self.element_index, "p_mw"], self._p_mw_feasible(net)))

    def control_step(self, net):
        """Write the allowed value to net.storage.p_mw. Called in the control loop if is_converged is False."""
        if net.storage.at[self.element_index, "in_service"]:
            net.storage.at[self.element_index, "p_mw"] = self._p_mw_feasible(net)

    def finalize_step(self, net, time):
        """Advance the state of charge by one time step and write soc_percent.

        Called once per time step after the output writer, so the output shows the state at the start of the step.
        Skipped if the power flow did not converge, the storage is out of service, or the step is already committed.
        Returns True if the step was committed.
        """
        idx = self.element_index
        if time == self.last_committed_time or not net.get("converged", False):
            return False
        if not net.storage.at[idx, "in_service"]:
            return False
        p_grid = net.storage.at[idx, "p_mw"] * net.storage.at[idx, "scaling"]
        dt = self.delta_t_h
        e_next = self.e_mwh + (self.eta_charge * max(p_grid, 0.0) - max(-p_grid, 0.0) / self.eta_discharge) * dt
        if not self.min_e_mwh - 1e-9 <= e_next <= self.max_e_mwh + 1e-9:
            logger.warning(
                "storage %s: p_mw was changed after control_step, energy %.4g MWh at time %s is clamped",
                idx,
                e_next,
                time,
            )
        self.e_mwh = min(max(e_next, self.min_e_mwh), self.max_e_mwh)
        self.energy_not_charged_mwh = max(self.p_request_mw - p_grid, 0.0) * dt
        self.energy_not_discharged_mwh = max(p_grid - self.p_request_mw, 0.0) * dt
        self.energy_not_charged_total_mwh += self.energy_not_charged_mwh
        self.energy_not_discharged_total_mwh += self.energy_not_discharged_mwh
        net.storage.at[idx, "soc_percent"] = self.e_mwh / self.max_e_mwh * 100
        self.last_committed_time = time
        return True
