# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import logging

from pandapower.control.controller.storage_control import StorageController

logger = logging.getLogger(__name__)


class SelfConsumptionController(StorageController):
    """
    Rule-based battery for a site of loads and static generators on the storage bus.

    The storage requests the negative site power, i.e. it charges from a surplus and discharges to cover a deficit.
    The request is set in ``initialize_control`` from the in-service loads and sgens of the site
    (``p_mw * scaling``, in MW). The rule is instantaneous: it uses no forecasts and is not an optimal dispatch.
    Power that the battery cannot charge or discharge is exchanged with the grid, which is not curtailment.

    A warning is logged once per committed time step if the battery is full or empty and the energy limit clips
    the request.

    Parameters:
        net (pandapowerNet): The net that contains the storage, loads and sgens
        element_index (int): Index of the storage in net.storage
        max_charge_mw (float): Maximum grid-side charging power [MW]
        max_discharge_mw (float): Maximum grid-side discharging power [MW]
        load_indices (list of int): Loads of the site
        sgen_indices (list of int): Static generators of the site
        **kwargs: Passed to StorageController

    Raises:
        ValueError: If a site element is not connected to the storage bus.
    """

    def __init__(self, net, element_index, max_charge_mw, max_discharge_mw, load_indices, sgen_indices, **kwargs):
        """Create the controller. Raises ValueError if a site element is not on the storage bus."""
        self.load_indices = list(load_indices)
        self.sgen_indices = list(sgen_indices)
        bus = net.storage.at[element_index, "bus"]
        if (net.load.loc[self.load_indices, "bus"] != bus).any() or (
            net.sgen.loc[self.sgen_indices, "bus"] != bus
        ).any():
            raise ValueError(f"loads {self.load_indices} and sgens {self.sgen_indices} must be on storage bus {bus}")
        super().__init__(net, element_index, max_charge_mw, max_discharge_mw, **kwargs)

    def initialize_control(self, net):
        """Set the request to the negative site power.

        Called at the start of each control run, after ConstControl has written the values of the time step.
        """
        loads = net.load.loc[self.load_indices]
        sgens = net.sgen.loc[self.sgen_indices]
        p_site = (loads.p_mw * loads.scaling * loads.in_service).sum() - (
            sgens.p_mw * sgens.scaling * sgens.in_service
        ).sum()
        self.p_request_mw = -p_site

    def finalize_step(self, net, time):
        """Commit the time step like StorageController and log a warning if the battery is full or empty.

        Returns True if the step was committed.
        """
        if not super().finalize_step(net, time):
            return False
        if self.energy_not_charged_mwh > 1e-9 and self.e_mwh >= self.max_e_mwh - 1e-9:
            logger.warning(
                "storage %s is full at time %s: the surplus is exchanged with the grid", self.element_index, time
            )
        if self.energy_not_discharged_mwh > 1e-9 and self.e_mwh <= self.min_e_mwh + 1e-9:
            logger.warning(
                "storage %s is empty at time %s: the deficit is exchanged with the grid", self.element_index, time
            )
        return True
