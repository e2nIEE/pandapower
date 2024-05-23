# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy

from pandapower.auxiliary import get_free_id
from pandapower.control.util.auxiliary import \
    drop_same_type_existing_controllers, log_same_type_existing_controllers
from pandapower.io_utils import JSONSerializableClass

try:
    import pandaplan.core.pplog as pplog
except:
    import logging as pplog

logger = pplog.getLogger(__name__)


class BasicCtrl(JSONSerializableClass):
    """
    Base-Class of all controllable elements within a network.
    """

    def __init__(self, container, index=None, **kwargs):
        super().__init__()
        # add oneself to net, creating the ['controller'] DataFrame, if necessary
        if index is None:
            index = get_free_id(container.controller)
        self.index = index

    def __repr__(self):
        rep = "This " + self.__class__.__name__ + " has the following parameters: \n"

        for member in ["index", "json_excludes"]:
            rep += ("\n" + member + ": ").ljust(20)
            d = locals()
            exec('value = self.' + member, d)
            rep += str(d['value'])

        return rep

    def __str__(self):
        s = self.__class__.__name__
        return s

    def __getstate__(self):
        state = copy.deepcopy(self.__dict__)

        for attr in self.json_excludes:
            try:
                del state[attr]
            except:
                continue

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def time_step(self, container, time):
        """
        It is the first call in each time step, thus suited for things like
        reading profiles or prepare the controller for the next control step.

        .. note:: This method is ONLY being called during time-series simulation!
        """
        pass

    def initialize_control(self, container):
        """
        Some controller require extended initialization in respect to the
        current state of the net (or their view of it). This method is being
        called after an initial loadflow but BEFORE any control strategies are
        being applied.

        This method may be interesting if you are aiming for a global
        controller or if it has to be aware of its initial state.
        """
        pass

    def is_converged(self, container):
        """
        This method calculated whether or not the controller converged. This is
        where any target values are being calculated and compared to the actual
        measurements. Returns convergence of the controller.
        """
        logger.warning("Method is_converged() has not been overwritten "
                       "(and will always return True)!")
        return True

    def control_step(self, container):
        """
        If the is_converged method returns false, the control_step will be
        called. In other words: if the controller did not converge yet, this
        method should implement actions that promote convergence e.g. adapting
        actuating variables and writing them back to the data structure.
        """
        pass

    def repair_control(self, container):
        """
        Some controllers can cause net to not converge. In this case, they can implement a method to
        try and catch the load flow error by altering some values in net, for example load scaling.
        This method is being called in the except block in run_control.
        Either implement this in a controller that is likely to cause the error, or define
        a special "load flow police" controller for your use case
        """
        pass

    def restore_init_state(self, container):
        """
        Some controllers manipulate values in net and then restore them back to initial values, e.g.
        DistributedSlack.
        This method should be used for such a purpose because it is executed in the except block of
        run_control to make sure that the net condition is restored even if load flow calculation
        doesn't converge
        """
        pass

    def finalize_control(self, container):
        """
        Some controller require extended finalization. This method is being
        called at the end of a loadflow.
        It is a separate method from restore_init_state because it is possible that control
        finalization does not only restore the init state but also something in addition to that,
        that would require the results in net
        """
        pass

    def finalize_step(self, container, time):
        """
        .. note:: This method is ONLY being called during time-series simulation!

        After each time step, this method is being called to clean things up or
        similar. The OutputWriter is a class specifically designed to store
        results of the loadflow. If the ControlHandler.output_writer got an
        instance of this class, it will be called before the finalize step.
        """
        pass

    def set_active(self, container, in_service):
        """
        Sets the controller in or out of service
        """
        container.controller.loc[self.index, 'in_service'] = in_service


    def level_reset(self, prosumer):
        pass


class Controller(BasicCtrl):
    """
    Base-Class of all controllable elements within a network.
    """

    def __init__(self, net, in_service=True, order=0, level=0, index=None, recycle=False,
                 drop_same_existing_ctrl=False, initial_run=True, overwrite=False,
                 matching_params=None, **kwargs):
        super(Controller, self).__init__(net, index, **kwargs)
        self.matching_params = dict() if matching_params is None else matching_params
        # add oneself to net, creating the ['controller'] DataFrame, if necessary
        # even though this code is repeated in JSONSerializableClass, it is necessary because of how drop_same_existing_controller works
        # it is still needed in JSONSerializableClass because it is used for characteristics
        if index is None and "controller" in net.keys():
            index = get_free_id(net.controller)
        self.index = self.add_controller_to_net(net=net, in_service=in_service, initial_run=initial_run,
                                                order=order, level=level, index=index, recycle=recycle,
                                                drop_same_existing_ctrl=drop_same_existing_ctrl,
                                                overwrite=overwrite, matching_params=matching_params, **kwargs)

    def add_controller_to_net(self, net, in_service, initial_run, order, level, index, recycle,
                              drop_same_existing_ctrl, overwrite, **kwargs):
        """
        adds the controller to net['controller'] dataframe.

        INPUT:
            **in_service** (bool) - in service status

            **order** (int) - order

            **index** (int) - index

            **recycle** (bool) - if controller needs a new bbm (ppc, Ybus...) or if it can be used \
                                 with prestored values. This is mostly needed for time series \
                                 calculations

        """
        if drop_same_existing_ctrl:
            drop_same_type_existing_controllers(net, type(self), index=index, **kwargs)
        else:
            log_same_type_existing_controllers(net, type(self), index=index, **kwargs)

        # use base class method to raise an error if the object is in DF and overwrite = False
        # if the index is None, the base class is in charge of obtaining the next free index in the data frame
        fill_dict = {"in_service": in_service, "initial_run": initial_run, "recycle": recycle,
                     "order": order, "level": level}
        added_index = super().add_to_net(net=net, element='controller', index=index, overwrite=overwrite,
                           fill_dict=fill_dict, preserve_dtypes=True)
        return added_index

    def time_step(self, net, time):
        super().time_step(net, time)

    def initialize_control(self, net):
        super().initialize_control(net)

    def is_converged(self, net):
        return super().is_converged(net)

    def control_step(self, net):
        super().control_step(net)

    def repair_control(self, net):
        super().repair_control(net)

    def restore_init_state(self, net):
        super().restore_init_state(net)

    def finalize_control(self, net):
        super().finalize_control(net)

    def finalize_step(self, net, time):
        super().finalize_step(net, time)

    def set_active(self, net, in_service):
        super().set_active(net, in_service)

    def set_recycle(self, net):
        """
        Checks the recyclability of this controller and changes the recyclability of the control handler if
        necessary. With this a faster time series calculation can be achieved since not everything must be
        recalculated.

        Beware: Setting recycle wrong can mess up your results. Set it to False in init if in doubt!
        """
        # checks what can be reused from this controller - default is False in base controller
        pass

