from pandapower.control import ConstControl

try:
    import pplog
except:
    import logging as pplog

logger = pplog.getLogger(__name__)


class MetaControl(ConstControl):
    """
    Class representing a generic time series controller for a specified controller and variable of the controller
    Control strategy: "No Control" -> just updates timeseries, is used to read timeseries data and set a controller variable accordingly

    INPUT:
        **net** (attrdict) - The net in which the controller resides

        **controller_index** - index of the controller in net.controller

        **variable** - variable ('uset', etc.)

        **index** (int[]) - IDs of the controlled elements

    OPTIONAL:

        **data_source** (, None) - The data source that provides profile data

        **profile_name** - the profile names of the elements in the data source

        **in_service** (bool, True) - Indicates if the controller is currently in_service

        **drop_same_existing_ctrl** (bool, False) - Indicates if already existing controllers of the
            same type and with the same matching parameters (e.g. at same element) should be dropped
    """

    def __init__(self, net, controller_index, variable, data_source, profile_name,
                 scale_factor=1.0, in_service=True, order=1, drop_same_existing_ctrl=False,
                 matching_params=None, **kwargs):
        if matching_params is None:
            matching_params = {"controller_index": controller_index, "variable": variable}
        # just calling init of the parent
        super().__init__(net, element='controller', variable=variable,
                         element_index=controller_index, data_source=data_source,
                         profile_name=profile_name, scale_factor=scale_factor,
                         in_service=in_service, order=order, matching_params=matching_params,
                         drop_same_existing_ctrl=drop_same_existing_ctrl, **kwargs)

    def write_to_net(self, net):
        # write to pandapower net
        # write p, q to bus within the net
        if hasattr(self.element_index, '__iter__') and len(self.element_index) > 1:
            for idx, val in zip(self.element_index, self.values):
                setattr(net.controller.object.at[idx], self.variable, val)
        else:
            setattr(net.controller.object.at[self.element_index], self.variable,
                    self.values)

    def control_step(self, net):
        self.write_to_net(net)
        self.applied = True

    def __str__(self):
        return super().__str__() + " of %s" % self.element_index
