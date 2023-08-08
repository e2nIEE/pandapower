import copy

from pandapower.auxiliary import get_free_id
from pandapower.io_utils import JSONSerializableClass

try:
    import pandaplan.core.pplog as pplog
except ImportError:
    import logging as pplog

logger = pplog.getLogger(__name__)


class ProtectionDevice(JSONSerializableClass):
    """
    Base Class of protection devices found in protection module
    """
    def __init__(self, net, index=None, in_service=True, overwrite=False, **kwargs):
        super().__init__()
        # add oneself to net, creating the ['controller'] DataFrame, if necessary
        if index is None and "protection" in net.keys():
            index = get_free_id(net.protection)
        fill_dict = {"in_service": in_service}
        self.index = super().add_to_net(net=net, element='protection', index=index, overwrite=overwrite,
                                        fill_dict=fill_dict, preserve_dtypes=True)

    def reset_device(self):
        pass

    def has_tripped(self):
        logger.warning("Method has not been overwritten")
        return False

    def protection_function(self, net, scenario):
        logger.warning("Method has not been overwritten")

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


if __name__ == "__main__":
    import pandapower as pp

    net = pp.create_empty_network()
    ProtectionDevice(net)
