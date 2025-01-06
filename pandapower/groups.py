import numpy as np
import pandas as pd
from pandapower.auxiliary import ensure_iterability


def detach_from_groups(net, element_type, element_index, index=None):
    """Detaches elements from one or multiple groups, defined by 'index'.
    No errors are raised if elements are passed to be dropped from groups which alread don't have
    these elements as members.
    A reverse function is available -> pp.group.attach_to_group().

    Parameters
    ----------
    net : pandapowerNet
        pandapower net
    element_type : str
        The element type of which elements should be dropped from the group(s), e.g. "bus"
    element_index : int or list of integers
        indices of the elements which should be dropped from the group
    index : int or list of integers, optional
        Indices of the group(s) from which the element should be dropped. If None, the elements are
        dropped from all groups, by default None
    """
    if index is None:
        index = net.group.index
    element_index = pd.Index(ensure_iterability(element_index), dtype=np.int64)

    to_check = np.isin(net.group.index.values, index)
    to_check &= net.group.element_type.values == element_type
    keep = np.ones(net.group.shape[0], dtype=bool)

    for i in np.arange(len(to_check), dtype=np.int64)[to_check]:
        rc = net.group.reference_column.iat[i]
        if rc is None or pd.isnull(rc):
            net.group.element_index.iat[i] = pd.Index(net.group.element_index.iat[i]).difference(
                element_index).tolist()
        else:
            net.group.element_index.iat[i] = pd.Index(net.group.element_index.iat[i]).difference(
                pd.Index(net[element_type][rc].loc[element_index.intersection(
                    net[element_type].index)])).tolist()

        if not len(net.group.element_index.iat[i]):
            keep[i] = False
    net.group = net.group.loc[keep]
