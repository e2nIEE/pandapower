import numpy as np
import pandas as pd

from pandapower.build_branch import _calc_line_parameter, branch_cols

# this is only used by pm tnep
CONSTRUCTION_COST = 23


def read_tnep_results(net):
    ne_branch = net._pm_result["solution"]["ne_branch"]
    line_idx = net["res_ne_line"].index
    for pm_branch_idx, branch_data in ne_branch.items():
        # get pandapower index from power models index
        pp_idx = line_idx[int(pm_branch_idx) - 1]
        # built is a float, which is not exactly 1.0 or 0. sometimes
        net["res_ne_line"].loc[pp_idx, "built"] = branch_data["built"] > 0.5


def build_ne_branch(net, ppc):
    if "ne_line" in net:
        length = len(net["ne_line"])
        ppc["ne_branch"] = np.zeros(shape=(length, branch_cols + 1), dtype=np.complex128)
        ppc["ne_branch"][:, :13] = np.array([0, 0, 0, 0, 0, 250, 250, 250, 1, 0, 1, -60, 60])
        # create branch array ne_branch like the common branch array in the ppc
        net._pd2ppc_lookups["ne_branch"] = dict()
        net._pd2ppc_lookups["ne_branch"]["ne_line"] = (0, length)
        _calc_line_parameter(net, ppc, "ne_line", "ne_branch")
        ppc["ne_branch"][:, CONSTRUCTION_COST] = net["ne_line"].loc[:, "construction_cost"].values
    return ppc


def init_ne_line(net, new_line_index, construction_costs=None):
    """
    init function for new line dataframe, which specifies the possible new lines being built by power models opt

    Parameters
    ----------
    net - pp net
    new_line_index (list) - indices of new lines. These are copied to the new dataframe net["ne_line"] from net["line"]
    construction_costs (list, 0.) - costs of newly constructed lines

    Returns
    -------

    """
    # init dataframe
    net["ne_line"] = net["line"].loc[new_line_index, :]
    # add costs, if None -> init with zeros
    construction_costs = np.zeros(len(new_line_index)) if construction_costs is None else construction_costs
    net["ne_line"].loc[new_line_index, "construction_cost"] = construction_costs
    # set in service, but only in ne line dataframe
    net["ne_line"].loc[new_line_index, "in_service"] = True
    # init res_ne_line to save built status afterwards
    net["res_ne_line"] = pd.DataFrame(data=0, index=new_line_index, columns=["built"], dtype=int)
