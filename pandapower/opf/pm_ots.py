from pandapower import BR_STATUS


def read_ots_results(net):
    """
    Reads the branch_status variable from ppc to pandapower net

    INPUT

        **net** - pandapower net
    """
    ppc = net._ppc
    for element, (f, t) in net._pd2ppc_lookups["branch"].items():
        # for trafo, line, trafo3w
        res = "res_" + element
        if "in_service" not in net[res]:
            # copy in service state from inputs
            net[res].loc[:, "in_service"] = None
            net[res].loc[:, "in_service"] = net[res].loc[:, "in_service"].values
        branch_status = ppc["branch"][f:t, BR_STATUS].real

        net[res]["in_service"].values[:] = branch_status
