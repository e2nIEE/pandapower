import numpy as np
from scipy.io import savemat, loadmat

from pandapower.converter.pypower import from_ppc, to_ppc

def from_mpc(mpc_file):
    ppc = _mpc_to_ppc(mpc_file)
    return from_ppc(ppc)
    
def to_mpc(net, filename):
    ppc = to_ppc(net)
    _ppc_to_mpc(ppc, filename)
    
def _mpc_to_ppc(mpc_file):
    mpc = loadmat(mpc_file , squeeze_me=True, struct_as_record=False)
    if "mpc" in mpc:
        mpc = mpc["mpc"]
    ppc = dict()
    # generate ppc
    ppc['version'] = mpc["version"]
    ppc["baseMVA"] = mpc["baseMVA"]
    ppc["bus"] = mpc["bus"]
    ppc["gen"] = mpc["gen"].reshape(1, 21)
    ppc["branch"] = mpc["branch"]

    # ppc must start at 0 rather than 1 (matlab)
    ppc["bus"][:, 0] -= 1
    ppc["branch"][:, 0] -= 1
    ppc["branch"][:, 1] -= 1
    ppc["gen"][:, 0] -= 1

    # adjust for the matpower converter -> taps should be 0 when there is no transformer, but are 1
    ppc["branch"][np.where(ppc["branch"][:, 8] == 0), 8] = 1

    return ppc
    

def _ppc_to_mpc(ppc, filename):
    """
    Store network in Pypower/Matpower format as mat-file
    Convert 0-based python to 1-based Matlab
    Take care of a few small details

    **INPUT**:
        * net - The Pandapower format network
        * filename - File path + name of the mat file which is created
    """

    # convert to matpower
    # Matlab is one-based, so all entries (buses, lines, gens) have to start with 1 instead of 0
    if len(np.where(ppc["bus"][:, 0] == 0)[0]):
        ppc["bus"][:, 0] = ppc["bus"][:, 0] + 1
        ppc["gen"][:, 0] = ppc["gen"][:, 0] + 1
        ppc["branch"][:, 0:2] = ppc["branch"][:, 0:2] + 1
    # adjust for the matpower converter -> taps should be 0 when there is no transformer, but are 1
    ppc["branch"][np.where(ppc["branch"][:, 8] == 1), 8] = 0
    # version is a string
    ppc["version"] = str(ppc["version"])
    savemat(filename, ppc)
    
if __name__ == '__main__':
    net = from_mpc("test.m")
    to_mpc(net, "test2.m")
    mpc = loadmat("test.m" , squeeze_me=True, struct_as_record=False)
    mpc2 = loadmat("test2.m" , squeeze_me=True, struct_as_record=False)

    