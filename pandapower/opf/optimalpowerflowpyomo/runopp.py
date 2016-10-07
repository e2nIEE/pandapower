__author__ = "fmeier"

from pypower.runopf import runopf
from .build_opf import _make_constraints, _make_objective
from pypower.opf import opf
from pypower.ppoption import ppoption
from pandapower.run import _pd2mpc



def runopp(net, objectivetype = "simple", verbose=False):
    ppopt = ppoption(OPF_VIOLATION=1e-6, PDIPM_GRADTOL=1e-8,
                   PDIPM_COMPTOL=1e-8, PDIPM_COSTTOL=1e-9)
    ppopt = ppoption(ppopt, OUT_ALL=0, VERBOSE=verbose, OPF_ALG=560)
    
    gen_is = net["gen"][net["gen"]["in_service"].values.astype(bool)]
    eg_is = net["ext_grid"][net["ext_grid"]["in_service"].values.astype(bool)]

    mpc, bus_lookup = _pd2mpc(net, gen_is, eg_is)
    
    
   
    A, l, u = _make_constraints(net, mpc)
    N, fparm, H, Cw = _make_objective(mpc, objectivetype = "simple")
    
   
    
    
    opf(mpc, A, l, u, ppopt, N, fparm, H, Cw)
    
    # get results and copy to pp net