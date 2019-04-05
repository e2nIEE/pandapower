# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Appends a userfcn to the list to be called for a case.
"""

from sys import stderr

def add_userfcn(ppc, stage, fcn, args=None, allow_multiple=False):
    """Appends a userfcn to the list to be called for a case.

    A userfcn is a callback function that can be called automatically by
    PYPOWER at one of various stages in a simulation.

    Currently there are 5 different callback stages defined. Each stage has
    a name, and by convention, the name of a user-defined callback function
    ends with the name of the stage. The following is a description of each
    stage, when it is called and the input and output arguments which vary
    depending on the stage. The reserves example (see L{runopf_w_res}) is used
    to illustrate how these callback userfcns might be used.

      1. C{'ext2int'}

      Called from L{ext2int} immediately after the case is converted from
      external to internal indexing. Inputs are a PYPOWER case dict (C{ppc}),
      freshly converted to internal indexing and any (optional) C{args} value
      supplied via L{add_userfcn}. Output is the (presumably updated) C{ppc}.
      This is typically used to reorder any input arguments that may be needed
      in internal ordering by the formulation stage.

      E.g. C{ppc = userfcn_reserves_ext2int(ppc, args)}

      2. C{'formulation'}

      Called from L{opf} after the OPF Model (C{om}) object has been
      initialized with the standard OPF formulation, but before calling the
      solver. Inputs are the C{om} object and any (optional) C{args} supplied
      via L{add_userfcn}. Output is the C{om} object. This is the ideal place
      to add any additional vars, constraints or costs to the OPF formulation.

      E.g. C{om = userfcn_reserves_formulation(om, args)}

      3. C{'int2ext'}

      Called from L{int2ext} immediately before the resulting case is converted
      from internal back to external indexing. Inputs are the C{results} dict
      and any (optional) C{args} supplied via C{add_userfcn}. Output is the
      C{results} dict. This is typically used to convert any results to
      external indexing and populate any corresponding fields in the
      C{results} dict.

      E.g. C{results = userfcn_reserves_int2ext(results, args)}

      4. C{'printpf'}

      Called from L{printpf} after the pretty-printing of the standard OPF
      output. Inputs are the C{results} dict, the file descriptor to write to,
      a PYPOWER options dict, and any (optional) C{args} supplied via
      L{add_userfcn}. Output is the C{results} dict. This is typically used for
      any additional pretty-printing of results.

      E.g. C{results = userfcn_reserves_printpf(results, fd, ppopt, args)}

      5. C{'savecase'}

      Called from L{savecase} when saving a case dict to a Python file after
      printing all of the other data to the file. Inputs are the case dict,
      the file descriptor to write to, the variable prefix (typically 'ppc')
      and any (optional) C{args} supplied via L{add_userfcn}. Output is the
      case dict. This is typically used to write any non-standard case dict
      fields to the case file.

      E.g. C{ppc = userfcn_reserves_printpf(ppc, fd, prefix, args)}

    @param ppc: the case dict
    @param stage: the name of the stage at which this function should be
        called: ext2int, formulation, int2ext, printpf
    @param fcn: the name of the userfcn
    @param args: (optional) the value to be passed as an argument to the
        userfcn
    @param allow_multiple: (optional) if True, allows the same function to
        be added more than once.

    @see: L{run_userfcn}, L{remove_userfcn}, L{toggle_reserves},
          L{toggle_iflims}, L{runopf_w_res}.

    @author: Ray Zimmerman (PSERC Cornell)
    """
    if args is None:
        args = []

    if stage not in ['ext2int', 'formulation', 'int2ext', 'printpf', 'savecase']:
        stderr.write('add_userfcn : \'%s\' is not the name of a valid callback stage\n' % stage)

    n = 0
    if 'userfcn' in ppc:
        if stage in ppc['userfcn']:
            n = len(ppc['userfcn'][stage]) #+ 1
            if not allow_multiple:
                for k in range(n):
                    if ppc['userfcn'][stage][k]['fcn'] == fcn:
                        stderr.write('add_userfcn: the function \'%s\' has already been added\n' % fcn.__name__)
        else:
            ppc['userfcn'][stage] = []
    else:
        ppc['userfcn'] = {stage: []}

    ppc['userfcn'][stage].append({'fcn': fcn})
    if len(args) > 0:
        ppc['userfcn'][stage][n]['args'] = args

    return ppc
