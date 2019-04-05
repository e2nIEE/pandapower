# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Runs the userfcn callbacks for a given stage.
"""


def run_userfcn(userfcn, stage, *args2):
    """Runs the userfcn callbacks for a given stage.

    Example::
        ppc = om.get_mpc()
        om = run_userfcn(ppc['userfcn'], 'formulation', om)

    @param userfcn: the 'userfcn' field of ppc, populated by L{add_userfcn}
    @param stage: the name of the callback stage begin executed
    (additional arguments) some stages require additional arguments.

    @see: L{add_userfcn}, L{remove_userfcn}, L{toggle_reserves},
          L{toggle_iflims}, L{runopf_w_res}.

    @author: Ray Zimmerman (PSERC Cornell)
    """
    rv = args2[0]
    if (len(userfcn) > 0) and (stage in userfcn):
        for k in range(len(userfcn[stage])):
            if 'args' in userfcn[stage][k]:
                args = userfcn[stage][k]['args']
            else:
                args = []

            if stage in ['ext2int', 'formulation', 'int2ext']:
                # ppc     = userfcn_*_ext2int(ppc, args)
                # om      = userfcn_*_formulation(om, args)
                # results = userfcn_*_int2ext(results, args)
                rv = userfcn[stage][k]['fcn'](rv, args)
            elif stage in ['printpf', 'savecase']:
                # results = userfcn_*_printpf(results, fd, ppopt, args)
                # ppc     = userfcn_*_savecase(mpc, fd, prefix, args)
                fdprint = args2[1]
                ppoptprint = args2[2]
                rv = userfcn[stage][k]['fcn'](rv, fdprint, ppoptprint, args)

    return rv
