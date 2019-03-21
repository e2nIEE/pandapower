# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Implements the OPF model object used to encapsulate a given OPF
problem formulation.
"""

from sys import stderr

from numpy import array, zeros, ones, Inf, dot, arange, r_
from numpy import flatnonzero as find
from scipy.sparse import lil_matrix, csr_matrix as sparse


class opf_model(object):
    """This class implements the OPF model object used to encapsulate
    a given OPF problem formulation. It allows for access to optimization
    variables, constraints and costs in named blocks, keeping track of the
    ordering and indexing of the blocks as variables, constraints and costs
    are added to the problem.

    @author: Ray Zimmerman (PSERC Cornell)
    """

    def __init__(self, ppc):
        #: PYPOWER case dict used to build the object.
        self.ppc = ppc

        #: data for optimization variable sets that make up the
        #  full optimization variable x
        self.var = {
            'idx': {
                'i1': {},  ## starting index within x
                'iN': {},  ## ending index within x
                'N': {}    ## number of elements in this variable set
            },
            'N': 0,        ## total number of elements in x
            'NS': 0,       ## number of variable sets or named blocks
            'data': {      ## bounds and initial value data
                'v0': {},  ## vector of initial values
                'vl': {},  ## vector of lower bounds
                'vu': {},  ## vector of upper bounds
            },
            'order': []    ## list of names for variable blocks in the order they appear in x
        }

        #: data for nonlinear constraints that make up the
        #  full set of nonlinear constraints ghn(x)
        self.nln = {
            'idx': {
                'i1': {},   ## starting index within ghn(x)
                'iN': {},   ## ending index within ghn(x)
                'N': {}     ## number of elements in this constraint set
            },
            'N': 0 ,        ## total number of elements in ghn(x)
            'NS': 0,        ## number of nonlinear constraint sets or named blocks
            'order': []     ## list of names for nonlinear constraint blocks in the order they appear in ghn(x)
        }

        #: data for linear constraints that make up the
        #  full set of linear constraints ghl(x)
        self.lin = {
            'idx': {
                'i1': {},   ## starting index within ghl(x)
                'iN': {},   ## ending index within ghl(x)
                'N': {}     ## number of elements in this constraint set
            },
            'N': 0,         ## total number of elements in ghl(x)
            'NS': 0,        ## number of linear constraint sets or named blocks
            'data': {       ## data for l <= A*xx <= u linear constraints
                'A': {},    ## sparse linear constraint matrix
                'l': {},    ## left hand side vector, bounding A*x below
                'u': {},    ## right hand side vector, bounding A*x above
                'vs': {}    ## cell array of variable sets that define the xx for this constraint block
            },
            'order': []     ## list of names for linear constraint blocks in the order they appear in ghl(x)
        }

        #: data for user-defined costs
        self.cost = {
            'idx': {
                'i1': {},   ## starting row index within full N matrix
                'iN': {},   ## ending row index within full N matrix
                'N':  {}    ## number of rows in this cost block in full N matrix
            },
            'N': 0,         ## total number of rows in full N matrix
            'NS': 0,        ## number of cost blocks
            'data': {       ## data for each user-defined cost block
                'N': {},    ## see help for add_costs() for details
                'H': {},    ##               "
                'Cw': {},   ##               "
                'dd': {},   ##               "
                'rh': {},   ##               "
                'kk': {},   ##               "
                'mm': {},   ##               "
                'vs': {}    ## list of variable sets that define xx for this cost block, where the N for this block multiplies xx'
            },
            'order': []     ## of names for cost blocks in the order they appear in the rows of the full N matrix
        }

        self.user_data = {}


    def __repr__(self): # pragma: no cover
        """String representation of the object.
        """
        s = ''
        if self.var['NS']:
            s += '\n%-22s %5s %8s %8s %8s\n' % ('VARIABLES', 'name', 'i1', 'iN', 'N')
            s += '%-22s %5s %8s %8s %8s\n' % ('=========', '------', '-----', '-----', '------')
            for k in range(self.var['NS']):
                name = self.var['order'][k]
                idx = self.var['idx']
                s += '%15d:%12s %8d %8d %8d\n' % (k, name, idx['i1'][name], idx['iN'][name], idx['N'][name])

            s += '%15s%31s\n' % (('var[\'NS\'] = %d' % self.var['NS']), ('var[\'N\'] = %d' % self.var['N']))
            s += '\n'
        else:
            s += '%s  :  <none>\n', 'VARIABLES'

        if self.nln['NS']:
            s += '\n%-22s %5s %8s %8s %8s\n' % ('NON-LINEAR CONSTRAINTS', 'name', 'i1', 'iN', 'N')
            s += '%-22s %5s %8s %8s %8s\n' % ('======================', '------', '-----', '-----', '------')
            for k in range(self.nln['NS']):
                name = self.nln['order'][k]
                idx = self.nln['idx']
                s += '%15d:%12s %8d %8d %8d\n' % (k, name, idx['i1'][name], idx['iN'][name], idx['N'][name])

            s += '%15s%31s\n' % (('nln.NS = %d' % self.nln['NS']), ('nln.N = %d' % self.nln['N']))
            s += '\n'
        else:
            s += '%s  :  <none>\n', 'NON-LINEAR CONSTRAINTS'

        if self.lin['NS']:
            s += '\n%-22s %5s %8s %8s %8s\n' % ('LINEAR CONSTRAINTS', 'name', 'i1', 'iN', 'N')
            s += '%-22s %5s %8s %8s %8s\n' % ('==================', '------', '-----', '-----', '------')
            for k in range(self.lin['NS']):
                name = self.lin['order'][k]
                idx = self.lin['idx']
                s += '%15d:%12s %8d %8d %8d\n' % (k, name, idx['i1'][name], idx['iN'][name], idx['N'][name])

            s += '%15s%31s\n' % (('lin.NS = %d' % self.lin['NS']), ('lin.N = %d' % self.lin['N']))
            s += '\n'
        else:
            s += '%s  :  <none>\n', 'LINEAR CONSTRAINTS'

        if self.cost['NS']:
            s += '\n%-22s %5s %8s %8s %8s\n' % ('COSTS', 'name', 'i1', 'iN', 'N')
            s += '%-22s %5s %8s %8s %8s\n' % ('=====', '------', '-----', '-----', '------')
            for k in range(self.cost['NS']):
                name = self.cost['order'][k]
                idx = self.cost['idx']
                s += '%15d:%12s %8d %8d %8d\n' % (k, name, idx['i1'][name], idx['iN'][name], idx['N'][name])

            s += '%15s%31s\n' % (('cost.NS = %d' % self.cost['NS']), ('cost.N = %d' % self.cost['N']))
            s += '\n'
        else:
            s += '%s  :  <none>\n' % 'COSTS'

        #s += '  ppc = '
        #if len(self.ppc):
        #    s += '\n'
        #
        #s += str(self.ppc) + '\n'

        s += '  userdata = '
        if len(self.user_data):
            s += '\n'

        s += str(self.user_data)

        return s


    def add_constraints(self, name, AorN, l, u=None, varsets=None):
        """Adds a set of constraints to the model.

        Linear constraints are of the form C{l <= A * x <= u}, where
        C{x} is a vector made of of the vars specified in C{varsets} (in
        the order given). This allows the C{A} matrix to be defined only
        in terms of the relevant variables without the need to manually
        create a lot of zero columns. If C{varsets} is empty, C{x} is taken
        to be the full vector of all optimization variables. If C{l} or
        C{u} are empty, they are assumed to be appropriately sized vectors
        of C{-Inf} and C{Inf}, respectively.

        For nonlinear constraints, the 3rd argument, C{N}, is the number
        of constraints in the set. Currently, this is used internally
        by PYPOWER, but there is no way for the user to specify
        additional nonlinear constraints.
        """
        if u is None:  ## nonlinear
            ## prevent duplicate named constraint sets
            if name in self.nln["idx"]["N"]:
                stderr.write("opf_model.add_constraints: nonlinear constraint set named '%s' already exists\n" % name)

            ## add info about this nonlinear constraint set
            self.nln["idx"]["i1"][name] = self.nln["N"] #+ 1    ## starting index
            self.nln["idx"]["iN"][name] = self.nln["N"] + AorN ## ing index
            self.nln["idx"]["N"][name]  = AorN            ## number of constraints

            ## update number of nonlinear constraints and constraint sets
            self.nln["N"]  = self.nln["idx"]["iN"][name]
            self.nln["NS"] = self.nln["NS"] + 1

            ## put name in ordered list of constraint sets
#            self.nln["order"][self.nln["NS"]] = name
            self.nln["order"].append(name)
        else:                ## linear
            ## prevent duplicate named constraint sets
            if name in self.lin["idx"]["N"]:
                stderr.write('opf_model.add_constraints: linear constraint set named ''%s'' already exists\n' % name)

            if varsets is None:
                varsets = []

            N, M = AorN.shape
            if len(l) == 0:                   ## default l is -Inf
                l = -Inf * ones(N)

            if len(u) == 0:                   ## default u is Inf
                u = Inf * ones(N)

            if len(varsets) == 0:
                varsets = self.var["order"]

            ## check sizes
            if (l.shape[0] != N) or (u.shape[0] != N):
                stderr.write('opf_model.add_constraints: sizes of A, l and u must match\n')

            nv = 0
            for k in range(len(varsets)):
                nv = nv + self.var["idx"]["N"][varsets[k]]

            if M != nv:
                stderr.write('opf_model.add_constraints: number of columns of A does not match\nnumber of variables, A is %d x %d, nv = %d\n' % (N, M, nv))

            ## add info about this linear constraint set
            self.lin["idx"]["i1"][name]  = self.lin["N"] #+ 1   ## starting index
            self.lin["idx"]["iN"][name]  = self.lin["N"] + N   ## ing index
            self.lin["idx"]["N"][name]   = N              ## number of constraints
            self.lin["data"]["A"][name]  = AorN
            self.lin["data"]["l"][name]  = l
            self.lin["data"]["u"][name]  = u
            self.lin["data"]["vs"][name] = varsets

            ## update number of vars and var sets
            self.lin["N"]  = self.lin["idx"]["iN"][name]
            self.lin["NS"] = self.lin["NS"] + 1

            ## put name in ordered list of var sets
#            self.lin["order"][self.lin["NS"]] = name
            self.lin["order"].append(name)


    def add_vars(self, name, N, v0=None, vl=None, vu=None):
        """ Adds a set of variables to the model.

        Adds a set of variables to the model, where N is the number of
        variables in the set, C{v0} is the initial value of those variables,
        and C{vl} and C{vu} are the lower and upper bounds on the variables.
        The defaults for the last three arguments, which are optional,
        are for all values to be initialized to zero (C{v0 = 0}) and unbounded
        (C{VL = -Inf, VU = Inf}).
        """
        ## prevent duplicate named var sets
        if name in self.var["idx"]["N"]:
            stderr.write('opf_model.add_vars: variable set named ''%s'' already exists\n' % name)

        if v0 is None or len(v0) == 0:
            v0 = zeros(N)           ## init to zero by default

        if vl is None or len(vl) == 0:
            vl = -Inf * ones(N)     ## unbounded below by default

        if vu is None or len(vu) == 0:
            vu = Inf * ones(N)      ## unbounded above by default


        ## add info about this var set
        self.var["idx"]["i1"][name]  = self.var["N"] #+ 1   ## starting index
        self.var["idx"]["iN"][name]  = self.var["N"] + N   ## ing index
        self.var["idx"]["N"][name]   = N              ## number of vars
        self.var["data"]["v0"][name] = v0             ## initial value
        self.var["data"]["vl"][name] = vl             ## lower bound
        self.var["data"]["vu"][name] = vu             ## upper bound

        ## update number of vars and var sets
        self.var["N"]  = self.var["idx"]["iN"][name]
        self.var["NS"] = self.var["NS"] + 1

        ## put name in ordered list of var sets
#        self.var["order"][self.var["NS"]] = name
        self.var["order"].append(name)


    def build_cost_params(self):
        """Builds and saves the full generalized cost parameters.

        Builds the full set of cost parameters from the individual named
        sub-sets added via L{add_costs}. Skips the building process if it has
        already been done, unless a second input argument is present.

        These cost parameters can be retrieved by calling L{get_cost_params}
        and the user-defined costs evaluated by calling L{compute_cost}.
        """
        ## initialize parameters
        nw = self.cost["N"]
#        nnzN = 0
#        nnzH = 0
#        for k in range(self.cost["NS"]):
#            name = self.cost["order"][k]
#            nnzN = nnzN + nnz(self.cost["data"]["N"][name])
#            if name in self.cost["data"]["H"]:
#                nnzH = nnzH + nnz(self.cost["data"]["H"][name])

        ## FIXME Zero dimensional sparse matrices
        N = zeros((nw, self.var["N"]))
        H = zeros((nw, nw))  ## default => no quadratic term

        Cw = zeros(nw)
        dd = ones(nw)                        ## default => linear
        rh = zeros(nw)                       ## default => no shift
        kk = zeros(nw)                       ## default => no dead zone
        mm = ones(nw)                        ## default => no scaling


        ## save in object
        self.cost["params"] = {
            'N': N, 'Cw': Cw, 'H': H, 'dd': dd, 'rh': rh, 'kk': kk, 'mm': mm }


    def get_cost_params(self):
        """Returns the cost parameter struct for user-defined costs.

        Requires calling L{build_cost_params} first to build the full set of
        parameters. Returns the full cost parameter struct for all user-defined
        costs that incorporates all of the named cost sets added via
        L{add_costs}, or, if a name is provided it returns the cost dict
        corresponding to the named set of cost rows (C{N} still has full number
        of columns).

        The cost parameters are returned in a dict with the following fields::
            N      - nw x nx sparse matrix
            Cw     - nw x 1 vector
            H      - nw x nw sparse matrix (optional, all zeros by default)
            dd, mm - nw x 1 vectors (optional, all ones by default)
            rh, kk - nw x 1 vectors (optional, all zeros by default)
        """
        if not 'params' in self.cost:
            stderr.write('opf_model.get_cost_params: must call build_cost_params first\n')

        cp = self.cost["params"]

        return cp


    def get_idx(self):
        """ Returns the idx struct for vars, lin/nln constraints, costs.

        Returns a structure for each with the beginning and ending
        index value and the number of elements for each named block.
        The 'i1' field (that's a one) is a dict with all of the
        starting indices, 'iN' contains all the ending indices and
        'N' contains all the sizes. Each is a dict whose keys are
        the named blocks.

        Examples::
            [vv, ll, nn] = get_idx(om)

        For a variable block named 'z' we have::
                vv['i1']['z'] - starting index for 'z' in optimization vector x
                vv['iN']['z'] - ending index for 'z' in optimization vector x
                vv["N"]    - number of elements in 'z'

        To extract a 'z' variable from x::
                z = x(vv['i1']['z']:vv['iN']['z'])

        To extract the multipliers on a linear constraint set
        named 'foo', where mu_l and mu_u are the full set of
        linear constraint multipliers::
                mu_l_foo = mu_l(ll['i1']['foo']:ll['iN']['foo'])
                mu_u_foo = mu_u(ll['i1']['foo']:ll['iN']['foo'])

        The number of nonlinear constraints in a set named 'bar'::
                nbar = nn["N"].bar
        (note: the following is preferable ::
                nbar = getN(om, 'nln', 'bar')
        ... if you haven't already called L{get_idx} to get C{nn}.)
        """
        vv = self.var["idx"]
        ll = self.lin["idx"]
        nn = self.nln["idx"]
        cc = self.cost["idx"]

        return vv, ll, nn, cc


    def get_ppc(self):
        """Returns the PYPOWER case dict.
        """
        return self.ppc


    def getN(self, selector, name=None):
        """Returns the number of variables, constraints or cost rows.

        Returns either the total number of variables/constraints/cost rows
        or the number corresponding to a specified named block.

        Examples::
            N = getN(om, 'var')         : total number of variables
            N = getN(om, 'lin')         : total number of linear constraints
            N = getN(om, 'nln')         : total number of nonlinear constraints
            N = getN(om, 'cost')        : total number of cost rows (in N)
            N = getN(om, 'var', name)   : number of variables in named set
            N = getN(om, 'lin', name)   : number of linear constraints in named set
            N = getN(om, 'nln', name)   : number of nonlinear cons. in named set
            N = getN(om, 'cost', name)  : number of cost rows (in N) in named set
        """
        if name is None:
            N = getattr(self, selector)["N"]
        else:
            if name in getattr(self, selector)["idx"]["N"]:
                N = getattr(self, selector)["idx"]["N"][name]
            else:
                N = 0
        return N


    def getv(self):
        """Returns initial value, lower bound and upper bound for opt variables.

        Returns the initial value, lower bound and upper bound for the full
        optimization variable vector, or for a specific named variable set.

        Examples::
            x, xmin, xmax = getv(om)
            Pg, Pmin, Pmax = getv(om, 'Pg')
        """
        v0 = array([]); vl = array([]); vu = array([])
        for k in range(self.var["NS"]):
            name = self.var["order"][k]
            v0 = r_[ v0, self.var["data"]["v0"][name] ]
            vl = r_[ vl, self.var["data"]["vl"][name] ]
            vu = r_[ vu, self.var["data"]["vu"][name] ]

        return v0, vl, vu


    def linear_constraints(self):
        """Builds and returns the full set of linear constraints.

        Builds the full set of linear constraints based on those added by
        L{add_constraints}::

            L <= A * x <= U
        """

        ## initialize A, l and u
#        nnzA = 0
#        for k in range(self.lin["NS"]):
#            nnzA = nnzA + nnz(self.lin["data"].A.(self.lin.order{k}))

        if self.lin["N"]:
            A = lil_matrix((self.lin["N"], self.var["N"]))
            u = Inf * ones(self.lin["N"])
            l = -u
        else:
            A = None
            u = array([])
            l = array([])

            return A, l, u

        ## fill in each piece
        for k in range(self.lin["NS"]):
            name = self.lin["order"][k]
            N = self.lin["idx"]["N"][name]
            if N:                                   ## non-zero number of rows to add
                Ak = self.lin["data"]["A"][name]    ## A for kth linear constrain set
                i1 = self.lin["idx"]["i1"][name]    ## starting row index
                iN = self.lin["idx"]["iN"][name]    ## ing row index
                vsl = self.lin["data"]["vs"][name]  ## var set list
                kN = 0                              ## initialize last col of Ak used
                # FIXME: Sparse matrix with fancy indexing
                Ai = zeros((N, self.var["N"]))
                for v in vsl:
                    j1 = self.var["idx"]["i1"][v]      ## starting column in A
                    jN = self.var["idx"]["iN"][v]      ## ing column in A
                    k1 = kN                            ## starting column in Ak
                    kN = kN + self.var["idx"]["N"][v]  ## ing column in Ak
                    Ai[:, j1:jN] = Ak[:, k1:kN].todense()

                A[i1:iN, :] = Ai

                l[i1:iN] = self.lin["data"]["l"][name]
                u[i1:iN] = self.lin["data"]["u"][name]

        return A.tocsr(), l, u


    def userdata(self, name, val=None):
        """Used to save or retrieve values of user data.

        This function allows the user to save any arbitrary data in the object
        for later use. This can be useful when using a user function to add
        variables, constraints, costs, etc. For example, suppose some special
        indexing is constructed when adding some variables or constraints.
        This indexing data can be stored and used later to "unpack" the results
        of the solved case.
        """
        if val is not None:
            self.user_data[name] = val
            return self
        else:
            if name in self.user_data:
                return self.user_data[name]
            else:
                return array([])
