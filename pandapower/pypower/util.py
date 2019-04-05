# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""PYPOWER utilities.
"""


def sub2ind(shape, I, J, row_major=False):
    """Returns the linear indices of subscripts
    """
    if row_major:
        ind = (I % shape[0]) * shape[1] + (J % shape[1])
    else:
        ind = (J % shape[1]) * shape[0] + (I % shape[0])

    return ind.astype(int)


def feval(func, *args, **kw_args):
    """Evaluates the function C{func} using positional arguments C{args}
    and keyword arguments C{kw_args}.
    """
    return eval(func)(*args, **kw_args)


def have_fcn(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False
