# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""PYPOWER utilities."""

from numpy import int64


def sub2ind(shape, i, j, row_major=False):
    """Returns the linear indices of subscripts"""
    if row_major:
        ind = (i % shape[0]) * shape[1] + (j % shape[1])
    else:
        ind = (j % shape[1]) * shape[0] + (i % shape[0])

    return ind.astype(int64)


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
