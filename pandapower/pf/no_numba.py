# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from typing import Any, Callable, ParamSpec, TypeVar, overload, Union


P = ParamSpec("P")
R = TypeVar("R")


@overload
def jit(f: Callable[P, R]) -> Callable[P, R]: ...


@overload
def jit(*args: Any, **kwargs: Any) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def jit(*args: Any, **kwargs: Any) -> Union[Callable[[Callable[P, R]], Callable[P, R]], Callable[..., Callable[P, R]]]:
    def wrapper(f: Callable[P, R]) -> Callable[P, R]:
        return f

    if len(args) > 0 and (args[0] is marker or not callable(args[0])) \
            or len(kwargs) > 0:
        # @jit(int32(int32, int32)), @jit(signature="void(int32)")
        return wrapper
    elif len(args) == 0:
        # @jit()
        return wrapper
    else:
        # @jit
        return args[0]


def marker(*args: Any, **kwargs: Any) -> Callable[..., Any]:
    return marker


int32 = marker