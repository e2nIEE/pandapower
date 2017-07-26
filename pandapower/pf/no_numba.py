def jit(*args, **kwargs):
    def wrapper(f):
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


def marker(*args, **kwargs):
    return marker


int32 = marker