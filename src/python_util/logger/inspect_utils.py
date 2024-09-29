import inspect


def walk_back_exc(num_frames_walk_back: int = 8):
    """
    Walk backwards on stack frames finding any local variables that are exceptions and adding the string representation
    of them to a list.
    :return: exception list
    """
    b = inspect.currentframe().f_back
    exc = []
    for i in range(num_frames_walk_back):
        for j in filter(lambda x: isinstance(x, Exception) and str(x) not in exc, [i for i in b.f_locals.values()]):
            exc.append(str(j))
        b = b.f_back

    return exc
