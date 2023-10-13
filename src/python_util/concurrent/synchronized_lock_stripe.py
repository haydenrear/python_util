import functools
import threading
import typing

from python_util.logger.logger import LoggerFacade

CallableT = typing.TypeVar('CallableT', bound=typing.Callable)
LockId = str


class LockStripingLocks:
    def __init__(self):
        self.locks: dict[LockId, threading.RLock] = {}


def synchronized_lock_striping(locks: LockStripingLocks, lock_arg_arg_name: str):
    def outside_wrapper(function):

        @functools.wraps(function)
        def wrapper(self, *args, **kwargs):
            lock_id = kwargs[lock_arg_arg_name]
            if lock_id not in locks.locks.keys():
                LoggerFacade.info(f"Creating new lock with id {locks}.")
                locks.locks[lock_id] = threading.RLock()
            with locks.locks[lock_id]:
                return function(self, *args, **kwargs)


        return wrapper

    return outside_wrapper
