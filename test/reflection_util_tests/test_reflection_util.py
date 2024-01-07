import typing
import unittest
from typing import Optional

from python_util.reflection.reflection_utils import is_empty_inspect, get_fn_param_types, get_all_fn_param_types, \
    is_optional_ty


class ReflectionUtilTests(unittest.TestCase):
    def test_empty(self):
        def first(cls=None):
            pass

        found = get_fn_param_types(first)
        assert len(found) == 0

        assert len(get_all_fn_param_types(first)) != 0
        assert is_empty_inspect(get_all_fn_param_types(first)['cls'][0])

    def test_optional(self):
        def second(cls: Optional[str] = None):
            pass

        def third(cls: Optional[int] = None):
            pass

        T = typing.TypeVar("T")

        class Something(typing.Generic[T]):
            pass

        def fourth(cls: Optional[Something[str]] = None):
            pass

        assert is_optional_ty(get_fn_param_types(second)['cls'][0])
        assert is_optional_ty(get_fn_param_types(third)['cls'][0])
        assert is_optional_ty(get_fn_param_types(fourth)['cls'][0])


if __name__ == '__main__':
    unittest.main()
