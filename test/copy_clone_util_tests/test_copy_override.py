import unittest

from python_util.copy_clone_util import copy_override
from python_util.copy_clone_util.copy_override import copy_with_override


class One:
    def __init__(self, one: str, two: int):
        self.one = one
        self.two = two


class Two:
    def __init__(self, one_one: str, two_one: str, one_value: One):
        self.two_one = two_one
        self.one_one = one_one
        self.one_value = one_value


class CopyCloneTest(unittest.TestCase):
    def test_copy_with_override(self):
        test_copy = Two("one", "two", One("one_two", 1))
        whatever = copy_with_override(test_copy, **{"one": "three"})
        assert whatever.one == "three"
