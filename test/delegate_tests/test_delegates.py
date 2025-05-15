import unittest

from delegate_tests.delegate_fixutres import Delegator, TestDelegate


class DelegatesTest(unittest.TestCase):
    def test_delegates(self):
        delegator = Delegator('hello')
        delegator_2 = Delegator('goodbye')
        one = TestDelegate('hello')
        one.delegate = delegator
        two = TestDelegate('goodbye')
        two.delegate = delegator_2
        assert one.hello == 'hello'
        assert one.return_hello() == 'hello'
        assert two.hello == 'goodbye'
        assert two.return_hello() == 'goodbye'
        assert isinstance(one, TestDelegate)


