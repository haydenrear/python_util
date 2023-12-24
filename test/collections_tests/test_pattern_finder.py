import unittest

from python_util.collections.pattern_keeping_list import PatternKeepingList, has_subpattern, is_repetition, \
    create_rolling_patterns


class TestPatternFinder(unittest.TestCase):
    def test_find_pattern_creation(self):
        finder = PatternKeepingList(10)
        assert create_rolling_patterns([1, 2, 3]) == [[1, 2, 3], [2, 3, 1], [3, 1, 2]]

    def test_find_pattern(self):
        ls = [1, 2, 3, 1, 2]
        assert len([i for i in range(len(ls)) if ls[i: i + 2] == [1, 2]]) == 2
        finder = PatternKeepingList(10)
        assert has_subpattern([1, 2, 3], [1, 2, 3, 1, 2, 3])
        assert is_repetition([1, 2, 3], [1, 2, 3, 1, 2, 3, 1])
        assert not is_repetition([2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3, 1, 2, 3])
        assert is_repetition([1, 2, 3], [1, 2, 3, 1, 2, 3, 1, 2, 3, 1])
        assert not is_repetition([2, 3, 4], [2, 3, 4, 1, 2, 3])
        assert is_repetition([1, 2, 3], [3, 1, 2, 3, 1, 2, 3])
        assert is_repetition([1, 2], [2, 1, 2, 1, 2, 1, 2, 1])
        assert is_repetition([3, 4, 1, 2], [4, 1, 2, 3, 4, 1])

    def test_no_repetition(self):
        finder = PatternKeepingList(10)
        for element in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            finder.add(element)
        self.assertEqual(finder.get_pattern(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        for element in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            finder.add(element)
        self.assertEqual(finder.get_pattern(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        for element in [1, 2, 3, 4]:
            finder.add(element)
        self.assertEqual(finder.get_pattern(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_full_repetition(self):
        finder = PatternKeepingList(10)
        for element in [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]:
            finder.add(element)
        patt = finder.get_pattern()
        assert all([i in patt for i in [1, 2, 3]])
        for element in [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]:
            finder.add(element)
        patt = finder.get_pattern()
        assert all([i in patt for i in [1, 2, 3]])

    def test_partial_repetition(self):
        finder = PatternKeepingList(10)
        for element in [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]:
            finder.add(element)
        self.assertEqual(finder.get_pattern(), [1, 2, 3, 4, 5])
        for element in [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]:
            finder.add(element)
        self.assertEqual(finder.get_pattern(), [1, 2, 3, 4, 5])

    def test_single_element_repetition(self):
        finder = PatternKeepingList(10)
        for element in [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]:
            finder.add(element)
        self.assertEqual(finder.get_pattern(), [1])
        for element in [1, 1, 1, 1, 1, 1, 1]:
            finder.add(element)
        self.assertEqual(finder.get_pattern(), [1])

    def test_double_element_repetition(self):
        finder = PatternKeepingList(10)
        for element in [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]:
            finder.add(element)
        self.assertEqual(finder.get_pattern(), [1, 2])
        finder = PatternKeepingList(10)
        for element in [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]:
            finder.add(element)
        self.assertEqual(finder.get_pattern(), [2, 1])
        self.assertEqual(finder.get_pattern(), [2, 1])
        for element in [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]:
            finder.add(element)
        self.assertEqual(finder.get_pattern(), [2, 1])

    def test_late_repetition(self):
        finder = PatternKeepingList(10)
        for element in [3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2]:
            finder.add(element)
        patt = finder.get_pattern()
        self.assertTrue(len(patt) == 4)
        self.assertTrue(all(i in patt for i in [1, 2, 3, 4]))
        for element in [3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2]:
            finder.add(element)
        self.assertTrue(len(patt) == 4)
        self.assertTrue(all(i in patt for i in [1, 2, 3, 4]))

    def test_single_patt(self):
        finder = PatternKeepingList(10)
        for element in [3]:
            finder.add(element)
        patt = finder.get_pattern()
        self.assertTrue(patt == [3])


if __name__ == '__main__':
    unittest.main()
