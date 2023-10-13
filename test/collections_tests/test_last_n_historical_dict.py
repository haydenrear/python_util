import unittest

from python_util.collections.last_n_historical_dict import LastNHistoricalDict


class TestLastNHistoricalDict(unittest.TestCase):
    def test_big(self):
        my_dict = LastNHistoricalDict(10, 4)
        for i in range(40):
            my_dict.force_index_key(i)
            my_dict.insert(i)

        assert my_dict.dict == {
            i: [i] for i in range(30, 40)
        }

    def test_insertion(self):
        my_dict = LastNHistoricalDict(3, 2)
        my_dict.insert('a')
        my_dict.insert('b')
        my_dict.insert('c')
        self.assertEqual(my_dict.dict, {0: ['a', 'b'], 1: ['c']})

    def test_key_rollover(self):
        my_dict = LastNHistoricalDict(3, 2)
        for i in range(10):
            my_dict.insert(i)
        self.assertEqual(my_dict.dict, {2: [4, 5], 3: [6, 7], 4: [8, 9]})
        my_dict.force_index_key(6)
        for i in range(10, 16):
            my_dict.insert(i)
        self.assertEqual(my_dict.dict, {6: [10, 11], 7: [12, 13], 8: [14, 15]})


    def test_list_rollover(self):
        my_dict = LastNHistoricalDict(3, 2)
        my_dict.insert('a')
        my_dict.insert('b')
        my_dict.insert('c')
        my_dict.insert('d')
        self.assertEqual(my_dict.dict, {0: ['a', 'b'], 1: ['c', 'd']})

    def test_last_nhistorical_dict(self):
        dict_found = LastNHistoricalDict(10, 2)
        for i in range(20):
            dict_found.insert(i)

        assert len(dict_found.dict) == 10
        assert all([len(i) == 2 for i in dict_found.dict.values()])
