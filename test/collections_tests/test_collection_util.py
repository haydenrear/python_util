import collections
import unittest

import python_util.collections.collection_util
from python_util.collections.collection_util import retrieve_all_indices_of

class CollectionUtilTest(unittest.TestCase):
    def test_first_key(self):
        out = collections.OrderedDict({
            3: "one",
            1: "two",
            2: "three"
        })
        assert python_util.collections.collection_util.first_key(out) == 1

    def test_retrieve(self):
        out = retrieve_all_indices_of([1,2,3,4,1,2,3,4], 3)
        print(out)
        assert out == [2, 6]


if __name__ == '__main__':
    unittest.main()
