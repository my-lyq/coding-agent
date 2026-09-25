import unittest
from collections_utils import stable_unique
class TestStableUnique(unittest.TestCase):
    def test_order_and_duplicates(self):
        self.assertEqual(stable_unique([3,1,3,2,1]),[3,1,2])
        self.assertEqual(stable_unique([]),[])
if __name__=="__main__": unittest.main()
