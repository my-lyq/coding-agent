import unittest
from strings import is_palindrome
class TestStrings(unittest.TestCase):
    def test_normalized_palindromes(self):
        self.assertTrue(is_palindrome("A man, a plan, a canal: Panama!"))
        self.assertTrue(is_palindrome("RaceCar"))
    def test_non_palindrome(self): self.assertFalse(is_palindrome("coding agent"))
if __name__=="__main__": unittest.main()
