import unittest

from examples.calculator import add, subtract


class CalculatorTest(unittest.TestCase):
    def test_add(self) -> None:
        self.assertEqual(add(2, 3), 5)

    def test_subtract(self) -> None:
        self.assertEqual(subtract(7, 4), 3)
        self.assertEqual(subtract(-2, 3), -5)


if __name__ == "__main__":
    unittest.main()

