import unittest
from app import handle_user_input 

class TestLongInput(unittest.TestCase):
    def test_long_input(self):
        long_input = "A" * 1000 
        result = handle_user_input(long_input)
        self.assertEqual(result, "Input accepted", "Should accept long input without crash")

if __name__ == '__main__':
    unittest.main()
