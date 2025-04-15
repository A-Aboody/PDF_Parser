import unittest
import os
from unittest.mock import patch
from app import process_documents 

class TestApiTokenFailure(unittest.TestCase):

    @patch.dict(os.environ, {"API_TOKEN": ""})
    def test_missing_api_token(self):
        with self.assertRaises(ValueError) as context:
            process_documents()
        self.assertIn("missing API token", str(context.exception).lower())

    @patch.dict(os.environ, {"API_TOKEN": "INVALID_TOKEN"})
    def test_invalid_api_token(self):
        with self.assertRaises(ValueError) as context:
            process_documents()
        self.assertIn("invalid API token", str(context.exception).lower())

if __name__ == '__main__':
    unittest.main()
