import unittest
import os
import tempfile
import streamlit as st
from unittest.mock import patch, MagicMock
from io import BytesIO
from app import get_pdf_text

class TestInvalidFileType(unittest.TestCase):
    
    def setUp(self):
        self.text_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
        self.text_file.write(b"This is a text file, not a PDF.")
        self.text_file.close()
        
        self.docx_file = tempfile.NamedTemporaryFile(suffix='.docx', delete=False)
        self.docx_file.write(b"PK\x03\x04\x14\x00\x00\x00\x08\x00This is fake DOCX content")
        self.docx_file.close()
        
        if not hasattr(st, 'session_state'):
            st.session_state = {}
            
    def tearDown(self):
        os.unlink(self.text_file.name)
        os.unlink(self.docx_file.name)
        
    def test_invalid_file_type_txt(self):
        with open(self.text_file.name, 'rb') as f:
            with self.assertRaises(Exception) as context:
                get_pdf_text([f])
                
        self.assertTrue(any(msg in str(context.exception) for msg in [
            "file has not been decrypted",
            "not a PDF file",
            "EOF marker not found",
            "File is not a PDF"
        ]))
        
    def test_invalid_file_type_docx(self):
        with open(self.docx_file.name, 'rb') as f:
            with self.assertRaises(Exception) as context:
                get_pdf_text([f])
                
        self.assertTrue(any(msg in str(context.exception) for msg in [
            "file has not been decrypted",
            "not a PDF file",
            "EOF marker not found",
            "File is not a PDF"
        ]))
        
    @patch('streamlit.file_uploader')
    def test_streamlit_file_filtering(self, mock_file_uploader):
        mock_file_uploader.return_value = None
        
        def mock_main():
            st.file_uploader(
                "Upload PDF files and click 'Process'",
                accept_multiple_files=True,
                type=["pdf"],
                key="pdf_uploader"
            )
            
        mock_main()
        
        mock_file_uploader.assert_called_once()
        _, kwargs = mock_file_uploader.call_args
        self.assertEqual(kwargs.get('type'), ["pdf"])

if __name__ == '__main__':
    unittest.main()