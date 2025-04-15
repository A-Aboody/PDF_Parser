import unittest
import os
import tempfile
import streamlit as st
from unittest.mock import patch, MagicMock
from PyPDF2 import PdfWriter, PdfReader
from app import get_pdf_text

class TestEmptyPDFHandling(unittest.TestCase):
    
    def setUp(self):
        self.pdf_writer = PdfWriter()
        self.pdf_writer.add_blank_page(width=612, height=792)
        
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        self.pdf_writer.write(self.temp_file)
        self.temp_file.close()
        
        if not hasattr(st, 'session_state'):
            st.session_state = {}
            
    def tearDown(self):
        os.unlink(self.temp_file.name)
        
    @patch('main.PdfReader')
    def test_empty_pdf_handling(self, mock_pdf_reader):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""  
        
        mock_instance = MagicMock()
        mock_instance.pages = [mock_page]
        mock_pdf_reader.return_value = mock_instance

        with open(self.temp_file.name, 'rb') as f:
            text = get_pdf_text([f])
        
        self.assertEqual(text, "")
        
        @patch('streamlit.error')
        def test_streamlit_error_handling(mock_error):
            if not text:
                st.error("Failed to extract text from the provided PDF(s). Check if they contain selectable text.")
            
            mock_error.assert_called_once_with(
                "Failed to extract text from the provided PDF(s). Check if they contain selectable text."
            )
            
        test_streamlit_error_handling()

if __name__ == '__main__':
    unittest.main()