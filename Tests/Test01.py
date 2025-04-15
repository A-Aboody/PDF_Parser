import unittest
import os
import tempfile
import streamlit as st
from unittest.mock import patch, MagicMock
from io import BytesIO
from PyPDF2 import PdfWriter, PdfReader
from app import get_pdf_text, get_text_chunks, get_vectorstore, get_conversation_chain

class TestBasicDocumentProcessing(unittest.TestCase):
    
    def setUp(self):
        self.pdf_writer = PdfWriter()
        page = self.pdf_writer.add_blank_page(width=612, height=792)
        page.extract_text = MagicMock(return_value="This is a test document for basic processing.")
        
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        self.pdf_writer.write(self.temp_file)
        self.temp_file.close()
        
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = "fake_token_for_testing"
        
        if not hasattr(st, 'session_state'):
            st.session_state = {}
            
    def tearDown(self):
        os.unlink(self.temp_file.name)
        
    @patch('main.PdfReader')
    def test_basic_document_processing(self, mock_pdf_reader):
        mock_instance = MagicMock()
        mock_instance.pages = [MagicMock()]
        mock_instance.pages[0].extract_text.return_value = "This is a test document for basic processing."
        mock_pdf_reader.return_value = mock_instance
        
        with open(self.temp_file.name, 'rb') as f:
            text = get_pdf_text([f])
        self.assertIn("This is a test document", text)
        
        chunks = get_text_chunks(text)
        self.assertTrue(len(chunks) > 0)
        
        @patch('main.HuggingFaceEmbeddings')
        @patch('main.FAISS')
        def test_vectorstore(mock_faiss, mock_embeddings):
            mock_embeddings_instance = MagicMock()
            mock_embeddings.return_value = mock_embeddings_instance
            
            mock_vectorstore = MagicMock()
            mock_faiss.from_texts.return_value = mock_vectorstore
            
            vectorstore = get_vectorstore(chunks)
            self.assertIsNotNone(vectorstore)
            mock_faiss.from_texts.assert_called_once()
            
            return vectorstore
        
        vectorstore = test_vectorstore()
        
        @patch('main.HuggingFaceHub')
        @patch('main.ConversationBufferMemory')
        @patch('main.ConversationalRetrievalChain')
        def test_conversation_chain(mock_chain, mock_memory, mock_hub):
            mock_hub_instance = MagicMock()
            mock_hub.return_value = mock_hub_instance
            
            mock_memory_instance = MagicMock()
            mock_memory.return_value = mock_memory_instance
            
            mock_chain_instance = MagicMock()
            mock_chain.from_llm.return_value = mock_chain_instance
            
            chain = get_conversation_chain(vectorstore)
            self.assertIsNotNone(chain)
            mock_chain.from_llm.assert_called_once()
            
        test_conversation_chain()

if __name__ == '__main__':
    unittest.main()