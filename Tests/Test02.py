import unittest
import os
import tempfile
import streamlit as st
from unittest.mock import patch, MagicMock
from PyPDF2 import PdfWriter, PdfReader
from app import get_pdf_text, get_text_chunks, get_vectorstore, get_conversation_chain

class TestMultipleDocumentProcessing(unittest.TestCase):
    
    def setUp(self):
        self.temp_files = []
        
        test_contents = [
            "This is the first test document with basic content.",
            "This is the second document with some more extensive content for testing multiple document processing capabilities of our system.",
            "This is the third document that contains additional information about our testing methodology and approaches."
        ]
        
        for i, content in enumerate(test_contents):
            pdf_writer = PdfWriter()
            page = pdf_writer.add_blank_page(width=612, height=792)
            page.extract_text = MagicMock(return_value=content)
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            pdf_writer.write(temp_file)
            temp_file.close()
            self.temp_files.append(temp_file)
        
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = "fake_token_for_testing"
        
        if not hasattr(st, 'session_state'):
            st.session_state = {}
            
    def tearDown(self):
        for temp_file in self.temp_files:
            os.unlink(temp_file.name)
        
    @patch('main.PdfReader')
    def test_multiple_document_processing(self, mock_pdf_reader):
        mock_pages = []
        for i, temp_file in enumerate(self.temp_files):
            mock_page = MagicMock()
            mock_page.extract_text.return_value = f"This is document {i+1} content."
            mock_pages.append(mock_page)
        
        mock_pdf_reader.side_effect = [
            MagicMock(pages=[mock_pages[0]]),
            MagicMock(pages=[mock_pages[1]]),
            MagicMock(pages=[mock_pages[2]])
        ]
        
        pdf_file_objects = []
        for temp_file in self.temp_files:
            with open(temp_file.name, 'rb') as f:
                pdf_file_objects.append(f)
        
        text = get_pdf_text(pdf_file_objects)
        
        for i in range(3):
            self.assertIn(f"This is document {i+1} content.", text)
        
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
            args, kwargs = mock_faiss.from_texts.call_args
            self.assertEqual(args[0], chunks)
            
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