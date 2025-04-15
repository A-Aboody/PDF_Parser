import unittest
import os
import tempfile
import streamlit as st
import time
from unittest.mock import patch, MagicMock
from PyPDF2 import PdfWriter, PdfReader
from langchain.schema import HumanMessage, AIMessage
from app import get_pdf_text, get_text_chunks, get_vectorstore, get_conversation_chain, handle_userinput

class TestBasicQuestionAnswering(unittest.TestCase):
    
    def setUp(self):
        self.pdf_content = """
        # Research Paper: Machine Learning for Climate Change Prediction
        
        Abstract:
        This paper presents a novel approach to climate change prediction using advanced machine learning techniques.
        We demonstrate that neural networks can accurately forecast temperature trends using historical data.
        
        Methods:
        We employed a combination of convolutional neural networks (CNNs) and recurrent neural networks (RNNs)
        trained on a global dataset of temperature measurements from 1950 to 2020.
        
        Results:
        Our model achieved 87% accuracy in predicting temperature anomalies with a six-month lead time.
        """
        
        self.pdf_writer = PdfWriter()
        page = self.pdf_writer.add_blank_page(width=612, height=792)
        page.extract_text = MagicMock(return_value=self.pdf_content)
        
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        self.pdf_writer.write(self.temp_file)
        self.temp_file.close()
        
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = "fake_token_for_testing"
        
        if not hasattr(st, 'session_state'):
            st.session_state = {}
            
        st.session_state.chat_history = []
        st.session_state.metrics = {
            "response_times": [], 
            "questions": [], 
            "interactions": []
        }
            
    def tearDown(self):
        os.unlink(self.temp_file.name)
        
    @patch('main.PdfReader')
    @patch('main.HuggingFaceEmbeddings')
    @patch('main.FAISS')
    @patch('main.HuggingFaceHub')
    @patch('main.ConversationBufferMemory')
    @patch('main.ConversationalRetrievalChain')
    def test_basic_question_answering(self, mock_chain, mock_memory, mock_hub, 
                                     mock_faiss, mock_embeddings, mock_pdf_reader):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = self.pdf_content
        mock_instance = MagicMock()
        mock_instance.pages = [mock_page]
        mock_pdf_reader.return_value = mock_instance
        
        with open(self.temp_file.name, 'rb') as f:
            text = get_pdf_text([f])
        self.assertEqual(text, self.pdf_content)
        
        chunks = get_text_chunks(text)
        
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_vectorstore = MagicMock()
        mock_faiss.from_texts.return_value = mock_vectorstore
        
        vectorstore = get_vectorstore(chunks)
        
        mock_llm = MagicMock()
        mock_hub.return_value = mock_llm
        
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        
        mock_chain_instance = MagicMock()
        mock_chain.from_llm.return_value = mock_chain_instance
        
        expected_answer = "The paper is about machine learning for climate change prediction. It uses neural networks to forecast temperature trends."
        mock_chain_instance.return_value = {
            'answer': expected_answer,
            'source_documents': []
        }
        
        conversation = get_conversation_chain(vectorstore)
        st.session_state.conversation = conversation
        
        user_question = "What is the main topic of this paper?"
        
        with patch('time.time') as mock_time:
            mock_time.side_effect = [0, 1] 
            handle_userinput(user_question)
        
        mock_chain_instance.assert_called_once()
        call_args = mock_chain_instance.call_args[0][0]
        self.assertEqual(call_args['question'], user_question)
        
        self.assertEqual(len(st.session_state.chat_history), 1)
        self.assertTrue(isinstance(st.session_state.chat_history[0], AIMessage))
        self.assertEqual(st.session_state.chat_history[0].content, expected_answer)
        
        self.assertEqual(len(st.session_state.metrics["questions"]), 1)
        self.assertEqual(st.session_state.metrics["questions"][0], user_question)
        self.assertEqual(len(st.session_state.metrics["response_times"]), 1)
        self.assertEqual(st.session_state.metrics["response_times"][0], 1) 
        self.assertEqual(len(st.session_state.metrics["interactions"]), 1)
        self.assertEqual(st.session_state.metrics["interactions"][0], 1)

if __name__ == '__main__':
    unittest.main()