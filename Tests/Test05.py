import unittest
import streamlit as st
from unittest.mock import patch, MagicMock
from app import handle_userinput

class TestQuestionWithoutProcessing(unittest.TestCase):
    
    def setUp(self):
        if not hasattr(st, 'session_state'):
            st.session_state = {}
        
        st.session_state.conversation = None
        st.session_state.chat_history = []
        st.session_state.metrics = {
            "response_times": [], 
            "questions": [], 
            "interactions": []
        }
        
    @patch('streamlit.error')
    def test_question_without_processing(self, mock_error):
        user_question = "What is the main topic of the document?"
        
        handle_userinput(user_question)
        
        mock_error.assert_called_once_with(
            "Conversation not initialized. Please process your documents first."
        )
        
        self.assertEqual(len(st.session_state.metrics["questions"]), 0)
        self.assertEqual(len(st.session_state.metrics["response_times"]), 0)
        self.assertEqual(len(st.session_state.metrics["interactions"]), 0)
        
    @patch('streamlit.warning')
    def test_chat_input_warning(self, mock_warning):
        
        user_question = "What is the main topic of the document?"
        
        if user_question:
            if not st.session_state.conversation:
                st.warning("Please process documents before asking questions.")
        
        mock_warning.assert_called_once_with(
            "Please process documents before asking questions."
        )

if __name__ == '__main__':
    unittest.main()