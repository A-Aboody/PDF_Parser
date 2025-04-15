import streamlit as st
import os
import time
import pandas as pd
import altair as alt
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.schema import HumanMessage, AIMessage

st.set_page_config(
    page_title="ENGR 493",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

css = '''
<style>
    .main-header {
        background-color: #1E3A8A;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        border-radius: 0.375rem;
        padding: 0.5rem 1rem;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    .sidebar-section {
        margin-bottom: 1.5rem;
    }
    .sidebar-section h3 {
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        color: #1E3A8A;
    }
    /* Ensure chat input doesn't overlap content when page scrolls */
    .stChatInputContainer {
        position: sticky;
        bottom: 0;
        background-color: white; /* Adjust background as needed */
        padding-top: 10px;
        padding-bottom: 10px;
        z-index: 1000;
    }
</style>
'''

def load_api_token():
    env_file = ".env.example"
    token = None

    if os.path.exists(env_file):
        load_dotenv(env_file)
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
            return token
        else:
            try:
                with open(env_file, 'r') as file:
                    content = file.read()
                    for line in content.split('\n'):
                        if line.startswith("HUGGINGFACEHUB_API_TOKEN="):
                            token = line.split('=', 1)[1].strip()
                            if token:
                                os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
                                return token
            except Exception as e:
                st.warning(f"Could not read API token from {env_file}: {str(e)}")

    st.error("HuggingFace API token not found. Please ensure a `.env.example` file exists in the same directory with `HUGGINGFACEHUB_API_TOKEN=your_token`.")
    return None

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.error(f"Error reading file {pdf.name}: {e}")
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.warning(f"Primary embedding method failed: {str(e)}. Trying alternative...")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
            )
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            return vectorstore
        except Exception as e2:
            st.error(f"Alternative embedding method also failed: {str(e2)}")
            st.stop()

def get_conversation_chain(vectorstore):
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        st.error("API Token is not set. Cannot create conversation chain.")
        return None

    try:
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-large",
            model_kwargs={
                "temperature": 0.2,    
                "max_length": 1024    
                }
        )

        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
            )
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_kwargs={'k': 4} 
                ),
            memory=memory,
            return_source_documents=True
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        raise e

def handle_userinput(user_question):
    if not st.session_state.conversation:
        st.error("Conversation not initialized. Please process your documents first.")
        return

    try:
        start_time = time.time()

        st.session_state.metrics["questions"].append(user_question)

        response = st.session_state.conversation({'question': user_question})

        end_time = time.time()
        response_time = end_time - start_time
        st.session_state.metrics["response_times"].append(response_time)

        ai_answer = response.get('answer', "Sorry, I couldn't generate a response.")

        st.session_state.chat_history.append(AIMessage(content=ai_answer))

        st.session_state.metrics["interactions"].append(len(st.session_state.metrics["questions"]))

    except Exception as e:
        st.error(f"Error getting response: {str(e)}")
        st.session_state.chat_history.append(AIMessage(content=f"Sorry, an error occurred: {e}"))


def display_metrics():
    if not st.session_state.metrics["response_times"]:
        st.info("No chat metrics available yet. Ask questions after processing documents.")
        return

    st.subheader("📊 Chat Performance Metrics")
    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.metrics["response_times"]:
            df_response_times = pd.DataFrame({
                'Interaction': range(1, len(st.session_state.metrics["response_times"]) + 1),
                'Response Time (s)': st.session_state.metrics["response_times"]
            })

            chart = alt.Chart(df_response_times).mark_line(point=True).encode(
                x=alt.X('Interaction:O', axis=alt.Axis(title='Interaction Number')),
                y=alt.Y('Response Time (s):Q', axis=alt.Axis(title='Response Time (s)')),
                tooltip=['Interaction', 'Response Time (s)']
            ).properties(
                title='Response Times per Interaction'
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No response time data yet.")

    with col2:
        if st.session_state.metrics["questions"]:
            df_question_length = pd.DataFrame({
                'Interaction': range(1, len(st.session_state.metrics["questions"]) + 1),
                'Question Length (chars)': [len(q) for q in st.session_state.metrics["questions"]]
            })

            chart = alt.Chart(df_question_length).mark_bar().encode(
                x=alt.X('Interaction:O', axis=alt.Axis(title='Interaction Number')),
                y=alt.Y('Question Length (chars):Q', axis=alt.Axis(title='Question Length (chars)')),
                tooltip=['Interaction', 'Question Length (chars)']
            ).properties(
                title='Question Length per Interaction'
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No question length data yet.")


def show_how_it_works():
    with st.expander("ℹ️ How it Works", expanded=False):
        st.markdown("""
        ## How the PDF Knowledge Base Works

        This application allows you to chat with multiple PDF documents using advanced NLP techniques:

        1.  **Upload PDFs**: Use the sidebar to upload one or more PDF documents.
        2.  **Process Documents**: Click the 'Process Documents' button. The system then:
            * Extracts text from your PDFs (`PyPDF2`).
            * Splits the text into manageable chunks (`LangChain`).
            * Creates vector embeddings (numerical representations) of these chunks using a sentence transformer model (`HuggingFace Embeddings`).
            * Stores these embeddings in a searchable vector database (`FAISS`).
            * Initializes a conversational AI chain (`LangChain`, `HuggingFace Hub LLM`) linked to the vector database and conversation memory.
        3.  **Ask Questions**: Type your questions in the chat input box at the bottom.
        4.  **Get Answers**: The system retrieves relevant text chunks based on your question and chat history, then uses the Large Language Model (`google/flan-t5-large`) to generate an answer based on that context.
        5.  **View Metrics**: Performance metrics for document processing and chat interactions are displayed below the chat and in the sidebar.
        """)

def main():
    api_token = load_api_token()

    st.markdown(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "metrics" not in st.session_state:
        st.session_state.metrics = {
            "response_times": [], "questions": [], "interactions": []
        }
    if "processing_metrics" not in st.session_state:
        st.session_state.processing_metrics = None

    st.markdown('<div class="main-header"><h1>🧠 PDF Knowledge Base Chat</h1></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Controls & Info")

        st.subheader("📄 Document Upload")
        pdf_docs = st.file_uploader(
            "Upload PDF files and click 'Process'",
            accept_multiple_files=True,
            type=["pdf"],
            key="pdf_uploader"
        )

        if st.button("⚙️ Process Documents", key="process_button", use_container_width=True):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            elif not api_token:
                 st.error("Cannot process documents. API token is missing.")
            else:
                with st.spinner("Processing documents... This may take a moment."):
                    try:
                        st.session_state.conversation = None
                        st.session_state.chat_history = []
                        st.session_state.metrics = {"response_times": [], "questions": [], "interactions": []}
                        st.session_state.processing_metrics = None # Clear old metrics
                        start_total_time = time.time()

                        step_start_time = time.time()
                        raw_text = get_pdf_text(pdf_docs)
                        extraction_time = time.time() - step_start_time
                        if not raw_text:
                            st.error("Failed to extract text from the provided PDF(s). Check if they contain selectable text.")
                            st.stop()
                        st.info(f"📚 Extracted {len(raw_text):,} characters.")

                        step_start_time = time.time()
                        text_chunks = get_text_chunks(raw_text)
                        chunking_time = time.time() - step_start_time
                        if not text_chunks:
                            st.error("Failed to split text into chunks.")
                            st.stop()
                        st.info(f"✂️ Split into {len(text_chunks):,} text chunks.")

                        step_start_time = time.time()
                        vectorstore = get_vectorstore(text_chunks)
                        embedding_time = time.time() - step_start_time
                        if vectorstore is None:
                            st.stop()
                        st.info("🧠 Created vector embeddings.")

                        step_start_time = time.time()
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        chain_time = time.time() - step_start_time
                        if st.session_state.conversation is None:
                            st.stop()
                        st.info("🔗 Initialized conversation chain.")

                        st.session_state.processing_metrics = {
                            "extraction_time": extraction_time,
                            "chunking_time": chunking_time,
                            "embedding_time": embedding_time,
                            "chain_time": chain_time,
                            "total_time": time.time() - start_total_time,
                            "num_docs": len(pdf_docs),
                            "num_chunks": len(text_chunks),
                            "text_length": len(raw_text)
                        }

                        st.success(f"✅ Processing complete! ({st.session_state.processing_metrics['total_time']:.2f}s)")

                    except Exception as e:
                        st.error(f"An error occurred during processing: {e}")
                        st.exception(e)

        if st.session_state.processing_metrics:
            st.subheader("⚙️ Processing Metrics")
            metrics = st.session_state.processing_metrics
            df_times = pd.DataFrame({
                'Step': ['Text Extraction', 'Text Chunking', 'Vector Embedding', 'Chain Creation'],
                'Time (s)': [
                    metrics['extraction_time'], metrics['chunking_time'],
                    metrics['embedding_time'], metrics['chain_time']
                ]
            })
            chart = alt.Chart(df_times).mark_bar().encode(
                x=alt.X('Step:N', sort='-y'),
                y=alt.Y('Time (s):Q'),
                color='Step:N',
                tooltip=['Step', alt.Tooltip('Time (s):Q', format='.2f')]
            ).properties(
                title='Processing Time Breakdown', height=250
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", metrics['num_docs'])
                st.metric("Text Length", f"{metrics['text_length']:,} chars")
            with col2:
                st.metric("Text Chunks", f"{metrics['num_chunks']:,}")
                st.metric("Total Time", f"{metrics['total_time']:.2f}s")

        show_how_it_works()

    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
             st.info("Upload and process PDFs, then ask a question below to start chatting!")

        for message in st.session_state.chat_history:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(message.content)

    user_question = st.chat_input("Ask a question about your documents...")

    if user_question:
        if not st.session_state.conversation:
            st.warning("Please process documents before asking questions.")
        else:
            st.session_state.chat_history.append(HumanMessage(content=user_question))

            with chat_container:
                 with st.chat_message("user"):
                     st.markdown(user_question)

            with st.spinner("Thinking..."):
                 handle_userinput(user_question)

            st.rerun()

    with st.expander("📊 View Chat Performance Metrics", expanded=False):
        display_metrics()


if __name__ == '__main__':
    main()