# 🧠 PDF Knowledge Base Chat

This project was developed as my final project for **ENGR 493** at the University of Michigan–Dearborn. Over the course of the semester, I built an interactive AI-powered application that allows users to upload, process, and chat with PDF documents using state-of-the-art natural language processing techniques.

---

## 📘 Overview

**PDF Knowledge Base Chat** enables users to ask natural language questions about one or more PDF documents. The app processes uploaded PDFs, extracts their text, transforms the content into vector embeddings, and uses a conversational LLM chain to respond to user queries based on document context.

---

## 🚀 Features

- 📄 Upload and process multiple PDF files at once.
- 🧠 Embed documents using `HuggingFace` sentence transformers.
- 🔗 Use `LangChain` to connect embeddings to a conversational LLM.
- 💬 Ask follow-up questions in a natural and persistent chat.
- 📈 View processing and chat performance metrics.
- 🌐 Clean, responsive UI with an interactive sidebar.
- 🔒 Uses a HuggingFace API token for secure LLM access.

---

## 🛠️ Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **NLP & LLM**: [LangChain](https://www.langchain.com/), [HuggingFace Hub](https://huggingface.co/)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **PDF Parsing**: PyPDF2
- **Environment Management**: `dotenv`

---

## 📊 Metric Dashboard

**The app tracks:**

- Chat response times per question
- Length of each user query
- PDF processing time per step (extraction, chunking, embedding, chain creation)
- These metrics help evaluate system performance and optimize interactions.

---

## 📚 How It Works

**Upload PDFs**
- Users upload one or more PDF files.

**Process Documents**

    - Text is extracted using PyPDF2.
    - Text is chunked using LangChain.
    - Each chunk is embedded using a HuggingFace transformer.
    - Embeddings are stored in a FAISS vector store.
    - A conversational retrieval chain is built.

**Ask Questions**
- User queries are matched with relevant text chunks, and an LLM generates answers based on context.

**Analyze Metrics**
- The app displays charts and breakdowns of processing and interaction data.

---

## 📅 Project Background

This tool was developed as a semester-long final project for **ENGR 493**. The project goal was to apply modern NLP and LLM techniques to build a practical, user-friendly tool.
It showcases:

- A real-world application of retrieval-augmented generation (RAG)
- Integration of various AI/NLP libraries
- End-to-end development from data processing to deployment

---

## 🧪 Sample Use Cases

- Summarizing academic papers
- Analyzing meeting notes or reports
- Answering questions about documentation
- Legal or contract review

---

## ✅ Future Enhancements

- PDF highlighting of source text in responses
- User authentication and session saving
- Multi-modal input support (e.g., scanned OCR PDFs)
- Deployment via Docker or cloud services
- Enhance the replies by utilizing a higher end LLM