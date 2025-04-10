# Fix SQLite version issue (MUST BE FIRST)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # Fall back to default sqlite3 if pysqlite3 not available

import os
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Try Chroma first, fall back to FAISS if unavailable
try:
    from langchain_chroma import Chroma
    USE_CHROMA = True
except ImportError:
    USE_CHROMA = False
    from langchain_community.vectorstores import FAISS
    st.warning("Using FAISS instead of Chroma - some features may be limited")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# [Rest of your existing configuration code...]

@st.cache_resource
def build_or_load_vector_store():
    try:
        from langchain_community.embeddings import FastEmbedEmbeddings
        embedding_model = FastEmbedEmbeddings()

        if USE_CHROMA:
            if os.path.exists(f"{PERSIST_DIRECTORY}/chroma.sqlite3"):
                with st.spinner("Loading existing Chroma database..."):
                    return Chroma(
                        persist_directory=PERSIST_DIRECTORY,
                        embedding_function=embedding_model
                    )
            else:
                with st.spinner("Building new Chroma database..."):
                    docs = PyPDFLoader(PDF_PATH).load()
                    # [Rest of your document processing code...]
                    return Chroma.from_documents(
                        documents=documents,
                        embedding=embedding_model,
                        persist_directory=PERSIST_DIRECTORY
                    )
        else:
            # Fallback to FAISS
            with st.spinner("Using FAISS vector store..."):
                docs = PyPDFLoader(PDF_PATH).load()
                # [Rest of your document processing code...]
                return FAISS.from_documents(documents, embedding_model)
                
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        return None

# [Rest of your existing application code...]