import torch
import os

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS  # Changed from Chroma to FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile
import asyncio
torch.classes.__path__ = []
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
st.set_page_config(
    page_title="Pakistan Constitution Assistant",
    page_icon="🇵🇰",  
    layout="wide"
)


DATA_DIR = os.path.join(tempfile.gettempdir(), "pakistan_constitution_db")
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")  

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
if not GROQ_API_KEY:
    st.error("GROQ API key is missing. Please set it in your Streamlit secrets or as an environment variable.")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #01411C;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #01411C;
        margin-bottom: 1rem;
    }
    .response-container {
        background-color: rgba(248, 249, 250, 0.05);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #01411C;
        margin-bottom: 60px;  /* Space for fixed footer */
        color: inherit;  /* Inherit text color from parent */
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        text-align: center;
        padding: 10px;
        background-color: rgba(255, 255, 255, 0.1);  /* Semi-transparent background */
        color: inherit;  /* Inherit text color from parent */
        font-size: 0.8rem;
        border-top: 1px solid rgba(240, 240, 240, 0.1);
        z-index: 100;
    }
    .loading {
        text-align: center;
        color: #01411C;
    }
    .chat-container {
        margin-bottom: 70px;  /* Ensure content doesn't hide behind footer */
    }
    /* Ensure emoji displays properly across browsers */
    .emoji-fix {
        font-family: "Segoe UI Emoji", "Noto Color Emoji", "Apple Color Emoji", "Android Emoji", sans-serif;
    }
    /* Dark mode compatibility */
    @media (prefers-color-scheme: dark) {
        .main-header, .sub-header {
            color: #3ed160;  /* Brighter green for dark mode */
        }
        .response-container {
            border-left-color: #3ed160;
        }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'><span class='emoji-fix'>🇵🇰</span> Pakistan Constitution Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask questions about the Constitution of Pakistan and get expert legal analysis</p>", unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown("<h2 class='sub-header'>About</h2>", unsafe_allow_html=True)
    st.markdown("""
    This application uses RAG (Retrieval Augmented Generation) to answer questions about the Constitution of Pakistan.
    
    It retrieves relevant sections from the constitution and generates expert legal responses based on the constitutional text.
    
    **How it works:**
    1. Enter your question about Pakistan's Constitution
    2. The system retrieves relevant constitutional provisions
    3. An AI legal expert generates a structured analysis
    
    **Data Source:** Official Constitution of Pakistan (2024 Edition)
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.history = []
        st.rerun()
    
    st.markdown("""
    **Note:** This application uses a pre-loaded Constitution of Pakistan PDF located in the data directory.
    """)

@st.cache_resource
def build_or_load_vector_store():
    """Build a new vector store if it doesn't exist, or load the existing one."""
    try:
        
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        
        if os.path.exists(INDEX_PATH):
            with st.spinner("Loading existing vector database..."):
                vector_store = FAISS.load_local(
                    folder_path=INDEX_PATH,
                    embeddings=embedding_model,
                    allow_dangerous_deserialization=True  # Added this parameter
                )
                return vector_store
        
        
        with st.spinner("Building new vector database (this may take a few minutes)..."):
            
            pdf_path = "data/constitution_of_pakistan.pdf"
            
            if not pdf_path:
                st.error("No PDF path defined. Please check your configuration.")
                return None
            
            if not os.path.exists(pdf_path):
                st.error(f"PDF file not found at path: {pdf_path}")
                return None
            
            
            docs = PyPDFLoader(pdf_path).load()
            
            
            def clean_text(text):
                return " ".join(text.split())
            
            cleaned_docs = [Document(page_content=clean_text(doc.page_content)) for doc in docs]
            
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200
            )
            documents = text_splitter.split_documents(cleaned_docs)
            
        
            vector_store = FAISS.from_documents(
                documents=documents,
                embedding=embedding_model
            )
            
            
            vector_store.save_local(INDEX_PATH)
            
            
                
            return vector_store

    except ImportError as e:
        st.error(f"Required dependency not available: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        return None


vector_db = build_or_load_vector_store()
db_initialized = vector_db is not None

def generate_response(question):
    if not GROQ_API_KEY:
        return "Error: GROQ API key is missing. Please set it in your Streamlit secrets or as an environment variable."
        
    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0.1,
        max_tokens=4096
    )
    
    template = """
    You are an expert legal assistant specializing in the Constitution of Pakistan (2024 Edition).
    CONTEXT INFORMATION:
    {context}
    QUESTION: {question}
    INSTRUCTIONS:
    1. Parse the question to identify the precise constitutional provisions, principles, or mechanisms being queried.
    2. Provide a structured, evidence-based response derived exclusively from the constitutional text provided in the context.
    3. When citing specific provisions, use the standardized citation format: "Article X(Y)" for sections/clauses and "Part Z" for larger divisions.
    4. For complex constitutional concepts, employ a hierarchical structure:
       - Primary heading: Constitutional principle/mechanism
       - Subheadings: Component elements
       - Bullet points: Specific provisions and their implications
    5. If the question falls outside the scope of the provided constitutional text:
       - Clearly state the information gap
       - Identify the specific constitutional provisions that would be needed
       - Avoid speculative interpretation
    6. Distinguish between:
       - Explicit constitutional text (direct quotations)
       - Constitutional mechanisms (procedural elements)
       - Constitutional principles (underlying concepts)
    RESPONSE FORMAT:
    CONSTITUTIONAL ANALYSIS: [Concise summary of the relevant constitutional framework]
    DETAILED RESPONSE:
    [Structured explanation with appropriate headings and citation-backed statements]
    RELEVANT PROVISIONS: [Complete list of all constitutional articles, sections, and clauses referenced]
    LIMITATIONS: [If applicable, note any constraints in addressing the question based on the provided context]
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    retriever = vector_db.as_retriever(
        search_type="similarity", 
        search_kwargs={
            "k": 8,  
        }
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {
            "context": retriever | format_docs, 
            "question": RunnablePassthrough()
        } 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    try:
        response = chain.invoke(question)
        return response if response.strip() else "The provided context does not contain the requested information."
    except Exception as e:
        return f"An error occurred: {str(e)}"

chat_container = st.container()

with chat_container:
    for message in st.session_state.history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").markdown(
                f"<div class='response-container'>{message['content']}</div>",
                unsafe_allow_html=True
            )

if db_initialized:
    user_question = st.chat_input("Ask a question about the Constitution of Pakistan...")

    if user_question:
        st.chat_message("user").write(user_question)
        st.session_state.history.append({"role": "user", "content": user_question})

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = generate_response(user_question)
            
                st.markdown(f"<div class='response-container'>{response}</div>", unsafe_allow_html=True)

        st.session_state.history.append({"role": "assistant", "content": response})
else:
    st.warning("Vector database not initialized. Please check that the Constitution PDF exists at 'data/constitution_of_pakistan.pdf'.")

st.markdown("""
<div class='footer'>
    © 2025 Pakistan Constitution Assistant | Not legal advice | For educational purposes only
</div>
""", unsafe_allow_html=True)