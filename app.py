import os
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile

st.set_page_config(
    page_title="Pakistan Constitution Assistant",
    page_icon="🇵🇰",
    layout="wide"
)

# Create a data directory that works in Streamlit Cloud
DATA_DIR = os.path.join(tempfile.gettempdir(), "pakistan_constitution_faiss")
os.makedirs(DATA_DIR, exist_ok=True)

# Get API key from Streamlit secrets or environment variable
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
if not GROQ_API_KEY:
    st.error("GROQ API key is missing. Please set it in your Streamlit secrets or as an environment variable.")

# Set environment variables
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
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #01411C;
        margin-bottom: 60px;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        text-align: center;
        padding: 10px;
        background-color: white;
        color: gray;
        font-size: 0.8rem;
        border-top: 1px solid #f0f0f0;
        z-index: 100;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🇵🇰 Pakistan Constitution Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask questions about the Constitution of Pakistan and get expert legal analysis</p>", unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown("<h2 class='sub-header'>About</h2>", unsafe_allow_html=True)
    st.markdown("""
    This application uses RAG (Retrieval Augmented Generation) to answer questions about the Constitution of Pakistan.
    
    **How it works:**
    1. Enter your question about Pakistan's Constitution
    2. The system retrieves relevant constitutional provisions
    3. An AI legal expert generates a structured analysis
    
    **Data Source:** Official Constitution of Pakistan
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.history = []
        st.rerun()

@st.cache_resource
def build_or_load_vector_store():
    """Build a new vector store if it doesn't exist, or load the existing one."""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        index_path = os.path.join(DATA_DIR, "faiss_index")
        
        if not os.path.exists(index_path):
            with st.spinner("Building new vector database (this may take a few minutes)..."):
                pdf_path = "data/constitution_of_pakistan.pdf"
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
                
                db = FAISS.from_documents(documents, embedding_model)
                db.save_local(index_path)
                return db
        else:
            with st.spinner("Loading existing vector database..."):
                return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)

    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        return None

# Initialize the vector store
faiss_db = build_or_load_vector_store()
db_initialized = faiss_db is not None

def generate_response(question):
    if not GROQ_API_KEY:
        return "Error: GROQ API key is missing."
        
    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0.1,
        max_tokens=4096
    )
    
    template = """
    You are an expert legal assistant specializing in the Constitution of Pakistan.
    CONTEXT INFORMATION:
    {context}
    QUESTION: {question}
    
    Provide a structured response with:
    1. Relevant constitutional articles
    2. Legal interpretation
    3. Practical implications
    
    Format citations as "Article X(Y)".
    If unsure, state you couldn't find relevant provisions.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    retriever = faiss_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10}
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    try:
        return chain.invoke(question)
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Chat interface
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
            with st.spinner("Analyzing constitutional provisions..."):
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