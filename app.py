import os
os.environ["CHROMA_DB_IMPL"] = "duckdb"
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
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
DATA_DIR = os.path.join(tempfile.gettempdir(), "pakistan_constitution_db")
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
        margin-bottom: 60px;  /* Space for fixed footer */
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
    .loading {
        text-align: center;
        color: #01411C;
    }
    .chat-container {
        margin-bottom: 70px;  /* Ensure content doesn't hide behind footer */
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
    
    It retrieves relevant sections from the constitution and generates expert legal responses based on the constitutional text.
    
    **How it works:**
    1. Enter your question about Pakistan's Constitution
    2. The system retrieves relevant constitutional provisions
    3. An AI legal expert generates a structured analysis
    
    **Data Source:** Official Constitution of Pakistan (2024 Edition)
    """)
    
    uploaded_pdf = st.file_uploader("Upload Constitution PDF", type="pdf")
    
    if st.button("Clear Chat History"):
        st.session_state.history = []
        st.rerun()

@st.cache_resource
def build_or_load_vector_store(pdf_file=None):
    """Build a new vector store if it doesn't exist, or load the existing one using FastEmbed."""
    try:
        embedding_model = FastEmbedEmbeddings()
        db_file = os.path.join(DATA_DIR, "chroma.sqlite3")
        
        # If we have a PDF file uploaded or the database doesn't exist yet
        if pdf_file is not None or not os.path.exists(db_file):
            with st.spinner("Building new vector database (this may take a few minutes)..."):
                # If a file was uploaded, use it
                if pdf_file is not None:
                    # Save the uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(pdf_file.getvalue())
                        pdf_path = tmp_file.name
                else:
                    # Use the default file path - first we need to check if the file exists
                    default_pdf_path = "constitution_of_pakistan.pdf"
                    if not os.path.exists(default_pdf_path):
                        st.error(f"Default PDF file not found: {default_pdf_path}")
                        st.info("Please upload a PDF of the Pakistan Constitution.")
                        return None
                    pdf_path = default_pdf_path
                
                # Load the document
                docs = PyPDFLoader(pdf_path).load()
                
                # Clean the text
                def clean_text(text):
                    return " ".join(text.split())
                
                cleaned_docs = [Document(page_content=clean_text(doc.page_content)) for doc in docs]
                
                # Split the documents
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=200
                )
                documents = text_splitter.split_documents(cleaned_docs)
                
                # Create the database
                Chroma.from_documents(
                    documents=documents,
                    embedding=embedding_model,
                    persist_directory=DATA_DIR
                )
                
                # Delete the temporary file if it was created
                if pdf_file is not None:
                    try:
                        os.unlink(pdf_path)
                    except:
                        pass
                
                return Chroma(
                    persist_directory=DATA_DIR,
                    embedding_function=embedding_model
                )
                
        else:
            # Load the existing database
            with st.spinner("Loading existing vector database..."):
                return Chroma(
                    persist_directory=DATA_DIR,
                    embedding_function=embedding_model
                )

    except ImportError:
        st.error("FastEmbed not available. Please install with: pip install fastembed")
        return None
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        return None

# Initialize the vector store based on the uploaded file or existing database
chroma_db = build_or_load_vector_store(uploaded_pdf if 'uploaded_pdf' in locals() else None)
db_initialized = chroma_db is not None

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
    
    retriever = chroma_db.as_retriever(
        search_type="mmr", 
        search_kwargs={
            "fetch_k": 15,  
            "k": 7,  
            "lambda_mult": 0.7,
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
    if 'uploaded_pdf' not in locals() or uploaded_pdf is None:
        st.warning("Please upload the Constitution of Pakistan PDF to initialize the application.")

st.markdown("""
<div class='footer'>
    © 2025 Pakistan Constitution Assistant | Not legal advice | For educational purposes only
</div>
""", unsafe_allow_html=True)