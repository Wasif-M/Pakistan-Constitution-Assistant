import os
import streamlit as st
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


st.set_page_config(
    page_title="Pakistan Constitution Assistant",
    page_icon="🇵🇰",
    layout="wide"
)

PERSIST_DIRECTORY = r"C:\Users\WasifMehmood\Desktop\Agent\Pk-Constitution-Assistant\Data\faiss_index"
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
FAISS_INDEX_PATH = os.path.join(PERSIST_DIRECTORY, "faiss_index.pkl")

os.environ["GROQ_API_KEY"] = "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"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Styling
st.markdown("""...""", unsafe_allow_html=True)  # Keep your original style block
st.markdown("<h1 class='main-header'>🇵🇰 Pakistan Constitution Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask questions about the Constitution of Pakistan and get expert legal analysis</p>", unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown("<h2 class='sub-header'>About</h2>", unsafe_allow_html=True)
    st.markdown("""...""")  # Keep your original About block
    if st.button("Clear Chat History"):
        st.session_state.history = []
        st.rerun()


@st.cache_resource
def build_or_load_vector_store():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        if os.path.exists(FAISS_INDEX_PATH):
            with st.spinner("Loading existing FAISS index..."):
                with open(FAISS_INDEX_PATH, "rb") as f:
                    return pickle.load(f)

        with st.spinner("Building FAISS index..."):
            loader = PyPDFLoader(r"C:\Users\WasifMehmood\Desktop\Agent\Pk-Constitution-Assistant\Data\constitution_of_pakistan.pdf")
            docs = loader.load()

            def clean_text(text):
                return " ".join(text.split())

            cleaned_docs = [Document(page_content=clean_text(doc.page_content)) for doc in docs]

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            documents = text_splitter.split_documents(cleaned_docs)

            faiss_index = FAISS.from_documents(documents, embedding_model)

            with open(FAISS_INDEX_PATH, "wb") as f:
                pickle.dump(faiss_index, f)

            return faiss_index

    except Exception as e:
        st.error(f"Error building/loading FAISS vector store: {str(e)}")
        raise


try:
    faiss_db = build_or_load_vector_store()
    db_initialized = True
except Exception as e:
    st.error(f"Vector DB initialization failed: {e}")
    db_initialized = False


def generate_response(question):
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.1, max_tokens=4096)

    template = """..."""  # Keep your original system prompt

    prompt = ChatPromptTemplate.from_template(template)

    retriever = faiss_db.as_retriever(
        search_kwargs={
            "k": 7
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
        return response.strip() or "The provided context does not contain the requested information."
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
    st.warning("Vector database not initialized. Please check the error message above.")

st.markdown("""<div class='footer'>© 2025 Pakistan Constitution Assistant | Not legal advice | For educational purposes only</div>""", unsafe_allow_html=True)
