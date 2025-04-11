
# Pakistan Constitution Assistant

![Pakistan Flag](https://img.shields.io/badge/-Pakistan%20Constitution-01411C?style=for-the-badge&logo=bookstack&logoColor=white)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pakistan-constitution-assistant-pkconst24.streamlit.app/)

An AI-powered legal assistant for the Constitution of Pakistan.

## Description

This application leverages Retrieval-Augmented Generation (RAG) to provide accurate information and expert analysis about Pakistan's constitutional framework. Ask any question about the Constitution in natural language and receive structured legal responses with proper citations.

## Features

- **Natural Language Interface** - Ask questions in plain English
- **Structured Responses** - Get organized answers with proper legal citations
- **Contextual Understanding** - All answers grounded in the constitutional text
- **Advanced Retrieval** - Semantic search finds relevant provisions
- **Powered by GROQ** - Fast responses using state-of-the-art LLMs

## Technology Stack

- **Backend**: Python, LangChain
- **Vector Database**: FAISS
- **Embeddings**: HuggingFace's all-MiniLM-L6-v2
- **LLM**: Groq's LLaMA3-70b 
- **Frontend**: Streamlit
- **Document Processing**: PyPDFLoader, RecursiveCharacterTextSplitter

## Project Structure

```
.
├── .streamlit/               # Streamlit configuration
├── data/                     # Data directory containing the Constitution PDF
├── pakistan_constitution_db/ # Vector database storage
├── app.py                    # Main application file
├── main.py                   # Alternative entry point
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Installation & Local Development

1. Clone the repository:
   ```
   git clone https://github.com/Wasif-M/Pakistan-Constitution-Assistant
   cd pakistan-constitution-assistant
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your GROQ API key:
   - Create a `.streamlit/secrets.toml` file
   - Add your GROQ API key: `GROQ_API_KEY = "your_api_key_here"`

4. Run the application:
   ```
   streamlit run app.py
   ```

## How It Works

1. The Constitution of Pakistan is processed and stored in a vector database
2. User questions are analyzed and converted to vector embeddings
3. Relevant sections of the Constitution are retrieved based on semantic similarity
4. An AI language model generates a comprehensive answer using the retrieved context

## Limitations

- Limited to information contained in the Constitution of Pakistan
- Not a substitute for professional legal advice
- Educational purposes only

## Disclaimer

This application is not affiliated with the Government of Pakistan or any official legal entity. It is designed for educational and informational purposes only.

## Contact

For questions or feedback, please open an issue in this repository.

---


## Disclaimer

This application is for educational purposes only and does not constitute legal advice. For official legal matters, please consult a qualified attorney.
