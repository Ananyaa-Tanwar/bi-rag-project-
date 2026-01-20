import streamlit as st #ui framework
import os
import tempfile
from dotenv import load_dotenv
import pandas as pd
from langchain.document_loaders import PyPDFLoader # reads PDFs into langchain Document objects
from langchain.text_splitter import RecursiveCharacterTextSplitter # splits text into overlapping chunks
from langchain.embeddings import OpenAIEmbeddings # converts text chunks to embeddings
from langchain.vectorstores import FAISS # stores vector embeddings for semantic retrieval
from langchain.chains import RetrievalQA # chain that connects LLMs with retrievers
from langchain.chat_models import ChatOpenAI # gpt wrapper
from langchain.schema import Document # LangChain document type

# Load environment variables
load_dotenv()

# --- Helper Functions ---
def load_pdf(path):
    '''
    This functions loads and parses a PDF into LangChain Document objects.
    '''
    loader = PyPDFLoader(path)
    return loader.load()

def load_csv(path):
    '''
    This function reads a CSV file and converts its entire content into a string.
    '''
    df = pd.read_csv(path)
    return df.to_string(index=False)

def create_documents_from_csv(csv_path):
    '''
    Wraps the CSV string content into a Document for downstream processing.
    '''
    content = load_csv(csv_path)
    return [Document(page_content=content, metadata={"source": csv_path})]

def embed_docs(docs):
    '''
    This function does 3 things:
    1. Splits large texts into overlapping chunks.
    2. Embeds the chunks into vectors using OpenAI's API.
    3. It stores embeddings in an FAISS index for fast retrieval.
    '''
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def build_rag_chain(vectorstore):
    '''
    This functions sets up a retrieval-augmented generation (RAG) chain.
    On query, it retrieves top k (here, 4) matching documents and passes them to GPT-4 for answer generation.
    '''
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )
    return chain

# Streamlit App Code
st.set_page_config(page_title="Business Intelligence Document Q&A", layout="wide")
st.title("Business Intelligence Document Q&A (CSV + PDF)")

# File upload: allows users to upload multiple files
uploaded_files = st.file_uploader(
    "Upload PDF and/or CSV files",
    type=["pdf", "csv"],
    accept_multiple_files=True
)

# Initialize session state
# Used to preserve state across reruns, enabling continuous Q&A without reloading documents.
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
    st.session_state.sources = []
    st.session_state.vectorstore = None

# Initial setup on upload
# Reads uploaded files and creates LangChain documents. 
# Calls embed_docs() → build_rag_chain() → stores in session state.
if uploaded_files and st.session_state.qa_chain is None:
    with st.spinner("Processing uploaded documents..."):
        docs = []
        temp_paths = []

        for uploaded_file in uploaded_files:
            suffix = os.path.splitext(uploaded_file.name)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                temp_paths.append(tmp.name)

                if suffix == ".pdf":
                    docs.extend(load_pdf(tmp.name))
                elif suffix == ".csv":
                    docs.extend(create_documents_from_csv(tmp.name))

        if not docs:
            st.error("No valid documents were processed.")
        else:
            # Embed and store chain
            st.session_state.vectorstore = embed_docs(docs)
            st.session_state.qa_chain = build_rag_chain(st.session_state.vectorstore)
            st.success("Documents processed. You can now ask questions!")

# Question input and response
# If chain is initialized, shows a question box.
# Sends query to the chain and returns the answer and relevant source documents.
if st.session_state.qa_chain:
    query = st.text_input("Ask a question about your documents:")
    if query:
        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain.invoke({"query": query})
            st.session_state.sources = response['source_documents']

        # Display answer
        # It renders the answer and expands to show the most relevant retrieved texts (max 50 characters).
        st.success("Answer:")
        st.markdown(f"**{response['result']}**")

        with st.expander("📎 Source Documents"):
            for doc in st.session_state.sources:
                st.markdown(f"- **Source**: `{doc.metadata['source']}`")
                st.code(doc.page_content[:500] + "...")

