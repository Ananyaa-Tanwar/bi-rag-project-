# BI RAG Application: PDF & CSV Question Answering

A Retrieval-Augmented Generation (RAG) application that enables users to upload **business intelligence documents (PDFs and CSVs)** and ask natural-language questions over them.  
The system combines **semantic search (FAISS)** with **LLM-based generation** to produce grounded, context-aware answers from uploaded data.

Built as an interactive **Streamlit** application using **LangChain** and **OpenAI embeddings**.

---

## Key Features

- Upload and analyze multiple PDFs and CSVs
- Semantic search using vector embeddings with FAISS
- Natural-language Q&A over structured and unstructured data
- Source-aware responses grounded in retrieved document chunks
- Modular architecture suitable for BI and analytics use cases
- Prototype notebook included for experimentation and testing

---

## Architecture Overview

1. **Document Ingestion**
   - PDFs are parsed page-by-page
   - CSVs are chunked by rows for scalable retrieval

2. **Embedding & Indexing**
   - Documents are split into chunks
   - Each chunk is embedded using OpenAI embeddings
   - Embeddings are stored in a FAISS vector index

3. **Retrieval-Augmented Generation**
   - User queries are embedded
   - Top-k relevant chunks are retrieved
   - An LLM generates answers grounded in retrieved context

---

## Tech Stack

- Python
- Streamlit (UI)
- LangChain
- OpenAI Embeddings and LLMs
- FAISS (vector store)
- Pandas (CSV handling)
- PyPDF (PDF parsing)

---

## Project Structure

```text
bi-rag-project/
├── app.py                  # Streamlit application
├── app.ipynb               # Prototyping and experimentation notebook
├── requirements.txt
├── README.md
├── .gitignore
└── .env.example
```
---

## Notebook

The `app.ipynb` notebook contains:

- Early experimentation and prototyping
- Chunking and retrieval testing
- Prompt and response evaluation

The notebook is **not required** to run the application but is included for transparency and reproducibility.

---

## Use Cases

- Business intelligence document analysis
- Ad-hoc analytics over CSV exports
- Internal knowledge base Q&A
- Rapid prototyping of RAG pipelines

---

## Future Improvements

- Full-document summarization using map-reduce chains
- Query-time source citations
- Support for additional file formats
- Persistent vector stores across sessions
- Role-based document access

---

## Author

**Ananyaa Tanwar**  
Graduate student in Information Management at UIUC with a focus on data science, BI systems, and applied machine learning.
