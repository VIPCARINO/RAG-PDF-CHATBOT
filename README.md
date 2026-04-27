# 📚 PDF RAG Assistant

A fully local **Retrieval-Augmented Generation (RAG)** application built with **Streamlit**, **ChromaDB**, **SentenceTransformers**, and **Ollama**.

Upload PDFs, process them into embeddings, store them in ChromaDB, and chat with your documents using a local LLM.

---

# ✨ Features

- 📄 PDF upload and processing
- 🧩 Structured PDF parsing
- ✂️ Intelligent chunking
- 🧠 Local embeddings with SentenceTransformers
- 🔎 Hybrid retrieval pipeline
  - Semantic Search
  - BM25 Ranking
  - Cross-Encoder Reranking
- 💾 Persistent ChromaDB vector storage
- 🗂 SQLite metadata management
- 💬 Streaming AI responses
- 📚 Multi-document support
- 🗑 Delete PDFs and embeddings
- ⚡ Fully local AI stack
- 🎨 Streamlit interface

---

# 🏗 Architecture

```text
PDF Upload
    ↓
PDF Extraction & Structuring
    ↓
Chunking
    ↓
Embedding Generation
    ↓
ChromaDB Vector Storage
    ↓
User Query
    ↓
Query Expansion
    ↓
Semantic Search
    ↓
BM25 Ranking
    ↓
Cross-Encoder Reranking
    ↓
Context Building
    ↓
LLM Response Generation
```

---

# 📂 Project Structure

```text
.
│
├── app.py
├── database.py
├── helper.py
├── ingestion_pipeline_class_based.py
├── RAG_Phase_5_query_answer_class_based.py
│
├── database/
│   ├── metadata.db
│   └── chroma_db/
│
├── json_files/
│   └── chat_histories.json
│
├── models/
│   ├── all-MiniLM-L6-v2/
│   └── cross-encoder/
│
└── README.md
```

---

# ⚙️ Tech Stack

## Frontend
- Streamlit

## Vector Database
- ChromaDB

## Embeddings
- SentenceTransformers
- all-MiniLM-L6-v2

## Reranking
- CrossEncoder
- ms-marco-MiniLM-L-6-v2

## LLM
- Ollama
- Llama 3.2

## Search
- BM25Okapi

## Database
- SQLite

---

# 🚀 Installation

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/pdf-rag-assistant.git
cd pdf-rag-assistant
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🦙 Install Ollama

Download Ollama:

https://ollama.com/download

---

# 📥 Pull Llama Model

```bash
ollama pull llama3.2
```

---

# 📦 Required Models

Place these models inside the `models/` directory.

## Embedding Model

```text
models/all-MiniLM-L6-v2
```

## Cross Encoder

```text
models/cross-encoder/ms-marco-MiniLM-L-6-v2
```

You can download them from Hugging Face.

---

# ▶️ Run Application

```bash
streamlit run app.py
```

---

# 💬 Example Questions

```text
What is the main topic of chapter 3?
```

```text
Summarize this document.
```

```text
What does the author say about machine learning?
```

---

# 🧠 How It Works

## PDF Ingestion Pipeline

The ingestion pipeline performs:

1. PDF extraction
2. Structure parsing
3. Chunk generation
4. Embedding generation
5. ChromaDB indexing

Implemented in:

```python
PDFIngestionPipeline
```

---

## Retrieval Pipeline

The RAG pipeline performs:

### Query Expansion
Generates multiple semantic versions of the query.

### Semantic Search
Retrieves relevant chunks using embeddings.

### BM25 Ranking
Adds keyword-based retrieval.

### Cross-Encoder Reranking
Improves retrieval accuracy.

### Context Building
Creates token-safe context windows.

### LLM Generation
Streams responses from Ollama.

Implemented in:

```python
LocalRAGPipeline
```

---

# 📊 Retrieval Pipeline

The system combines:

- Dense vector retrieval
- Sparse keyword retrieval
- Neural reranking

This hybrid approach improves answer quality significantly.

---

# 🔥 Streaming Responses

Responses are streamed token-by-token using:

```python
ask_llm_stream()
```

This improves responsiveness and user experience.

---

# 🗃 Persistent Storage

## SQLite

Stores:
- document metadata
- filenames
- upload dates
- collection names

## ChromaDB

Stores:
- embeddings
- chunks
- metadata
- document IDs

---

# 🛠 Main Classes

| Class | Purpose |
|---|---|
| `PDFIngestionPipeline` | Full ingestion pipeline |
| `LocalRAGPipeline` | Retrieval + generation |
| `MetadataDB` | SQLite manager |
| `EmbeddingPipeline` | Embedding generation |
| `PDFStructurer` | PDF extraction |
| `ChromaIndexer` | Vector indexing |

---

# 🔒 Fully Local AI

This project runs entirely locally.

- No OpenAI API
- No cloud vector database
- No external embedding APIs

Your documents remain private on your machine.

---

# 📌 Future Improvements

- Multi-PDF querying
- OCR support
- Citation highlighting
- Table extraction
- Image understanding
- Async ingestion
- GPU acceleration
- Agentic workflows
- Web search integration

---

# 🐛 Troubleshooting

## Ollama Not Running

Start Ollama:

```bash
ollama serve
```

---

## Model Not Found

Pull the model again:

```bash
ollama pull llama3.2
```

---

## ChromaDB Issues

Delete the database folder and rebuild:

```text
database/chroma_db
```

---

# 🙌 Acknowledgements

Built with:

- Streamlit
- ChromaDB
- Ollama
- SentenceTransformers
- Hugging Face
- BM25Okapi

---

# ⭐ Support

If you found this project useful:

- Star the repository
- Fork the project
- Contribute improvements

---

# 👨‍💻 Author

Chukwuemeka Samuel Okpala