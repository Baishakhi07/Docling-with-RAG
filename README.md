
# Building a Local RAG-Powered Q&A System with Docling
Docling is a robust open-source tool for parsing and converting diverse document formats into structured, AI-friendly representations. In this project, we extend Docling into a fully offline Retrieval Augmented Generation (RAG) pipeline that enables semantic search and real-time Q&A over documents—all without calling external APIs.

Designed to run on standard hardware, this system is modular, lightweight, and ideal for building private knowledge assistants or document retrieval tools.

**Key Features**
Universal Document Processing: Ingest PDFs, HTML pages, full websites (via sitemap), and convert them to structured chunks.

Smart Chunking: Leveraging Docling’s HybridChunker to split document content into meaningful, context-aware chunks tailored to embedding models.

Local Embeddings: Generate 384-dimensional embeddings using the HuggingFace all-MiniLM-L6-v2 model.

Semantic Vector Store: Store embeddings with rich metadata in a local LanceDB vector database.

Interactive Q&A App: Powered by Streamlit and Ollama (local mistral model), your own document-based chatbot runs entirely offline.

Privacy-First Design: All processing and inference happen on-device—no external cloud dependencies.

**Repository Structure**
```bash
├── 1-extraction.py      # Converts PDFs/HTML into structured documents
├── 2-chunking.py        # (Optional) chunk documents into smaller segments
├── 3-embedding.py       # Generates embeddings and stores them in LanceDB
├── 4-search.py          # Tests vector search for semantic retrieval
├── 5-chat.py            # Streamlit chatbot interface using testimonial context
├── utils/
│   └── sitemap.py       # Sitemap scraper to collect URLs for ingestion
├── data/
│   └── lancedb/         # Local LanceDB storage for vector index
└── README.md            # This documentation
```
**Getting Started**

Prerequisites
Python 3.9+

**Install dependencies:**

```bash
pip install -r requirements.txt
```
Install and configure Ollama to run the mistral model locally.

**Usage Workflow**

Extract content from documents:

```bash
python 1-extraction.py
```
Customize chunking logic (if needed):

```bash
python 2-chunking.py
```
Embed and store chunked text:

```bash
python 3-embedding.py
```
Test semantic search manually:

```bash
python 4-search.py
```
Launch the chat-based Q&A interface:

```bash
streamlit run 5-chat.py
```
Then navigate to http://localhost:8501 in your browser to start querying your documents.

**Why It Works for RAG**
This pipeline intelligently segments documents into chunks that respect document hierarchy and semantic boundaries (headers, tables, lists). That preserves context for more accurate embedding and retrieval—critical for RAG performance. Leveraging Docling’s rich metadata and layout-aware conversions ensures the system can answer questions that rely on real document structure.

**How It Works**
Extraction with DocumentConverter handles complex layout and formats.

Chunking uses Docling’s HybridChunker to create clean, meaningful text segments optimized for embedding.

Embedding & Vector Storage: Converted chunks are embedded and stored with metadata (filename, page numbers, title) in LanceDB.

Q&A Interface: The Streamlit app retrieves the top relevant chunks, presents them as context, and uses the local LLM to answer—transparent and private.

**Example Use Cases**
  1. Internal knowledge assistant for company documents

  2. Research assistant for technical papers or academic PDFs

  3. Offline Q&A over manuals, SOPs, or internal documentation












