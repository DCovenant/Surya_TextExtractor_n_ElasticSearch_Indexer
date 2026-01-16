# RAG Document Processing Pipeline

A production-ready document extraction and indexing pipeline for Retrieval-Augmented Generation (RAG) systems. Built with Surya OCR for high-accuracy text extraction and Elasticsearch for hybrid search capabilities.

## Overview

This pipeline processes PDF documents through two main stages:

1. **Document Extraction** (`text_parser_surya.py`) - Extracts structured content from PDFs using Surya OCR with layout analysis, table recognition, and optimized batch processing
2. **Embedding & Indexing** (`embeddings_indexer.py`) - Chunks documents intelligently and indexes them with dense vector embeddings for semantic search

## Features

### Document Extraction
- **Advanced OCR** - Leverages Surya's foundation models for accurate text recognition
- **Layout Analysis** - Preserves document structure (headers, paragraphs, tables, etc.)
- **Table Recognition** - Extracts tables with cell structure and converts to markdown
- **Smart Filtering** - Removes headers, footers, and metadata tables automatically
- **Batch Processing** - GPU-optimized with memory management for large-scale processing
- **Resume Support** - Continue interrupted processing jobs

### Embedding & Indexing
- **Dual Chunking Strategies**:
  - **Recursive Chunking** - Fast, deterministic text splitting with overlap
  - **Semantic Chunking** - AI-powered breakpoint detection for coherent chunks
- **Table Preservation** - Tables indexed as complete units
- **Section Awareness** - Maintains document hierarchy and context
- **Dense Vector Search** - sentence-transformers embeddings for semantic retrieval
- **Elasticsearch Integration** - Enables hybrid BM25 + kNN search
- **Debug Mode** - Export chunks as JSON for inspection

## Architecture

```
PDF Documents
    ↓
[text_parser_surya.py]
    ├─ PDF → Images (2048px)
    ├─ Layout Detection
    ├─ OCR Recognition
    ├─ Table Extraction
    └─ Structure Preservation
    ↓
Structured JSON
    ↓
[embeddings_indexer.py]
    ├─ Semantic/Recursive Chunking
    ├─ Embedding Generation (all-mpnet-base-v2)
    └─ Elasticsearch Indexing
    ↓
Searchable Knowledge Base
```

## Requirements

### System Requirements
- CUDA-capable GPU (required for Surya OCR)
- Elasticsearch 7.x or 8.x running on `localhost:9200`
- Python 3.8+

### Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install surya-ocr pdf2image Pillow
pip install sentence-transformers elasticsearch
pip install langchain langchain-huggingface langchain-experimental langchain-text-splitters
pip install orjson tqdm
```

**System packages** (Ubuntu/Debian):
```bash
sudo apt-get install poppler-utils  # For pdf2image
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-document-pipeline.git
cd rag-document-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start Elasticsearch:
```bash
docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.11.0
```

## Usage

### Step 1: Extract Documents

Basic extraction:
```bash
python3 text_parser_surya.py /path/to/pdfs -o documents.json
```

Resume interrupted job:
```bash
python3 text_parser_surya.py /path/to/pdfs -o documents.json --resume
```

**Output**: JSON file with structured document content including:
- Text elements with bounding boxes
- Section headers
- Tables in structured format
- Page numbers and positions

### Step 2: Index Documents

**Recursive chunking** (faster, deterministic):
```bash
python embeddings_indexer.py documents.json -i rag_index
```

**Semantic chunking** (better coherence):
```bash
python embeddings_indexer.py documents.json -i rag_index --semantic
```

**Debug mode** (export chunks without indexing):
```bash
python embeddings_indexer.py documents.json --json chunks.json
```

**Reset index** (delete and recreate):
```bash
python embeddings_indexer.py documents.json -i rag_index --reset
```

## Configuration

### text_parser_surya.py

```python
# Batch sizes (adjust based on GPU memory)
DETECTOR_BATCH_SIZE = 18
RECOGNITION_BATCH_SIZE = 128
LAYOUT_BATCH_SIZE = 20
TABLE_REC_BATCH_SIZE = 64

# Processing batches
PAGE_BATCH = 4    # Pages per batch
PDF_BATCH = 10    # PDFs before checkpoint save
```

### embeddings_indexer.py

```python
# Chunking parameters
CHUNK_SIZE = 512              # Characters per chunk (recursive)
BREAKPOINT_THRESHOLD = 75     # Percentile for semantic chunking

# Model
MODEL = "all-mpnet-base-v2"  # Embedding model (768 dims)
```

## Output Format

### Document Structure (text_parser_surya.py)
```json
{
  "file": "folder/document.pdf",
  "pages": [
    {
      "page_number": 1,
      "elements": [
        {
          "type": "SectionHeader",
          "text": "Introduction",
          "bbox": [x0, y0, x1, y1],
          "pos": 0
        },
        {
          "type": "Table",
          "info": {
            "rows": 3,
            "cols": 4,
            "cells": [
              {
                "row": 0,
                "col": 0,
                "text": "Header",
                "rowspan": 1,
                "colspan": 1,
                "is_header": true
              }
            ]
          },
          "bbox": [x0, y0, x1, y1],
          "pos": 1
        }
      ]
    }
  ]
}
```

### Indexed Chunks (Elasticsearch)
```json
{
  "chunk_id": "md5_hash",
  "file_name": "folder/document.pdf",
  "page_number": 1,
  "content_type": "text",
  "section": "Introduction",
  "chunk_text": "The content of the chunk...",
  "embedding": [0.123, -0.456, ...]
}
```

## Performance

- **Extraction**: ~15-30 seconds per page (GPU-dependent)
- **Indexing**: ~1000 chunks/minute (GPU embeddings)
- **Memory**: ~4-8GB GPU RAM for extraction, ~2-4GB for indexing

## Troubleshooting

### CUDA Out of Memory
Reduce batch sizes in `text_parser_surya.py`:
```python
os.environ["DETECTOR_BATCH_SIZE"] = "10"
os.environ["RECOGNITION_BATCH_SIZE"] = "64"
```

### Elasticsearch Connection Failed
```bash
# Check if Elasticsearch is running
curl http://localhost:9200

# Start Elasticsearch
docker start elasticsearch
```

### Poor Table Recognition
Tables with complex layouts may require manual post-processing. Check the debug output:
```bash
python embeddings_indexer.py documents.json --json debug_chunks.json
```

## Advanced Usage

### Custom Chunking Strategy

Modify `chunk_recursive()` in `embeddings_indexer.py`:
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,           # Larger chunks
    chunk_overlap=128,         # More overlap
    separators=["\n\n", "\n", ". ", " "]
)
```

### Hybrid Search Example

```python
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

# Dense vector search
knn_results = es.search(
    index="rag_index",
    knn={
        "field": "embedding",
        "query_vector": query_embedding,
        "k": 10,
        "num_candidates": 100
    }
)

# BM25 text search
bm25_results = es.search(
    index="rag_index",
    query={"match": {"chunk_text": query_text}}
)
```

## Roadmap

- [ ] Add support for images and diagrams extraction
- [ ] Implement cross-encoder reranking
- [ ] Add evaluation metrics and benchmarks
- [ ] Support for multi-language documents
- [ ] Integration with LangGraph for production workflows

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## License

MIT License - see LICENSE file for details

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{rag_document_pipeline,
  author = {Your Name},
  title = {RAG Document Processing Pipeline},
  year = {2025},
  url = {https://github.com/yourusername/rag-document-pipeline}
}
```

## Acknowledgments

- [Surya OCR](https://github.com/VikParuchuri/surya) - State-of-the-art multilingual OCR
- [sentence-transformers](https://www.sbert.net/) - Dense embedding models
- [LangChain](https://www.langchain.com/) - Text splitting and RAG utilities
