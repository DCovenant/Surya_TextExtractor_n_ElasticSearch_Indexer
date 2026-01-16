#!/usr/bin/env python3
"""
RAG Embeddings Indexer with Semantic Chunking.

Usage:
  python embeddings_indexer.py output.json -i my_index
  python embeddings_indexer.py output.json -i my_index --semantic  # Semantic chunking
  python embeddings_indexer.py output.json --json chunks.json     # Debug output
"""
import argparse
import hashlib
import orjson
from typing import List, Dict
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

CHUNK_SIZE = 512
BREAKPOINT_THRESHOLD = 75


def table_to_markdown(info: dict) -> str:
    if not info or not info.get("cells"):
        return ""
    rows, cols = info["rows"], info["cols"]
    grid = [[""] * cols for _ in range(rows)]
    for cell in info["cells"]:
        r, c = cell["row"], cell["col"]
        if r < rows and c < cols:
            grid[r][c] = cell.get("text", "").strip()
    lines = []
    for i, row in enumerate(grid):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            lines.append("| " + " | ".join(["---"] * cols) + " |")
    return "\n".join(lines)


def flatten_document(doc: dict) -> List[Dict]:
    segments = []
    current_section = None
    for page in doc["pages"]:
        for elem in page["elements"]:
            if elem["type"] == "SectionHeader":
                current_section = elem.get("text", "").replace("<b>", "").replace("</b>", "")
            text = table_to_markdown(elem.get("info")) if elem["type"] == "Table" else elem.get("text", "")
            if text and text.strip():
                segments.append({"text": text.strip(), "page": page["page_number"],
                                 "type": elem["type"], "section": current_section, "pos": elem["pos"]})
    return segments


def chunk_recursive(doc: dict, splitter) -> List[Dict]:
    chunks = []
    for seg in flatten_document(doc):
        if seg["type"] == "Table":
            chunks.append({"chunk_id": hashlib.md5(f"{doc['file']}:{seg['page']}:{seg['pos']}".encode()).hexdigest(),
                           "file_name": doc["file"], "page_number": seg["page"], "content_type": "table",
                           "section": seg["section"], "chunk_text": seg["text"]})
        else:
            for i, t in enumerate(splitter.split_text(seg["text"])):
                chunks.append({"chunk_id": hashlib.md5(f"{doc['file']}:{seg['page']}:{seg['pos']}:{i}".encode()).hexdigest(),
                               "file_name": doc["file"], "page_number": seg["page"],
                               "content_type": seg["type"].lower(), "section": seg["section"], "chunk_text": t})
    return chunks


def chunk_semantic(doc: dict, chunker) -> List[Dict]:
    chunks = []
    text_buffer, buffer_meta = [], None
    
    def flush():
        nonlocal text_buffer, buffer_meta
        if not text_buffer:
            return
        combined = " ".join(text_buffer)
        if len(combined) > CHUNK_SIZE:
            for i, d in enumerate(chunker.create_documents([combined])):
                chunks.append({"chunk_id": hashlib.md5(f"{doc['file']}:{buffer_meta['page']}:{buffer_meta['pos']}:{i}".encode()).hexdigest(),
                               "file_name": doc["file"], "page_number": buffer_meta["page"],
                               "content_type": buffer_meta["type"].lower(), "section": buffer_meta["section"],
                               "chunk_text": d.page_content})
        else:
            chunks.append({"chunk_id": hashlib.md5(f"{doc['file']}:{buffer_meta['page']}:{buffer_meta['pos']}".encode()).hexdigest(),
                           "file_name": doc["file"], "page_number": buffer_meta["page"],
                           "content_type": buffer_meta["type"].lower(), "section": buffer_meta["section"],
                           "chunk_text": combined})
        text_buffer, buffer_meta = [], None
    
    for seg in flatten_document(doc):
        if seg["type"] == "Table":
            flush()
            chunks.append({"chunk_id": hashlib.md5(f"{doc['file']}:{seg['page']}:{seg['pos']}".encode()).hexdigest(),
                           "file_name": doc["file"], "page_number": seg["page"], "content_type": "table",
                           "section": seg["section"], "chunk_text": seg["text"]})
        elif seg["type"] == "SectionHeader":
            flush()
            buffer_meta, text_buffer = seg, [seg["text"]]
        else:
            if buffer_meta is None:
                buffer_meta = seg
            text_buffer.append(seg["text"])
    flush()
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--index', '-i', default='rag_documents')
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('--json', '-j', metavar='FILE')
    parser.add_argument('--semantic', '-s', action='store_true')
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        documents = orjson.loads(f.read())

    if args.semantic:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_experimental.text_splitter import SemanticChunker
        embed = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2", model_kwargs={"device": "cuda"})
        chunker = SemanticChunker(embed, breakpoint_threshold_type="percentile",
                                   breakpoint_threshold_amount=BREAKPOINT_THRESHOLD)
        chunk_fn = lambda doc: chunk_semantic(doc, chunker)
    else:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=64,
                                                   separators=["\n\n", "\n", ". ", " ", ""])
        chunk_fn = lambda doc: chunk_recursive(doc, splitter)

    all_chunks = []
    for doc in tqdm(documents, desc="Chunking"):
        all_chunks.extend(chunk_fn(doc))
    print(f"Generated {len(all_chunks)} chunks")

    if args.json:
        with open(args.json, 'wb') as f:
            f.write(orjson.dumps(all_chunks, option=orjson.OPT_INDENT_2))
        print(f"Saved to {args.json}")
        return

    model = SentenceTransformer("all-mpnet-base-v2")
    if torch.cuda.is_available():
        model = model.to("cuda").half()
    dims = model.get_sentence_embedding_dimension()

    embeddings = model.encode([c["chunk_text"] for c in all_chunks], batch_size=32,
                               show_progress_bar=True, normalize_embeddings=True)
    for chunk, emb in zip(all_chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    es = Elasticsearch("http://localhost:9200")
    if not es.ping():
        print("ES unavailable")
        return

    if args.reset and es.indices.exists(index=args.index):
        es.indices.delete(index=args.index)

    if not es.indices.exists(index=args.index):
        es.indices.create(index=args.index, body={"mappings": {"properties": {
            "chunk_id": {"type": "keyword"}, "file_name": {"type": "keyword"},
            "page_number": {"type": "integer"}, "content_type": {"type": "keyword"},
            "section": {"type": "keyword"}, "chunk_text": {"type": "text"},
            "embedding": {"type": "dense_vector", "dims": dims, "index": True, "similarity": "cosine"}
        }}})

    success, _ = bulk(es, [{"_index": args.index, "_id": c["chunk_id"], "_source": c} for c in all_chunks], chunk_size=100)
    print(f"Indexed {success} chunks")


if __name__ == '__main__':
    main()