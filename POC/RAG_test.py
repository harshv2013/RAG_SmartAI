# semantic_chunker.py
"""
Modular semantic chunking, indexing, retrieval and evaluation module.

Usage:
 - Put this file in your project or paste into a notebook cell.
 - Ensure you have get_embeddings(...) available (or set EMBEDDING_BACKEND to "azure").
 - Configure FAISS availability and DB path as needed.
"""

from __future__ import annotations
import os
import time
import math
import json
import logging
import sqlite3
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import nltk

# try to import faiss (optional)
try:
    import faiss
except Exception:
    faiss = None

# ensure NLTK sentence tokenizer
nltk.download("punkt", quiet=True)

# ---------------------------
# CONFIG (adjust as needed)
# ---------------------------
BATCH_SIZE = int(os.getenv("CHUNKER_BATCH_SIZE", "16"))
TARGET_TOKENS = int(os.getenv("CHUNKER_TARGET_TOKENS", "400"))
OVERLAP_TOKENS = int(os.getenv("CHUNKER_OVERLAP_TOKENS", "60"))
MIN_SENT_TOKENS = int(os.getenv("CHUNKER_MIN_SENT_TOKENS", "2"))
MIN_CHUNK_TOKENS = int(os.getenv("CHUNKER_MIN_CHUNK_TOKENS", "60"))
LOCAL_MERGE_WINDOW = int(os.getenv("CHUNKER_LOCAL_MERGE_WINDOW", "2"))
MERGE_SIM_THRESHOLD = float(os.getenv("CHUNKER_MERGE_SIM_THRESHOLD", "0.78"))
EMBED_DIM_FALLBACK = 1536
SQLITE_PATH = os.getenv("CHUNKER_SQLITE", "chunks_meta.db")
FAISS_INDEX_PATH = os.getenv("CHUNKER_FAISS_PATH", "chunks.index")

# ---------------------------
# Embedding interface
# ---------------------------
# The module will try to import get_embeddings(texts: List[str]) -> np.ndarray from
# a file named azure_openai_openai_sdk.py if present. If you already have get_embeddings
# in your environment, that will be used automatically.
def _load_get_embeddings():
    try:
        # prefer user's module if present
        from azure_openai_openai_sdk import get_embeddings  # type: ignore
        return get_embeddings
    except Exception:
        # fallback: check global name
        try:
            return globals()["get_embeddings"]
        except Exception:
            raise RuntimeError(
                "Embedding function not found. Provide a function `get_embeddings(texts: List[str]) -> np.ndarray` "
                "in azure_openai_openai_sdk.py or define get_embeddings in the environment."
            )

_get_embeddings = None  # lazy load


def get_embedding_vectors(texts: List[str]) -> np.ndarray:
    """
    Wrapper used by this module. Calls user-provided get_embeddings function.
    Returns a numpy array shape (len(texts), dim).
    """
    global _get_embeddings
    if _get_embeddings is None:
        _get_embeddings = _load_get_embeddings()
    embs = _get_embeddings(texts)
    arr = np.asarray(embs, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Embeddings must be 2-D array (N, D)")
    return arr


# ---------------------------
# Token counting
# ---------------------------
def count_tokens(text: str) -> int:
    """
    Token count: uses tiktoken if available otherwise fallbacks to word count.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


# ---------------------------
# Sentence splitting
# ---------------------------
def sentence_split(text: str) -> List[str]:
    return nltk.tokenize.sent_tokenize(text)


# ---------------------------
# Batching helper
# ---------------------------
def batch_call(fn, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Batch-call wrapper: collects outputs and stacks them into np.ndarray."""
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        res = fn(batch)
        out.append(np.asarray(res, dtype=np.float32))
    if not out:
        return np.zeros((0, EMBED_DIM_FALLBACK), dtype=np.float32)
    return np.vstack(out)


# ---------------------------
# Chunking: sentence-accumulation (provenance-friendly)
# ---------------------------
def make_chunks_by_sentence(
    doc_id: str,
    text: str,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    min_sent_tokens: int = MIN_SENT_TOKENS,
) -> List[Dict[str, Any]]:
    """
    Build contiguous chunks by accumulating sentences until target_tokens.
    Returns list of chunk metadata dicts (chunk_id, doc_id, chunk_index, text,
    token_start, token_end, sentence_start, sentence_end, tokens).
    """
    sents = sentence_split(text)
    if not sents:
        return []

    token_counts = [count_tokens(s) for s in sents]
    chunks: List[Dict[str, Any]] = []
    cur_idxs: List[int] = []
    cur_tokens = 0
    idx = 0
    cid = 0
    n = len(sents)

    while idx < n:
        s_tok = token_counts[idx]
        if s_tok < min_sent_tokens:
            idx += 1
            continue

        # if a single sentence is huge, emit it as one chunk
        if s_tok >= target_tokens and not cur_idxs:
            token_start = sum(token_counts[:idx])
            token_end = token_start + s_tok
            chunks.append(
                {
                    "chunk_id": f"{doc_id}::chunk::{cid}",
                    "doc_id": doc_id,
                    "chunk_index": cid,
                    "text": sents[idx],
                    "token_start": token_start,
                    "token_end": token_end,
                    "sentence_start": idx,
                    "sentence_end": idx,
                    "tokens": s_tok,
                }
            )
            cid += 1
            idx += 1
            continue

        # flush if adding current sentence would exceed budget
        if cur_tokens + s_tok > target_tokens and cur_idxs:
            st = cur_idxs[0]
            ed = cur_idxs[-1]
            token_start = sum(token_counts[:st])
            token_end = sum(token_counts[: ed + 1])
            chunk_text = " ".join([sents[j] for j in cur_idxs])
            chunks.append(
                {
                    "chunk_id": f"{doc_id}::chunk::{cid}",
                    "doc_id": doc_id,
                    "chunk_index": cid,
                    "text": chunk_text,
                    "token_start": token_start,
                    "token_end": token_end,
                    "sentence_start": st,
                    "sentence_end": ed,
                    "tokens": cur_tokens,
                }
            )
            cid += 1
            # simple approach: do not carry token overlap across sentence boundaries to avoid provenance mismatch
            cur_idxs = []
            cur_tokens = 0
            continue

        # accumulate
        cur_idxs.append(idx)
        cur_tokens += s_tok
        idx += 1

    # flush leftover
    if cur_idxs:
        st = cur_idxs[0]
        ed = cur_idxs[-1]
        token_start = sum(token_counts[:st])
        token_end = sum(token_counts[: ed + 1])
        chunk_text = " ".join([sents[j] for j in cur_idxs])
        chunks.append(
            {
                "chunk_id": f"{doc_id}::chunk::{cid}",
                "doc_id": doc_id,
                "chunk_index": cid,
                "text": chunk_text,
                "token_start": token_start,
                "token_end": token_end,
                "sentence_start": st,
                "sentence_end": ed,
                "tokens": cur_tokens,
            }
        )

    return chunks


# ---------------------------
# Local semantic merge (only nearby chunks)
# ---------------------------
def local_semantic_merge(
    chunks: List[Dict[str, Any]],
    merge_window: int = LOCAL_MERGE_WINDOW,
    sim_threshold: float = MERGE_SIM_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Compute chunk embeddings in batches and merge neighboring chunks if cosine similarity >= threshold.
    Only merges neighbors within merge_window distance. Returns (new_chunks, embeddings_normalized).
    """
    if not chunks:
        return [], np.zeros((0, EMBED_DIM_FALLBACK), dtype=np.float32)

    texts = [c["text"] for c in chunks]
    emb = batch_call(get_embedding_vectors, texts, batch_size=BATCH_SIZE)  # (m, d)
    # normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_norm = emb / norms
    m = len(chunks)

    used = [False] * m
    merged: List[Dict[str, Any]] = []
    i = 0
    new_idx = 0

    while i < m:
        if used[i]:
            i += 1
            continue

        # look for best neighbor in window
        left = max(0, i - merge_window)
        right = min(m - 1, i + merge_window)
        best_j = None
        best_sim = -1.0
        for j in range(left, right + 1):
            if j == i or used[j]:
                continue
            sim = float(np.dot(emb_norm[i], emb_norm[j]))
            if sim > best_sim:
                best_sim = sim
                best_j = j

        if best_j is not None and best_sim >= sim_threshold and abs(best_j - i) <= merge_window:
            a = min(i, best_j)
            b = max(i, best_j)
            merged_text = chunks[a]["text"] + " " + chunks[b]["text"]
            merged_chunk = {
                "chunk_id": f"{chunks[a]['doc_id']}::chunk::merged::{new_idx}",
                "doc_id": chunks[a]["doc_id"],
                "chunk_index": new_idx,
                "text": merged_text,
                "token_start": min(chunks[a]["token_start"], chunks[b]["token_start"]),
                "token_end": max(chunks[a]["token_end"], chunks[b]["token_end"]),
                "sentence_start": min(chunks[a]["sentence_start"], chunks[b]["sentence_start"]),
                "sentence_end": max(chunks[a]["sentence_end"], chunks[b]["sentence_end"]),
                "tokens": chunks[a]["tokens"] + chunks[b]["tokens"],
            }
            merged.append(merged_chunk)
            used[i] = True
            used[best_j] = True
            new_idx += 1
            i += 1
        else:
            single = dict(chunks[i])
            single["chunk_id"] = f"{single['doc_id']}::chunk::merged::{new_idx}"
            single["chunk_index"] = new_idx
            merged.append(single)
            used[i] = True
            new_idx += 1
            i += 1

    # compute embeddings for merged result
    new_texts = [c["text"] for c in merged]
    if new_texts:
        new_emb = batch_call(get_embedding_vectors, new_texts, batch_size=BATCH_SIZE)
        norms = np.linalg.norm(new_emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        new_emb = new_emb / norms
    else:
        new_emb = np.zeros((0, EMBED_DIM_FALLBACK), dtype=np.float32)

    return merged, new_emb


# ---------------------------
# Merge tiny chunks into neighbor
# ---------------------------
def merge_small_chunks(chunks: List[Dict[str, Any]], min_tokens: int = MIN_CHUNK_TOKENS) -> List[Dict[str, Any]]:
    if not chunks:
        return []
    out: List[Dict[str, Any]] = []
    i = 0
    n = len(chunks)
    new_idx = 0
    while i < n:
        cur = chunks[i]
        if cur.get("tokens", 0) < min_tokens and i + 1 < n:
            nxt = chunks[i + 1]
            merged_text = cur["text"] + " " + nxt["text"]
            merged_chunk = {
                "chunk_id": f"{cur['doc_id']}::chunk::merged::{new_idx}",
                "doc_id": cur["doc_id"],
                "chunk_index": new_idx,
                "text": merged_text,
                "token_start": cur["token_start"],
                "token_end": nxt["token_end"],
                "sentence_start": cur["sentence_start"],
                "sentence_end": nxt["sentence_end"],
                "tokens": cur.get("tokens", 0) + nxt.get("tokens", 0),
            }
            out.append(merged_chunk)
            i += 2
        else:
            single = dict(cur)
            single["chunk_id"] = f"{single['doc_id']}::chunk::merged::{new_idx}"
            single["chunk_index"] = new_idx
            out.append(single)
            i += 1
        new_idx += 1
    return out


# ---------------------------
# Metadata store (SQLite)
# ---------------------------
class MetadataStore:
    def __init__(self, path: str = SQLITE_PATH):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._init_table()

    def _init_table(self):
        cur = self.conn.cursor()
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT UNIQUE,
            doc_id TEXT,
            chunk_index INTEGER,
            token_start INTEGER,
            token_end INTEGER,
            sentence_start INTEGER,
            sentence_end INTEGER,
            text TEXT,
            embedding BLOB,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        )
        self.conn.commit()

    def persist_chunk(self, chunk: Dict[str, Any], embedding: Optional[np.ndarray] = None, model_name: str = "emb"):
        cur = self.conn.cursor()
        emb_blob = None
        if embedding is not None:
            emb_blob = embedding.astype(np.float32).tobytes()
        cur.execute(
            """
            INSERT OR REPLACE INTO chunks
            (chunk_id, doc_id, chunk_index, token_start, token_end, sentence_start, sentence_end, text, embedding, model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk["chunk_id"],
                chunk["doc_id"],
                chunk["chunk_index"],
                chunk["token_start"],
                chunk["token_end"],
                chunk["sentence_start"],
                chunk["sentence_end"],
                chunk["text"],
                emb_blob,
                model_name,
            ),
        )
        self.conn.commit()

    def fetch_chunk_by_chunk_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT chunk_id, doc_id, chunk_index, token_start, token_end, sentence_start, sentence_end, text FROM chunks WHERE chunk_id = ?", (chunk_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "chunk_id": row[0],
            "doc_id": row[1],
            "chunk_index": row[2],
            "token_start": row[3],
            "token_end": row[4],
            "sentence_start": row[5],
            "sentence_end": row[6],
            "text": row[7],
        }

    def fetch_all_chunks(self) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT chunk_id, doc_id, chunk_index, token_start, token_end, sentence_start, sentence_end, text FROM chunks ORDER BY id")
        rows = cur.fetchall()
        out = []
        for row in rows:
            out.append(
                {
                    "chunk_id": row[0],
                    "doc_id": row[1],
                    "chunk_index": row[2],
                    "token_start": row[3],
                    "token_end": row[4],
                    "sentence_start": row[5],
                    "sentence_end": row[6],
                    "text": row[7],
                }
            )
        return out


# ---------------------------
# FAISS index manager (optional)
# ---------------------------
class VectorIndex:
    def __init__(self, dim: int, path: Optional[str] = None, use_hnsw: bool = True):
        self.dim = dim
        self.path = path
        self.use_hnsw = use_hnsw
        self.index = None
        if faiss is None:
            logging.info("faiss not installed; VectorIndex will remain None until a backend is provided.")
        else:
            self._create_index()

    def _create_index(self):
        if self.use_hnsw:
            self.index = faiss.IndexHNSWFlat(self.dim, 32)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
        # make sure index is ready for inner product (we use normalized vectors)
        # no additional setup required here

    def add(self, vectors: np.ndarray):
        if self.index is None:
            raise RuntimeError("FAISS index not initialized")
        self.index.add(vectors.astype(np.float32))

    def save(self, path: Optional[str] = None):
        path = path or self.path
        if not path:
            raise ValueError("path required to save index")
        faiss.write_index(self.index, path)

    def load(self, path: str):
        self.index = faiss.read_index(path)

    def search(self, query_vec: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        query_vec shape (1, dim)
        returns (distances, indices)
        """
        if self.index is None:
            raise RuntimeError("FAISS index not initialized")
        return self.index.search(query_vec.astype(np.float32), k)


# ---------------------------
# Pipeline: create chunks, merge, embed, persist, index
# ---------------------------
def process_document(
    doc_id: str,
    text: str,
    metadata_store: Optional[MetadataStore] = None,
    vector_index: Optional[VectorIndex] = None,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    do_merge: bool = True,
    merge_threshold: float = MERGE_SIM_THRESHOLD,
    persist: bool = True,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    High-level pipeline:
      1) sentence-accumulation chunking
      2) local semantic merge (optional)
      3) merge small chunks
      4) compute chunk embeddings
      5) persist metadata and index vectors
    Returns (final_chunks, chunk_embeddings_normalized)
    """
    # 1) initial chunks
    initial = make_chunks_by_sentence(doc_id, text, target_tokens=target_tokens, overlap_tokens=overlap_tokens)

    if not initial:
        return [], np.zeros((0, EMBED_DIM_FALLBACK), dtype=np.float32)

    # 2) local merge
    if do_merge:
        merged, _ = local_semantic_merge(initial, merge_window=LOCAL_MERGE_WINDOW, sim_threshold=merge_threshold)
    else:
        merged = initial

    # 3) merge tiny chunks
    final = merge_small_chunks(merged, min_tokens=MIN_CHUNK_TOKENS)

    # 4) compute embeddings
    texts = [c["text"] for c in final]
    emb = batch_call(get_embedding_vectors, texts, batch_size=BATCH_SIZE)
    # normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms

    # 5) persist
    if persist and metadata_store is not None:
        for c, e in zip(final, emb):
            metadata_store.persist_chunk(c, e)

    # 6) index
    if vector_index is not None and emb.shape[0] > 0:
        vector_index.add(emb)

    return final, emb


# ---------------------------
# Retrieval helpers
# ---------------------------
def retrieve_top_k_in_memory(chunk_embs: np.ndarray, chunks: List[Dict[str, Any]], query: str, k: int = 5) -> List[Dict[str, Any]]:
    q = batch_call(get_embedding_vectors, [query], batch_size=1)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    sims = (chunk_embs @ q.T).reshape(-1)
    idxs = np.argsort(sims)[::-1][:k]
    return [{"chunk_id": chunks[i]["chunk_id"], "score": float(sims[i]), "text": chunks[i]["text"]} for i in idxs]


def retrieve_top_k_faiss(index: VectorIndex, chunks: List[Dict[str, Any]], query: str, k: int = 5) -> List[Dict[str, Any]]:
    q = batch_call(get_embedding_vectors, [query], batch_size=1)
    # index expects normalized vectors if built that way
    # ensure q is normalized
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    D, I = index.search(q, k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        c = chunks[idx]
        results.append({"chunk_id": c["chunk_id"], "score": float(dist), "text": c["text"]})
    return results


# ---------------------------
# Evaluation helpers
# ---------------------------
def evaluate_retrieval(
    test_queries: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    chunk_embs: np.ndarray,
    k_list: List[int] = [1, 3, 5],
) -> Dict[str, Any]:
    """
    test_queries: list of {"query": str, "relevant_chunk_ids": [id1,id2,...]}
    """
    import numpy as _np

    results = {k: {"recall": 0, "precision": 0} for k in k_list}
    rr_sum = 0.0
    lat = []
    n = len(test_queries)
    for tq in test_queries:
        q = tq["query"]
        rel = set(tq.get("relevant_chunk_ids", []))
        t0 = time.time()
        retrieved = retrieve_top_k_in_memory(chunk_embs, chunks, q, k=max(k_list))
        t1 = time.time()
        lat.append(t1 - t0)
        retrieved_ids = [r["chunk_id"] for r in retrieved]
        for k in k_list:
            topk = retrieved_ids[:k]
            hits = len(set(topk) & rel)
            if hits > 0:
                results[k]["recall"] += 1
            results[k]["precision"] += hits / float(k)
        rr = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in rel:
                rr = 1.0 / rank
                break
        rr_sum += rr

    out = {"n_queries": n, "avg_latency": float(np.mean(lat)) if lat else None}
    for k in k_list:
        out[f"recall@{k}"] = results[k]["recall"] / n if n else None
        out[f"precision@{k}"] = results[k]["precision"] / n if n else None
    out["MRR"] = rr_sum / n if n else None
    return out


# ---------------------------
# Example command-line test (if run as script)
# ---------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Basic smoke test - expects azure_openai_openai_sdk.get_embeddings present or similar
    try:
        # load sample text file if available
        sample_path = "sample_long_text.txt"
        if not os.path.exists(sample_path):
            print("Place a sample_long_text.txt file in current dir to run a smoke ingest.")
            exit(0)
        text = open(sample_path, "r", encoding="utf-8").read()
        doc_id = "doc_test"

        # init stores
        meta = MetadataStore(SQLITE_PATH)
        vec_idx = VectorIndex(dim=EMBED_DIM_FALLBACK, path=FAISS_INDEX_PATH) if faiss is not None else None

        # process document
        chunks, emb = process_document(doc_id, text, metadata_store=meta, vector_index=vec_idx, persist=True)
        print("Created chunks:", len(chunks))
        if emb is not None and emb.shape[0] > 0:
            print("Embedding matrix shape:", emb.shape)

        # quick interactive query
        while True:
            q = input("Query (empty to exit)> ").strip()
            if not q:
                break
            if vec_idx and vec_idx.index is not None:
                res = retrieve_top_k_faiss(vec_idx, chunks, q, k=5)
            else:
                res = retrieve_top_k_in_memory(emb, chunks, q, k=5)
            for r in res:
                print(r["score"], r["chunk_id"])
                print(r["text"][:300].replace("\n", " "))
                print("-" * 60)

    except Exception as e:
        logging.exception("Smoke test failed: %s", e)
