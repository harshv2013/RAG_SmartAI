from typing import Tuple, List, Dict, Any
import numpy as np
from .utils import sha256_hex
from .chunker import make_chunks_by_sentence, local_semantic_merge, merge_small_chunks
from .embeddings import get_embeddings_safe
from .store import MetadataStore
from .vector_index import VectorIndex
from os import environ

BATCH_SIZE = int(environ.get("CHUNKER_BATCH_SIZE", "16"))

def _batch_call(fn, texts: List[str], batch_size: int = BATCH_SIZE):
    out = []
    for i in range(0, len(texts), batch_size):
        out.append(fn(texts[i:i+batch_size]))
    return np.vstack(out) if out else np.zeros((0, 1536), dtype=np.float32)

def process_document(doc_id: str, text: str, metadata_store: MetadataStore, vector_index: VectorIndex = None, do_merge: bool = True, persist: bool = True) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    doc_hash = sha256_hex(text)
    existing = metadata_store.document_exists(doc_hash)
    if existing:
        # return stored chunks + embeddings (if available)
        all_chunks = metadata_store.fetch_all_chunks()
        chunks = [c for c in all_chunks if c["doc_id"] == existing]
        embs=[]
        for c in chunks:
            cur = metadata_store.conn.cursor()
            cur.execute("SELECT embedding FROM chunks WHERE chunk_id = ?", (c["chunk_id"],))
            r = cur.fetchone()
            if r and r[0]:
                embs.append(np.frombuffer(r[0], dtype=np.float32))
        if embs:
            mat = np.vstack(embs)
            norms = np.linalg.norm(mat, axis=1, keepdims=True); norms[norms==0]=1.0
            return chunks, mat / norms
        return chunks, np.zeros((0,1536), dtype=np.float32)

    # 1 initial chunks
    initial = make_chunks_by_sentence(doc_id, text)
    if not initial:
        return [], np.zeros((0,1536), dtype=np.float32)

    # 2 local merge
    if do_merge:
        merged, _ = local_semantic_merge(initial)
    else:
        merged = initial

    # 3 merge small
    final = merge_small_chunks(merged)

    # 4 embeddings
    texts = [c["text"] for c in final]
    emb = _batch_call(get_embeddings_safe, texts)
    norms = np.linalg.norm(emb, axis=1, keepdims=True); norms[norms==0]=1.0
    emb = emb / norms

    # persist document record
    metadata_store.insert_document(doc_id, doc_hash)

    # store chunks + embeddings
    if persist:
        for c, e in zip(final, emb):
            metadata_store.persist_chunk(c, e)
    # index
    if vector_index is not None and emb.shape[0] > 0:
        vector_index.add_with_mapping(emb, [c["chunk_id"] for c in final], metadata_store)
    return final, emb
