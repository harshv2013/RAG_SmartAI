from typing import List, Dict, Tuple, Any
import numpy as np
from .utils import sentence_split, count_tokens
from .embeddings import get_embeddings_safe
from os import environ

# config from env or defaults
BATCH_SIZE = int(environ.get("CHUNKER_BATCH_SIZE", "16"))
TARGET_TOKENS = int(environ.get("CHUNKER_TARGET_TOKENS", "400"))
MIN_SENT_TOKENS = int(environ.get("CHUNKER_MIN_SENT_TOKENS", "2"))
MIN_CHUNK_TOKENS = int(environ.get("CHUNKER_MIN_CHUNK_TOKENS", "60"))
LOCAL_MERGE_WINDOW = int(environ.get("CHUNKER_LOCAL_MERGE_WINDOW", "2"))
MERGE_SIM_THRESHOLD = float(environ.get("CHUNKER_MERGE_SIM_THRESHOLD", "0.78"))
EMBED_DIM_FALLBACK = int(environ.get("EMBED_DIM_FALLBACK", "1536"))

def make_chunks_by_sentence(doc_id: str, text: str, target_tokens: int = TARGET_TOKENS) -> List[Dict[str, Any]]:
    sents = sentence_split(text)
    if not sents:
        return []
    token_counts = [count_tokens(s) for s in sents]
    chunks = []
    cur_idxs = []
    cur_tokens = 0
    idx = 0
    cid = 0
    n = len(sents)
    while idx < n:
        s_tok = token_counts[idx]
        if s_tok < MIN_SENT_TOKENS:
            idx += 1
            continue
        if s_tok >= target_tokens and not cur_idxs:
            token_start = sum(token_counts[:idx])
            token_end = token_start + s_tok
            chunks.append({"chunk_id": f"{doc_id}::chunk::{cid}", "doc_id": doc_id, "chunk_index": cid,
                           "text": sents[idx], "token_start": token_start, "token_end": token_end,
                           "sentence_start": idx, "sentence_end": idx, "tokens": s_tok})
            cid += 1
            idx += 1
            continue
        if cur_tokens + s_tok > target_tokens and cur_idxs:
            st = cur_idxs[0]; ed = cur_idxs[-1]
            token_start = sum(token_counts[:st]); token_end = sum(token_counts[:ed+1])
            chunk_text = " ".join([sents[j] for j in cur_idxs])
            chunks.append({"chunk_id": f"{doc_id}::chunk::{cid}", "doc_id": doc_id, "chunk_index": cid,
                           "text": chunk_text, "token_start": token_start, "token_end": token_end,
                           "sentence_start": st, "sentence_end": ed, "tokens": cur_tokens})
            cid += 1
            cur_idxs = []; cur_tokens = 0
            continue
        cur_idxs.append(idx); cur_tokens += s_tok; idx += 1
    if cur_idxs:
        st = cur_idxs[0]; ed = cur_idxs[-1]
        token_start = sum(token_counts[:st]); token_end = sum(token_counts[:ed+1])
        chunk_text = " ".join([sents[j] for j in cur_idxs])
        chunks.append({"chunk_id": f"{doc_id}::chunk::{cid}", "doc_id": doc_id, "chunk_index": cid,
                       "text": chunk_text, "token_start": token_start, "token_end": token_end,
                       "sentence_start": st, "sentence_end": ed, "tokens": cur_tokens})
    return chunks

def _batch_embeddings(texts: List[str], batch_size: int = BATCH_SIZE):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        out.append(get_embeddings_safe(batch))
    if not out:
        return np.zeros((0, EMBED_DIM_FALLBACK), dtype=np.float32)
    return np.vstack(out)

def local_semantic_merge(chunks: List[Dict[str, Any]], merge_window: int = LOCAL_MERGE_WINDOW, sim_threshold: float = MERGE_SIM_THRESHOLD) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    if not chunks:
        return [], np.zeros((0, EMBED_DIM_FALLBACK), dtype=np.float32)
    texts = [c["text"] for c in chunks]
    emb = _batch_embeddings(texts)
    norms = np.linalg.norm(emb, axis=1, keepdims=True); norms[norms == 0] = 1.0
    emb_norm = emb / norms
    m = len(chunks)
    used = [False]*m
    merged = []
    i = 0; new_idx = 0
    while i < m:
        if used[i]:
            i += 1; continue
        left = max(0, i-merge_window); right = min(m-1, i+merge_window)
        best_j = None; best_sim = -1.0
        for j in range(left, right+1):
            if j==i or used[j]: continue
            sim = float(np.dot(emb_norm[i], emb_norm[j]))
            if sim > best_sim:
                best_sim = sim; best_j = j
        if best_j is not None and best_sim >= sim_threshold and abs(best_j - i) <= merge_window:
            a = min(i, best_j); b = max(i, best_j)
            merged_text = chunks[a]["text"] + " " + chunks[b]["text"]
            merged_chunk = {"chunk_id": f"{chunks[a]['doc_id']}::chunk::merged::{new_idx}", "doc_id": chunks[a]["doc_id"],
                            "chunk_index": new_idx, "text": merged_text,
                            "token_start": min(chunks[a]["token_start"], chunks[b]["token_start"]),
                            "token_end": max(chunks[a]["token_end"], chunks[b]["token_end"]),
                            "sentence_start": min(chunks[a]["sentence_start"], chunks[b]["sentence_start"]),
                            "sentence_end": max(chunks[a]["sentence_end"], chunks[b]["sentence_end"]),
                            "tokens": chunks[a]["tokens"] + chunks[b]["tokens"]}
            merged.append(merged_chunk)
            used[i]=True; used[best_j]=True; new_idx+=1; i+=1
        else:
            single = dict(chunks[i])
            single["chunk_id"] = f"{single['doc_id']}::chunk::merged::{new_idx}"
            single["chunk_index"] = new_idx
            merged.append(single)
            used[i]=True; new_idx+=1; i+=1
    new_texts = [c["text"] for c in merged]
    if new_texts:
        new_emb = _batch_embeddings(new_texts)
        norms = np.linalg.norm(new_emb, axis=1, keepdims=True); norms[norms == 0] = 1.0
        new_emb = new_emb / norms
    else:
        new_emb = np.zeros((0, EMBED_DIM_FALLBACK), dtype=np.float32)
    return merged, new_emb

def merge_small_chunks(chunks: List[Dict[str, Any]], min_tokens: int = MIN_CHUNK_TOKENS) -> List[Dict[str, Any]]:
    if not chunks:
        return []
    out=[]; i=0; n=len(chunks); new_idx=0
    while i < n:
        cur = chunks[i]
        if cur.get("tokens",0) < min_tokens and i+1 < n:
            nxt = chunks[i+1]
            merged_text = cur["text"] + " " + nxt["text"]
            merged_chunk = {"chunk_id": f"{cur['doc_id']}::chunk::merged::{new_idx}", "doc_id": cur["doc_id"],
                            "chunk_index": new_idx, "text": merged_text, "token_start": cur["token_start"],
                            "token_end": nxt["token_end"], "sentence_start": cur["sentence_start"],
                            "sentence_end": nxt["sentence_end"], "tokens": cur.get("tokens",0) + nxt.get("tokens",0)}
            out.append(merged_chunk); i += 2
        else:
            single = dict(cur); single["chunk_id"] = f"{single['doc_id']}::chunk::merged::{new_idx}"
            single["chunk_index"] = new_idx; out.append(single); i += 1
        new_idx += 1
    return out
