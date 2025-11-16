import sqlite3
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

class MetadataStore:
    def __init__(self, path: str = "chunks_meta.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._init_tables()
        try:
            cur = self.conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass

    def _init_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (doc_id TEXT PRIMARY KEY, doc_hash TEXT UNIQUE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
        """)
        cur.execute("""
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
          embedding_model TEXT,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS faiss_map (faiss_idx INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE)
        """)
        self.conn.commit()

    def document_exists(self, doc_hash: str) -> Optional[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT doc_id FROM documents WHERE doc_hash = ?", (doc_hash,))
        row = cur.fetchone()
        return row[0] if row else None

    def insert_document(self, doc_id: str, doc_hash: str):
        cur = self.conn.cursor()
        cur.execute("INSERT OR REPLACE INTO documents(doc_id, doc_hash) VALUES(?,?)", (doc_id, doc_hash))
        self.conn.commit()

    def persist_chunk(self, chunk: Dict[str, Any], embedding: Optional[np.ndarray] = None, model_name: str = "emb"):
        cur = self.conn.cursor()
        emb_blob = None
        if embedding is not None:
            emb_blob = embedding.astype(np.float32).tobytes()
        cur.execute("""INSERT OR REPLACE INTO chunks
            (chunk_id, doc_id, chunk_index, token_start, token_end, sentence_start, sentence_end, text, embedding, embedding_model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                chunk["chunk_id"], chunk["doc_id"], chunk["chunk_index"],
                chunk["token_start"], chunk["token_end"], chunk["sentence_start"], chunk["sentence_end"],
                chunk["text"], emb_blob, model_name
            ))
        self.conn.commit()

    def insert_faiss_map(self, faiss_idx: int, chunk_id: str):
        cur = self.conn.cursor()
        cur.execute("INSERT OR REPLACE INTO faiss_map(faiss_idx, chunk_id) VALUES (?, ?)", (faiss_idx, chunk_id))
        self.conn.commit()

    def fetch_all_chunks(self) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT chunk_id, doc_id, chunk_index, token_start, token_end, sentence_start, sentence_end, text, embedding FROM chunks ORDER BY id")
        rows = cur.fetchall()
        out=[]
        for r in rows:
            out.append({"chunk_id": r[0], "doc_id": r[1], "chunk_index": r[2], "token_start": r[3], "token_end": r[4], "sentence_start": r[5], "sentence_end": r[6], "text": r[7], "embedding": r[8]})
        return out

    def fetch_embeddings_and_ids(self) -> Tuple[List[str], List[np.ndarray]]:
        cur = self.conn.cursor()
        cur.execute("SELECT chunk_id, embedding FROM chunks ORDER BY id")
        rows = cur.fetchall()
        ids=[]; vecs=[]
        for cid, emb_blob in rows:
            if emb_blob is None:
                continue
            v = np.frombuffer(emb_blob, dtype=np.float32)
            ids.append(cid); vecs.append(v)
        return ids, vecs
