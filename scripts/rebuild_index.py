# # scripts/rebuild_faiss_direct.py
# import os, sqlite3, numpy as np
# try:
#     import faiss
# except Exception:
#     raise SystemExit("faiss not installed")

# DB = os.getenv("DB_PATH", "chunks_meta.db")
# FAISS_PATH = os.getenv("FAISS_PATH", "faiss.index")
# DIM = int(os.getenv("EMBED_DIM_FALLBACK", "1536"))

# def fetch(db):
#     conn = sqlite3.connect(db); cur = conn.cursor()
#     cur.execute("SELECT chunk_id, embedding FROM chunks WHERE embedding IS NOT NULL AND LENGTH(embedding)>0 ORDER BY id")
#     rows = cur.fetchall(); conn.close()
#     chunk_ids = [r[0] for r in rows]
#     vecs = [np.frombuffer(r[1], dtype=np.float32) for r in rows]
#     return chunk_ids, vecs

# def write_map(db, id_map):
#     conn = sqlite3.connect(db); cur = conn.cursor()
#     cur.execute("CREATE TABLE IF NOT EXISTS faiss_map (faiss_idx INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE)")
#     cur.execute("DELETE FROM faiss_map")
#     cur.executemany("INSERT INTO faiss_map (faiss_idx, chunk_id) VALUES (?, ?)", id_map)
#     conn.commit(); conn.close()

# def main():
#     chunk_ids, vecs = fetch(DB)
#     if not vecs:
#         print("No embeddings found in DB. Run ingest/re-embed first.")
#         return
#     mat = np.vstack(vecs).astype('float32')
#     faiss.normalize_L2(mat)

#     dim = mat.shape[1]
#     # build HNSW (same as your VectorIndex settings)
#     index = faiss.IndexHNSWFlat(dim, 32)
#     id_map = faiss.IndexIDMap(index)

#     ids = np.arange(len(mat), dtype='int64')
#     id_map.add_with_ids(mat, ids)
#     faiss.write_index(id_map, FAISS_PATH)
#     # persist mapping faiss_idx -> chunk_id (ids are 0..n-1 here)
#     id_map_rows = [(int(i), cid) for i, cid in zip(ids.tolist(), chunk_ids)]
#     write_map(DB, id_map_rows)
#     print("Wrote FAISS:", FAISS_PATH, "ntotal:", id_map.ntotal)

# if __name__ == "__main__":
#     main()




# # scripts/rebuild_index.py

# from dotenv import load_dotenv
# import os
# from src.store import MetadataStore
# import sqlite3, numpy as np
# import faiss

# load_dotenv(override=True)

# DB_PATH = os.getenv("DB_PATH", "chunks_meta.db")
# FAISS_PATH = os.getenv("FAISS_PATH", "faiss.index")

# # 1) Ensure tables exist
# metadata_store = MetadataStore(DB_PATH)

# # 2) Fetch embeddings
# conn = sqlite3.connect(DB_PATH)
# cur = conn.cursor()
# cur.execute("SELECT chunk_id, embedding FROM chunks WHERE embedding IS NOT NULL AND LENGTH(embedding)>0")
# rows = cur.fetchall()
# conn.close()

# if not rows:
#     print("No embeddings found. Ingest documents first.")
#     exit()

# chunk_ids = [r[0] for r in rows]
# vectors = [np.frombuffer(r[1], dtype=np.float32) for r in rows]
# mat = np.vstack(vectors).astype("float32")
# faiss.normalize_L2(mat)

# # 3) Build FAISS index using explicit IDs
# index = faiss.IndexHNSWFlat(mat.shape[1], 32)
# idmap = faiss.IndexIDMap(index)
# ids = np.arange(len(mat), dtype="int64")
# idmap.add_with_ids(mat, ids)

# faiss.write_index(idmap, FAISS_PATH)

# # 4) Store faiss_map
# conn = sqlite3.connect(DB_PATH)
# cur = conn.cursor()
# cur.execute("DELETE FROM faiss_map")
# cur.executemany("INSERT INTO faiss_map(faiss_idx, chunk_id) VALUES (?,?)", zip(ids, chunk_ids))
# conn.commit()
# conn.close()

# print("FAISS rebuilt:", FAISS_PATH, "ntotal =", idmap.ntotal)



#!/usr/bin/env python3
"""
scripts/rebuild_faiss.py
Rebuild FAISS index from embeddings stored in DB and populate faiss_map.

Usage:
  python scripts/rebuild_faiss.py

Environment:
  DB_PATH (default: chunks_meta.db)
  FAISS_PATH (default: faiss.index)
  EMBED_DIM_FALLBACK (default: 1536)  # used only for sanity checks

Notes:
 - This script expects chunk embeddings to be stored in `chunks.embedding` as float32 bytes.
 - It creates/overwrites FAISS_PATH and replaces faiss_map table entries.
"""
from dotenv import load_dotenv
import os
import sqlite3
import numpy as np
import sys

load_dotenv(override=True)

DB_PATH = os.getenv("DB_PATH", "chunks_meta.db")
FAISS_PATH = os.getenv("FAISS_PATH", "faiss.index")
EMBED_DIM_FALLBACK = int(os.getenv("EMBED_DIM_FALLBACK", "1536"))

# Ensure MetadataStore/tables exist (so rebuild works even on fresh DB)
from src.store import MetadataStore
ms = MetadataStore(DB_PATH)

# fetch embeddings
def fetch_embeddings(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT chunk_id, embedding FROM chunks WHERE embedding IS NOT NULL AND LENGTH(embedding) > 0 ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    chunk_ids = [r[0] for r in rows]
    vecs = [np.frombuffer(r[1], dtype=np.float32) for r in rows]
    return chunk_ids, vecs

def write_faiss_map(db_path, id_tuples):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS faiss_map (
        faiss_idx INTEGER PRIMARY KEY,
        chunk_id TEXT UNIQUE
    )""")
    cur.execute("DELETE FROM faiss_map")
    cur.executemany("INSERT INTO faiss_map (faiss_idx, chunk_id) VALUES (?, ?)", id_tuples)
    conn.commit()
    conn.close()

def main():
    try:
        import faiss
    except Exception as e:
        print("faiss not installed or cannot be imported. Install faiss-cpu (pip) or faiss-gpu.", e)
        sys.exit(1)

    chunk_ids, vecs = fetch_embeddings(DB_PATH)
    n = len(vecs)
    if n == 0:
        print("No embeddings found in DB. Ingest documents first.")
        return

    mat = np.vstack(vecs).astype("float32")
    dim = mat.shape[1]
    if dim <= 0:
        print("Embedding dimension appears invalid:", dim)
        return

    print(f"Found {n} embeddings, dim={dim}. Normalizing vectors and building FAISS index...")

    # normalize for cosine (we will use inner product / normalized vectors)
    faiss.normalize_L2(mat)

    # build index (HNSW), wrap in IndexIDMap to store explicit ids
    index = faiss.IndexHNSWFlat(dim, 32)  # same parameters used in your code
    id_index = faiss.IndexIDMap(index)

    ids = np.arange(n, dtype="int64")  # assign deterministic ids 0..n-1
    id_index.add_with_ids(mat, ids)

    # write index to disk (overwrite)
    faiss.write_index(id_index, FAISS_PATH)
    print(f"Wrote FAISS index to {FAISS_PATH}. ntotal={id_index.ntotal}")

    # persist mapping faiss_idx -> chunk_id
    id_map_rows = [(int(i), cid) for i, cid in zip(ids.tolist(), chunk_ids)]
    write_faiss_map(DB_PATH, id_map_rows)
    print("faiss_map table updated with", len(id_map_rows), "rows")

if __name__ == "__main__":
    main()
