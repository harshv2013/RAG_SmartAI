import os
import logging
import numpy as np
try:
    import faiss
except Exception:
    faiss = None

# class VectorIndex:
#     def __init__(self, dim: int = 1536, use_hnsw: bool = True, path: str = "faiss.index"):
#         self.dim = dim; self.path = path; self.use_hnsw = use_hnsw; self.index = None
#         if faiss is not None:
#             self._create_index()
#         else:
#             logging.info("faiss not available")

#     def _create_index(self):
#         if self.use_hnsw:
#             self.index = faiss.IndexHNSWFlat(self.dim, 32)
#         else:
#             self.index = faiss.IndexFlatIP(self.dim)

#     # def add_with_mapping(self, vectors: np.ndarray, chunk_ids: list, metadata_store):
#     #     if self.index is None:
#     #         raise RuntimeError("faiss not initialized")
#     #     start = int(self.index.ntotal)
#     #     self.index.add(vectors.astype(np.float32))
#     #     for offset, cid in enumerate(chunk_ids):
#     #         metadata_store.insert_faiss_map(start + offset, cid)
#     # vector_index.py (replace add_with_mapping)

#     def add_with_mapping(self, vectors: np.ndarray, chunk_ids: list, metadata_store):
#         """
#         Add vectors and persist a mapping faiss_idx -> chunk_id.
#         We assign integer IDs sequentially starting from current ntotal.
#         Uses IndexIDMap so searches return the IDs we assigned.
#         """
#         if self.index is None:
#             raise RuntimeError("faiss not initialized")

#         # ensure IndexIDMap wrapper
#         try:
#             # If already an IDMap, leave as is
#             if not isinstance(self.index, faiss.IndexIDMap):
#                 base = self.index
#                 idmapped = faiss.IndexIDMap(base)
#                 self.index = idmapped
#         except Exception:
#             pass

#         start = int(self.index.ntotal)
#         n = vectors.shape[0]
#         faiss_ids = np.arange(start, start + n).astype('int64')

#         # add_with_ids expects (n,dim) matrix and ids ndarray
#         self.index.add_with_ids(vectors.astype(np.float32), faiss_ids)

#         # persist mapping to DB
#         for fid, cid in zip(faiss_ids.tolist(), chunk_ids):
#             metadata_store.insert_faiss_map(int(fid), cid)


#     def save(self, path: str = None):
#         if self.index is None:
#             return
#         p = path or self.path
#         if p:
#             faiss.write_index(self.index, p)

#     def load(self, path: str):
#         self.index = faiss.read_index(path)

#     def search(self, qvec: np.ndarray, k: int = 5):
#         if self.index is None:
#             raise RuntimeError("index not initialized")
#         return self.index.search(qvec.astype(np.float32), k)


# vector_index.py
import os, logging
# ... existing imports and faiss try ...

class VectorIndex:
    def __init__(self, dim: int = 1536, use_hnsw: bool = True, path: str = "faiss.index"):
        self.dim = dim
        self.path = path
        self.use_hnsw = use_hnsw
        self.index = None
        if faiss is not None:
            # try load index if file present; otherwise create empty index
            if self.path and os.path.exists(self.path):
                try:
                    self.load(self.path)
                    logging.info("Loaded FAISS index from %s (ntotal=%d)", self.path, int(self.index.ntotal))
                except Exception as e:
                    logging.warning("Failed to load FAISS index (%s). Creating new empty index. Error: %s", self.path, e)
                    self._create_index()
            else:
                logging.info("FAISS file not found at %s. Creating new empty index.", self.path)
                self._create_index()
        else:
            logging.info("faiss not available")

    def _create_index(self):
        if self.use_hnsw:
            self.index = faiss.IndexHNSWFlat(self.dim, 32)
        else:
            self.index = faiss.IndexFlatIP(self.dim)

    def add_with_mapping(self, vectors: np.ndarray, chunk_ids: list, metadata_store):
        if self.index is None:
            raise RuntimeError("faiss not initialized")

        # wrap in IndexIDMap if not already
        if not isinstance(self.index, faiss.IndexIDMap):
            base = self.index
            idmapped = faiss.IndexIDMap(base)
            self.index = idmapped

        start = int(self.index.ntotal)
        n = vectors.shape[0]
        if n == 0:
            return
        faiss_ids = np.arange(start, start + n).astype('int64')
        # add with IDs (explicit)
        self.index.add_with_ids(vectors.astype(np.float32), faiss_ids)

        # persist mapping to DB
        for fid, cid in zip(faiss_ids.tolist(), chunk_ids):
            metadata_store.insert_faiss_map(int(fid), cid)

        # persist the index file so it's available next startup
        try:
            self.save(self.path)
            logging.info("Saved FAISS index to %s (ntotal=%d)", self.path, int(self.index.ntotal))
        except Exception as e:
            logging.warning("Failed to save FAISS index to %s: %s", self.path, e)


    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"FAISS file not found: {path}")
        self.index = faiss.read_index(path)

    def save(self, path: str = None):
        if self.index is None:
            return
        p = path or self.path
        if p:
            # ensure parent dir exists
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            faiss.write_index(self.index, p)

    def search(self, qvec: np.ndarray, k: int = 5):
        if self.index is None:
            raise RuntimeError("index not initialized")
        return self.index.search(qvec.astype(np.float32), k)
