
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from typing import Optional
import tempfile
import os
import sys
from pydantic import BaseModel
from typing import Optional
from .store import MetadataStore
from .vector_index import VectorIndex
from .ingest import process_document
from .utils import sha256_hex
import os, numpy as np
import faiss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "chunks_meta.db")
FAISS_PATH = os.getenv("FAISS_PATH", "faiss.index")
USE_FAISS = os.getenv("USE_FAISS", "true").lower() == "true"
EMBED_DIM_FALLBACK = int(os.getenv("EMBED_DIM_FALLBACK", "1536"))

metadata_store = MetadataStore(DB_PATH)
vector_index = VectorIndex(dim=EMBED_DIM_FALLBACK, path=FAISS_PATH) if (USE_FAISS and faiss is not None) else None
# try load index if file exists
if vector_index and vector_index.path and os.path.exists(vector_index.path):
    try:
        vector_index.load(vector_index.path)
    except Exception:
        pass

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="Semantic Chunker API")

# serve static UI
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        # serve the UI
        return FileResponse(os.path.join(static_dir, "index.html"))
else:
    # if static not present, root returns a helpful message
    @app.get("/", include_in_schema=False)
    def index():
        return {"status": "ok", "note": "Place static/index.html in project-root/static/ to enable UI"}


class IngestRequest(BaseModel):
    doc_id: str
    text: str

class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 5


def _extract_text_from_pdf(path: str) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(path)
    pages = [p.get_text("text") for p in doc]
    doc.close()
    return "\n".join(pages)


# @app.post("/ingest")
# def ingest(
#     doc_id: Optional[str] = Form(None),
#     text: Optional[str] = Form(None),
#     file: Optional[UploadFile] = File(None),
#     do_merge: Optional[bool] = Form(True),
#     persist: Optional[bool] = Form(True),
# ):
#     """
#     Ingest endpoint supporting either:
#       - multipart/form-data with a PDF file (field name: 'file')
#       - form-data/text with 'text' field (or JSON body as before if using the old client)

#     Priority: file > text.
#     """
#     try:
#         # 1) Ensure we have either file or text
#         if file is None and (text is None or not text.strip()):
#             raise HTTPException(status_code=400, detail="Provide either a PDF file (file) or non-empty text (text).")

#         # 2) If file provided, extract text from PDF
#         if file is not None:
#             # If doc_id not given, use filename (without extension)
#             if not doc_id:
#                 filename = getattr(file, "filename", "uploaded")
#                 doc_id = os.path.splitext(filename)[0]

#             # Save uploaded file to a temp file and extract text
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpf:
#                 tmp_path = tmpf.name
#                 tmpf.write(file.file.read())

#             try:
#                 extracted_text = _extract_text_from_pdf(tmp_path)
#             finally:
#                 try:
#                     os.remove(tmp_path)
#                 except Exception:
#                     pass

#             if not extracted_text or not extracted_text.strip():
#                 raise HTTPException(status_code=400, detail="Uploaded PDF contains no extractable text.")

#             ingest_text = extracted_text

#         else:
#             # file not provided -> use text field
#             if not doc_id:
#                 raise HTTPException(status_code=400, detail="doc_id is required when sending text.")
#             ingest_text = text

#         # 3) Run ingestion pipeline (synchronous, simple)
#         chunks, emb = process_document(
#             doc_id,
#             ingest_text,
#             metadata_store,
#             vector_index,
#             do_merge=do_merge,
#             persist=persist,
#         )

#         return {"status": "ok", "doc_id": doc_id, "n_chunks": len(chunks)}

#     except HTTPException:
#         raise
#     except Exception as e:
#         # keep error messages readable for learning/debugging
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest(
    request: Request,
    file: Optional[UploadFile] = File(None),
    doc_id: Optional[str] = None,
):
    """
    Very simple ingestion:
    - If JSON is sent: { "doc_id": "...", "text": "..." }
    - If PDF file is sent (multipart): use ?doc_id=xxx and upload PDF
    """

    # -----------------------------
    # CASE 1: JSON with text
    # -----------------------------
    if request.headers.get("content-type", "").startswith("application/json"):
        body = await request.json()
        doc_id = body.get("doc_id")
        text = body.get("text")
        logger.info(f"text : {text}")

        if not doc_id or not text:
            raise HTTPException(400, "JSON must include doc_id and text")

        chunks, emb = process_document(doc_id, text, metadata_store, vector_index)
        return {"status": "ok", "doc_id": doc_id, "n_chunks": len(chunks)}

    # -----------------------------
    # CASE 2: PDF upload
    # -----------------------------
    if file is not None:
        if not doc_id:
            raise HTTPException(400, "doc_id is required when uploading PDF")

        # save temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(file.file.read())
        tmp.close()

        # extract PDF text
        extracted_text = _extract_text_from_pdf(tmp.name)
        logger.info(f"Extracted text : {extracted_text}")
        # print("EXTRACTED TEXT:", extracted_text[:1000])
        os.remove(tmp.name)

        if not extracted_text.strip():
            raise HTTPException(400, "Could not extract text from PDF")

        chunks, emb = process_document(doc_id, extracted_text, metadata_store, vector_index)
        return {"status": "ok", "doc_id": doc_id, "n_chunks": len(chunks)}

    # -----------------------------
    # No input
    # -----------------------------
    raise HTTPException(400, "Send JSON {doc_id,text} OR upload PDF as file")

# @app.post("/query")
# def query(req: QueryRequest):
#     try:
#         # load all chunks and embeddings from DB
#         all_chunks = metadata_store.fetch_all_chunks()
#         if vector_index and vector_index.index is not None:
#             # use FAISS
#             # get query embedding
#             from .embeddings import get_embeddings_safe
#             q = get_embeddings_safe([req.query])
#             q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
#             D, I = vector_index.search(q, req.k)
#             # map indices to chunk_ids using faiss_map
#             results=[]
#             cur = metadata_store.conn.cursor()
#             for dist, idx in zip(D[0], I[0]):
#                 if idx < 0: continue
#                 cur.execute("SELECT chunk_id, text FROM chunks LIMIT 1 OFFSET ?", (int(idx),))
#                 row = cur.fetchone()
#                 if row:
#                     results.append({"chunk_id": row[0], "score": float(dist), "text": row[1]})
#             return {"results": results}
#         else:
#             # in-memory fallback
#             chunks = []
#             vecs = []
#             for c in all_chunks:
#                 if c["embedding"] is None:
#                     continue
#                 v = np.frombuffer(c["embedding"], dtype=np.float32)
#                 vecs.append(v); chunks.append(c)
#             if not vecs:
#                 return {"results": []}
#             mat = np.vstack(vecs)
#             norms = np.linalg.norm(mat, axis=1, keepdims=True); norms[norms==0]=1.0; mat = mat / norms
#             from .embeddings import get_embeddings_safe
#             q = get_embeddings_safe([req.query])
#             q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
#             sims = (mat @ q.T).reshape(-1)
#             idxs = np.argsort(sims)[::-1][:req.k]
#             out = [{"chunk_id": chunks[i]["chunk_id"], "score": float(sims[i]), "text": chunks[i]["text"]} for i in idxs]
#             return {"results": out}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query(req: QueryRequest):
    """
    FAISS-only retrieval.
    - Requires FAISS index loaded and ntotal > 0.
    - Returns empty list if FAISS is empty or not initialized.
    """
    try:
        # --- Ensure FAISS index is available ---
        if not vector_index or vector_index.index is None:
            return {"results": []}

        ntotal = getattr(vector_index.index, "ntotal", 0)
        if ntotal == 0:
            # FAISS index is empty → no fallback
            return {"results": []}

        from .embeddings import get_embeddings_safe

        # --- Embed query ---
        q = get_embeddings_safe([req.query])
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

        # --- Search FAISS ---
        D, I = vector_index.search(q, req.k)

        # Flatten ids, remove negatives
        faiss_ids = [int(x) for x in I.reshape(-1) if int(x) >= 0]
        if not faiss_ids:
            return {"results": []}

        # --- Lookup FAISS id -> chunk_id mapping ---
        cur = metadata_store.conn.cursor()
        placeholders = ",".join(["?"] * len(faiss_ids))
        cur.execute(
            f"SELECT faiss_idx, chunk_id FROM faiss_map WHERE faiss_idx IN ({placeholders})",
            tuple(faiss_ids)
        )
        rows = cur.fetchall()
        id_map = {r[0]: r[1] for r in rows}

        if not id_map:
            return {"results": []}

        # --- Batch fetch chunk texts ---
        ordered_chunk_ids = []
        for raw_id in I[0]:
            fid = int(raw_id)
            cid = id_map.get(fid)
            if cid and cid not in ordered_chunk_ids:
                ordered_chunk_ids.append(cid)

        placeholders = ",".join(["?"] * len(ordered_chunk_ids))
        cur.execute(
            f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({placeholders})",
            tuple(ordered_chunk_ids)
        )
        text_rows = cur.fetchall()
        text_map = {r[0]: r[1] for r in text_rows}

        # --- Build results preserving FAISS order ---
        results = []
        for dist, raw_id in zip(D[0], I[0]):
            fid = int(raw_id)
            cid = id_map.get(fid)
            if not cid:
                continue
            txt = text_map.get(cid, "")
            results.append({
                "chunk_id": cid,
                "score": float(dist),
                "text": txt
            })

        return {"results": results}

    except Exception as e:
        logger.exception("Query error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    try:
        cur = metadata_store.conn.cursor(); cur.execute("SELECT 1")
        return {"status":"ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail="DB error")
