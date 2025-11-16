#!/usr/bin/env python3
"""
scripts/init_db.py
Create DB tables using your MetadataStore (idempotent).
Usage:
  python scripts/init_db.py
  DB_PATH=path/to/db.sqlite python scripts/init_db.py
"""
from dotenv import load_dotenv
import os
from src.store import MetadataStore

if __name__ == "__main__":
    load_dotenv(override=True)
    DB_PATH = os.getenv("DB_PATH", "chunks_meta.db")
    print("Initializing DB at:", DB_PATH)
    ms = MetadataStore(DB_PATH)
    # Quick table list for confirmation
    cur = ms.conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    print("Tables present:", tables)
    print("Done.")
