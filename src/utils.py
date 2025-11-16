import nltk
import hashlib
from typing import List
import os

nltk.download("punkt", quiet=True)

def sentence_split(text: str) -> List[str]:
    return nltk.tokenize.sent_tokenize(text)

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())
