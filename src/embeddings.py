# src/embeddings.py

import os
import time
import numpy as np
from typing import List
from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------
# Load Azure OpenAI Client
# ---------------------------------------
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version="2023-10-01-preview"
)

# Embedding deployment name
DEPLOYMENT = os.environ["OPENAI_DEPLOYMENT_NAME"]


def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    Low-level embedding function that calls Azure OpenAI directly.
    FOR INTERNAL USE BY SAFE WRAPPER BELOW.
    """
    resp = client.embeddings.create(
        model=DEPLOYMENT,
        input=texts
    )
    return np.vstack([np.array(item.embedding, dtype=np.float32) for item in resp.data])


def get_embeddings_safe(texts: List[str], retries: int = 3) -> np.ndarray:
    """
    Safe wrapper: retries on transient Azure / network errors.
    Used everywhere in the pipeline.
    """
    attempts = 0
    while True:
        try:
            return get_embeddings(texts)
        except Exception as e:
            attempts += 1
            if attempts > retries:
                print("❌ Embeddings failed after retries:", e)
                raise
            wait = 0.5 * (2 ** (attempts - 1))
            print(f"⚠️ Embedding error. Retrying in {wait:.1f}s...")
            time.sleep(wait)
