import os
from typing import List, Dict, Tuple
import numpy as np

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
    Returns np.ndarray of shape (N, dim)
    """
    resp = client.embeddings.create(
        model=DEPLOYMENT,
        input=texts
    )
    embs = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
    return np.vstack(embs)
