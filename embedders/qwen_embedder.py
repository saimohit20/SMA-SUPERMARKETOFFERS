import ollama
from qdrant_client import QdrantClient, models
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

QWEN_EMBED_MODEL = "qwen3-embedding:4b"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60.0)

def embed_texts(texts):
    vectors = []
    for t in texts:
        resp = ollama.embeddings(model=QWEN_EMBED_MODEL, prompt=t)

        if "embedding" in resp:
            vectors.append(resp["embedding"])

        elif "embeddings" in resp:
            vectors.append(resp["embeddings"][0])

        else:
            raise RuntimeError(f"Unexpected Ollama response: {resp}")

    return vectors

def embed_one(text):
    resp = ollama.embeddings(model=QWEN_EMBED_MODEL, prompt=text)
    if "embedding" in resp:
        return resp["embedding"]
    return resp["embeddings"][0]

def qwen_embed(texts):

    vectors = []
    for t in texts:
        resp = ollama.embeddings(model=QWEN_EMBED_MODEL, prompt=t)

        if "embedding" in resp:
            vectors.append(resp["embedding"])
        elif "embeddings" in resp:
            vectors.append(resp["embeddings"][0])
        else:
            raise RuntimeError(f"Unexpected embedding response: {resp}")

    return vectors

def ensure_qwen_collection():

    try:
        qdrant.get_collection("offers_qwen")
    except Exception:
        test_dim = len(qwen_embed(["test"])[0])
        qdrant.create_collection(
            collection_name="offers_qwen",
            vectors_config=models.VectorParams(size=test_dim, distance=models.Distance.COSINE)
        )

def chunk_upsert(points, batch_size=100):
    
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        tries = 0
        while tries < 2:
            try:
                qdrant.upsert("offers_qwen", batch)
                break
            except Exception as e:
                if "timeout" in str(e).lower() and batch_size > 20 and tries == 0:
                    batch_size = max(20, batch_size // 2)
                    batch = points[i:i+batch_size]
                    tries += 1
                else:
                    raise