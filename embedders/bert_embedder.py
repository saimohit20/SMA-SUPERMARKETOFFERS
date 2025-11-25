import os
import warnings
from typing import List, Union

# Suppress PyTorch warnings for compatibility
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

from sentence_transformers import SentenceTransformer

# You can change the BERT / SBERT model here if you want
_BERT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print(f"Loading BERT model: {_BERT_MODEL_NAME}")

try:
    _bert_model = SentenceTransformer(_BERT_MODEL_NAME)
    print("BERT model loaded successfully")
except Exception as e:
    print(f"Warning: BERT model loading issue: {e}")
    _bert_model = None


def bert_embed(texts: Union[str, List[str]]) -> List[List[float]]:

    if _bert_model is None:
        raise RuntimeError("BERT model failed to load. Please check PyTorch installation.")
    
    if isinstance(texts, str):
        texts = [texts]

    embeddings = _bert_model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()