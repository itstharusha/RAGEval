import numpy as np
from backend.ingest import get_embedder

def cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def answer_relevance(question: str, generated_answer: str) -> float:
    """
    Answer Relevance = cosine similarity between question and answer embeddings.
    Measures whether the answer is topically aligned with the question.
    """
    if not generated_answer.strip():
        return 0.0
    embedder = get_embedder()
    vecs = embedder.encode([question, generated_answer])
    return round(cosine_similarity(vecs[0].tolist(), vecs[1].tolist()), 4)
