import time
from typing import Tuple, List
from backend.retriever import retrieve
from backend.generator import generate_answer

def run_rag_pipeline(
    question: str,
    namespace: str = "default"
) -> Tuple[str, List[str], float]:
    """
    Run retrieval + generation for one question.
    Returns: (generated_answer, retrieved_contexts, latency_ms)
    """
    start = time.perf_counter()
    contexts = retrieve(question, namespace=namespace)
    answer = generate_answer(question, contexts)
    latency_ms = (time.perf_counter() - start) * 1000
    return answer, contexts, latency_ms
