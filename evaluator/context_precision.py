from backend.generator import get_groq_client
from config import CONFIG

RELEVANCE_PROMPT = """
Is the following context passage relevant to answering the question?
Answer ONLY "yes" or "no".

Question: {question}

Context passage: {context}

Answer (yes/no):"""

def context_precision(question: str, retrieved_contexts: list[str]) -> float:
    """
    Context Precision = fraction of retrieved chunks relevant to the question.
    """
    if not retrieved_contexts:
        return 0.0
    client = get_groq_client()
    relevant_count = 0
    for ctx in retrieved_contexts:
        resp = client.chat.completions.create(
            model=CONFIG["GROQ_MODEL"],
            messages=[{"role": "user", "content": RELEVANCE_PROMPT.format(
                question=question, context=ctx
            )}],
            temperature=0.0, max_tokens=8
        )
        verdict = resp.choices[0].message.content.strip().lower()
        if verdict.startswith("yes"):
            relevant_count += 1
    return round(relevant_count / len(retrieved_contexts), 4)
