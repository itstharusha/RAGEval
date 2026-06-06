from backend.generator import get_groq_client
from config import CONFIG

RECALL_PROMPT = """
Is the information in the reference passage captured (fully or partially) in the retrieved passages below?
Answer ONLY "yes" or "no".

Reference passage:
{reference}

Retrieved passages:
{retrieved}

Answer (yes/no):"""

def context_recall(
    ground_truth_contexts: list[str],
    retrieved_contexts: list[str]
) -> float:
    """
    Context Recall = fraction of ground-truth contexts covered by retrieved contexts.
    Returns 1.0 if ground_truth_contexts is empty (no reference to miss).
    """
    if not ground_truth_contexts:
        return 1.0
    if not retrieved_contexts:
        return 0.0
    client = get_groq_client()
    retrieved_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(retrieved_contexts))
    covered = 0
    for ref in ground_truth_contexts:
        resp = client.chat.completions.create(
            model=CONFIG["GROQ_MODEL"],
            messages=[{"role": "user", "content": RECALL_PROMPT.format(
                reference=ref, retrieved=retrieved_text
            )}],
            temperature=0.0, max_tokens=8
        )
        verdict = resp.choices[0].message.content.strip().lower()
        if verdict.startswith("yes"):
            covered += 1
    return round(covered / len(ground_truth_contexts), 4)
