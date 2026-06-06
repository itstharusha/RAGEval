from groq import Groq
from config import CONFIG
from backend.generator import get_groq_client
import json, re

DECOMPOSE_PROMPT = """
Given the following answer, extract all atomic factual claims as a JSON list of strings.
Each claim must be a single, self-contained statement.
Return ONLY a JSON array, no explanation.

Answer: {answer}

JSON array of claims:"""

VERIFY_PROMPT = """
Given the following context passages, is the claim directly supported by them?
Answer with ONLY "yes" or "no".

Context:
{context}

Claim: {claim}

Answer (yes/no):"""

def faithfulness(generated_answer: str, retrieved_contexts: list[str]) -> float:
    """
    Faithfulness = fraction of answer claims supported by retrieved contexts.
    Returns 0.0 if answer is empty or no claims extracted.
    """
    if not generated_answer.strip():
        return 0.0
    client = get_groq_client()
    context_text = "\n\n".join(retrieved_contexts)
    # Step 1: decompose answer into claims
    decomp_resp = client.chat.completions.create(
        model=CONFIG["GROQ_MODEL"],
        messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(answer=generated_answer)}],
        temperature=0.0, max_tokens=512
    )
    raw = decomp_resp.choices[0].message.content.strip()
    # Strip markdown fences if present
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        claims = json.loads(raw)
    except json.JSONDecodeError:
        return 0.0
    if not claims:
        return 0.0
    # Step 2: verify each claim against contexts
    supported = 0
    for claim in claims:
        verify_resp = client.chat.completions.create(
            model=CONFIG["GROQ_MODEL"],
            messages=[{"role": "user", "content": VERIFY_PROMPT.format(
                context=context_text, claim=claim
            )}],
            temperature=0.0, max_tokens=8
        )
        verdict = verify_resp.choices[0].message.content.strip().lower()
        if verdict.startswith("yes"):
            supported += 1
    return round(supported / len(claims), 4)
