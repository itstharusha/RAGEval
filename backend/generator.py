from groq import Groq
from config import CONFIG

_client = None

def get_groq_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=CONFIG["GROQ_API_KEY"])
    return _client

ANSWER_PROMPT = """You are a precise question-answering assistant.
Answer the question using ONLY the information in the provided context.
If the context does not contain enough information, say "I don't know based on the provided context."
Be concise and direct.

Context:
{context}

Question: {question}

Answer:"""

def generate_answer(question: str, contexts: list[str]) -> str:
    """Generate an answer from Groq given question + retrieved contexts."""
    context_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = ANSWER_PROMPT.format(context=context_text, question=question)
    client = get_groq_client()
    response = client.chat.completions.create(
        model=CONFIG["GROQ_MODEL"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,   # deterministic for eval
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()
