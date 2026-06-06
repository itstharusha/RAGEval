from typing import List
from backend.ingest import get_embedder, get_pinecone_index
from config import CONFIG

def retrieve(query: str, namespace: str = "default", top_k: int = None) -> List[str]:
    """Embed query, search Pinecone, return list of text chunks."""
    top_k = top_k or CONFIG["TOP_K_CHUNKS"]
    embedder = get_embedder()
    index = get_pinecone_index()
    query_vec = embedder.encode([query])[0].tolist()
    results = index.query(
        vector=query_vec,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )
    return [match["metadata"]["text"] for match in results["matches"]]
