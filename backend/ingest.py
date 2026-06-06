import uuid
from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from config import CONFIG
import logging

logger = logging.getLogger(__name__)

_embedder = None
_pinecone_index = None

def get_embedder() -> SentenceTransformer:
    """Lazy-load sentence transformer (cached after first call)."""
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model: {CONFIG['EMBEDDING_MODEL']}")
        _embedder = SentenceTransformer(CONFIG["EMBEDDING_MODEL"])
        logger.info("Embedding model loaded successfully")
    return _embedder

def get_pinecone_index():
    """Lazy-load and return Pinecone index, creating it if it does not exist."""
    global _pinecone_index
    if _pinecone_index is None:
        logger.info("Initializing Pinecone connection")
        pc = Pinecone(api_key=CONFIG["PINECONE_API_KEY"])
        index_name = CONFIG["PINECONE_INDEX_NAME"]
        logger.info(f"Listing Pinecone indexes...")
        existing = [i.name for i in pc.list_indexes()]
        logger.info(f"Existing indexes: {existing}")
        if index_name not in existing:
            logger.info(f"Creating index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=384,   # all-MiniLM-L6-v2 output dim
                metric="cosine",
                spec=ServerlessSpec(cloud=CONFIG["PINECONE_CLOUD"],
                                    region=CONFIG["PINECONE_REGION"])
            )
            logger.info(f"Index created: {index_name}")
        _pinecone_index = pc.Index(index_name)
        logger.info(f"Connected to index: {index_name}")
    return _pinecone_index

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping word-boundary chunks."""
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

def ingest_text(text: str, namespace: str = "default", metadata: dict = None) -> Dict:
    """Chunk text, embed, and upsert into Pinecone. Returns ingestion stats."""
    metadata = metadata or {}
    chunks = chunk_text(text, CONFIG["CHUNK_SIZE"], CONFIG["CHUNK_OVERLAP"])
    embedder = get_embedder()
    index = get_pinecone_index()
    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
    vectors = [
        {
            "id": str(uuid.uuid4()),
            "values": emb,
            "metadata": {"text": chunk, **metadata}
        }
        for chunk, emb in zip(chunks, embeddings)
    ]
    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i+batch_size], namespace=namespace)
    return {"chunks_ingested": len(chunks), "namespace": namespace}
