from dotenv import load_dotenv
import os

load_dotenv()

CONFIG = {
    "GROQ_API_KEY":        os.getenv("GROQ_API_KEY"),
    "PINECONE_API_KEY":    os.getenv("PINECONE_API_KEY"),
    "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME", "rageval-index"),
    "PINECONE_CLOUD":      os.getenv("PINECONE_CLOUD", "aws"),
    "PINECONE_REGION":     os.getenv("PINECONE_REGION", "us-east-1"),
    "GROQ_MODEL":          os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    "EMBEDDING_MODEL":     os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    "TOP_K_CHUNKS":        int(os.getenv("TOP_K_CHUNKS", 5)),
    "CHUNK_SIZE":          int(os.getenv("CHUNK_SIZE", 512)),
    "CHUNK_OVERLAP":       int(os.getenv("CHUNK_OVERLAP", 64)),
    "HOST":                os.getenv("HOST", "0.0.0.0"),
    "PORT":                int(os.getenv("PORT", 8000)),
}

def validate_config():
    required = ["GROQ_API_KEY", "PINECONE_API_KEY"]
    missing = [k for k in required if not CONFIG.get(k)]
    if missing:
        raise ValueError(f"Missing required env vars: {missing}")
