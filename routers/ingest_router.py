from fastapi import APIRouter, HTTPException
from models.requests import IngestTextRequest
from backend.ingest import ingest_text
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["ingest"])

@router.post("/ingest")
async def ingest_endpoint(body: IngestTextRequest):
    """
    Ingest raw text into Pinecone.
    Chunks the text, embeds with sentence-transformers, upserts to Pinecone.
    Returns: { "chunks_ingested": int, "namespace": str }
    """
    try:
        logger.info(f"Starting ingest for namespace: {body.namespace}, text length: {len(body.text)}")
        result = ingest_text(body.text, namespace=body.namespace, metadata=body.metadata)
        logger.info(f"Ingest successful: {result}")
        return result
    except Exception as e:
        logger.error(f"Ingest failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
