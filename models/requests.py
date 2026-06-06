from pydantic import BaseModel, Field
from typing import List, Optional

class IngestTextRequest(BaseModel):
    """Request body for ingesting raw text into Pinecone."""
    text: str = Field(..., description="Raw text corpus to chunk and ingest")
    namespace: str = Field(default="default", description="Pinecone namespace")
    metadata: dict = Field(default_factory=dict, description="Optional metadata tags")

class EvalSample(BaseModel):
    """Single evaluation sample: a question with ground truth."""
    question: str
    ground_truth_answer: str
    ground_truth_contexts: List[str] = Field(
        default_factory=list,
        description="List of text chunks that should ideally be retrieved"
    )

class EvaluateRequest(BaseModel):
    """Request body for running a full evaluation suite."""
    dataset: List[EvalSample]
    namespace: str = Field(default="default")
    run_name: Optional[str] = Field(default=None, description="Human-readable run label")
    metrics: List[str] = Field(
        default=["faithfulness","context_precision","context_recall","answer_relevance","latency"],
        description="Which metrics to compute"
    )
