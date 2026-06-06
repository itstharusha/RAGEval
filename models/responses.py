from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

class SampleResult(BaseModel):
    """Evaluation result for one question-answer pair."""
    question: str
    generated_answer: str
    retrieved_contexts: List[str]
    ground_truth_answer: str
    faithfulness: Optional[float]
    context_precision: Optional[float]
    context_recall: Optional[float]
    answer_relevance: Optional[float]
    latency_ms: float

class AggregateScores(BaseModel):
    """Mean scores across all samples in a run."""
    faithfulness: Optional[float]
    context_precision: Optional[float]
    context_recall: Optional[float]
    answer_relevance: Optional[float]
    avg_latency_ms: float
    num_samples: int

class EvalRunResult(BaseModel):
    """Full evaluation run result returned to the dashboard."""
    run_id: str
    run_name: Optional[str]
    created_at: datetime
    aggregate: AggregateScores
    samples: List[SampleResult]
    status: str   # "completed" | "error"
    error_message: Optional[str] = None
