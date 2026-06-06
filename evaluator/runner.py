import uuid
from datetime import datetime, timezone
from typing import Dict
from models.requests import EvaluateRequest, EvalSample
from models.responses import EvalRunResult, SampleResult, AggregateScores
from backend.pipeline import run_rag_pipeline
from evaluator.faithfulness import faithfulness
from evaluator.context_precision import context_precision
from evaluator.context_recall import context_recall
from evaluator.answer_relevance import answer_relevance

# In-memory store: run_id -> EvalRunResult
EVAL_STORE: Dict[str, EvalRunResult] = {}

def run_evaluation(request: EvaluateRequest) -> EvalRunResult:
    """
    Full evaluation pipeline:
    1. For each sample: run RAG pipeline, compute metrics.
    2. Aggregate scores.
    3. Store and return EvalRunResult.
    """
    run_id = str(uuid.uuid4())[:8]
    sample_results = []
    enabled = set(request.metrics)

    for sample in request.dataset:
        answer, contexts, latency_ms = run_rag_pipeline(
            sample.question, namespace=request.namespace
        )
        result = SampleResult(
            question=sample.question,
            generated_answer=answer,
            retrieved_contexts=contexts,
            ground_truth_answer=sample.ground_truth_answer,
            faithfulness=faithfulness(answer, contexts) if "faithfulness" in enabled else None,
            context_precision=context_precision(sample.question, contexts) if "context_precision" in enabled else None,
            context_recall=context_recall(sample.ground_truth_contexts, contexts) if "context_recall" in enabled else None,
            answer_relevance=answer_relevance(sample.question, answer) if "answer_relevance" in enabled else None,
            latency_ms=round(latency_ms, 2),
        )
        sample_results.append(result)

    def mean(vals): return round(sum(v for v in vals if v is not None) / max(1, sum(1 for v in vals if v is not None)), 4)

    aggregate = AggregateScores(
        faithfulness=mean([r.faithfulness for r in sample_results]),
        context_precision=mean([r.context_precision for r in sample_results]),
        context_recall=mean([r.context_recall for r in sample_results]),
        answer_relevance=mean([r.answer_relevance for r in sample_results]),
        avg_latency_ms=mean([r.latency_ms for r in sample_results]),
        num_samples=len(sample_results),
    )

    run_result = EvalRunResult(
        run_id=run_id,
        run_name=request.run_name,
        created_at=datetime.now(timezone.utc),
        aggregate=aggregate,
        samples=sample_results,
        status="completed",
    )
    EVAL_STORE[run_id] = run_result
    return run_result
