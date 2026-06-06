from fastapi import APIRouter, HTTPException, BackgroundTasks
from models.requests import EvaluateRequest
from evaluator.runner import run_evaluation

router = APIRouter(prefix="/api", tags=["evaluate"])

@router.post("/evaluate")
async def evaluate_endpoint(body: EvaluateRequest):
    """
    Run a full evaluation suite on the provided dataset.
    Synchronous — waits for all metrics to complete.
    Returns: EvalRunResult (full JSON)
    Warning: for large datasets (>20 samples), expect 30-120s response time due to LLM calls.
    """
    try:
        result = run_evaluation(body)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
