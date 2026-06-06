from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from evaluator.runner import EVAL_STORE
import csv, io

router = APIRouter(prefix="/api", tags=["results"])

@router.get("/results")
async def list_results():
    """Return list of all run IDs and their aggregate scores."""
    return [
        {"run_id": k, "run_name": v.run_name, "created_at": v.created_at, "aggregate": v.aggregate}
        for k, v in EVAL_STORE.items()
    ]

@router.get("/results/{run_id}")
async def get_result(run_id: str):
    """Return full evaluation result for a specific run."""
    result = EVAL_STORE.get(run_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    return result

@router.get("/results/{run_id}/export/csv")
async def export_csv(run_id: str):
    """Export per-sample results as CSV download."""
    result = EVAL_STORE.get(run_id)
    if not result:
        raise HTTPException(status_code=404, detail="Run not found")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "question","generated_answer","faithfulness","context_precision",
        "context_recall","answer_relevance","latency_ms"
    ])
    writer.writeheader()
    for s in result.samples:
        writer.writerow({
            "question": s.question, "generated_answer": s.generated_answer,
            "faithfulness": s.faithfulness, "context_precision": s.context_precision,
            "context_recall": s.context_recall, "answer_relevance": s.answer_relevance,
            "latency_ms": s.latency_ms
        })
    output.seek(0)
    return StreamingResponse(
        iter([output.read()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=rageval_{run_id}.csv"}
    )
