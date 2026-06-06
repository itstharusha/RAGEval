import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import validate_config
from backend.ingest import ingest_text
from models.requests import EvaluateRequest, EvalSample
from evaluator.runner import run_evaluation

validate_config()

# 1. Ingest corpus
print("Ingesting sample corpus into Pinecone...")
with open("sample_data/sample_corpus.txt") as f:
    text = f.read()
stats = ingest_text(text, namespace="demo")
print(f"Ingested: {stats}")

# 2. Load eval dataset
with open("sample_data/sample_eval_dataset.json") as f:
    raw = json.load(f)
dataset = [EvalSample(**item) for item in raw]

# 3. Run evaluation
print("Running evaluation (this takes ~60s due to LLM calls)...")
request = EvaluateRequest(dataset=dataset, namespace="demo", run_name="sample-demo")
result = run_evaluation(request)

# 4. Print results
print("\n=== AGGREGATE SCORES ===")
agg = result.aggregate
print(f"  Faithfulness:      {agg.faithfulness:.4f}")
print(f"  Context Precision: {agg.context_precision:.4f}")
print(f"  Context Recall:    {agg.context_recall:.4f}")
print(f"  Answer Relevance:  {agg.answer_relevance:.4f}")
print(f"  Avg Latency:       {agg.avg_latency_ms:.1f} ms")
print(f"\nFull result saved to run_id: {result.run_id}")
