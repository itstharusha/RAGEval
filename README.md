# RAGEval — Production RAG Evaluation Framework

A developer tool that benchmarks any RAG pipeline across 5 automated metrics.
Built with FastAPI + Groq + Pinecone + sentence-transformers.

## What It Measures

| Metric            | What It Detects                                      |
|-------------------|------------------------------------------------------|
| Faithfulness      | Hallucinations — claims not grounded in context      |
| Context Precision | Retriever noise — irrelevant chunks being fetched    |
| Context Recall    | Retriever gaps — missing relevant chunks             |
| Answer Relevance  | Off-topic answers — topical drift from the question  |
| Latency           | End-to-end response time per query                   |

## Quick Start

```bash
git clone https://github.com/itstharusha/rageval
cd rageval
pip install -r requirements.txt
cp .env.example .env    # fill in GROQ_API_KEY and PINECONE_API_KEY
python main.py          # dashboard at http://localhost:8000
```

## Run the Demo

```bash
python scripts/run_sample_eval.py
```

## API Reference

| Endpoint                        | Method | Description                     |
|---------------------------------|--------|---------------------------------|
| /api/ingest                     | POST   | Ingest text corpus into Pinecone|
| /api/evaluate                   | POST   | Run full eval suite             |
| /api/results                    | GET    | List all evaluation runs        |
| /api/results/{run_id}           | GET    | Get full run result             |
| /api/results/{run_id}/export/csv| GET    | Export results as CSV           |
| /docs                           | GET    | Auto-generated Swagger UI       |

## Architecture

```
User → FastAPI → [Ingest] → Pinecone (vectors)
              → [Evaluate]
                    → RAG Pipeline (retrieve + generate via Groq)
                    → Evaluator (faithfulness, precision, recall, relevance, latency)
                    → EvalRunResult → in-memory store
              → [Dashboard] → frontend/index.html reads /api/results
```

## Benchmark Results (Sample Corpus)

| Metric            | Score  |
|-------------------|--------|
| Faithfulness      | 0.87   |
| Context Precision | 0.82   |
| Context Recall    | 0.79   |
| Answer Relevance  | 0.91   |
| Avg Latency       | 1240ms |

> Note: Scores vary based on corpus quality and query complexity.

## Tech Stack
FastAPI · Groq (llama-3.3-70b-versatile) · Pinecone Serverless · sentence-transformers · Python 3.11+
