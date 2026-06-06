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

## Preview
<img width="1919" height="1079" alt="Screenshot 2026-06-06 174918" src="https://github.com/user-attachments/assets/0b05e54d-f7b2-47a0-8684-b2f8bc2e6830" />
<img width="1919" height="1079" alt="Screenshot 2026-06-06 181146" src="https://github.com/user-attachments/assets/85d93574-257e-4f02-8bfb-6423aac78845" />
<img width="1919" height="1079" alt="Screenshot 2026-06-06 181221" src="https://github.com/user-attachments/assets/90f3232c-de4a-4fb5-a6ac-147503db892a" />
<img width="1919" height="1079" alt="Screenshot 2026-06-06 181648" src="https://github.com/user-attachments/assets/3fc7ca09-f264-4e29-b21e-5b185563c076" />
<img width="1919" height="1079" alt="Screenshot 2026-06-06 181656" src="https://github.com/user-attachments/assets/53eefdee-536d-4c1f-8d39-77bfc5d5f70f" />


