from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from config import validate_config, CONFIG
from routers.ingest_router import router as ingest_router
from routers.evaluate_router import router as evaluate_router
from routers.results_router import router as results_router
import uvicorn

validate_config()

app = FastAPI(
    title="RAGEval — RAG Evaluation Framework",
    description="Benchmark any RAG pipeline with 5 automated metrics",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(evaluate_router)
app.include_router(results_router)

# Serve frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run("main:app", host=CONFIG["HOST"], port=CONFIG["PORT"], reload=True)
