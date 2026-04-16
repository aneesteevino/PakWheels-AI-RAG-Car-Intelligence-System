"""
PakWheels RAG System - FastAPI Backend
Retrieval-Augmented Generation for used car queries
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag_pipeline_simple import RAGPipeline

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="PakWheels RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State ─────────────────────────────────────────────────────────────────────
rag: Optional[RAGPipeline] = None


@app.on_event("startup")
async def startup():
    global rag
    log.info("Initialising RAG pipeline …")
    data_path = Path(__file__).parent.parent / "data" / "pakwheels_used_cars.csv"
    rag = RAGPipeline(str(data_path))
    rag.build_index()
    log.info("RAG pipeline ready.")


# ── Schemas ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    retrieved_cars: list
    stats: dict
    query_type: str
    elapsed_ms: float


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if rag is None:
        raise HTTPException(503, "RAG pipeline not ready")
    t0 = time.time()
    result = rag.query(req.query, top_k=req.top_k)
    result["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
    return result


@app.get("/api/suggestions")
async def suggestions():
    return {
        "suggestions": [
            "What is the average price of Toyota Corolla in Lahore?",
            "Show me low mileage Honda Civic under 3 million",
            "Which cars after 2020 are diesel and automatic?",
            "What is the cheapest SUV in Karachi?",
            "Compare Honda and Toyota prices",
            "Show me imported automatic cars under 5 million",
            "What are the most popular car makes?",
            "Show me Suzuki Alto listings under 1.5 million",
            "What is the average price of cars in Karachi vs Lahore?",
            "Show me hybrid cars with low mileage",
        ]
    }


@app.get("/api/stats")
async def global_stats():
    if rag is None:
        raise HTTPException(503, "RAG pipeline not ready")
    df = rag.df
    return {
        "total_listings": len(df),
        "cities": sorted(df["ad_city"].dropna().unique().tolist()),
        "makes": df["make"].value_counts().head(15).to_dict(),
        "price_range": {
            "min": int(df["price"].min()),
            "max": int(df["price"].max()),
            "avg": int(df["price"].mean()),
        },
        "fuel_types": df["fuel_type"].dropna().unique().tolist(),
        "transmissions": df["transmission"].dropna().unique().tolist(),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "rag_ready": rag is not None}


# ── Serve frontend ────────────────────────────────────────────────────────────
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    # Mount static files (CSS, JS) at root level
    app.mount("/style.css", StaticFiles(directory=str(frontend_dir)), name="css")
    app.mount("/app.js", StaticFiles(directory=str(frontend_dir)), name="js")
    
    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(frontend_dir / "index.html"))
    
    @app.get("/style.css")
    async def serve_css():
        return FileResponse(str(frontend_dir / "style.css"))
    
    @app.get("/app.js")
    async def serve_js():
        return FileResponse(str(frontend_dir / "app.js"))
