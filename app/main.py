from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import router

app = FastAPI(title="DAG QA")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api")

web_dist = Path(os.getenv("DAGQA_WEB_DIST", "app/web/dist"))
if web_dist.exists():
    app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
