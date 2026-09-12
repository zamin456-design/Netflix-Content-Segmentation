"""
fastapi_backend.py
===================
Serves the trained PyTorch K-Means model over HTTP.

Endpoints:
  GET  /health     - is the API alive, is the model loaded
  GET  /clusters    - profile of each cluster (size, dominant genre, avg year...)
  GET  /titles       - every title with its assigned cluster + PCA coords (for plotting)
  POST /predict       - given a new title's attributes, returns its predicted cluster

Run:  uvicorn fastapi_backend:app --reload --port 8000
(run `python pytorch_model.py` first so artifacts.pkl exists)
"""

import pickle

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pytorch_model import build_features, parse_duration  # reuse the exact same feature logic

ARTIFACTS_PATH = "artifacts.pkl"

app = FastAPI(title="Netflix Content Segmentation API")

# Lets the Streamlit app (a different port) call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    with open(ARTIFACTS_PATH, "rb") as f:
        artifacts = pickle.load(f)
    scaler = artifacts["scaler"]
    encoder = artifacts["encoder"]
    centroids = artifacts["centroids"]
    cluster_profiles = artifacts["cluster_profiles"]
    clustered_titles = artifacts["clustered_titles"]
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False


class TitleInput(BaseModel):
    type: str          # "Movie" or "TV Show"
    rating: str          # e.g. "TV-MA", "PG-13"
    listed_in: str        # primary genre, e.g. "Dramas"
    release_year: int
    duration: str          # e.g. "90 min" or "3 Seasons"


def _require_model():
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run `python pytorch_model.py` first to create artifacts.pkl.",
        )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_LOADED}


@app.get("/clusters")
def get_clusters():
    _require_model()
    return cluster_profiles


@app.get("/titles")
def get_titles():
    _require_model()
    cols = ["title", "type", "listed_in", "release_year", "rating", "cluster", "pca_x", "pca_y"]
    return clustered_titles[cols].to_dict(orient="records")


@app.post("/predict")
def predict_cluster(item: TitleInput):
    _require_model()

    raw = pd.DataFrame([{
        "type": item.type,
        "rating": item.rating,
        "listed_in": item.listed_in,
        "release_year": item.release_year,
        "duration": item.duration,
    }])
    features = build_features(raw)

    numeric = scaler.transform(features[["release_year", "duration_value"]])
    categorical = encoder.transform(features[["type", "rating", "listed_in"]])
    X = np.hstack([numeric, categorical]).astype(np.float32)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    distances = torch.cdist(X_tensor, centroids)
    cluster_id = int(torch.argmin(distances, dim=1).item())

    return {
        "cluster": cluster_id,
        "cluster_profile": cluster_profiles.get(cluster_id, {}),
    }
