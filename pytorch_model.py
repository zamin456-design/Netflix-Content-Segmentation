"""
pytorch_model.py
=================
Task 4 (Medium) — Netflix Content Segmentation.

Everything about the MODEL lives in this one file:
  - feature preparation (Step 1)
  - a K-Means clustering algorithm implemented directly in PyTorch (Step 2)
  - cluster assignment (Step 3)
  - PCA reduction to 2D so clusters can be plotted (Step 4 groundwork)
  - cluster interpretation / profiling (Step 5)

Running this file trains the model on netflix_titles.csv and saves everything
needed for the backend to reuse it (scaler, encoder, PCA, centroids, and the
clustered dataset) into artifacts.pkl.

Run:  python pytorch_model.py
"""

import re
import pickle
import json

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

DATA_PATH = "Dataset.csv"
ARTIFACTS_PATH = "artifacts.pkl"
N_CLUSTERS = 5


# ---------------------------------------------------------------------------
# Step 1: Prepare numerical and categorical features
# ---------------------------------------------------------------------------
def parse_duration(duration_str: str) -> float:
    """'90 min' -> 90.0, '3 Seasons' -> 3.0"""
    if not isinstance(duration_str, str):
        return 0.0
    match = re.search(r"(\d+)", duration_str)
    return float(match.group(1)) if match else 0.0


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raw Netflix columns -> a clean feature frame.
    Only 5 columns matter for clustering:
      type, rating, listed_in (primary genre), release_year, duration_value
    This function is reused by the backend at prediction time, so training
    and inference always compute features the exact same way.
    """
    out = pd.DataFrame()
    out["type"] = df["type"].fillna("Unknown")
    out["rating"] = df["rating"].fillna("Unrated")
    out["listed_in"] = df["listed_in"].apply(lambda g: str(g).split(",")[0].strip())
    out["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(2000)
    out["duration_value"] = df["duration"].apply(parse_duration)
    return out


def vectorize(features: pd.DataFrame, scaler: StandardScaler = None, encoder: OneHotEncoder = None):
    """
    Numeric columns -> scaled (mean 0, std 1).
    Categorical columns -> one-hot encoded.
    Concatenated into one numeric matrix a model can consume.
    If scaler/encoder are None, fits new ones (training time).
    If they're passed in, reuses them (prediction time) so a single new
    title is transformed into the exact same feature space as training data.
    """
    numeric_cols = ["release_year", "duration_value"]
    categorical_cols = ["type", "rating", "listed_in"]

    if scaler is None:
        scaler = StandardScaler()
        numeric_scaled = scaler.fit_transform(features[numeric_cols])
    else:
        numeric_scaled = scaler.transform(features[numeric_cols])

    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        categorical_encoded = encoder.fit_transform(features[categorical_cols])
    else:
        categorical_encoded = encoder.transform(features[categorical_cols])

    X = np.hstack([numeric_scaled, categorical_encoded]).astype(np.float32)
    return X, scaler, encoder


# ---------------------------------------------------------------------------
# Step 2: Apply clustering — K-Means written with PyTorch tensors
# ---------------------------------------------------------------------------
class TorchKMeans:
    """
    K-Means, implemented by hand with PyTorch (no sklearn.cluster).

    The algorithm:
      1. Pick K random data points as starting centroids.
      2. Assign every point to its nearest centroid (torch.cdist + argmin).
      3. Move each centroid to the mean of the points assigned to it.
      4. Repeat 2-3 until centroids barely move, or max_iter is reached.
    """

    def __init__(self, n_clusters=N_CLUSTERS, max_iter=200, tol=1e-4, seed=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.centroids = None

    def fit(self, X: torch.Tensor):
        torch.manual_seed(self.seed)
        n_samples = X.shape[0]

        init_idx = torch.randperm(n_samples)[: self.n_clusters]
        self.centroids = X[init_idx].clone()

        for _ in range(self.max_iter):
            distances = torch.cdist(X, self.centroids)          # (n_samples, k)
            labels = torch.argmin(distances, dim=1)               # (n_samples,)

            new_centroids = self.centroids.clone()
            for k in range(self.n_clusters):
                members = X[labels == k]
                if len(members) > 0:
                    new_centroids[k] = members.mean(dim=0)
                # else: keep the old centroid instead of collapsing to NaN

            shift = torch.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            if shift < self.tol:
                break

        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(X, self.centroids)
        return torch.argmin(distances, dim=1)


# ---------------------------------------------------------------------------
# Step 5: Interpret cluster characteristics
# ---------------------------------------------------------------------------
def summarize_clusters(df: pd.DataFrame, features: pd.DataFrame, labels: np.ndarray) -> dict:
    profiles = {}
    for k in sorted(set(labels)):
        mask = labels == k
        subset = features[mask]
        titles_subset = df[mask]
        profiles[int(k)] = {
            "size": int(mask.sum()),
            "avg_release_year": round(float(subset["release_year"].mean()), 1),
            "avg_duration_value": round(float(subset["duration_value"].mean()), 1),
            "most_common_type": subset["type"].mode().iloc[0],
            "most_common_genre": subset["listed_in"].mode().iloc[0],
            "most_common_rating": subset["rating"].mode().iloc[0],
            "sample_titles": titles_subset["title"].head(5).tolist(),
        }
    return profiles


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------
def main():
    print("Step 1: Preparing numerical and categorical features...")
    df = pd.read_csv(DATA_PATH)
    features = build_features(df)
    X, scaler, encoder = vectorize(features)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    print(f"Step 2: Applying PyTorch K-Means (k={N_CLUSTERS}) on {len(df)} titles...")
    kmeans = TorchKMeans(n_clusters=N_CLUSTERS)
    kmeans.fit(X_tensor)
    labels = kmeans.predict(X_tensor).numpy()

    print("Step 3: Identifying content groups...")
    profiles = summarize_clusters(df, features, labels)
    for cid, info in profiles.items():
        print(f"  Cluster {cid}: {info['size']} titles | "
              f"mostly {info['most_common_type']} / {info['most_common_genre']} "
              f"(~{info['avg_release_year']})")

    print("Step 4: Fitting PCA for 2D visualization...")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    print("Step 5: Cluster profiles saved.")
    result_df = df.copy()
    result_df["cluster"] = labels
    result_df["pca_x"] = coords[:, 0]
    result_df["pca_y"] = coords[:, 1]

    artifacts = {
        "scaler": scaler,
        "encoder": encoder,
        "pca": pca,
        "centroids": kmeans.centroids,          # torch.Tensor
        "cluster_profiles": profiles,
        "clustered_titles": result_df,          # pandas DataFrame
    }
    with open(ARTIFACTS_PATH, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"\nDone. Saved trained model + data to {ARTIFACTS_PATH}")


if __name__ == "__main__":
    main()
