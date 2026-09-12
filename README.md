# 🎬 Netflix Content Segmentation

Groups Netflix titles into meaningful clusters using **unsupervised machine learning** — a K-Means algorithm built from scratch in **PyTorch**, served through a **FastAPI** backend, with an interactive **Streamlit** frontend for exploring clusters and predicting where a new title would land.

> Task 4 (Medium) of the **Auspify Internship Program**.

---

## 📖 Overview

Netflix's catalog doesn't come with built-in labels for "what kind of content is this, really." This project groups thousands of titles into natural clusters based on their type, genre, rating, release year, and duration — without ever telling the model what the "correct" groups are.

The core learning goal wasn't just to get clusters out of a black-box function, but to actually implement K-Means clustering by hand using PyTorch tensors, then turn that model into something usable: a real API and a real UI.

---

## ⚙️ How It Works

1. **Prepare features** — numeric columns (release year, duration) are scaled, and categorical columns (type, rating, primary genre) are one-hot encoded.
2. **Apply clustering** — a K-Means algorithm implemented directly with PyTorch (`torch.cdist`, `argmin`, `mean`) assigns every title to a cluster.
3. **Identify content groups** — each title receives a cluster label.
4. **Visualize clusters** — PCA reduces the feature space to 2D so clusters can be plotted.
5. **Interpret cluster characteristics** — each cluster is profiled by its dominant genre, type, rating, and average release year.

---

## 🧰 Tech Stack

| Layer          | Technology                        |
|----------------|------------------------------------|
| Model           | PyTorch (custom K-Means)          |
| Preprocessing | scikit-learn (StandardScaler, OneHotEncoder, PCA) |
| Backend        | FastAPI                             |
| Frontend       | Streamlit                            |
| Data           | pandas / NumPy                    |

---

## 📂 Project Structure

```
netflix-content-segmentation/
├── pytorch_model.py        # Feature engineering + PyTorch K-Means + training
├── fastapi_backend.py      # REST API serving the trained model
├── streamlit_frontend.py   # Interactive UI (cluster explorer + prediction form)
├── netflix_titles.csv      # Dataset
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/netflix-content-segmentation.git
cd netflix-content-segmentation
```

### 2. Install dependencies
```bash
pip install torch fastapi uvicorn streamlit scikit-learn pandas numpy requests
```

### 3. Train the model
```bash
python pytorch_model.py
```
This preprocesses `netflix_titles.csv`, trains the K-Means model, and saves everything the API needs into `artifacts.pkl`.

### 4. Start the backend
```bash
uvicorn fastapi_backend:app --reload --port 8000
```

### 5. Start the frontend (in a new terminal)
```bash
streamlit run streamlit_frontend.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`) to use the app.

> **Note:** the backend and frontend run as two separate live servers that talk to each other — run them on a regular machine (laptop, cloud IDE, VPS), not on a mobile-only Python environment.

---

## 🔌 API Endpoints

| Method | Endpoint     | Description                                  |
|--------|--------------|-----------------------------------------------|
| GET     | `/health`     | Checks the API is running and the model is loaded |
| GET     | `/clusters`   | Returns a profile summary for each cluster    |
| GET     | `/titles`      | Returns every title with its assigned cluster and 2D coordinates |
| POST    | `/predict`     | Takes a new title's attributes, returns its predicted cluster |

**Example request to `/predict`:**
```json
{
  "type": "Movie",
  "rating": "PG-13",
  "listed_in": "Dramas",
  "release_year": 2023,
  "duration": "110 min"
}
```

---

## 🧠 Skills Demonstrated

- K-Means clustering, implemented from first principles in PyTorch
- Data scaling and categorical encoding
- Cluster analysis and interpretation
- Turning an ML model into a served API
- Building a functional frontend around a live model

---

## 📌 About

Built as part of my self-directed journey toward becoming an AI engineer, and as Project 4 of the Auspify Internship Program. Follow my learning progress on [LinkedIn].
