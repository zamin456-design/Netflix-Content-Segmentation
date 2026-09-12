"""
streamlit_frontend.py
======================
UI only — all real logic lives in fastapi_backend.py. This file just sends
HTTP requests to the API and displays the results.

Run:  streamlit run streamlit_frontend.py
(make sure fastapi_backend.py is already running on http://localhost:8000)
"""

import requests
import pandas as pd
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Netflix Content Segmentation", layout="wide")
st.title("🎬 Netflix Content Segmentation")
st.caption("Unsupervised clustering of Netflix titles, powered by a PyTorch K-Means model.")


def api_get(path):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the backend at {API_URL}. Is it running? ({e})")
        return None


tab1, tab2 = st.tabs(["📊 Cluster Explorer", "🔮 Predict a New Title's Cluster"])

# ---------------- Tab 1: Explore existing clusters ----------------
with tab1:
    titles_data = api_get("/titles")
    profiles = api_get("/clusters")

    if titles_data and profiles:
        df = pd.DataFrame(titles_data)

        st.subheader("Cluster Visualization (PCA-reduced)")
        st.scatter_chart(df, x="pca_x", y="pca_y", color="cluster", size=20)

        st.subheader("What each cluster looks like")
        sorted_profiles = sorted(profiles.items(), key=lambda x: int(x[0]))
        cols = st.columns(len(sorted_profiles))
        for (cluster_id, info), col in zip(sorted_profiles, cols):
            with col:
                st.metric(f"Cluster {cluster_id}", f"{info['size']} titles")
                st.write(f"**Mostly:** {info['most_common_type']}")
                st.write(f"**Genre:** {info['most_common_genre']}")
                st.write(f"**Rating:** {info['most_common_rating']}")
                st.write(f"**Avg. year:** {info['avg_release_year']}")

        st.subheader("Browse titles by cluster")
        selected_cluster = st.selectbox("Choose a cluster", sorted(df["cluster"].unique()))
        st.dataframe(
            df[df["cluster"] == selected_cluster][["title", "type", "listed_in", "release_year", "rating"]],
            use_container_width=True,
        )

# ---------------- Tab 2: Predict a new title ----------------
with tab2:
    st.subheader("Enter a new title's attributes")

    col1, col2 = st.columns(2)
    with col1:
        content_type = st.selectbox("Type", ["Movie", "TV Show"])
        rating = st.selectbox(
            "Rating",
            ["TV-MA", "TV-14", "TV-PG", "R", "PG-13", "TV-Y7", "TV-Y", "PG", "TV-G", "NR"],
        )
        genre = st.selectbox(
            "Primary Genre",
            ["Dramas", "Comedies", "Action & Adventure", "Documentaries",
             "International TV Shows", "Children & Family Movies", "Crime TV Shows",
             "Kids' TV", "Stand-Up Comedy", "Horror Movies", "British TV Shows",
             "Docuseries", "Anime Series", "International Movies", "Reality TV"],
        )
    with col2:
        release_year = st.number_input("Release Year", min_value=1950, max_value=2026, value=2024)
        if content_type == "Movie":
            duration_val = st.number_input("Duration (minutes)", min_value=1, max_value=400, value=100)
            duration_str = f"{duration_val} min"
        else:
            duration_val = st.number_input("Number of Seasons", min_value=1, max_value=20, value=2)
            duration_str = f"{duration_val} Season{'s' if duration_val > 1 else ''}"

    if st.button("Predict Cluster", type="primary"):
        payload = {
            "type": content_type,
            "rating": rating,
            "listed_in": genre,
            "release_year": int(release_year),
            "duration": duration_str,
        }
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            r.raise_for_status()
            result = r.json()
            st.success(f"This title belongs to **Cluster {result['cluster']}**")
            st.json(result["cluster_profile"])
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the backend at {API_URL}. Is it running? ({e})")
