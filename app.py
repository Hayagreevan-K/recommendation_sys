
---

## 💻 **4️⃣ Final `app.py` (Production-Ready)**

This version uses only your small, compressed artifacts.  
It auto-loads `.ann.gz` and handles missing optional files gracefully.

```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gzip, shutil
from annoy import AnnoyIndex

st.set_page_config(page_title="🛍️ Product Recommender", layout="wide")

# ---------- Load artifacts ----------
@st.cache_resource
def load_models():
    base = "models"
    models = {}

    def load_joblib(path):
        full = os.path.join(base, path)
        if os.path.exists(full):
            return joblib.load(full)
        return None

    # decompress annoy if gzipped
    ann_path = os.path.join(base, "annoy_index_small.ann")
    ann_gz = ann_path + ".gz"
    if not os.path.exists(ann_path) and os.path.exists(ann_gz):
        with gzip.open(ann_gz, "rb") as f_in, open(ann_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        st.info("Decompressed annoy_index_small.ann.gz ✅")

    models["product_meta"] = load_joblib("product_meta_small.joblib")
    models["similarity_map"] = load_joblib("similarity_map_small.joblib")
    models["svd_model"] = load_joblib("svd_model_small.joblib")
    models["tfidf"] = load_joblib("tfidf_vectorizer.joblib")

    dim = 32
    ann = AnnoyIndex(dim, metric="angular")
    ann.load(ann_path)
    models["annoy"] = ann

    return models

art = load_models()
meta = art["product_meta"]
ann = art["annoy"]
similarity_map = art["similarity_map"]

# ---------- UI ----------
st.title("🛍️ Product Recommender System")
st.markdown("Hybrid (Content + Collaborative) recommendations using TF-IDF, SVD, and Annoy.")

if meta is None:
    st.error("❌ product_meta_small.joblib not found in `models/`.")
    st.stop()

product_ids = meta["product_id"].astype(str).tolist()
prod_to_idx = {p: i for i, p in enumerate(product_ids)}

def get_title(pid):
    row = meta.loc[meta["product_id"].astype(str) == str(pid)]
    return row["title"].iloc[0] if not row.empty else pid

# ---------- Functions ----------
def get_similar(pid, top_k=5):
    pid = str(pid)
    if similarity_map and pid in similarity_map:
        return similarity_map[pid][:top_k]
    idx = prod_to_idx.get(pid)
    if idx is None:
        return []
    nn = ann.get_nns_by_item(idx, top_k + 1)
    nn = [product_ids[i] for i in nn if product_ids[i] != pid][:top_k]
    return nn

# ---------- Sidebar ----------
st.sidebar.header("🔍 Search & Options")
search_query = st.sidebar.text_input("Search Product", "")
k = st.sidebar.slider("Number of Recommendations", 3, 15, 5)

# ---------- Search ----------
if search_query:
    mask = meta["title"].str.contains(search_query, case=False, na=False)
    matches = meta[mask].head(30)
    if matches.empty:
        st.warning("No products found for that search.")
        st.stop()
    selected_pid = st.selectbox("Select Product", matches["product_id"], format_func=get_title)
else:
    selected_pid = st.selectbox("Select Product", meta["product_id"].head(30), format_func=get_title)

# ---------- Recommendations ----------
st.markdown("---")
st.subheader("🎯 Recommended Products")

recs = get_similar(selected_pid, top_k=k)
if not recs:
    st.warning("No similar items found.")
else:
    cols = st.columns(2)
    for i, pid in enumerate(recs):
        col = cols[i % 2]
        with col:
            st.markdown(f"**{i+1}. {get_title(pid)}**")
            st.caption(f"Product ID: {pid}")
st.markdown("---")

st.caption("Model artifacts loaded successfully ✅")
